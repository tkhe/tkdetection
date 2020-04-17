import datetime
import itertools
import logging
import os
import tempfile
import time
from collections import Counter

import torch
from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
from fvcore.nn.precise_bn import get_bn_modules
from fvcore.nn.precise_bn import update_bn_stats

import tkdet.utils.comm as comm
from tkdet.evaluation.testing import flatten_results_dict
from tkdet.utils.events import EventStorage
from tkdet.utils.events import EventWriter
from .train_loop import HookBase

__all__ = [
    "CallbackHook",
    "IterationTimer",
    "PeriodicWriter",
    "PeriodicCheckpointer",
    "LRScheduler",
    "AutogradProfiler",
    "EvalHook",
    "PreciseBN",
]


class CallbackHook(HookBase):
    def __init__(self, *, before_train=None, after_train=None, before_step=None, after_step=None):
        self._before_train = before_train
        self._before_step = before_step
        self._after_step = after_step
        self._after_train = after_train

    def before_train(self):
        if self._before_train:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:
            self._after_train(self.trainer)
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_step(self):
        if self._before_step:
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:
            self._after_step(self.trainer)


class IterationTimer(HookBase):
    def __init__(self, warmup_iter=3):
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


class PeriodicWriter(HookBase):
    def __init__(self, writers, period=20):
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (
            (self.trainer.iter + 1) % self._period == 0
            or self.trainer.iter == self.trainer.max_iter - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            writer.close()


class PeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        self.step(self.trainer.iter)


class LRScheduler(HookBase):
    def __init__(self, optimizer, scheduler):
        self._optimizer = optimizer
        self._scheduler = scheduler

        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self._scheduler.step()


class AutogradProfiler(HookBase):
    def __init__(self, enable_predicate, output_dir, *, use_cuda=True):
        self._enable_predicate = enable_predicate
        self._use_cuda = use_cuda
        self._output_dir = output_dir

    def before_step(self):
        if self._enable_predicate(self.trainer):
            self._profiler = torch.autograd.profiler.profile(use_cuda=self._use_cuda)
            self._profiler.__enter__()
        else:
            self._profiler = None

    def after_step(self):
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        PathManager.mkdirs(self._output_dir)
        out_file = os.path.join(
            self._output_dir,
            "profiler-trace-iter{}.json".format(self.trainer.iter)
        )
        if "://" not in out_file:
            self._profiler.export_chrome_trace(out_file)
        else:
            with tempfile.TemporaryDirectory(prefix="tkdet_profiler") as d:
                tmp_file = os.path.join(d, "tmp.json")
                self._profiler.export_chrome_trace(tmp_file)
                with open(tmp_file) as f:
                    content = f.read()
            with PathManager.open(out_file, "w") as f:
                f.write(content)


class EvalHook(HookBase):
    def __init__(self, eval_period, eval_function):
        self._period = eval_period
        self._func = eval_function

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(results, dict), \
                f"Eval function must return a dict. Got {results} instead."

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    )
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_eval()

    def after_train(self):
        del self._func


class PreciseBN(HookBase):
    def __init__(self, period, model, data_loader, num_iter):
        self._logger = logging.getLogger(__name__)
        if len(get_bn_modules(model)) == 0:
            self._logger.info(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
            )
            self._disabled = True
            return

        self._model = model
        self._data_loader = data_loader
        self._num_iter = num_iter
        self._period = period
        self._disabled = False

        self._data_iter = None

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self.update_stats()

    def update_stats(self):
        if self._disabled:
            return

        if self._data_iter is None:
            self._data_iter = iter(self._data_loader)

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    self._logger.info(
                        f"Running precise-BN ... {num_iter}/{self._num_iter} iterations."
                    )
                yield next(self._data_iter)

        with EventStorage():
            self._logger.info(
                "Running precise-BN for {} iterations...  ".format(self._num_iter)
                + "Note that this could produce different statistics every time."
            )
            update_bn_stats(self._model, data_loader(), self._num_iter)
