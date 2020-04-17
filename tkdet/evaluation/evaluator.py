import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch

from tkdet.utils.comm import get_world_size
from tkdet.utils.comm import is_main_process
from tkdet.utils.logger import log_every_n_seconds

__all__ = [
    "DatasetEvaluator",
    "DatasetEvaluators",
    "inference_context",
    "inference_on_dataset",
]


class DatasetEvaluator(object):
    def reset(self):
        pass

    def process(self, inputs, outputs):
        pass

    def evaluate(self):
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        super().__init__()

        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert k not in results, \
                        f"Different evaluators produce results with the same key {k}"
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)
    if evaluator is None:
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1,
                        total,
                        seconds_per_img,
                        str(eta)
                    ),
                    n=5,
                )

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str,
            total_time / (total - num_warmup),
            num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices
        )
    )

    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
