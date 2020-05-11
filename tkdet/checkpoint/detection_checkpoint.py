from typing import Optional
from typing import List

from fvcore.common.checkpoint import Checkpointer

import tkdet.utils.comm as comm

__all__ = ["DetectionCheckpointer"]


class DetectionCheckpointer(Checkpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

        self.align_model_prefix = None

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> object:
        if path.startswith("torchvision://") or path.startswith("local://"):
            self.align_model_prefix = _align_models_prefix
        return super().load(path, checkpointables)

    def _load_file(self, filename):
        loaded = super()._load_file(filename)
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if self.align_model_prefix:
            self.align_model_prefix(checkpoint["model"])

        incompatible = super()._load_model(checkpoint)
        if incompatible is None:
            return None

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

def _align_models_prefix(state_dict):
    loaded_keys = sorted(state_dict.keys())

    for key in loaded_keys:
        newkey = "backbone." + key
        state_dict[newkey] = state_dict.pop(key)
