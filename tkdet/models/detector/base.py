from abc import ABC

import torch
import torch.nn as nn

__all__ = ["Detector"]


class Detector(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()

        self.register_buffer("pixel_mean", torch.Tensor(cfg.INPUT.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.INPUT.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    @classmethod
    def forward(cls, batched_inputs):
        pass

    @classmethod
    def losses(cls):
        pass

    @classmethod
    def inference(cls):
        pass

    def preprocess_inputs(self, batched_inputs):
        raise NotImplementedError
