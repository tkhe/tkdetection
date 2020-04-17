from abc import ABC

import torch.nn as nn

__all__ = ["Detector"]


class Detector(nn.Module, ABC):
    def __init__(self):
        super().__init__()

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
