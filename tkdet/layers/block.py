from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from .batch_norm import FrozenBatchNorm2d

__all__ = ["Block"]


class Block(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self
