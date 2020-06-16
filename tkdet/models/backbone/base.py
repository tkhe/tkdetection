from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from tkdet.layers import ShapeSpec

__all__ = ["Backbone"]


class Backbone(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def size_divisibility(self):
        return 0

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @abstractmethod
    def freeze(self, freeze_at=0):
        raise NotImplementedError
