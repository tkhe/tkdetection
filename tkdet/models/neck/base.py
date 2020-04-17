from abc import ABC

import torch.nn as nn

from tkdet.layers import ShapeSpec

__all__ = ["Neck", "NeckSequential"]


class Neck(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def forward(cls, bottom_up_features):
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


class NeckSequential(Neck):
    def __init__(self, *args):
        assert all(isinstance(m, Neck) for m in args)

        super().__init__()

        self.sequential = nn.Sequential(*args)

    @property
    def size_divisibility(self):
        return self.sequential[0].size_divisibility

    def output_shape(self):
        return self.sequential[-1].output_shape()

    def forward(self, x):
        return self.sequential(x)
