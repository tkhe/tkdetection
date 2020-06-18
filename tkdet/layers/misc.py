import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple

__all__ = [
    "cat",
    "interpolate",
    "make_divisible",
]


def cat(tensors: List[torch.Tensor], dim: int = 0):
    assert isinstance(tensors, (list, tuple))

    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input,
            size,
            scale_factor,
            mode,
            align_corners=align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        return [math.floor(input.size(i + 2) * scale_factors[i]) for i in range(dim)]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)
