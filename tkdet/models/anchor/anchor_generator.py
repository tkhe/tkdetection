import math
from typing import List

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from tkdet.config import configurable
from tkdet.layers import ShapeSpec
from tkdet.structures import Boxes
from .build import ANCHOR_REGISTRY

__all__ = ["DefaultAnchorGenerator", "SSDAnchorGenerator"]


class BufferList(nn.Module):
    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(size: List[int], stride: int, offset: float, device: torch.device):
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride,
        grid_width * stride,
        step=stride,
        dtype=torch.float32,
        device=device
    )
    shifts_y = torch.arange(
        offset * stride,
        grid_height * stride,
        step=stride,
        dtype=torch.float32,
        device=device
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def _broadcast_params(params, num_features, name):
    assert isinstance(params, (list, tuple)), \
        f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"

    if not isinstance(params[0], (list, tuple)):
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        "but the number of input features is {num_features}!"
    )

    return params


@ANCHOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    box_dim: int = 4

    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
        """
        NOTE: This interface is experimental.
        """
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)
        assert 0.0 <= offset < 1.0, offset

        self.offset = offset

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "sizes": cfg.MODEL.ANCHOR.SIZES,
            "aspect_ratios": cfg.MODEL.ANCHOR.ASPECT_RATIOS,
            "strides": [x.stride for x in input_shape],
            "offset": cfg.MODEL.ANCHOR.OFFSET,
        }

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]

        return BufferList(cell_anchors)

    @property
    def num_cell_anchors(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features):
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)

        return [Boxes(x) for x in anchors_over_all_feature_maps]


@ANCHOR_REGISTRY.register()
class SSDAnchorGenerator(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        self.strides = cfg.SSD.STRIDES
        sizes = cfg.MODEL.ANCHOR.SIZES
        self.min_sizes = sizes[:-1]
        self.max_sizes = sizes[1:]
        self.aspect_ratios = cfg.MODEL.ANCHOR.ASPECT_RATIOS
        self.offset = cfg.MODEL.ANCHOR.OFFSET

        self.cell_anchors = self._calculate_anchors(
            self.min_sizes,
            self.max_sizes,
            self.aspect_ratios
        )

    @property
    def num_cell_anchors(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def _calculate_anchors(self, min_sizes, max_sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(min_size, max_size, a)
            for min_size, max_size, a in zip(min_sizes, max_sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    def generate_cell_anchors(self, min_size, max_size, aspect_ratios):
        anchors = []
        ratios = [1]
        for r in aspect_ratios:
            ratios += [r, 1 / r]

        base_size = min_size
        for r in ratios:
            w = base_size * math.sqrt(r)
            h = base_size / math.sqrt(r)
            x0, y0, x1, y1 = - w / 2.0, - h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])

        base_size = math.sqrt(min_size * max_size)
        w = h = base_size
        x0, y0, x1, y1 = - w / 2.0, - h / 2.0, w / 2.0, h / 2.0
        anchors.insert(1, [x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features):
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)

        return [Boxes(x) for x in anchors_over_all_feature_maps]
