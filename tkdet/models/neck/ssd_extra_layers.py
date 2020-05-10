from typing import Dict

import torch.nn as nn

from tkdet.layers import Conv2d
from tkdet.layers import L2Norm
from tkdet.layers import ShapeSpec
from .base import Neck
from .build import NECK_REGISTRY

__all__ = ["VGGExtraLayers", "build_ssd_extra_layers"]


class VGGExtraLayers(Neck):
    extra_setting = {
        300:  [
            ((256, 1, 1, 0), (512, 3, 2, 1)),
            ((128, 1, 1, 0), (256, 3, 2, 1)),
            ((128, 1, 1, 0), (256, 3, 1, 0)),
            ((128, 1, 1, 0), (256, 3, 1, 0))
        ],
        512: [
            ((256, 1, 1, 0), (512, 3, 2, 1)),
            ((128, 1, 1, 0), (256, 3, 2, 1)),
            ((128, 1, 1, 0), (256, 3, 2, 1)),
            ((128, 1, 1, 0), (256, 3, 2, 1)),
            ((128, 1, 1, 0), (256, 4, 1, 1))
        ],
    }

    def __init__(
        self,
        cfg,
        input_shape: Dict[str, ShapeSpec],
        norm=""
    ):
        super().__init__()

        input_size = cfg.SSD.SIZE
        extra_cfg = self.extra_setting[input_size]
        in_channels = 1024

        layers = []
        previous_features = [f for f in input_shape]
        self._out_features = [f for f in input_shape]
        previous_channels = [input_shape[f].channels for f in input_shape]
        extra_channels = []
        for i, config in enumerate(extra_cfg):
            name = f"extra_{i + 1}"
            module = []
            for c, k, s, p in config:
                module.append(Conv2d(in_channels, c, k, s, p, norm=norm, activation="ReLU"))
                in_channels = c
            self.add_module(name, nn.Sequential(*module))
            self._out_features.append(name)
            extra_channels.append(in_channels)
        self._out_feature_strides = dict(
            zip(self._out_features, cfg.SSD.STRIDES)
        )
        self._out_feature_channels = dict(
            zip(self._out_features, previous_channels + extra_channels)
        )
        self.l2_norm = L2Norm(previous_channels[0])

    @property
    def size_divisibility(self):
        return -1

    def forward(self, x):
        outputs = x

        x = [x[f] for f in x]
        out = x[-1]

        for stage in self._out_features[2:]:
            out = getattr(self, stage)(out)
            outputs[stage] = out

        return outputs


@NECK_REGISTRY.register("VGGExtraLayers")
def build_ssd_extra_layers(cfg, input_shape):
    return VGGExtraLayers(cfg, input_shape, cfg.VGG.NORM)
