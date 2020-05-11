import torch.nn as nn

from tkdet.layers import Conv2d
from .base import Backbone
from .build import BACKBONE_REGISTRY

__all__ = ["VGG", "vgg16"]


class VGG(Backbone):
    def __init__(self, vgg_cfg, norm="", out_features=None):
        super().__init__()

        self._out_feature_channels = {}
        self._out_feature_strides = {}

        layers = []
        in_channels = 3
        idx = 0
        stride = 1
        for v in vgg_cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
                stride *= 2
            else:
                layers.append(Conv2d(in_channels, v, 3, 1, 1, norm=norm, activation="ReLU"))
                in_channels = v
            self._out_feature_channels[str(idx)] = v
            self._out_feature_strides[str(idx)] = stride
            idx += 1

        layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self._out_feature_channels[str(idx)] = in_channels
        self._out_feature_strides[str(idx)] = stride
        idx += 1
        layers.append(
            Conv2d(in_channels, 1024, 3, padding=6, dilation=6, norm=norm, activation="ReLU")
        )
        self._out_feature_channels[str(idx)] = 1024
        self._out_feature_strides[str(idx)] = stride
        idx += 1
        layers.append(Conv2d(1024, 1024, 1, norm=norm, activation="ReLU"))
        self._out_feature_channels[str(idx)] = 1024
        self._out_feature_strides[str(idx)] = stride

        self.features = nn.Sequential(*layers)

        self._out_features = out_features

    def forward(self, x):
        outputs = {}

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if str(idx) in self._out_features:
                outputs[str(idx)] = x
        return outputs


vgg_cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
}


def _vgg(cfg, vgg_type):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    norm = cfg.VGG.NORM
    return VGG(vgg_cfg[vgg_type], norm=norm, out_features=out_features)


@BACKBONE_REGISTRY.register("VGG-16")
def vgg16(cfg):
    return _vgg(cfg, "D")


@BACKBONE_REGISTRY.register("VGG-19")
def vgg19(cfg):
    return _vgg(cfg, "E")
