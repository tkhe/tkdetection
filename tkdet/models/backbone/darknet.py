import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.layers import Conv2d
from .base import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "DarkNet",
    "darknet53",
]


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm="BN",
        activation="LeakyReLU",
        **kwargs
    ):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=1,
            bias=False,
            norm=norm,
            activation=activation,
            **kwargs
        )
        self.conv2 = Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            bias=False,
            norm=norm,
            activation=activation,
            **kwargs
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class DarkNet(Backbone):
    def __init__(
        self,
        layers,
        channels,
        stem_channels=32,
        norm="BN",
        activation="LeakyReLU",
        out_features=None,
        num_classes=1000
    ):
        super().__init__()

        assert len(layers) == len(channels), \
            f"len(layers) should equal to len(channels), given {len(layers)} vs {len(channels)}"

        self.stem = Conv2d(3, stem_channels, 3, 1, bias=False, norm=norm, activation=activation)
        self.stage1 = _make_stage(layers[0], stem_channels, channels[0], norm, activation)
        self.stage2 = _make_stage(layers[1], channels[0], channels[1], norm, activation)
        self.stage3 = _make_stage(layers[2], channels[1], channels[2], norm, activation)
        self.stage4 = _make_stage(layers[3], channels[2], channels[3], norm, activation)
        self.stage5 = _make_stage(layers[4], channels[3], channels[4], norm, activation)
        self._out_feature_channels = {f"stage{i}": c for i, c in zip(range(1, 6), channels)}
        self._out_feature_strides = {f"stage{i}": 2 ** i for i in range(1, 6)}

        if not out_features:
            out_features = ["linear"]
        if "linear" in out_features and num_classes is not None:
            self.fc = nn.Linear(channels[4], num_classes)
        self._out_features = out_features

    def forward(self, x):
        outputs = {}

        x = self.stem(x)
        x = self.stage1(x)
        if "stage1" in self._out_features:
            outputs["stage1"] = x

        x = self.stage2(x)
        if "stage2" in self._out_features:
            outputs["stage2"] = x

        x = self.stage3(x)
        if "stage3" in self._out_features:
            outputs["stage3"] = x

        x = self.stage4(x)
        if "stage4" in self._out_features:
            outputs["stage4"] = x

        x = self.stage5(x)
        if "stage5" in self._out_features:
            outputs["stage5"] = x

        if "linear" in self._out_features:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.fc(x)
            outputs["linear"] = x

        return outputs


def _make_stage(layers, in_channels, out_channels, norm="BN", activation="LeakyReLU"):
    module = [
        Conv2d(in_channels, out_channels, 3, 2, bias=False, norm=norm, activation=activation)
    ]
    for _ in range(layers):
        module.append(BasicBlock(out_channels, out_channels, norm, activation))
    return nn.Sequential(*module)


def get_darknet(layers, channels, cfg):
    norm = cfg.DARKNET.NORM
    activation = cfg.DARKNET.ACTIVATION
    stem_channels = cfg.DARKNET.STEM_CHANNELS
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    return DarkNet(layers, channels, stem_channels, norm, activation, out_features)


@BACKBONE_REGISTRY.register("DarkNet-53")
def darknet53(cfg):
    return get_darknet([1, 2, 8, 8, 4], [64, 128, 256, 512, 1024], cfg)
