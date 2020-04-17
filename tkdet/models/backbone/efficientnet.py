import math

import torch
import torch.nn as nn

from tkdet.layers import Conv2d
from tkdet.layers import SEModule
from tkdet.layers import make_divisible
from .base import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "EfficientNet",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        expand_ratio,
        kernel_size,
        stride,
        se_ratio,
        out_channels,
        norm="BN",
        activation="Swish"
    ):
        super().__init__()

        expand_channels = int(in_channels * expand_ratio)
        self.expand = expand_channels != in_channels
        if self.expand:
            self.expand_conv = Conv2d(
                in_channels,
                expand_channels,
                1,
                bias=False,
                norm=norm,
                activation=activation
            )
        padding = (kernel_size - 1) // 2
        self.dw = Conv2d(
            expand_channels,
            expand_channels,
            kernel_size,
            stride,
            padding,
            groups=expand_channels,
            bias=False,
            norm=norm,
            activation=activation
        )
        self.use_se = se_ratio > 0
        if self.use_se:
            self.se = SEModule(expand_channels, se_channels=int(in_channels * se_ratio))
        self.pw = Conv2d(expand_channels, out_channels, 1, bias=False, norm=norm)
        self.use_res = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x
        if self.expand:
            out = self.expand_conv(out)
        out = self.dw(out)
        if self.use_se:
            out = self.se(out)
        out = self.pw(out)
        if self.use_res:
            out = out + x
        return out


class EfficientStage(nn.Sequential):
    def __init__(
        self,
        in_channels,
        expand_ratio,
        kernel_size,
        stride,
        se_ratio,
        out_channels,
        depth,
        norm="BN",
        activation="Swish"
    ):
        blocks = []
        for i in range(depth):
            s = stride if i == 0 else 1
            ic = in_channels if i == 0 else out_channels
            blocks.append(
                MBConv(ic, expand_ratio, kernel_size, s, se_ratio, out_channels, norm, activation)
            )

        super().__init__(*blocks)


class EfficientNet(Backbone):
    """
    Implement EfficientNet (https://arxiv.org/abs/1905.11946).
    """

    def __init__(
        self,
        stem_channels,
        depths,
        widths,
        exp_ratios,
        se_ratio,
        strides,
        kernels,
        last_channels,
        norm="BN",
        activation="Swish",
        num_classes=1000,
        out_features=None
    ):
        super().__init__()

        stage_params = list(zip(depths, widths, exp_ratios, strides, kernels))

        self.stem = Conv2d(3, stem_channels, 3, 2, 1, bias=False, norm=norm, activation="Swish")
        self._out_feature_channels = {"stem": stem_channels}
        stride = 2
        self._out_feature_strides = {"stem": stride}
        prev_channels = stem_channels
        self.stages = ["stem"]
        for i, (depth, width, expand_ratio, s, k) in enumerate(stage_params):
            name = f"stage{i + 1}"
            stage = EfficientStage(
                prev_channels,
                expand_ratio,
                k,
                s,
                se_ratio,
                width,
                depth,
                norm,
                activation
            )
            self.add_module(name, stage)
            self.stages.append(name)
            prev_channels = width
            stride *= s
            self._out_feature_strides[name] = stride
            self._out_feature_channels[name] = width

        if not out_features:
            out_features = ["linear"]
        if "linear" in out_features and num_classes is not None:
            self.last_conv = Conv2d(
                prev_channels,
                last_channels,
                1,
                bias=False,
                norm="BN",
                activation=activation
            )
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(last_channels, num_classes)
        self._out_features = out_features

    def forward(self, x):
        outputs = {}

        for stage in self.stages:
            x = getattr(self, stage)(x)
            if stage in self._out_features:
                outputs[stage] = x

        if "linear" in self._out_features:
            x = self.last_conv(x)
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            outputs["linear"] = x

        return outputs


def get_efficientnet(depth_factor, width_factor, cfg):
    norm = cfg.EFFICIENTNET.NORM
    activation = cfg.EFFICIENTNET.ACTIVATION
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    stem_channels = make_divisible(int(32 * width_factor))
    depths = [math.ceil(d * depth_factor) for d in [1, 2, 2, 3, 3, 4, 1]]
    widths = [make_divisible(w * width_factor) for w in [16, 24, 40, 80, 112, 192, 320]]
    exp_ratios = [1, 6, 6, 6, 6, 6, 6]
    se_ratio = 0.25
    strides = [1, 2, 2, 2, 1, 2, 1]
    kernels = [3, 3, 5, 3, 5, 5, 3]
    last_channels = make_divisible(int(1280 * width_factor))
    return EfficientNet(
        stem_channels,
        depths,
        widths,
        exp_ratios,
        se_ratio,
        strides,
        kernels,
        last_channels,
        norm=norm,
        activation=activation,
        out_features=out_features
    )


@BACKBONE_REGISTRY.register("EfficientNet-B0")
def efficientnet_b0(cfg):
    return get_efficientnet(1.0, 1.0, cfg)


@BACKBONE_REGISTRY.register("EfficientNet-B1")
def efficientnet_b1(cfg):
    return get_efficientnet(1.1, 1.0, cfg)


@BACKBONE_REGISTRY.register("EfficientNet-B2")
def efficientnet_b2(cfg):
    return get_efficientnet(1.2, 1.1, cfg)


@BACKBONE_REGISTRY.register("EfficientNet-B3")
def efficientnet_b3(cfg):
    return get_efficientnet(1.4, 1.2, cfg)


@BACKBONE_REGISTRY.register("EfficientNet-B4")
def efficientnet_b4(cfg):
    return get_efficientnet(1.8, 1.4, cfg)


@BACKBONE_REGISTRY.register("EfficientNet-B5")
def efficientnet_b5(cfg):
    return get_efficientnet(2.2, 1.6, cfg)


@BACKBONE_REGISTRY.register("EfficientNet-B6")
def efficientnet_b6(cfg):
    return get_efficientnet(2.6, 1.8, cfg)


@BACKBONE_REGISTRY.register("EfficientNet-B7")
def efficientnet_b7(cfg):
    return get_efficientnet(3.1, 2.0, cfg)
