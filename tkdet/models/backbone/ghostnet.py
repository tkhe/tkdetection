import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.layers import Conv2d
from tkdet.layers import SEModule
from tkdet.layers import make_divisible
from .base import Backbone
from .build import BACKBONE_REGISTRY

__all__ = ["GhostNet", "ghostnet_1_0"]


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        norm="BN",
        relu=True
    ):
        super().__init__()

        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = Conv2d(
            in_channels,
            init_channels,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            bias=False,
            norm=norm,
            activation="ReLU" if relu else ""
        )
        self.cheap_operation = Conv2d(
            init_channels,
            new_channels,
            dw_size,
            1,
            (dw_size - 1) // 2,
            groups=init_channels,
            bias=False,
            norm=norm,
            activation="ReLU" if relu else ""
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :,]


class GhostBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        dw_kernel_size=3,
        stride=1,
        se_ratio=0,
        norm="BN"
    ):
        super().__init__()

        self.use_se = se_ratio is not None and se_ratio > 0

        self.stride = stride

        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        if stride > 1:
            self.dw_conv = Conv2d(
                mid_channels,
                mid_channels,
                dw_kernel_size,
                stride,
                (dw_kernel_size - 1) // 2,
                groups=mid_channels,
                bias=False,
                norm=norm
            )

        if self.use_se:
            self.se = SEModule(
                mid_channels,
                se_ratio,
                activation=("ReLU", "HardSigmoid"),
                divisor=4
            )

        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        if (in_channels == out_channels and stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                Conv2d(
                    in_channels,
                    in_channels,
                    dw_kernel_size,
                    stride,
                    (dw_kernel_size - 1) // 2,
                    groups=in_channels,
                    bias=False,
                    norm=norm
                ),
                Conv2d(in_channels, out_channels, 1, bias=False, norm=norm)
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.stride > 1:
            x = self.dw_conv(x)

        if self.use_se:
            x = self.se(x)

        x = self.ghost2(x)

        x += self.shortcut(residual)

        return x


class GhostNet(Backbone):
    def __init__(
        self,
        ghostnet_cfg=None,
        multiplier=1.0,
        dropout=0.2,
        norm="BN",
        num_classes=1000,
        out_features=None
    ):
        super().__init__()

        if ghostnet_cfg is None:
            ghostnet_cfg = [
                [3,  16,  16, 0, 1],
                [3,  48,  24, 0, 2],
                [3,  72,  24, 0, 1],
                [5,  72,  40, 0.25, 2],
                [5, 120,  40, 0.25, 1],
                [3, 240,  80, 0, 2],
                [3, 200,  80, 0, 1],
                [3, 184,  80, 0, 1],
                [3, 184,  80, 0, 1],
                [3, 480, 112, 0.25, 1],
                [3, 672, 112, 0.25, 1],
                [5, 672, 160, 0.25, 2],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1]
            ]

        output_channel = make_divisible(16 * multiplier, 4)
        layers = []
        layers.append(Conv2d(3, output_channel, 3, 2, 1, bias=False, norm=norm, activation="ReLU"))
        self._out_feature_channels = {"0": output_channel}
        stride = 2
        self._out_feature_strides = {"0": stride}

        input_channel = output_channel
        block = GhostBottleneck
        index = 1
        for k, exp_size, c, se_ratio, s in ghostnet_cfg:
            output_channel = make_divisible(c * multiplier, 4)
            hidden_channel = make_divisible(exp_size * multiplier, 4)
            layers.append(
                block(input_channel, hidden_channel, output_channel, k, s, se_ratio)
            )
            input_channel = output_channel
            stride *= s
            self._out_feature_channels[str(index)] = output_channel
            self._out_feature_strides[str(index)] = stride
            index += 1

        output_channel = make_divisible(exp_size * multiplier, 4)
        layers.append(Conv2d(input_channel, output_channel, 1, norm=norm, activation="ReLU"))
        self._out_feature_channels[str(index)] = output_channel
        self._out_feature_strides[str(index)] = stride

        self.features = nn.Sequential(*layers)

        if not out_features:
            out_features = ["linear"]
        if "linear" in out_features and num_classes is not None:
            self.conv_head = Conv2d(input_channel, 1280, 1, activation="ReLU")
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(last_channel, num_classes)
            )
        self._out_features = out_features

    def forward(self, x):
        outputs = {}

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if str(idx) in self._out_features:
                outputs[str(idx)] = x

        if "linear" in self._out_features:
            x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            x = self.conv_head(x)
            x = self.classifier(x)
            outputs["linear"] = x

        return outputs


def _ghostnet(multiplier, cfg):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    norm = cfg.GHOSTNET.NORM
    return GhostNet(multiplier=multiplier, norm=norm, out_features=out_features)


@BACKBONE_REGISTRY.register("GhostNet-1.0")
def ghostnet_1_0(cfg):
    return _ghostnet(1.0, cfg)
