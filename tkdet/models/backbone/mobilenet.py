import torch.nn as nn
import torch.nn.functional as F

from tkdet.layers import get_norm
from tkdet.layers import make_divisible
from .base import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "MobileNetV2",
    "mobilenetv2_1_0",
    "mobilenetv2_0_75",
    "mobilenetv2_0_5",
    "mobilenetv2_0_25",
]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, norm="BN"):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False
            ),
            get_norm(norm, out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, norm="BN"):
        super().__init__()

        assert stride in (1, 2)
        self.stride = stride

        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, 1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                get_norm(norm, out_channels)
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(Backbone):
    """
    Implement MobileNet v2 (https://arxiv.org/abs/1801.04381).
    """

    def __init__(
        self,
        multiplier=1.0,
        inverted_residual_setting=None,
        block=InvertedResidual,
        norm="BN",
        out_features=None,
        num_classes=1000
    ):
        super().__init__()

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list."
            )

        input_channel = make_divisible(input_channel * multiplier)
        last_channel = make_divisible(last_channel * max(1.0, multiplier))
        features = [ConvBNReLU(3, input_channel, stride=2, norm=norm)]
        strides = 2
        self._out_feature_strides = {"0": strides}
        self._out_feature_channels = {"0": input_channel}
        index = 1
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                strides *= stride
                features.append(block(input_channel, output_channel, stride, t, norm))
                input_channel = output_channel
                self._out_feature_strides[str(index)] = strides
                self._out_feature_channels[str(index)] = output_channel
                index += 1
        features.append(ConvBNReLU(input_channel, last_channel, 1, norm=norm))
        self.features = nn.Sequential(*features)
        self._out_feature_channels[str(index)] = last_channel
        self._out_feature_strides[str(index)] = strides

        if not out_features:
            out_features = ["linear"]
        if "linear" in out_features and num_classes is not None:
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
            x = self.classifier(x)
            outputs["linear"] = x

        return outputs


def get_mobilenet_v2(multiplier, cfg):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    norm = cfg.MOBILENET_V2.NORM
    return MobileNetV2(multiplier, norm=norm, out_features=out_features)


@BACKBONE_REGISTRY.register("MobileNet-V2-1.0")
def mobilenetv2_1_0(cfg):
    return get_mobilenet_v2(1.0, cfg)


@BACKBONE_REGISTRY.register("MobileNet-V2-0.75")
def mobilenetv2_0_75(cfg):
    return get_mobilenet_v2(0.75, cfg)


@BACKBONE_REGISTRY.register("MobileNet-V2-0.5")
def mobilenetv2_0_5(cfg):
    return get_mobilenet_v2(0.5, cfg)


@BACKBONE_REGISTRY.register("MobileNet-V2-0.25")
def mobilenetv2_0_25(cfg):
    return get_mobilenet_v2(0.25, cfg)
