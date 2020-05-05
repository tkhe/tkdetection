import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import weight_init

from tkdet.layers import Block
from tkdet.layers import FrozenBatchNorm2d
from tkdet.layers import get_norm
from .base import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet50_32x4d",
    "resnet101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, dilation, dilation, groups, False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)


class BasicBlock(Block):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm="BN"
    ):
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = get_norm(norm, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = get_norm(norm, out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(Block):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm="BN"
    ):
        super().__init__()

        width = int(out_channels * (base_width / 64)) * groups
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = get_norm(norm, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = get_norm(norm, width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = get_norm(norm, out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNet(Backbone):
    """
    Implement ResNet (https://arxiv.org/abs/1512.03385).
    """

    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm="",
        out_features=None,
        freeze_at=0,
        num_classes=1000
    ):
        super().__init__()

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.in_channels = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.norm = norm

        self.conv1 = nn.Conv2d(3, self.in_channels, 7, 2, 3, bias=False)
        self.bn1 = get_norm(norm, self.in_channels)
        self.layer1 = self._make_layers(block, 64, layers[0])
        self.layer2 = self._make_layers(block, 128, layers[1], 2, replace_stride_with_dilation[0])
        self.layer3 = self._make_layers(block, 256, layers[2], 2, replace_stride_with_dilation[1])
        self.layer4 = self._make_layers(block, 512, layers[3], 2, replace_stride_with_dilation[2])
        self._out_feature_channels = {
            "stem": 64,
            "layer1": 64 * block.expansion,
            "layer2": 128 * block.expansion,
            "layer3": 256 * block.expansion,
            "layer4": 512 * block.expansion
        }
        self._out_feature_strides = {
            "stem": 4,
            "layer1": 4,
            "layer2": 8,
            "layer3": 16,
            "layer4": 32,
        }

        if not out_features:
            out_features = ["linear"]
        if "linear" in out_features and num_classes is not None:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._out_features = out_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.freeze(freeze_at)

    def _make_layers(self, block, channels, blocks, stride=1, dilate=False):
        norm = self.norm
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.expansion, stride),
                get_norm(norm, channels * block.expansion)
            )

        layers = [
            block(
                self.in_channels,
                channels,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm
            )
        ]
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm=norm
                )
            )
        return nn.Sequential(*layers)

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.bn1.parameters():
                p.requires_grad = False
            for idx in range(1, freeze_at):
                module = getattr(self, f"layer{idx}")
                for block in module.children():
                    block.freeze()
        return self

    def forward(self, x):
        outputs = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        if "stem" in self._out_features:
            outputs["stem"] = x

        x = self.layer1(x)
        if "layer1" in self._out_features:
            outputs["layer1"] = x

        x = self.layer2(x)
        if "layer2" in self._out_features:
            outputs["layer2"] = x

        x = self.layer3(x)
        if "layer3" in self._out_features:
            outputs["layer3"] = x

        x = self.layer4(x)
        if "layer4" in self._out_features:
            outputs["layer4"] = x

        if "linear" in self._out_features:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.fc(x)
            outputs["linear"] = x

        return outputs


def get_resnet(cfg, block, layers, groups=1, width_per_group=64):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    norm = cfg.RESNET.NORM
    zero_init_residual = cfg.RESNET.ZERO_INIT_RESIDUAL
    replace_stride_with_dilation = cfg.RESNET.REPLACE_STRIDE_WITH_DILATION
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    model = ResNet(
        block,
        layers,
        zero_init_residual,
        groups,
        width_per_group,
        replace_stride_with_dilation,
        norm,
        out_features,
        freeze_at
    )
    return model


@BACKBONE_REGISTRY.register("ResNet-18")
def resnet18(cfg):
    return get_resnet(cfg, BasicBlock, [2, 2, 2, 2])


@BACKBONE_REGISTRY.register("ResNet-34")
def resnet34(cfg):
    return get_resnet(cfg, BasicBlock, [3, 4, 6, 3])


@BACKBONE_REGISTRY.register("ResNet-50")
def resnet50(cfg):
    return get_resnet(cfg, Bottleneck, [3, 4, 6, 3])


@BACKBONE_REGISTRY.register("ResNet-101")
def resnet101(cfg):
    return get_resnet(cfg, Bottleneck, [3, 4, 23, 3])


@BACKBONE_REGISTRY.register("ResNet-152")
def resnet152(cfg):
    return get_resnet(cfg, Bottleneck, [3, 8, 36, 3])


@BACKBONE_REGISTRY.register("ResNeXt-50-32x4d")
def resnet50_32x4d(cfg):
    return get_resnet(cfg, Bottleneck, [3, 4, 6, 3], 32, 4)


@BACKBONE_REGISTRY.register("ResNeXt-101-32x8d")
def resnet101_32x8d(cfg):
    return get_resnet(cfg, Bottleneck, [3, 4, 23, 3], 32, 8)


@BACKBONE_REGISTRY.register("Wide-ResNet-50-2")
def wide_resnet50_2(cfg):
    return get_resnet(cfg, Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2)


@BACKBONE_REGISTRY.register("Wide-ResNet-101-2")
def wide_resnet101_2(cfg):
    return get_resnet(cfg, Bottleneck, [3, 4, 23, 3], width_per_group=64 * 2)
