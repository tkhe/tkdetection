import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.layers import get_norm
from .base import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "ShuffleNetV2",
    "channel_shuffle",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


def channel_shuffle(x: torch.Tensor, groups: int):
    N, C, H, W = x.data.size()
    channels_per_group = C // groups

    x = x.view(N, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2,).contiguous()

    x = x.view(N, -1, H, W)

    return x


def depthwise_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=in_channels,
        bias=bias
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm="BN"):
        super().__init__()

        if not 1 <= stride <= 3:
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = out_channels // 2
        assert stride != 1 or in_channels == branch_features << 1

        if stride > 1:
            self.branch1 = nn.Sequential(
                depthwise_conv2d(in_channels, in_channels, 3, stride, 1),
                get_norm(norm, in_channels),
                nn.Conv2d(in_channels, branch_features, 1, bias=False),
                get_norm(norm, branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()
            in_channels = branch_features

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, 1, bias=False),
            get_norm(norm, branch_features),
            nn.ReLU(inplace=True),
            depthwise_conv2d(branch_features, branch_features, 3, stride, 1),
            get_norm(norm, branch_features),
            nn.Conv2d(branch_features, branch_features, 1, bias=False),
            get_norm(norm, branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(Backbone):
    """
    Implement ShuffleNet v2 (https://arxiv.org/abs/1807.11164).
    """

    def __init__(
        self,
        stages_repeats,
        stages_out_channels,
        inverted_residual=InvertedResidual,
        norm="BN",
        num_classes=1000,
        out_features=None
    ):
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stages_out_channels = stages_out_channels

        input_channels = 3
        output_channels = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            get_norm(norm, output_channels)
        )
        self._out_feature_strides = {"stem": 4}
        self._out_feature_channels = {"stem": output_channels}
        input_channels = output_channels

        stride = 4
        stage_names = [f"stage{i}" for i in (2, 3, 4)]
        for name, repeats, output_channels in zip(
            stage_names,
            stages_repeats,
            self._stages_out_channels[1:]
        ):
            seq = [inverted_residual(input_channels, output_channels, 2, norm)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1, norm))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
            self._out_feature_channels[name] = output_channels
            stride *= 2
            self._out_feature_strides[name] = stride

        output_channels = self._stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            get_norm(norm, output_channels)
        )
        self._out_feature_channels["conv5"] = output_channels
        self._out_feature_strides["conv5"] = stride

        if not out_features:
            out_features = ["linear"]
        if "linear" in out_features and num_classes is not None:
            self.fc = nn.Linear(output_channels, num_classes)
        self._out_features = out_features

    def forward(self, x):
        outputs = {}

        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        if "stem" in self._out_features:
            outputs["stem"] = x

        x = self.stage2(x)
        if "stage2" in self._out_features:
            outputs["stage2"] = x

        x = self.stage3(x)
        if "stage3" in self._out_features:
            outputs["stage3"] = x

        x = self.stage4(x)
        if "stage4" in self._out_features:
            outputs["stage4"] = x

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        if "conv5" in self._out_features:
            outputs["conv5"] = x
        
        if "linear" in self._out_features:
            x = x.mean([2, 3])
            x = self.fc(x)
            outputs["linear"] = x

        return outputs


def _shufflenet_v2(cfg, stages_repeats, stages_out_channels):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    norm = cfg.SHUFFLENET_V2.NORM
    return ShuffleNetV2(
        stages_repeats,
        stages_out_channels,
        norm=norm,
        out_features=out_features
    )


@BACKBONE_REGISTRY.register("ShuffleNet-V2-0.5")
def shufflenet_v2_x0_5(cfg):
    return _shufflenet_v2(cfg, [4, 8, 4], [24, 48, 96, 192, 1024])


@BACKBONE_REGISTRY.register("ShuffleNet-V2-1.0")
def shufflenet_v2_x1_0(cfg):
    return _shufflenet_v2(cfg, [4, 8, 4], [24, 116, 232, 464, 1024])


@BACKBONE_REGISTRY.register("ShuffleNet-V2-1.5")
def shufflenet_v2_x1_5(cfg):
    return _shufflenet_v2(cfg, [4, 8, 4], [24, 176, 352, 704, 1024])


@BACKBONE_REGISTRY.register("ShuffleNet-V2-2.0")
def shufflenet_v2_x2_0(cfg):
    return _shufflenet_v2(cfg, [4, 8, 4], [24, 244, 488, 976, 2048])
