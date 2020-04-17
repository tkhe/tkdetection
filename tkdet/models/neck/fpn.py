import math

import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import weight_init

from tkdet.layers import Conv2d
from tkdet.utils.registry import Registry
from .base import Neck
from .build import NECK_REGISTRY

__all__ = [
    "FPN",
    "LastLevelMaxPool",
    "LastLevelP6P7",
    "build_fpn_neck",
    "build_fpn_top_block",
]

FPN_TOP_BLOCK_REGISTRY = Registry("FPN_TOP_BLOCK")


class FPN(Neck):
    """
    Implement Feature Pyramid Network (https://arxiv.org/abs/1612.03144).
    """

    def __init__(
        self,
        input_shape,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum"
    ):
        super().__init__()

        in_strides = [input_shape[f].stride for f in in_features]
        in_channels = [input_shape[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            lateral_conv = Conv2d(in_channel, out_channels, 1, bias=use_bias, norm=norm)
            output_conv = Conv2d(out_channels, out_channels, 3, 1, 1, bias=use_bias, norm=norm)
            stage = int(math.log2(in_strides[idx]))
            self.add_module(f"fpn_lateral{stage}", lateral_conv)
            self.add_module(f"fpn_output{stage}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in in_strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        if fuse_type not in ("avg", "sum"):
            raise ValueError(f"Unsupported fuse_type, got '{fuse_type}'")
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, bottom_up_features):
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:],
            self.lateral_convs[1:],
            self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results)

        return dict(zip(self._out_features, results))


@FPN_TOP_BLOCK_REGISTRY.register()
class LastLevelMaxPool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.in_feature = kwargs.get("in_feature", "p5")
        self.num_levels = 1

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2)]


@FPN_TOP_BLOCK_REGISTRY.register()
class LastLevelP6P7(nn.Module):
    def __init__(self, in_channels, out_channels, in_feature):
        super().__init__()

        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.in_feature = in_feature
        self.num_levels = 2

        for m in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(m)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


def _assert_strides_are_log2_contiguous(strides):
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], \
            f"Strides {stride} {strides[i - 1]} are not log2 contiguous"


def build_fpn_top_block(name, in_channels, out_channels, in_feature):
    return FPN_TOP_BLOCK_REGISTRY.get(name)(in_channels, out_channels, in_feature=in_feature)


@NECK_REGISTRY.register("FPN")
def build_fpn_neck(cfg, input_shape):
    in_features = cfg.FPN.IN_FEATURES
    out_channels = cfg.FPN.OUT_CHANNELS
    top_block_name = cfg.FPN.TOP_BLOCK.NAME
    norm = cfg.FPN.NORM
    top_block_in_feature = cfg.FPN.TOP_BLOCK.IN_FEATURE
    in_channels = (
        input_shape[top_block_in_feature].channels
        if top_block_in_feature != "p5"
        else out_channels
    )
    top_block = build_fpn_top_block(
        top_block_name,
        in_channels,
        out_channels,
        top_block_in_feature
    )
    fuse_type = cfg.FPN.FUSE_TYPE
    fpn = FPN(input_shape, in_features, out_channels, norm, top_block, fuse_type)
    return fpn
