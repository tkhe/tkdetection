from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import weight_init

from tkdet.config import configurable
from tkdet.layers import Conv2d
from tkdet.layers import ShapeSpec
from tkdet.utils.registry import Registry

__all__ = ["BOX_HEAD_REGISTRY", "build_box_head"]

BOX_HEAD_REGISTRY = Registry("BOX_HEAD")


@BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        conv_dims: List[int],
        fc_dims: List[int],
        conv_norm=""
    ):
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                3,
                bias=(conv_norm == ""),
                norm=conv_norm,
                activation="ReLU",
            )
            self.add_module(f"conv{k + 1}", conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
        }

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_shape(self):
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


def build_box_head(cfg, input_shape):
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
