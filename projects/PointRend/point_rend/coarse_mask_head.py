import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import weight_init

from tkdet.layers import Conv2d
from tkdet.layers import ShapeSpec
from tkdet.models.roi_head.mask_head import MASK_HEAD_REGISTRY

__all__ = ["CoarseMaskHead"]


@MASK_HEAD_REGISTRY.register()
class CoarseMaskHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(CoarseMaskHead, self).__init__()

        self.num_classes = cfg.MODEL.NUM_CLASSES
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.fc_dim = cfg.MODEL.ROI_MASK_HEAD.FC_DIM
        num_fc = cfg.MODEL.ROI_MASK_HEAD.NUM_FC
        self.output_side_resolution = cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION
        self.input_channels = input_shape.channels
        self.input_h = input_shape.height
        self.input_w = input_shape.width

        self.conv_layers = []
        if self.input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(
                self.input_channels,
                conv_dim,
                kernel_size=1,
                activation="ReLU"
            )
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            activation="ReLU"
        )
        self.conv_layers.append(self.reduce_spatial_dim_conv)

        input_dim = conv_dim * self.input_h * self.input_w
        input_dim //= 4

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(input_dim, self.fc_dim)
            self.add_module("coarse_mask_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = self.fc_dim

        output_dim = self.num_classes * self.output_side_resolution * self.output_side_resolution

        self.prediction = nn.Linear(self.fc_dim, output_dim)
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer.conv)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, self.input_channels, self.input_h, self.input_w)
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        return self.prediction(x).view(
            N,
            self.num_classes,
            self.output_side_resolution,
            self.output_side_resolution
        )
