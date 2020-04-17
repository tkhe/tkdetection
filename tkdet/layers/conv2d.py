import torch.nn as nn
from fvcore.nn import weight_init

from .activation import get_activation
from .batch_norm import get_norm

__all__ = ["Conv2d", "DepthwiseSeparableConv2d"]


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="auto",
        dilation=1,
        groups=1,
        bias=True,
        norm="",
        activation="",
        init_method=None,
        **kwargs
    ):
        super().__init__()

        if padding == "auto":
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )
        self.norm = get_norm(norm, out_channels, **kwargs)
        inplace = kwargs.get("inplace", True)
        self.activation = get_activation(activation, inplace)

        if init_method is None:
            weight_init.c2_msra_fill(self.conv)
        else:
            init_method(self.conv)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def fuse_conv_norm(self):
        raise NotImplementedError


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding='auto',
        dilation=1,
        bias=False,
        norm="",
        activation="ReLU6",
        **kwargs
    ):
        super().__init__()

        self.dw = Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            bias,
            norm=norm,
            activation=activation
            **kwargs
        )
        last_activation = kwargs.get("last_activation", "")
        self.pw = Conv2d(
            in_channels,
            out_channels,
            1,
            bias=bias,
            norm=norm,
            activation=last_activation
        )

    def forward(self):
        x = self.dw(x)
        x = self.pw(x)
        return x
