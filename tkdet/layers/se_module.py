import torch.nn as nn

from .conv2d import Conv2d

__all__ = ["SEModule"]


class SEModule(nn.Module):
    def __init__(
        self,
        in_channels,
        se_ratio=0.25,
        se_channels=None,
        activation=("Swish", "Sigmoid")
    ):
        super().__init__()

        if se_channels is None:
            se_channels = int(in_channels * se_ratio)

        if not isinstance(activation, (tuple, list)):
            assert isinstance(activation, str)

            activation = [activation, "Sigmoid"]

        assert len(activation) == 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(in_channels, se_channels, 1, activation=activation[0])
        self.conv2 = Conv2d(se_channels, in_channels, 1, activation=activation[1])

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out
