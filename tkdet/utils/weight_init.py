import torch.nn as nn

from tkdet.layers import Conv2d

__all__ = ["c2_msra_fill", "c2_xavier_fill"]


def c2_msra_fill(module: nn.Module) -> None:
    if isinstance(module, Conv2d):
        c2_msra_fill(module.conv)
    else:
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def c2_xavier_fill(module: nn.Module) -> None:
    if isinstance(module, Conv2d):
        c2_xavier_fill(module.conv)
    else:
        nn.init.kaiming_uniform_(module.weight, a=1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
