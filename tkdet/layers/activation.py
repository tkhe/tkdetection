import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Sigmoid",
    "Swish",
    "get_activation",
    "memory_efficient_swish",
    "sigmoid",
    "swish",
]


def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


def swish(x, inplace: bool = False):
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()

        self.inplace = inplace

    def forward(self, x):
        return sigmoid(x, self.inplace)

    def extra_repr(self):
        return "inplace=True" if self.inplace else ""


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()

        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)

    def extra_repr(self):
        return "inplace=True" if self.inplace else ""


class _MemoryEfficientSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.mul(torch.sigmoid(input))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


memory_efficient_swish = _MemoryEfficientSwish.apply


class MemoryEfficientSwish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return memory_efficient_swish(x)


def get_activation(activation, inplace=False):
    if isinstance(activation, str):
        if len(activation) == 0:
            return None
        activation = {
            "ReLU": nn.ReLU(inplace),
            "relu": lambda x: F.relu(x, inplace),
            "ReLU6": nn.ReLU6(inplace),
            "relu6": lambda x: F.relu6(x, inplace),
            "Sigmoid": Sigmoid(inplace),
            "sigmoid": lambda x: sigmoid(x, inplace),
            "Swish": Swish(inplace),
            "swish": lambda x: swish(x, inplace),
            "MemoryEfficientSwish": MemoryEfficientSwish(),
            "memory_efficient_swish": memory_efficient_swish,
            "LeakyReLU": nn.LeakyReLU(0.1, inplace),
            "leaky_relu": lambda x: F.leaky_relu(x, 0.1, inplace),
        }[activation]
    return activation
