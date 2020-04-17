import torch.nn as nn

__all__ = ["get_norm"]


def get_norm(norm, out_channels, **kwargs):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        eps = kwargs.get("eps", 1e-5)
        momentum = kwargs.get("momentum", 0.1)
        affine = kwargs.get("affine", True)
        track_running_stats = kwargs.get("track_running_stats", True)
        norm = {
            "BN": lambda x: nn.BatchNorm2d(x, eps, momentum, affine, track_running_stats),
            "GN": lambda x: nn.GroupNorm(32, x, eps, affine),
        }[norm]
    return norm(out_channels)
