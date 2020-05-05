from typing import List

from tkdet.utils.registry import Registry
from .base import Neck
from typing import Union

__all__ = [
    "NECK_REGISTRY",
    "build_neck",
    "get_neck_list",
]

NECK_REGISTRY = Registry("NECK")


def build_neck(cfg, input_shape) -> Union[Neck, None]:
    if not cfg.MODEL.NECK.ENABLE:
        return None

    neck = NECK_REGISTRY.get(cfg.MODEL.NECK.NAME)(cfg, input_shape)
    assert isinstance(neck, Neck)

    return neck


def get_neck_list() -> List[str]:
    return list(NECK_REGISTRY.keys())
