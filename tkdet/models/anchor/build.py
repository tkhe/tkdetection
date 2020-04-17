from typing import List

from tkdet.utils.registry import Registry

__all__ = [
    "ANCHOR_REGISTRY",
    "build_anchor_generator",
    "get_anchor_generator_list",
]

ANCHOR_REGISTRY = Registry("ANCHOR_GENERATOR")


def build_anchor_generator(cfg, input_shape):
    return ANCHOR_REGISTRY.get(cfg.MODEL.ANCHOR.NAME)(cfg, input_shape)
 

def get_anchor_generator_list() -> List[str]:
    return ANCHOR_REGISTRY.keys()
