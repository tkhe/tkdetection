from typing import List

from tkdet.utils.registry import Registry
from .base import Backbone

__all__ = [
    "BACKBONE_REGISTRY",
    "build_backbone",
    "get_backbone_list",
]

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_backbone(cfg) -> Backbone:
    backbone = BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)(cfg)

    assert isinstance(backbone, Backbone)

    return backbone


def get_backbone_list() -> List[str]:
    return list(BACKBONE_REGISTRY.keys())
