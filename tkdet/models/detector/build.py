from typing import List

from tkdet.utils.registry import Registry
from .base import Detector

DETECTOR_REGISTRY = Registry("DETECTOR")


def build_model(cfg) -> Detector:
    return DETECTOR_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg)


def get_model_list() -> List[str]:
    return list(DETECTOR_REGISTRY.keys())
