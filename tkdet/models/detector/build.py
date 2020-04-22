from typing import List

import torch
from tkdet.utils.registry import Registry
from .base import Detector

DETECTOR_REGISTRY = Registry("DETECTOR")


def build_model(cfg) -> Detector:
    model = DETECTOR_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model


def get_model_list() -> List[str]:
    return list(DETECTOR_REGISTRY.keys())
