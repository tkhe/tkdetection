import logging
import os

from fvcore.common.file_io import PathHandler
from fvcore.common.file_io import PathManager


class ModelZooURL(object):
    TORCHVISION_PREFIX = "https://download.pytorch.org/models"
    LOCAL_PREFIX = os.path.join(os.path.expanduser('~'), ".torch/fvcore_cache/models")

    IMAGENET_PRETRAINED_MODELS = {
        "ResNet-18": "resnet18-5c106cde.pth",
        "ResNet-34": "resnet34-333f7ec4.pth",
        "ResNet-50": "resnet50-19c8e357.pth",
        "ResNet-101": "resnet101-5d3b4d8f.pth",
        "ResNet-152": "resnet152-b121ed2d.pth",
        "ResNeXt-50-32x4d": "resnext50_32x4d-7cdf4587.pth",
        "ResNeXt-101-32x8d": "resnext101_32x8d-8ba56ff5.pth",
        "Wide-ResNet-50-2": "wide_resnet50_2-95faca4d.pth",
        "Wide-ResNet-101-2": "wide_resnet101_2-32ee1156.pth",
        "ShuffleNet-V2-0.5": "shufflenetv2_x0.5-f707e7126e.pth",
        "ShuffleNet-V2-1.0": "shufflenetv2_x1-5666bf0f80.pth",
        "MobileNet-V2-1.0": "mobilenet_v2-b0353104.pth",
    }

    @staticmethod
    def get(prefix, name):
        if prefix.startswith("torchvision://"):
            prefix = ModelZooURL.TORCHVISION_PREFIX
        elif prefix.startswith("local://"):
            prefix = ModelZooURL.LOCAL_PREFIX
        name = ModelZooURL.IMAGENET_PRETRAINED_MODELS[name]
        url = "/".join([prefix, name])
        return url


class TorchvisionModelCatalogHandler(PathHandler):
    PREFIX = "torchvision://"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        logger = logging.getLogger(__name__)
        catalog_path = ModelZooURL.get(self.PREFIX, path[len(self.PREFIX) :])
        logger.info("Catalog entry {} points to {}".format(path, catalog_path))
        return PathManager.get_local_path(catalog_path)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


class LocalModelCatalogHandler(PathHandler):
    PREFIX = "local://"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        logger = logging.getLogger(__name__)
        catalog_path = ModelZooURL.get(self.PREFIX, path[len(self.PREFIX) :])
        logger.info("Catalog entry {} points to {}".format(path, catalog_path))
        return PathManager.get_local_path(catalog_path)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(TorchvisionModelCatalogHandler())
PathManager.register_handler(LocalModelCatalogHandler())
