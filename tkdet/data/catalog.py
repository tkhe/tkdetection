import copy
import logging
import types
from typing import List

from tkdet.utils.logger import log_first_n

__all__ = ["DatasetCatalog", "MetadataCatalog", "Metadata"]


class DatasetCatalog(object):
    _REGISTERED = {}

    @staticmethod
    def register(name, func):
        assert callable(func), "You must register a function with `DatasetCatalog.register`!"
        assert name not in DatasetCatalog._REGISTERED, f"Dataset '{name}' is already registered!"

        DatasetCatalog._REGISTERED[name] = func

    @staticmethod
    def get(name):
        try:
            f = DatasetCatalog._REGISTERED[name]
        except KeyError:
            raise KeyError(
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    name,
                    ", ".join(DatasetCatalog._REGISTERED.keys())
                )
            )
        return f()

    @staticmethod
    def list() -> List[str]:
        return list(DatasetCatalog._REGISTERED.keys())

    @staticmethod
    def clear():
        DatasetCatalog._REGISTERED.clear()


class Metadata(types.SimpleNamespace):
    name: str = "N/A"

    _RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes",
    }

    def __getattr__(self, key):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            return getattr(self, self._RENAMED[key])

        raise AttributeError(
            "Attribute '{}' does not exist in the metadata of '{}'. Available keys are {}.".format(
                key,
                self.name,
                str(self.__dict__.keys())
            )
        )

    def __setattr__(self, key, val):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            setattr(self, self._RENAMED[key], val)

        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(key, self.name, oldval, val)
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class MetadataCatalog(object):
    _NAME_TO_META = {}

    @staticmethod
    def get(name):
        assert len(name)

        if name in MetadataCatalog._NAME_TO_META:
            ret = MetadataCatalog._NAME_TO_META[name]
            return ret
        else:
            m = MetadataCatalog._NAME_TO_META[name] = Metadata(name=name)
            return m

    @staticmethod
    def list():
        return list(MetadataCatalog._NAME_TO_META.keys())
