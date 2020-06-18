import inspect
import pprint
from abc import ABC
from abc import abstractmethod

import numpy as np
from fvcore.transforms.transform import Transform
from fvcore.transforms.transform import TransformList

__all__ = ["TransformGen", "apply_transform_gens"]


def check_dtype(img):
    assert isinstance(img, np.ndarray), \
        f"[TransformGen] Needs an numpy array, but got a {type(img)}!"
    assert not isinstance(img.dtype, np.integer) or img.dtype == np.uint8, \
        f"[TransformGen] Got image of type {img.dtype}, use uint8 or floating points instead!"
    assert img.ndim in [2, 3], img.ndim


class TransformGen(ABC):
    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD, \
                    "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )

                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


def apply_transform_gens(transform_gens, img):
    for g in transform_gens:
        assert isinstance(g, (Transform, TransformGen)), g

    check_dtype(img)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(img) if isinstance(g, TransformGen) else g
        assert isinstance(tfm, Transform), \
            f"TransformGen {g} must return an instance of Transform! Got {tfm} instead"

        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)
