import inspect
import pprint
import sys
from abc import ABC

import numpy as np
from PIL import Image
from fvcore.transforms.transform import BlendTransform
from fvcore.transforms.transform import CropTransform
from fvcore.transforms.transform import HFlipTransform
from fvcore.transforms.transform import NoOpTransform
from fvcore.transforms.transform import Transform
from fvcore.transforms.transform import TransformList
from fvcore.transforms.transform import VFlipTransform

from .transform import ExtentTransform, ResizeTransform, RotationTransform

__all__ = [
    "RandomApply",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "RandomRotation",
    "Resize",
    "ResizeShortestEdge",
    "TransformGen",
    "apply_transform_gens",
]


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

    @classmethod
    def get_transform(cls, img):
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


class RandomApply(TransformGen):
    def __init__(self, transform, prob=0.5):
        super().__init__()

        assert isinstance(transform, (Transform, TransformGen)), \
            f"The given transform must either be a Transform or TransformGen instance. "
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0, got {prob})"

        self.prob = prob
        self.transform = transform

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            if isinstance(self.transform, TransformGen):
                return self.transform.get_transform(img)
            else:
                return self.transform
        else:
            return NoOpTransform()


class RandomFlip(TransformGen):
    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(TransformGen):
    def __init__(self, shape, interp=Image.BILINEAR):
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img):
        return ResizeTransform(
            img.shape[0],
            img.shape[1],
            self.shape[0],
            self.shape[1],
            self.interp
        )


class ResizeShortestEdge(TransformGen):
    def __init__(
        self,
        short_edge_length,
        max_size=sys.maxsize,
        sample_style="range",
        interp=Image.BILINEAR
    ):
        super().__init__()

        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)


class RandomRotation(TransformGen):
    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        super().__init__()

        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class RandomCrop(TransformGen):
    def __init__(self, crop_type: str, crop_size):
        super().__init__()

        assert crop_type in ["relative_range", "relative", "absolute"]

        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, f"Shape computation in {self} has bugs."

        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError(f"Unknown crop type {self.crop_type}")


class RandomExtent(TransformGen):
    def __init__(self, scale_range, shift_range):
        super().__init__()

        self._init(locals())

    def get_transform(self, img):
        img_h, img_w = img.shape[:2]

        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )


class RandomContrast(TransformGen):
    def __init__(self, intensity_min, intensity_max):

        super().__init__()

        self._init(locals())

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(TransformGen):
    def __init__(self, intensity_min, intensity_max):
        super().__init__()

        self._init(locals())

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(TransformGen):
    def __init__(self, intensity_min, intensity_max):
        super().__init__()

        self._init(locals())

    def get_transform(self, img):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(TransformGen):
    def __init__(self, scale):
        super().__init__()

        self._init(locals())
        self.eigen_vecs = np.array(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203]
            ]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, img):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals),
            src_weight=1.0,
            dst_weight=1.0
        )


def apply_transform_gens(transform_gens, img):
    for g in transform_gens:
        assert isinstance(g, TransformGen), g

    check_dtype(img)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(img)
        assert isinstance(tfm, Transform), \
            f"TransformGen {g} must return an instance of Transform! Got {tfm} instead"
        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)
