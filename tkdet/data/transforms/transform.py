import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms.transform import CropTransform
from fvcore.transforms.transform import HFlipTransform
from fvcore.transforms.transform import NoOpTransform
from fvcore.transforms.transform import Transform
from fvcore.transforms.transform import TransformList
from PIL import Image

try:
    import cv2
except ImportError:
    pass

__all__ = [
    "ExpandTransform",
    "ExtentTransform",
    "PhotoMetricDistortionTransform",
    "ResizeTransform",
    "ResizeWithPaddingTransform",
    "RotationTransform",
]


class ExtentTransform(Transform):
    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        super().__init__()

        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        ret = Image.fromarray(img).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords):
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(Transform):
    def __init__(self, h, w, new_h, new_w, interp=None):
        super().__init__()

        if interp is None:
            interp = Image.BILINEAR

        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4

        if img.dtype == np.uint8:
            pil_image = Image.fromarray(img)
            interp_method = interp if interp is not None else self.interp
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
        else:
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic"
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            img = F.interpolate(img, (self.new_h, self.new_w), mode=mode, align_corners=False)
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class ResizeWithPaddingTransform(Transform):
    def __init__(self, h, w, size, interplotion, value=None):
        super().__init__()

        if value is None:
            value = [124, 116, 104]

        self._set_attributes(locals())

    def apply_image(self, img, interplotion=None):
        h, w = img.shape[:2]
        size = self.size[0]
        if h > w:
            new_h = size
            new_w = int(size / h * w + 0.5)
            right = size - new_w
            bottom = 0
        else:
            new_h = int(size / w * h + 0.5)
            new_w = size
            right = 0
            bottom = size - new_h

        interplotion_method = interplotion if interplotion is not None else self.interplotion
        img = cv2.resize(img, (new_w, new_h), interpolation=interplotion_method)
        img = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=self.value)
        return img

    def apply_coords(self, coords):
        size = self.size[0]
        scale = size / max(self.h, self.w)
        coords[:, 0] = coords[:, 0] * scale
        coords[:, 1] = coords[:, 1] * scale
        return coords

    def apply_segmentation(self, segmentation):
        raise NotImplementedError


class RotationTransform(Transform):
    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        super().__init__()

        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle)))
        if expand:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        if not self.expand:
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h,
            self.bound_w,
            -self.angle,
            True,
            None,
            self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2,
            (rotation.bound_h - self.h) // 2,
            self.w,
            self.h
        )
        return TransformList([rotation, crop])


class PhotoMetricDistortionTransform(Transform):
    def __init__(
        self,
        image_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()

        assert img_format in ["BGR", "RGB"]

        self.is_rgb = img_format == "RGB"
        del img_format
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = self.brightness(img)
        if np.random.randint(2):
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if np.random.randint(2):
            return self.convert(
                img,
                beta=np.random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        if np.random.randint(2):
            return self.convert(img, alpha=np.random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        if np.random.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=np.random.uniform(self.saturation_low, self.saturation_high)
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if np.random.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + np.random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img


class ExpandTransform(Transform):
    def __init__(self, min_ratio=1, max_ratio=4, mean=(123.675, 116.28, 103.53), prob=0.5):
        super().__init__()

        self._set_attributes(locals())

        self.ratio = np.random.uniform(min_ratio, max_ratio)
        self.left = 0
        self.top = 0

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] + self.left
        coords[:, 1] = coords[:, 1] + self.top
        return coords

    def apply_image(self, img, interp=None):
        if np.random.uniform(0, 1) > self.prob:
            return img

        self.ratio = np.random.uniform(self.min_ratio, self.max_ratio)

        h, w, c = img.shape
        expand_img = np.full(
            (int(h * self.ratio), int(w * self.ratio), c),
            self.mean
        ).astype(img.dtype)
        self.left = int(np.random.uniform(0, w * self.ratio - w))
        self.top = int(np.random.uniform(0, h * self.ratio - h))
        expand_img[self.top:self.top + h, self.left:self.left + w] = img
        return expand_img
