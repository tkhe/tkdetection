import logging

import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image
from PIL import ImageOps
from fvcore.common.file_io import PathManager

from tkdet.structures import BitMasks
from tkdet.structures import BoxMode
from tkdet.structures import Boxes
from tkdet.structures import Instances
from tkdet.structures import Keypoints
from tkdet.structures import PolygonMasks
from tkdet.structures import polygons_to_bitmask
from tkdet.utils.registry import Registry
from . import transforms as T
from .catalog import MetadataCatalog

TRANSFORM_REGISTRY = Registry("TRANSFORM")

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]


def convert_PIL_to_numpy(image, format):
    if format is not None:
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "L":
        image = np.expand_dims(image, -1)

    elif format == "BGR":
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def convert_image_to_rgb(image, format):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if format == "BGR":
        image = image[:, :, [2, 1, 0]]
    elif format == "YUV-BT.601":
        image = np.dot(image, np.array(_M_YUV2RGB).T)
        image = image * 255.0
    else:
        if format == "L":
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert("RGB"))
    return image


def read_image(file_name, format=None):
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        return convert_PIL_to_numpy(image, format)


def check_image_size(dataset_dict, image):
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise ValueError(
                "Mismatched (W,H){}, got {}, expect {}".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
            )

    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]


def transform_proposals(dataset_dict, image_shape, transforms, min_box_side_len, proposal_topk):
    if "proposal_boxes" in dataset_dict:
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(
            dataset_dict.pop("proposal_objectness_logits").astype("float32")
        )

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_side_len)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals


def transform_instance_annotations(
    annotation,
    transforms,
    image_size,
    *,
    keypoint_hflip_indices=None
):
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size

            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"],
            transforms,
            image_size,
            keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    return annotation


def transform_keypoint_annotations(keypoints, transforms, image_size, keypoint_hflip_indices=None):
    keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
    keypoints[:, :2] = transforms.apply_coords(keypoints[:, :2])

    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1

    if do_hflip:
        assert keypoint_hflip_indices is not None
        keypoints = keypoints[keypoint_hflip_indices, :]

    keypoints[keypoints[:, 2] == 0] = 0
    return keypoints


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, f"Expect segmentation of 2 dimensions, got {segm.ndim}."
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    assert by_box or by_mask

    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]


def create_keypoint_hflip_indices(dataset_names):
    check_metadata_consistency("keypoint_names", dataset_names)
    check_metadata_consistency("keypoint_flip_map", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    names = meta.keypoint_names
    flip_map = dict(meta.keypoint_flip_map)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices)


def gen_crop_transform_with_instance(crop_size, image_size, instance):
    crop_size = np.asarray(crop_size, dtype=np.int32)
    bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1], \
        "The annotation bounding box is outside of the image!"
    assert image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1], \
        "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names):
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                f"Metadata '{key}' for dataset '{dataset_names[idx]}' is '{str(entry)}'"
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key,
                    dataset_names[0],
                    str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))


def build_transform_gen(cfg, is_train):
    tfm_gens = TRANSFORM_REGISTRY.get(cfg.INPUT.TRANSFORM)(cfg, is_train)
    return tfm_gens


@TRANSFORM_REGISTRY.register("DefaultTransform")
def build_default_transform(cfg, is_train):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, \
            f"more than 2 ({len(min_size)}) min_size(s) are provided for ranges"

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


@TRANSFORM_REGISTRY.register("ResizeTransform")
def build_resize_transform(cfg, is_train):
    if is_train:
        size = cfg.INPUT.MAX_SIZE_TRAIN
    else:
        size = cfg.INPUT.MAX_SIZE_TEST

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.Resize(size))
    if is_train:
        tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


@TRANSFORM_REGISTRY.register("ResizeWithPaddingTransform")
def build_resize_with_padding_transform(cfg, is_train):
    if is_train:
        size = cfg.INPUT.MAX_SIZE_TRAIN
    else:
        size = cfg.INPUT.MAX_SIZE_TEST

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeWithPadding(size))
    if is_train:
        tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


@TRANSFORM_REGISTRY.register("SSDTransform")
def build_ssd_transform(cfg, is_train):
    logger = logging.getLogger(__name__)
    tfm_gens = []

    if is_train:
        size = cfg.INPUT.MAX_SIZE_TRAIN
        tfm_gens.append(T.PhotoMetricDistortion(cfg.INPUT.FORMAT))
        tfm_gens.append(T.Expand())
        tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training: " + str(tfm_gens))
    else:
        size = cfg.INPUT.MAX_SIZE_TEST
    tfm_gens.append(T.Resize(size))
    return tfm_gens
