import os

import numpy as np
from PIL import Image
from fvcore.common.file_io import PathManager

from tkdet.data import DatasetCatalog
from tkdet.data import MetadataCatalog
from tkdet.structures import BoxMode

__all__ = ["load_vis_instances", "register_visdrone"]

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]


def load_vis_instances(dirname: str, split: str):
    assert split in ("train", "val", "test")
    if split == "test":
        split = "test-challenge"
    path = os.path.join(dirname, "VisDrone2018-DET-{}".format(split))
    images_path = os.path.join(path, "images")
    annotations_path = os.path.join(path, "annotations")

    fileids = [name[:-4] for name in os.listdir(annotations_path)]
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotations_path, fileid + ".txt")
        jpeg_file = os.path.join(images_path, fileid + ".jpg")

        fp = open(jpeg_file, 'rb')
        im = Image.open(fp)
        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": im.size[1],
            "width": im.size[0],
        }
        fp.close()

        instances = []
        with open(anno_file, "r") as f:
            objs = f.readlines()
        for obj in objs:
            obj = obj.split(",")
            if obj[4] == "0":
                continue
            bbox = [float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])]
            instances.append(
                {
                    "category_id": int(obj[5]) - 1,
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                }
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_visdrone(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_vis_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES,
        dirname=dirname,
        split=split
    )
