from .coco import register_coco_instances


def register_fruits_nuts_instances(name, metadata, json_file, image_root):
    register_coco_instances(name, metadata, json_file, image_root)
