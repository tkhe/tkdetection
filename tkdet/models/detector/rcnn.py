import logging

import numpy as np
import torch
import torch.nn as nn

from tkdet.structures import ImageList
from tkdet.models.backbone import build_backbone
from tkdet.models.neck import build_neck
from tkdet.models.postprocessing import detector_postprocess
from tkdet.models.rpn import build_rpn
from tkdet.models.roi_head import build_roi_heads
from tkdet.utils.events import get_event_storage
from .base import Detector
from .build import DETECTOR_REGISTRY

__all__ = ["GeneralizedRCNN"]


@DETECTOR_REGISTRY.register()
class GeneralizedRCNN(Detector):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        if self.neck is not None:
            output_shape = self.neck.output_shape()
        else:
            output_shape = self.backbone.output_shape()
        self.rpn = build_rpn(cfg, output_shape)
        self.roi_heads = build_roi_heads(cfg, output_shape)
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.INPUT.PIXEL_MEAN) == len(cfg.INPUT.PIXEL_STD)

        num_channels = len(cfg.INPUT.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.INPUT.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.INPUT.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def visualize_training(self, batched_inputs, proposals):
        from tkdet.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."

            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        if self.neck is not None:
            features = self.neck(features)

        if self.rpn:
            proposals, proposal_losses = self.rpn(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if self.neck is not None:
            features = self.neck(features)

        if detected_instances is None:
            if self.rpn:
                proposals, _ = self.rpn(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances,
            batched_inputs,
            image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
