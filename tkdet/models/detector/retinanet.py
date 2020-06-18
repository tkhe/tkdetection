import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.data.detection_utils import convert_image_to_rgb
from tkdet.layers import ShapeSpec
from tkdet.layers import batched_nms
from tkdet.layers import cat
from tkdet.layers import sigmoid_focal_loss_jit
from tkdet.layers import smooth_l1_loss
from tkdet.models.anchor import build_anchor_generator
from tkdet.models.backbone import build_backbone
from tkdet.models.box_regression import Box2BoxTransform
from tkdet.models.matcher import Matcher
from tkdet.models.neck import build_neck
from tkdet.models.postprocessing import detector_postprocess
from tkdet.structures import Boxes
from tkdet.structures import ImageList
from tkdet.structures import Instances
from tkdet.structures import pairwise_iou
from tkdet.utils.events import get_event_storage
from .base import Detector
from .build import DETECTOR_REGISTRY

__all__ = ["RetinaNet"]


def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape

    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


@DETECTOR_REGISTRY.register()
class RetinaNet(Detector):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_classes = cfg.MODEL.NUM_CLASSES

        self.focal_loss_alpha = cfg.LOSS.FOCAL_LOSS.ALPHA
        self.focal_loss_gamma = cfg.LOSS.FOCAL_LOSS.GAMMA
        self.smooth_l1_loss_beta = cfg.LOSS.SMOOTH_L1_LOSS.BETA
        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

        self.score_threshold = cfg.MODEL.SCORE_THRESHOLD
        self.topk_candidates = cfg.RETINANET.TOPK_CANDIDATES
        self.nms_threshold = cfg.MODEL.NMS_THRESHOLD
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg, self.backbone.output_shape())
        output_shape = self.neck.output_shape()
        feature_shapes = [output_shape[f] for f in output_shape]
        self.head = RetinaNetHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        self.box2box_transform = Box2BoxTransform(weights=cfg.RETINANET.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.RETINANET.IOU_THRESHOLDS,
            cfg.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True
        )

    def visualize_training(self, batched_inputs, results):
        from tkdet.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(results), \
            "Cannot visualize inputs and results of different sizes"

        storage = get_event_storage()
        max_boxes = 20

        image_index = 0
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        images = self.preprocess_inputs(batched_inputs)

        features = self.backbone(images.tensor)
        features = self.neck(features)
        features = [features[f] for f in self.neck.output_shape()]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features)

        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if self.training:
            assert "instances" in batched_inputs[0], \
                "Instance annotations are missing in training!"

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors,
                        pred_logits,
                        pred_anchor_deltas,
                        images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)
        anchors = type(anchors[0]).cat(anchors).tensor
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        gt_labels_target = F.one_hot(
            gt_labels[valid_mask],
            num_classes=self.num_classes + 1
        )[:, :-1]
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        )

        return {
            "loss_cls": loss_cls / self.loss_normalizer,
            "loss_box_reg": loss_box_reg / self.loss_normalizer,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        anchors = Boxes.cat(anchors)
        gt_labels = []
        matched_gt_boxes = []

        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                gt_labels_i[anchor_labels == 0] = self.num_classes
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def inference(self, anchors, pred_logits, pred_anchor_deltas, image_sizes):
        results = []

        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors,
                pred_logits_per_image,
                deltas_per_image,
                tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta, image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            box_cls_i = box_cls_i.flatten().sigmoid_()

            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.neck.size_divisibility)
        return images


class RetinaNetHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.NUM_CLASSES
        num_convs = cfg.RETINANET.NUM_CONVS
        prior_prob = cfg.RETINANET.PRIOR_PROB
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors

        assert len(set(num_anchors)) == 1, \
            "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            cls_subnet.append(nn.ReLU(inplace=True))
            bbox_subnet.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            bbox_subnet.append(nn.ReLU(inplace=True))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, 1, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, 1, 1)

        for module in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)

        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg
