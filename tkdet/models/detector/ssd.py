from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.layers import ShapeSpec
from tkdet.layers import batched_nms
from tkdet.layers import cat
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
from tkdet.utils.registry import Registry
from .base import Detector
from .build import DETECTOR_REGISTRY

__all__ = ["SSD"]

SSD_HEAD_REGISTRY = Registry("SSD_HEAD")


def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape

    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


@DETECTOR_REGISTRY.register()
class SSD(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.score_threshold = cfg.MODEL.SCORE_THRESHOLD
        self.nms_threshold = cfg.MODEL.NMS_THRESHOLD
        self.smooth_l1_loss_beta = cfg.LOSS.SMOOTH_L1_LOSS.BETA
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg, self.backbone.output_shape())
        output_shape = self.neck.output_shape()
        feature_shapes = [output_shape[f] for f in output_shape]
        self.head = build_ssd_head(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)
        self.box2box_transform = Box2BoxTransform(cfg.SSD.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.SSD.IOU_THRESHOLDS,
            cfg.SSD.IOU_LABELS,
            allow_low_quality_matches=True
        )

    def forward(self, batched_inputs):
        images = self.preprocess_inputs(batched_inputs)

        features = self.backbone(images.tensor)
        features = self.neck(features)
        features = [features[f] for f in self.neck.output_shape()]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
            losses = self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)
            return losses
        else:
            results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results,
                batched_inputs,
                images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        batch_size = pred_class_logits[0].shape[0]

        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits,
            pred_anchor_deltas,
            self.num_classes + 1
        )

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        background_idxs = (gt_classes == self.num_classes)
        num_foreground = foreground_idxs.sum().item()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        softmax_ce = F.cross_entropy(
            pred_class_logits[valid_idxs],
            gt_classes[valid_idxs],
            reduction="none"
        )
        loss_for_all_indices = torch.zeros_like(pred_class_logits[:, 1])
        loss_for_all_indices[valid_idxs] = softmax_ce

        loss_for_neg_indices = torch.zeros_like(loss_for_all_indices)
        loss_for_neg_indices[background_idxs] = loss_for_all_indices[background_idxs]
        loss_for_neg_indices = loss_for_neg_indices.view(batch_size, -1)

        rank = (-loss_for_neg_indices).argsort(axis = -1).argsort(axis = -1)
        num_foreground_per_img = foreground_idxs.view(batch_size, -1).sum(1, keepdim=True)

        hard_negative = rank < (3.0 * num_foreground_per_img)

        hard_negative = hard_negative.view(-1)
        hard_negative = hard_negative & background_idxs
        loss_for_all_indices[~ (hard_negative | foreground_idxs)] = 0
        loss_cls = loss_for_all_indices[valid_idxs].sum(0, keepdims=False) / max(1, num_foreground)

        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        gt_classes = []
        gt_anchors_deltas = []
        anchors = Boxes.cat(anchors)

        for targets_per_image in targets:
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors.tensor,
                    matched_gt_boxes.tensor
                )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                gt_classes_i[anchor_labels == 0] = self.num_classes
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors.tensor)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    def inference(self, box_cls, box_delta, anchors, image_sizes):
        results = []
        box_cls = [permute_to_N_HWA_K(x, self.num_classes + 1) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]

        for img_idx, image_size in enumerate(image_sizes):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            results_per_image = self.inference_single_image(
                box_cls_per_image,
                box_reg_per_image,
                anchors,
                tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            box_cls_i = F.softmax(box_cls_i, dim=-1)[:, :-1].flatten()

            num_topk = box_reg_i.size(0)
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


@SSD_HEAD_REGISTRY.register()
class SSDHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        in_channels = [x.channels for x in input_shape]
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        num_classes = cfg.MODEL.NUM_CLASSES

        cls_score = []
        bbox_pred = []
        for i, c in enumerate(in_channels):
            cls_score.append(nn.Conv2d(c, num_anchors[i] * (num_classes + 1), 3, 1, 1))
            bbox_pred.append(nn.Conv2d(c, num_anchors[i] * 4, 3, 1, 1))

        self.cls_score = nn.ModuleList(cls_score)
        self.bbox_pred = nn.ModuleList(bbox_pred)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for idx, feature in enumerate(features):
            logits.append(self.cls_score[idx](feature))
            bbox_reg.append(self.bbox_pred[idx](feature))
        return logits, bbox_reg


def build_ssd_head(cfg, input_shape):
    return SSD_HEAD_REGISTRY.get(cfg.SSD.HEAD.NAME)(cfg, input_shape)
