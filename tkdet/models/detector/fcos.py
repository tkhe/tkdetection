import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.layers import Conv2d
from tkdet.layers import ShapeSpec
from tkdet.layers import batched_nms
from tkdet.layers import cat
from tkdet.layers import giou_loss
from tkdet.layers import sigmoid_focal_loss_jit
from tkdet.models.backbone import build_backbone
from tkdet.models.neck import build_neck
from tkdet.models.postprocessing import detector_postprocess
from tkdet.structures import Boxes
from tkdet.structures import ImageList
from tkdet.structures import Instances
from .base import Detector
from .build import DETECTOR_REGISTRY

INF = 1e8

__all__ = ["FCOS"]


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def compute_locations(features, neck_strides):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locations_per_level = compute_locations_per_level(
            h,
            w,
            neck_strides[level],
            feature.device
        )
        locations.append(locations_per_level)
    return locations


def get_sample_region(gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
    num_gts = gt.shape[0]
    K = len(gt_xs)
    gt = gt[None].expand(K, num_gts, 4)
    center_x = (gt[..., 0] + gt[..., 2]) / 2
    center_y = (gt[..., 1] + gt[..., 3]) / 2
    center_gt = gt.new_zeros(gt.shape)

    if center_x[..., 0].sum() == 0:
        return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

    beg = 0
    for level, n_p in enumerate(num_points_per):
        end = beg + n_p
        stride = strides[level] * radius
        xmin = center_x[beg:end] - stride
        ymin = center_y[beg:end] - stride
        xmax = center_x[beg:end] + stride
        ymax = center_y[beg:end] + stride

        center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
        center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
        center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
        center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
        beg = end

    left = gt_xs[:, None] - center_gt[..., 0]
    right = center_gt[..., 2] - gt_xs[:, None]
    top = gt_ys[:, None] - center_gt[..., 1]
    bottom = center_gt[..., 3] - gt_ys[:, None]
    center_bbox = torch.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
    return inside_gt_bbox_mask


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (
        (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
        * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    )
    return torch.sqrt(centerness)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


@DETECTOR_REGISTRY.register()
class FCOS(Detector):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg, self.backbone.output_shape())
        output_shape = self.neck.output_shape()
        feature_shapes = [output_shape[f] for f in output_shape]
        self.neck_strides = [x.stride for x in feature_shapes]
        self.head = FCOSHead(cfg, feature_shapes)

        self.center_sampling_radius = cfg.FCOS.CENTER_SAMPLING_RADIUS
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.score_threshold = cfg.MODEL.SCORE_THRESHOLD
        self.nms_threshold = cfg.MODEL.NMS_THRESHOLD
        self.norm_reg_targets = cfg.FCOS.NORM_REG_TARGETS
        self.topk_candidates = cfg.FCOS.TOPK_CANDIDATES
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.device = torch.device(cfg.MODEL.DEVICE)
        pixel_mean = torch.Tensor(cfg.INPUT.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.INPUT.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        images = self.preprocess_inputs(batched_inputs)
        
        features = self.backbone(images.tensor)
        features = self.neck(features)
        features = [features[f] for f in self.neck.output_shape()]
        box_cls, box_regression, centerness = self.head(features)
        locations = compute_locations(features, self.neck_strides)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            return self.losses(locations, box_cls, box_regression, centerness, gt_instances)
        else:
            results = self.inference(
                locations,
                box_cls,
                box_regression,
                centerness,
                images.image_sizes
            )
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

    def losses(self, locations, box_cls, box_regression, centerness, targets):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)

        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []

        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = cat(box_cls_flatten, dim=0)
        box_regression_flatten = cat(box_regression_flatten, dim=0)
        centerness_flatten = cat(centerness_flatten, dim=0)
        labels_flatten = cat(labels_flatten, dim=0)
        reg_targets_flatten = cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(
            (labels_flatten >= 0) & (labels_flatten < self.num_classes)
        ).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_foreground = pos_inds.numel()
        gt_labels = torch.zeros_like(box_cls_flatten)
        gt_labels[pos_inds, labels_flatten[pos_inds]] = 1

        cls_loss = sigmoid_focal_loss_jit(
            box_cls_flatten,
            gt_labels,
            alpha=0.25,
            gamma=2,
            reduction="sum"
        ) / num_foreground

        if num_foreground > 0:
            centerness_targets = compute_centerness_targets(reg_targets_flatten)
            reg_loss = giou_loss(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_flatten
            ) / centerness_targets.sum().item()
            centerness_loss = F.binary_cross_entropy_with_logits(
                centerness_flatten,
                centerness_targets,
                reduction="sum"
            ) / num_foreground
        else:
            reg_loss = box_cls_flatten.sum()
            centerness_loss = centerness_flatten.sum()
        return {
            "loss_cls": cls_loss,
            "loss_box_reg": reg_loss,
            "loss_centerness": centerness_loss
        }

    def inference(self, locations, box_cls, box_regression, centerness, image_sizes):
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.inference_on_single_feature_map(l, o, b, c, image_sizes)
            )

        instances = list(zip(*sampled_boxes))
        instances = [Instances.cat(x) for x in instances]

        results = self.select_over_all_levels(instances)

        return results

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for level, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = points_per_level.new_tensor(
                object_sizes_of_interest[level]
            )
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = cat(expanded_object_sizes_of_interest, 0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level,
            targets,
            expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(cat([labels_per_im[level] for labels_per_im in labels], 0))

            reg_targets_per_level = cat(
                [reg_targets_per_im[level] for reg_targets_per_im in reg_targets],
                dim=0
            )

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.neck_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i].get("gt_boxes")
            bboxes = targets_per_im.tensor
            labels_per_im = targets[im_i].get("gt_classes")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sampling_radius > 0:
                is_in_boxes = get_sample_region(
                    bboxes,
                    self.neck_strides,
                    self.num_points_per_level,
                    xs,
                    ys,
                    self.center_sampling_radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            is_cared_in_the_level = (
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]])
                & (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
            )

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def inference_on_single_feature_map(
        self,
        locations,
        box_cls,
        box_regression,
        centerness,
        image_sizes
    ):
        N, C, H, W = box_cls.shape

        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.score_threshold
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.topk_candidates)

        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack(
                [
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ],
                dim=1
            )

            result = Instances(image_sizes[i])
            detections = Boxes(detections)
            detections.clip(image_sizes[i])
            result.pred_boxes = detections
            result.scores = torch.sqrt(per_box_cls)
            result.pred_classes = per_class
            results.append(result)

        return results

    def select_over_all_levels(self, instances):
        num_images = len(instances)
        results = []
        for i in range(num_images):
            boxes = instances[i].get("pred_boxes")
            scores = instances[i].get("scores")
            pred_classes = instances[i].get("pred_classes")
            keep = batched_nms(boxes.tensor, scores, pred_classes, self.nms_threshold)
            keep = keep[: self.max_detections_per_image]
            result = Instances(instances[i].image_size)
            result.pred_boxes = boxes[keep]
            result.scores = scores[keep]
            result.pred_classes = pred_classes[keep]
            results.append(result)
        return results

    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.neck.size_divisibility)
        return images


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        num_classes = cfg.MODEL.NUM_CLASSES
        num_convs = cfg.FCOS.NUM_CONVS
        prior_prob = cfg.FCOS.PRIOR_PROB
        in_channels = input_shape[0].channels
        self.neck_strides = [x.stride for x in input_shape]
        self.norm_reg_targets = cfg.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.FCOS.CENTERNESS_ON_REG

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                Conv2d(in_channels, in_channels, 3, 1, norm="GN", activation="ReLU")
            )
            bbox_subnet.append(
                Conv2d(in_channels, in_channels, 3, 1, norm="GN", activation="ReLU")
            )

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels, num_classes, 3, 1, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, 1, 1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, 1, 1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(len(input_shape))])

        for module in [self.cls_score, self.bbox_pred, self.centerness]:
            nn.init.normal_(module.weight, std=0.01)
            nn.init.constant_(module.bias, 0)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        logits = []
        bbox_reg = []
        centerness = []

        for idx, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            if self.centerness_on_reg:
                centerness.append(self.centerness(bbox_subnet))
            else:
                centerness.append(self.centerness(cls_subnet))

            bbox_pred = self.scales[idx](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.neck_strides[idx])
            else:
                bbox_reg.append(torch.exp(bbox_pred))

        return logits, bbox_reg, centerness
