from typing import Dict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.layers import ShapeSpec
from tkdet.utils.registry import Registry
from tkdet.models.anchor import build_anchor_generator
from tkdet.models.box_regression import Box2BoxTransform
from tkdet.models.matcher import Matcher
from .build import RPN_REGISTRY
from .rpn_outputs import RPNOutputs
from .rpn_outputs import find_top_rpn_proposals

__all__ = ["RPN", "RPN_HEAD_REGISTRY", "build_rpn_head"]

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")


def build_rpn_head(cfg, input_shape):
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"

        in_channels = in_channels[0]
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert len(set(num_cell_anchors)) == 1, \
            "Each level must have the same number of cell anchors"

        num_cell_anchors = num_cell_anchors[0]
        
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors, 1)
        self.anchor_deltas = nn.Conv2d(in_channels, num_cell_anchors * box_dim, 1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@RPN_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by Faster R-CNN.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.min_box_side_len = cfg.MODEL.RPN.MIN_SIZE
        self.in_feature = cfg.MODEL.RPN.IN_FEATURE
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT
        self.smooth_l1_beta = cfg.LOSS.SMOOTH_L1.BETA

        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS,
            cfg.MODEL.RPN.IOU_LABELS,
            allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None):
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )

        return proposals, losses
