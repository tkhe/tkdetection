from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.config import configurable
from tkdet.layers import ShapeSpec
from tkdet.layers import cat
from tkdet.layers import smooth_l1_loss
from tkdet.structures import Boxes
from tkdet.structures import ImageList
from tkdet.structures import Instances
from tkdet.structures import pairwise_iou
from tkdet.utils.events import get_event_storage
from tkdet.utils.memory import retry_if_cuda_oom
from tkdet.utils.registry import Registry
from tkdet.models.anchor import build_anchor_generator
from tkdet.models.box_regression import Box2BoxTransform
from tkdet.models.matcher import Matcher
from tkdet.models.sampling import subsample_labels
from .build import RPN_REGISTRY
from .proposal_utils import find_top_rpn_proposals

__all__ = ["RPN", "RPN_HEAD_REGISTRY", "build_rpn_head"]

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")


def build_rpn_head(cfg, input_shape):
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"

        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (len(set(num_anchors) == 1)), \
            "Each level must have the same number of anchors per spatial position"

        return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim}

    def forward(self, features: List[torch.Tensor]):
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
        self.in_features = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT
        self.smooth_l1_beta = cfg.LOSS.SMOOTH_L1_LOSS.BETA

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

    def _subsample_labels(self, label):
        pos_idx, neg_idx = subsample_labels(
            label,
            self.batch_size_per_image,
            self.positive_fraction,
            0
        )
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.boundary_threshold >= 0:
                anchors_inside_image = anchors.inside_box(image_size_i, self.boundary_threshold)
                gt_labels_i[~anchors_inside_image] = -1

            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    def losses(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes,
    ):
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)
        anchors = type(anchors[0]).cat(anchors).tensor
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)

        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            self.smooth_l1_beta,
            reduction="sum",
        )
        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        return {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors,
                pred_objectness_logits,
                gt_labels,
                pred_anchor_deltas,
                gt_boxes
            )
            losses = {k: v * self.loss_weight for k, v in losses.items()}
        else:
            losses = {}

        proposals = self.predict_proposals(
            anchors,
            pred_objectness_logits,
            pred_anchor_deltas,
            images.image_sizes
        )
        return proposals, losses

    @torch.no_grad()
    def predict_proposals(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_rpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_side_len,
            self.training,
        )

        return proposals, losses

    def _decode_proposals(self, anchors, pred_anchor_deltas: List[torch.Tensor]):
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals
