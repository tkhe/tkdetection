import itertools
import logging

import numpy as np
import torch
import torch.nn.functional as F

from tkdet.layers import batched_nms
from tkdet.layers import cat
from tkdet.layers import smooth_l1_loss
from tkdet.models.sampling import subsample_labels
from tkdet.structures import Boxes
from tkdet.structures import Instances
from tkdet.structures import pairwise_iou
from tkdet.utils.events import get_event_storage
from tkdet.utils.memory import retry_if_cuda_oom

logger = logging.getLogger(__name__)

__all__ = [""]


def find_top_rpn_proposals(
    proposals,
    pred_objectness_logits,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    image_sizes = images.image_sizes
    num_images = len(image_sizes)
    device = proposals[0].device

    topk_scores = []
    topk_proposals = []
    level_ids = []
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(),
        proposals,
        pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device)
        )

    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        keep = boxes.nonempty(threshold=min_box_side_len)
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


def rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta,
):
    pos_masks = gt_objectness_logits == 1
    localization_loss = smooth_l1_loss(
        pred_anchor_deltas[pos_masks],
        gt_anchor_deltas[pos_masks],
        smooth_l1_beta,
        reduction="sum"
    )

    valid_masks = gt_objectness_logits >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(
        pred_objectness_logits[valid_masks],
        gt_objectness_logits[valid_masks].to(torch.float32),
        reduction="sum",
    )
    return objectness_loss, localization_loss


class RPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        gt_objectness_logits = []
        gt_anchor_deltas = []
        anchors = Boxes.cat(self.anchors)
        for image_size_i, gt_boxes_i in zip(self.image_sizes, self.gt_boxes):
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_objectness_logits_i = retry_if_cuda_oom(
                self.anchor_matcher
            )(match_quality_matrix)
            gt_objectness_logits_i = gt_objectness_logits_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.boundary_threshold >= 0:
                anchors_inside_image = anchors.inside_box(image_size_i, self.boundary_threshold)
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                gt_anchor_deltas_i = torch.zeros_like(anchors.tensor)
            else:
                matched_gt_boxes = gt_boxes_i[matched_idxs]
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors.tensor,
                    matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        return gt_objectness_logits, gt_anchor_deltas

    def losses(self):
        def resample(label):
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

        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_objectness_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        gt_objectness_logits = torch.stack(
            [resample(label) for label in gt_objectness_logits],
            dim=0
        )

        num_pos_anchors = (gt_objectness_logits == 1).sum().item()
        num_neg_anchors = (gt_objectness_logits == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        assert gt_objectness_logits.shape[1] == num_anchors_per_image

        gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
        gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)

        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        assert gt_anchor_deltas.shape[1] == num_anchors_per_image

        B = gt_anchor_deltas.shape[2]

        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
        gt_anchor_deltas = cat([x.reshape(-1, B) for x in gt_anchor_deltas], dim=0)

        pred_objectness_logits = cat(
            [x.permute(0, 2, 3, 1).flatten() for x in self.pred_objectness_logits],
            dim=0
        )
        pred_anchor_deltas = cat(
            [
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_anchor_deltas
            ],
            dim=0,
        )

        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )
        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_cls = objectness_loss * normalizer
        loss_loc = localization_loss * normalizer
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        proposals = []
        for anchors_i, pred_anchor_deltas_i in zip(self.anchors, self.pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            anchors_i = cat([anchors_i.tensor] * N, dim=0)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

    def predict_objectness_logits(self):
        pred_objectness_logits = [
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits
