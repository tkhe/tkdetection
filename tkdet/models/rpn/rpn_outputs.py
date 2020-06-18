import itertools
import logging

import torch
import torch.nn.functional as F

from tkdet.layers import batched_nms
from tkdet.layers import cat
from tkdet.layers import smooth_l1_loss
from tkdet.structures import Boxes
from tkdet.structures import Instances
from tkdet.utils.events import get_event_storage

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
    gt_labels,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta
):
    pos_masks = gt_labels == 1
    localization_loss = smooth_l1_loss(
        pred_anchor_deltas[pos_masks],
        gt_anchor_deltas[pos_masks],
        smooth_l1_beta,
        reduction="sum"
    )

    valid_masks = gt_labels >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(
        pred_objectness_logits[valid_masks],
        gt_labels[valid_masks].to(torch.float32),
        reduction="sum",
    )
    return objectness_loss, localization_loss


class RPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        batch_size_per_image,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        gt_labels=None,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image

        B = anchors[0].tensor.size(1)
        self.pred_objectness_logits = [
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        self.pred_anchor_deltas = [
            x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        self.anchors = anchors

        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.num_images = len(images)
        self.smooth_l1_beta = smooth_l1_beta

    def losses(self):
        gt_labels = torch.stack(self.gt_labels)
        anchors = self.anchors[0].cat(self.anchors).tensor
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in self.gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)

        num_pos_anchors = (gt_labels == 1).sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        objectness_loss, localization_loss = rpn_losses(
            gt_labels,
            gt_anchor_deltas,
            cat(self.pred_objectness_logits, dim=1),
            cat(self.pred_anchor_deltas, dim=1),
            self.smooth_l1_beta,
        )
        normalizer = self.batch_size_per_image * self.num_images
        return {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }

    def predict_proposals(self):
        proposals = []
        for anchors_i, pred_anchor_deltas_i in zip(self.anchors, self.pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            N = self.num_images
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

    def predict_objectness_logits(self):
        return self.pred_objectness_logits
