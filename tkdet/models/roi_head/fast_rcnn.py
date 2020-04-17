import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from tkdet.layers import smooth_l1_loss
from tkdet.config import configurable
from tkdet.layers import ShapeSpec
from tkdet.layers import batched_nms
from tkdet.layers import cat
from tkdet.models.box_regression import Box2BoxTransform
from tkdet.structures import Boxes
from tkdet.structures import Instances
from tkdet.utils.events import get_event_storage

logger = logging.getLogger(__name__)

__all__ = ["FastRCNNOutputLayers"]


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape,
    score_thresh,
    nms_thresh,
    topk_per_image
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)

    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0,
    ):
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert not self.proposals.tensor.requires_grad, \
                "Proposals should not require gradients!"

            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")

                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0

    def _log_accuracy(self):
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        if self._no_instances:
            return 0.0 * F.cross_entropy(
                self.pred_class_logits,
                torch.zeros(0, dtype=torch.long, device=self.pred_class_logits.device),
                reduction="sum",
            )
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def smooth_l1_loss(self):
        if self._no_instances:
            return 0.0 * smooth_l1_loss(
                self.pred_proposal_deltas,
                torch.zeros_like(self.pred_proposal_deltas),
                0.0,
                reduction="sum",
            )
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor,
            self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = torch.nonzero(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind),
            as_tuple=True
        )[0]
        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B)

    def losses(self):
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_boxes_for_gt_classes(self):
        predicted_boxes = self._predict_boxes()
        B = self.proposals.tensor.shape[1]
        if predicted_boxes.shape[1] > B:
            num_pred = len(self.proposals)
            num_classes = predicted_boxes.shape[1] // B
            gt_classes = torch.clamp(self.gt_classes, 0, num_classes - 1)
            predicted_boxes = predicted_boxes.view(
                num_pred,
                num_classes,
                B
            )[torch.arange(num_pred, dtype=torch.long, device=predicted_boxes.device), gt_classes]
        return predicted_boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image
        )


class FastRCNNOutputLayers(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
    ):
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            "num_classes": cfg.MODEL.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.SCORE_THRESHOLD,
            "test_nms_thresh": cfg.MODEL.NMS_THRESHOLD,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE
        }

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions
        return FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas, 
            proposals,
            self.smooth_l1_beta
        ).losses()

    def inference(self, predictions, proposals):
        scores, proposal_deltas = predictions
        return FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta
        ).inference(self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image)

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        scores, proposal_deltas = predictions
        return FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta
        ).predict_boxes_for_gt_classes()

    def predict_boxes(self, predictions, proposals):
        scores, proposal_deltas = predictions
        return FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta
        ).predict_boxes()

    def predict_probs(self, predictions, proposals):
        scores, proposal_deltas = predictions
        return FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta
        ).predict_probs()
