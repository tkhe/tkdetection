import numpy as np
import torch

from tkdet.layers import ShapeSpec
from tkdet.layers import cat
from tkdet.layers import interpolate
from tkdet.models.roi_head import ROI_HEADS_REGISTRY
from tkdet.models.roi_head import StandardROIHeads
from tkdet.models.roi_head.mask_head import build_mask_head
from tkdet.models.roi_head.mask_head import mask_rcnn_inference
from tkdet.models.roi_head.mask_head import mask_rcnn_loss
from tkdet.models.roi_head.roi_head import select_foreground_proposals
from .point_features import generate_regular_grid_point_coords
from .point_features import get_uncertain_point_coords_on_grid
from .point_features import get_uncertain_point_coords_with_randomness
from .point_features import point_sample
from .point_features import point_sample_fine_grained_features
from .point_head import build_point_head
from .point_head import roi_mask_point_loss


def calculate_uncertainty(logits, classes):
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -torch.abs(gt_class_logits)


@ROI_HEADS_REGISTRY.register()
class PointRendROIHeads(StandardROIHeads):
    def _init_mask_head(self, cfg, input_shape):
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        self.mask_coarse_in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
        self.mask_coarse_side_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self._feature_scales = {k: 1.0 / v.stride for k, v in input_shape.items()}

        in_channels = np.sum([input_shape[f].channels for f in self.mask_coarse_in_features])
        self.mask_coarse_head = build_mask_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                width=self.mask_coarse_side_size,
                height=self.mask_coarse_side_size,
            ),
        )
        self._init_point_head(cfg, input_shape)

    def _init_point_head(self, cfg, input_shape):
        self.mask_point_on = cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON
        if not self.mask_point_on:
            return

        self.mask_point_in_features = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.mask_point_train_num_points = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.mask_point_oversample_ratio = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.mask_point_importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        self.mask_point_subdivision_steps = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.mask_point_subdivision_num_points = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS

        in_channels = np.sum([input_shape[f].channels for f in self.mask_point_in_features])
        self.mask_point_head = build_point_head(
            cfg,
            ShapeSpec(channels=in_channels, width=1, height=1)
        )

    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_coarse_logits = self._forward_mask_coarse(features, proposal_boxes)

            losses = {"loss_mask": mask_rcnn_loss(mask_coarse_logits, proposals)}
            losses.update(self._forward_mask_point(features, mask_coarse_logits, proposals))
            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_coarse_logits = self._forward_mask_coarse(features, pred_boxes)

            mask_logits = self._forward_mask_point(features, mask_coarse_logits, instances)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_mask_coarse(self, features, boxes):
        point_coords = generate_regular_grid_point_coords(
            np.sum(len(x) for x in boxes),
            self.mask_coarse_side_size,
            boxes[0].device
        )
        mask_coarse_features_list = [features[k] for k in self.mask_coarse_in_features]
        features_scales = [self._feature_scales[k] for k in self.mask_coarse_in_features]
        mask_features, _ = point_sample_fine_grained_features(
            mask_coarse_features_list,
            features_scales,
            boxes,
            point_coords
        )
        return self.mask_coarse_head(mask_features)

    def _forward_mask_point(self, features, mask_coarse_logits, instances):
        if not self.mask_point_on:
            return {} if self.training else mask_coarse_logits

        mask_features_list = [features[k] for k in self.mask_point_in_features]
        features_scales = [self._feature_scales[k] for k in self.mask_point_in_features]

        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            gt_classes = cat([x.gt_classes for x in instances])
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    mask_coarse_logits,
                    lambda logits: calculate_uncertainty(logits, gt_classes),
                    self.mask_point_train_num_points,
                    self.mask_point_oversample_ratio,
                    self.mask_point_importance_sample_ratio,
                )

            fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
                mask_features_list,
                features_scales,
                proposal_boxes,
                point_coords
            )
            coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)
            point_logits = self.mask_point_head(fine_grained_features, coarse_features)
            return {
                "loss_mask_point": roi_mask_point_loss(
                    point_logits,
                    instances,
                    point_coords_wrt_image
                )
            }
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_classes = cat([x.pred_classes for x in instances])
            if len(pred_classes) == 0:
                return mask_coarse_logits

            mask_logits = mask_coarse_logits.clone()
            for subdivions_step in range(self.mask_point_subdivision_steps):
                mask_logits = interpolate(
                    mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                H, W = mask_logits.shape[-2:]
                if (
                    self.mask_point_subdivision_num_points >= 4 * H * W
                    and subdivions_step < self.mask_point_subdivision_steps - 1
                ):
                    continue
                uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map,
                    self.mask_point_subdivision_num_points
                )
                fine_grained_features, _ = point_sample_fine_grained_features(
                    mask_features_list,
                    features_scales,
                    pred_boxes,
                    point_coords
                )
                coarse_features = point_sample(
                    mask_coarse_logits,
                    point_coords,
                    align_corners=False
                )
                point_logits = self.mask_point_head(fine_grained_features, coarse_features)

                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(R, C, H, W)
                )
            return mask_logits
