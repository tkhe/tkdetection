import math

import torch

from tkdet.structures import Instances


def add_ground_truth_to_proposals(gt_boxes, proposals):
    assert gt_boxes is not None
    assert len(proposals) == len(gt_boxes)

    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i)
        for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
    ]


def add_ground_truth_to_proposals_single_image(gt_boxes, proposals):
    device = proposals.objectness_logits.device
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))

    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)
    gt_proposal = Instances(proposals.image_size)

    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals
