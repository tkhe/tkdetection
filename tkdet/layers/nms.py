import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms

__all__ = ["batched_nms"]


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float
):
    assert boxes.shape[-1] == 4

    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep
