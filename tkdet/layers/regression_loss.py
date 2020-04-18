import torch

__all__ = ["smooth_l1_loss", "giou_loss"]


def smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    reduction: str = "none",
) -> torch.Tensor:
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def giou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight=None,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    NOTE: Now GIoU loss is only used for FCOS.
    """
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect + eps
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    ious = (area_intersect + eps) / (area_union + eps)
    gious = ious - (ac_uion - area_union) / ac_uion
    losses = 1 - gious

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return losses.sum()
