import torch

__all__ = ["subsample_labels"]


def subsample_labels(
    labels: torch.Tensor,
    num_samples: int,
    positive_fraction: float,
    bg_label: int
):
    positive = torch.nonzero((labels != -1) & (labels != bg_label), as_tuple=True)[0]
    negative = torch.nonzero(labels == bg_label, as_tuple=True)[0]

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(negative.numel(), num_neg)

    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx
