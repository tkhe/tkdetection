from typing import List

import torch

__all__ = ["Matcher"]


class Matcher(object):
    def __init__(
        self,
        thresholds: List[float],
        labels: List[int],
        allow_low_quality_matches: bool = False
    ):
        thresholds = thresholds[:]
        assert thresholds[0] > 0

        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        assert all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:]))
        assert all(l in [-1, 0, 1] for l in labels)
        assert len(labels) == len(thresholds) - 1

        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        assert match_quality_matrix.dim() == 2

        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),),
                0,
                dtype=torch.int64
            )
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),),
                self.labels[0],
                dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        _, pred_inds_with_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None],
            as_tuple=True
        )
        match_labels[pred_inds_with_highest_quality] = 1
