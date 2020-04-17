import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler

__all__ = ["GroupedBatchSampler"]


class GroupedBatchSampler(BatchSampler):
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
  
        self.batch_size = batch_size
        groups = np.unique(self.group_ids).tolist()

        self.buffer_per_group = {k: [] for k in groups}

    def __iter__(self):
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]
                del group_buffer[:]

    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")
