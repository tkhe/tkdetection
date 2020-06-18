from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F

__all__ = ["ImageList"]


class ImageList(object):
    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: Sequence[torch.Tensor],
        size_divisibility: int = 0,
        pad_value: float = 0.0
    ) -> "ImageList":
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))

        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
        max_size = (
            torch.stack(
                [
                    torch.stack([torch.as_tensor(dim) for dim in size])
                    for size in [tuple(img.shape) for img in tensors]
                ]
            )
            .max(0)
            .values
        )

        if size_divisibility > 1:
            stride = size_divisibility
            max_size = torch.cat([max_size[:-2], (max_size[-2:] + (stride - 1)) // stride * stride])

        image_sizes = [tuple(im.shape[-2:]) for im in tensors]

        if len(tensors) == 1:
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            if all(x == 0 for x in padding_size):
                batched_imgs = tensors[0].unsqueeze(0)
            else:
                padded = F.pad(tensors[0], padding_size, value=pad_value)
                batched_imgs = padded.unsqueeze_(0)
        else:
            batch_shape = (len(tensors),) + tuple(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)
