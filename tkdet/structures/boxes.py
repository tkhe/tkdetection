from enum import IntEnum
from enum import unique
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch

__all__ = ["BoxMode", "Boxes", "pairwise_iou", "matched_boxlist_iou"]

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4, \
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor, where k == 4"
            arr = torch.tensor(box)[None, :]
        else:
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert (
            to_mode.value not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL]
            and from_mode.value not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL]
        ), "Relative mode not yet supported!"

        if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
            arr[:, 2] += arr[:, 0]
            arr[:, 3] += arr[:, 1]
        elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
            arr[:, 2] -= arr[:, 0]
            arr[:, 3] -= arr[:, 1]
        else:
            raise NotImplementedError(
                f"Conversion from BoxMode {from_mode} to {to_mode} is not supported yet"
            )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class Boxes(object):
    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        return Boxes(self.tensor.clone())

    def to(self, device: str) -> "Boxes":
        return Boxes(self.tensor.to(device))

    def area(self) -> torch.Tensor:
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: BoxSizeType) -> None:
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"

        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=h)
        self.tensor[:, 2].clamp_(min=0, max=w)
        self.tensor[:, 3].clamp_(min=0, max=h)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"

        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: BoxSizeType, boundary_threshold: int = 0) -> torch.Tensor:
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        assert isinstance(boxes_list, (list, tuple))
        
        if len(boxes_list) == 0:
            return cls(torch.empty(0))

        assert all(isinstance(box, Boxes) for box in boxes_list)

        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from self.tensor


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    width_height = rb - lt

    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)
    del width_height

    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    assert len(boxes1) == len(boxes2), \
        f"boxlists should have the same number of entries, got {len(boxes1)}, {len(boxes2)}"

    area1 = boxes1.area()
    area2 = boxes2.area()
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])
    rb = torch.min(box1[:, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    iou = inter / (area1 + area2 - inter)
    return iou
