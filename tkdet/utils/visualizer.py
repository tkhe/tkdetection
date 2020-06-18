import colorsys
import logging
import math
from enum import Enum
from enum import unique

import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from tkdet.structures import BitMasks
from tkdet.structures import BoxMode
from tkdet.structures import Boxes
from tkdet.structures import Keypoints
from tkdet.structures import PolygonMasks
from .colormap import random_color

logger = logging.getLogger(__name__)

__all__ = ["ColorMode", "VisImage", "Visualizer"]

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05


@unique
class ColorMode(Enum):
    IMAGE = 0
    SEGMENTATION = 1
    IMAGE_BW = 2


class GenericMask(object):
    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), m.shape
            self._mask = m.astype("uint8")
            return

        raise ValueError(f"GenericMask cannot handle object {m} of type '{type(m)}'")

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False
        return self._has_holes

    def mask_to_polygons(self, mask):
        mask = np.ascontiguousarray(mask)
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


class _PanopticPrediction(object):
    def __init__(self, panoptic_seg, segments_info):
        self._seg = panoptic_seg

        self._sinfo = {s["id"]: s for s in segments_info}
        segment_ids, areas = torch.unique(panoptic_seg, sorted=True, return_counts=True)
        areas = areas.numpy()
        sorted_idxs = np.argsort(-areas)
        self._seg_ids, self._seg_areas = segment_ids[sorted_idxs], areas[sorted_idxs]
        self._seg_ids = self._seg_ids.tolist()
        for sid, area in zip(self._seg_ids, self._seg_areas):
            if sid in self._sinfo:
                self._sinfo[sid]["area"] = float(area)

    def non_empty_mask(self):
        empty_ids = []
        for id in self._seg_ids:
            if id not in self._sinfo:
                empty_ids.append(id)
        if len(empty_ids) == 0:
            return np.zeros(self._seg.shape, dtype=np.uint8)
        assert len(empty_ids) == 1, \
            ">1 ids corresponds to no labels. This is currently not supported"
        return (self._seg != empty_ids[0]).numpy().astype(np.bool)

    def semantic_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)
            if sinfo is None or sinfo["isthing"]:
                continue
            yield (self._seg == sid).numpy().astype(np.bool), sinfo

    def instance_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)
            if sinfo is None or not sinfo["isthing"]:
                continue
            mask = (self._seg == sid).numpy().astype(np.bool)
            if mask.sum() > 0:
                yield mask, sinfo


def _create_text_labels(classes, scores, class_names):
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


class VisImage(object):
    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(self.height)

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        if filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
            cv2.imwrite(filepath, self.get_image()[:, :, ::-1])
        else:
            self.ax.imshow(self.img, interpolation="nearest")
            self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        if (self.width, self.height) != (width, height):
            img = cv2.resize(self.img, (width, height))
        else:
            img = self.img

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        try:
            import numexpr as ne

            visualized_image = ne.evaluate("img * (1 - alpha / 255.0) + rgb * (alpha / 255.0)")
        except ImportError:
            alpha = alpha.astype("float32") / 255.0
            visualized_image = img * (1 - alpha) + rgb * alpha

        visualized_image = visualized_image.astype("uint8")

        return visualized_image


class Visualizer(object):
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.metadata = metadata
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90,
            10 // scale
        )
        self._instance_mode = instance_mode

    def draw_instance_predictions(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = self.metadata.stuff_classes[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output

    def draw_panoptic_seg_predictions(
        self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7
    ):
        pred = _PanopticPrediction(panoptic_seg, segments_info)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(pred.non_empty_mask())

        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            text = self.metadata.stuff_classes[category_idx]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        labels = _create_text_labels(category_ids, scores, self.metadata.thing_classes)

        try:
            colors = [random_color(rgb=True, maximum=1) for k in category_ids]
        except AttributeError:
            colors = None
        self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        return self.output

    def draw_dataset_dict(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            colors = None
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in labels
                ]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                f"{i}" + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype=uint8)
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    ):
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output

        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bbox()

                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    def draw_and_connect_keypoints(self, keypoints):
        visible = {}
        keypoint_names = self.metadata.get("keypoint_names")
        for idx, keypoint in enumerate(keypoints):
            x, y, prob = keypoint
            if prob > _KEYPOINT_THRESHOLD:
                self.draw_circle((x, y), color=_RED)
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)

        if self.metadata.get("keypoint_connection_rules"):
            for kp0, kp1, color in self.metadata.keypoint_connection_rules:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    color = tuple(x / 255.0 for x in color)
                    self.draw_line([x0, x1], [y0, y1], color=color)

        try:
            ls_x, ls_y = visible["left_shoulder"]
            rs_x, rs_y = visible["right_shoulder"]
            mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
        except KeyError:
            pass
        else:
            nose_x, nose_y = visible.get("nose", (None, None))
            if nose_x is not None:
                self.draw_line([nose_x, mid_shoulder_x], [nose_y, mid_shoulder_y], color=_RED)

            try:
                lh_x, lh_y = visible["left_hip"]
                rh_x, rh_y = visible["right_hip"]
            except KeyError:
                pass
            else:
                mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
                self.draw_line([mid_hip_x, mid_shoulder_x], [mid_hip_y, mid_shoulder_y], color=_RED)
        return self.output

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0
    ):
        if not font_size:
            font_size = self._default_font_size

        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output

    def draw_binary_mask(
        self,
        binary_mask,
        color=None,
        *,
        edge_color=None,
        text=None,
        alpha=0.5,
        area_threshold=4096
    ):
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)
        if area_threshold is None:
            area_threshold = 4096

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes:
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < area_threshold:
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
        else:
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            has_valid_segment = True
            self.output.ax.imshow(rgba)

        if text is not None and has_valid_segment:
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
            largest_component_id = np.argmax(stats[1:, -1]) + 1

            for cid in range(1, _num_cc):
                if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                    center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                    self.draw_text(text, center, color=lighter_color)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        if edge_color is None:
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def _jitter(self, color):
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _create_grayscale_image(self, mask=None):
        img_bw = self.img.astype("f4").mean(axis=2)
        img_bw = np.stack([img_bw] * 3, axis=2)
        if mask is not None:
            img_bw[mask] = self.img[mask]
        return img_bw

    def _change_color_brightness(self, color, brightness_factor):
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0

        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def _convert_boxes(self, boxes):
        if isinstance(boxes, Boxes):
            return boxes.tensor.numpy()
        else:
            return np.asarray(boxes)

    def _convert_masks(self, masks_or_polygons):
        m = masks_or_polygons
        if isinstance(m, PolygonMasks):
            m = m.polygons
        if isinstance(m, BitMasks):
            m = m.tensor.numpy()
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        ret = []
        for x in m:
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                ret.append(GenericMask(x, self.output.height, self.output.width))
        return ret

    def _convert_keypoints(self, keypoints):
        if isinstance(keypoints, Keypoints):
            keypoints = keypoints.tensor
        keypoints = np.asarray(keypoints)
        return keypoints

    def get_output(self):
        return self.output
