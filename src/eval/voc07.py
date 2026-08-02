"""VOC07 11-point mAP@0.5 — the metric NIRNet actually reports.

WHY THIS EXISTS
---------------
Ultralytics reports COCO-style AP: 101-point interpolation, mAP50 and mAP50-95.
NIRNet reports "mAP" without stating a convention (see docs/comparison-baselines.md),
but its released code settles it: `DIORDataset.evaluate` in
`nirnet-main/mmrotate/datasets/dior.py` defaults to `use_07_metric=True` with
`iou_thr` starting at 0.5, i.e. **VOC07 11-point interpolated AP at IoU 0.50**.

Those two conventions do not produce the same number for the same detections.
Putting a 101-point mAP50 beside NIRNet's 11-point mAP is a category error, and
the difference is not a constant offset — it depends on the shape of the
precision/recall curve. So this module recomputes AP under NIRNet's convention
from the same predictions, letting the thesis report both and state which is
which.

This is exactly the "metrics beyond what Ultralytics gives" that `src/eval/`
exists for. It does NOT replace Ultralytics' numbers; `scripts/eval.py` reports
both side by side.

MATCHING PROTOCOL, copied from mmrotate's `tpfp_default`
--------------------------------------------------------
Per class, over the whole split: sort every detection by score descending; greedy
match each to the highest-IoU unmatched ground truth in its own image; IoU >= 0.5
is a true positive, anything else a false positive. DIOR-R carries no ignore
regions and NIRNet does not filter `difficult`, so neither do we.

IoU
---
HBB uses axis-aligned intersection-over-union. OBB uses exact polygon
intersection via `cv2.rotatedRectangleIntersection`, not the probabilistic
approximation Ultralytics trains with — an approximate IoU would make the
comparison to a paper's exact IoU meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..data.dior_classes import DIOR_CLASSES

__all__ = ["Detection", "GroundTruth", "voc07_ap", "evaluate_voc07", "VOC07Report"]

IOU_THRESHOLD = 0.5


@dataclass
class Detection:
    image_id: str
    cls: int
    score: float
    # HBB: (x1, y1, x2, y2). OBB: 8 polygon coordinates.
    coords: tuple[float, ...]


@dataclass
class GroundTruth:
    image_id: str
    cls: int
    coords: tuple[float, ...]


@dataclass
class VOC07Report:
    task: str
    iou_threshold: float
    per_class_ap: dict[str, float] = field(default_factory=dict)
    per_class_gt: dict[str, int] = field(default_factory=dict)
    per_class_det: dict[str, int] = field(default_factory=dict)

    @property
    def mean_ap(self) -> float:
        """Mean over classes that HAVE ground truth.

        mmrotate averages only over classes present in the split; including
        absent classes as 0.0 would silently deflate the mean and make the
        number incomparable.
        """
        present = [ap for name, ap in self.per_class_ap.items() if self.per_class_gt.get(name, 0) > 0]
        return float(np.mean(present)) if present else 0.0


def _poly_to_rotated_rect(coords: tuple[float, ...]):
    pts = np.array(coords, dtype=np.float32).reshape(4, 2)
    return cv2.minAreaRect(pts)


def _obb_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Exact IoU between two oriented boxes given as 4-corner polygons."""
    ra, rb = _poly_to_rotated_rect(a), _poly_to_rotated_rect(b)
    area_a = ra[1][0] * ra[1][1]
    area_b = rb[1][0] * rb[1][1]
    if area_a <= 0 or area_b <= 0:
        return 0.0

    retval, region = cv2.rotatedRectangleIntersection(ra, rb)
    if retval == 0 or region is None or len(region) < 3:
        return 0.0
    inter = float(cv2.contourArea(cv2.convexHull(region)))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _hbb_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def voc07_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """11-point interpolated AP, as in the PASCAL VOC 2007 devkit.

        AP = (1/11) * sum_{r in 0, 0.1, ..., 1.0} max{ p(r') : r' >= r }

    This is a coarser estimator than the all-point / 101-point interpolation
    COCO and Ultralytics use, which is exactly why it has to be computed
    separately rather than assumed equal.
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        mask = recalls >= t
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


def evaluate_voc07(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    task: str = "detect",
    iou_threshold: float = IOU_THRESHOLD,
) -> VOC07Report:
    """Per-class VOC07 AP@0.5 over a whole split."""
    iou_fn = _hbb_iou if task == "detect" else _obb_iou
    report = VOC07Report(task=task, iou_threshold=iou_threshold)

    for class_id, class_name in enumerate(DIOR_CLASSES):
        gts = [g for g in ground_truths if g.cls == class_id]
        dets = sorted(
            (d for d in detections if d.cls == class_id),
            key=lambda d: d.score,
            reverse=True,
        )
        report.per_class_gt[class_name] = len(gts)
        report.per_class_det[class_name] = len(dets)

        if not gts:
            report.per_class_ap[class_name] = 0.0
            continue
        if not dets:
            report.per_class_ap[class_name] = 0.0
            continue

        by_image: dict[str, list[GroundTruth]] = {}
        for g in gts:
            by_image.setdefault(g.image_id, []).append(g)
        matched: dict[str, np.ndarray] = {
            k: np.zeros(len(v), dtype=bool) for k, v in by_image.items()
        }

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))

        for i, det in enumerate(dets):
            candidates = by_image.get(det.image_id, [])
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(candidates):
                if matched[det.image_id][j]:
                    continue
                iou = iou_fn(det.coords, gt.coords)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0:
                matched[det.image_id][best_j] = True
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recalls = tp_cum / len(gts)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
        report.per_class_ap[class_name] = voc07_ap(recalls, precisions)

    return report
