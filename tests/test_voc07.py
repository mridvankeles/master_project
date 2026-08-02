"""Tests for the VOC07 11-point AP used in the NIRNet comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.voc07 import (  # noqa: E402
    Detection,
    GroundTruth,
    _hbb_iou,
    _obb_iou,
    evaluate_voc07,
    voc07_ap,
)


# --------------------------------------------------------------------------
# the 11-point rule itself
# --------------------------------------------------------------------------


def test_perfect_detector_scores_one():
    recalls = np.linspace(0.1, 1.0, 10)
    precisions = np.ones(10)
    assert voc07_ap(recalls, precisions) == pytest.approx(1.0)


def test_empty_recall_scores_zero():
    assert voc07_ap(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0)


def test_eleven_point_differs_from_all_point():
    """The whole reason this module exists.

    A detector reaching recall 0.5 at precision 1.0 and nothing beyond scores
    6/11 under VOC07 (the r = 0, 0.1 ... 0.5 points) rather than 0.5.
    """
    recalls = np.array([0.25, 0.5])
    precisions = np.array([1.0, 1.0])
    assert voc07_ap(recalls, precisions) == pytest.approx(6 / 11)


def test_ap_is_monotone_in_precision():
    recalls = np.linspace(0.1, 1.0, 10)
    low = voc07_ap(recalls, np.full(10, 0.5))
    high = voc07_ap(recalls, np.full(10, 0.9))
    assert high > low


# --------------------------------------------------------------------------
# IoU
# --------------------------------------------------------------------------


def test_hbb_iou_identical_boxes():
    b = (0.0, 0.0, 10.0, 10.0)
    assert _hbb_iou(b, b) == pytest.approx(1.0)


def test_hbb_iou_disjoint():
    assert _hbb_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_hbb_iou_half_overlap():
    # two 10x10 boxes overlapping in a 5x10 strip: inter 50, union 150
    assert _hbb_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_obb_iou_identical_polygons():
    p = (0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0)
    assert _obb_iou(p, p) == pytest.approx(1.0, abs=1e-3)


def test_obb_iou_disjoint():
    a = (0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0)
    b = (50.0, 50.0, 60.0, 50.0, 60.0, 60.0, 50.0, 60.0)
    assert _obb_iou(a, b) == pytest.approx(0.0)


def test_obb_iou_matches_hbb_for_axis_aligned_boxes():
    """A rotated-IoU implementation must agree with the axis-aligned one when
    both boxes happen to be axis-aligned. Catches sign and ordering errors."""
    a_poly = (0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0)
    b_poly = (5.0, 0.0, 15.0, 0.0, 15.0, 10.0, 5.0, 10.0)
    assert _obb_iou(a_poly, b_poly) == pytest.approx(
        _hbb_iou((0, 0, 10, 10), (5, 0, 15, 10)), abs=1e-3
    )


def test_obb_iou_rotation_matters():
    """Same centre and size, 45 degrees apart: IoU well below 1."""
    square = (0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0)
    diamond = (5.0, -2.07, 12.07, 5.0, 5.0, 12.07, -2.07, 5.0)
    iou = _obb_iou(square, diamond)
    assert 0.3 < iou < 0.95


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_perfect_detections_give_ap_one():
    gts = [GroundTruth("a", 0, (0, 0, 10, 10)), GroundTruth("b", 0, (0, 0, 10, 10))]
    dets = [Detection("a", 0, 0.9, (0, 0, 10, 10)), Detection("b", 0, 0.8, (0, 0, 10, 10))]
    r = evaluate_voc07(dets, gts, task="detect")
    assert r.per_class_ap["airplane"] == pytest.approx(1.0)
    assert r.mean_ap == pytest.approx(1.0)


def test_absent_classes_excluded_from_mean():
    """Only classes with ground truth count, as in mmrotate. Counting the other
    19 DIOR classes as 0.0 would deflate the mean twentyfold."""
    gts = [GroundTruth("a", 0, (0, 0, 10, 10))]
    dets = [Detection("a", 0, 0.9, (0, 0, 10, 10))]
    r = evaluate_voc07(dets, gts, task="detect")
    assert r.mean_ap == pytest.approx(1.0)
    assert r.per_class_gt["ship"] == 0


def test_duplicate_ranked_last_does_not_reduce_ap():
    """A property of interpolated AP that is easy to get wrong.

    One ground truth, one correct detection, then a duplicate scored lower. The
    duplicate IS counted a false positive — but it lands after full recall is
    already reached, so the interpolated precision at every recall level is
    still 1.0 and AP stays 1.0. This is genuine VOC07 behaviour, not a bug: FPs
    ranked below every TP are free.
    """
    gts = [GroundTruth("a", 0, (0, 0, 10, 10))]
    dets = [
        Detection("a", 0, 0.9, (0, 0, 10, 10)),
        Detection("a", 0, 0.8, (0, 0, 10, 10)),
    ]
    r = evaluate_voc07(dets, gts, task="detect")
    assert r.per_class_ap["airplane"] == pytest.approx(1.0)


def test_duplicate_ranked_above_a_true_positive_does_reduce_ap():
    """The case that actually costs you: a false positive outranking a real hit.

    det1 TP (recall 0.5, precision 1.0), det2 duplicate FP (recall 0.5,
    precision 0.5), det3 TP on the other image (recall 1.0, precision 2/3).
    11-point AP = (6 * 1.0 + 5 * 2/3) / 11.
    """
    gts = [GroundTruth("a", 0, (0, 0, 10, 10)), GroundTruth("b", 0, (0, 0, 10, 10))]
    dets = [
        Detection("a", 0, 0.90, (0, 0, 10, 10)),
        Detection("a", 0, 0.85, (0, 0, 10, 10)),  # duplicate, outranks b's hit
        Detection("b", 0, 0.80, (0, 0, 10, 10)),
    ]
    r = evaluate_voc07(dets, gts, task="detect")
    expected = (6 * 1.0 + 5 * (2 / 3)) / 11
    assert r.per_class_ap["airplane"] == pytest.approx(expected)
    assert r.per_class_ap["airplane"] < 1.0


def test_wrong_class_is_not_matched():
    gts = [GroundTruth("a", 0, (0, 0, 10, 10))]
    dets = [Detection("a", 13, 0.9, (0, 0, 10, 10))]
    r = evaluate_voc07(dets, gts, task="detect")
    assert r.per_class_ap["airplane"] == pytest.approx(0.0)
    assert r.per_class_ap["ship"] == pytest.approx(0.0)


def test_below_threshold_iou_is_a_false_positive():
    gts = [GroundTruth("a", 0, (0, 0, 10, 10))]
    dets = [Detection("a", 0, 0.9, (9, 9, 19, 19))]  # tiny overlap
    r = evaluate_voc07(dets, gts, task="detect")
    assert r.per_class_ap["airplane"] == pytest.approx(0.0)


def test_no_detections_gives_zero():
    gts = [GroundTruth("a", 0, (0, 0, 10, 10))]
    r = evaluate_voc07([], gts, task="detect")
    assert r.mean_ap == pytest.approx(0.0)


def test_obb_end_to_end():
    poly = (0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0)
    gts = [GroundTruth("a", 5, poly)]
    dets = [Detection("a", 5, 0.9, poly)]
    r = evaluate_voc07(dets, gts, task="obb")
    assert r.per_class_ap["chimney"] == pytest.approx(1.0)
