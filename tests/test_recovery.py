"""Tests for the renumbered-subtree id recovery.

The pixel-matching half needs the releases on disk and is exercised by
`scripts/recover_ids.py --verify-only`. What is unit-testable — and what would
mislabel the entire fog corpus if wrong — is the severity assignment: it turns
an index position into a haze severity, and every annotation attached to a fog
image depends on getting that right.
"""

from __future__ import annotations

from src.data.pairing import SEVERITIES
from src.data.recovery import RecoveryReport, _assign_severities, sources_by_id


def test_three_indices_per_id_map_to_severities_in_block_order():
    # The subtree is three concatenated blocks, so an id's lowest index is the
    # first severity and its highest is the last.
    resolved = {"00001": "00042", "18771": "00042", "37541": "00042"}
    severities, problems = _assign_severities(resolved)
    assert severities == {
        "00001": SEVERITIES[0],
        "18771": SEVERITIES[1],
        "37541": SEVERITIES[2],
    }
    assert problems == []


def test_sort_is_numeric_by_zero_padded_stem_not_insertion_order():
    resolved = {"37541": "00042", "00001": "00042", "18771": "00042"}
    severities, _ = _assign_severities(resolved)
    assert severities["00001"] == SEVERITIES[0]
    assert severities["37541"] == SEVERITIES[-1]


def test_incomplete_groups_are_dropped_not_guessed():
    """A partial group means a member failed to match.

    Assigning severities to the survivors would silently mislabel them — the
    two present might be thin+thick, not thin+moderate.
    """
    resolved = {"00001": "00042", "18771": "00042"}  # only two of three
    severities, problems = _assign_severities(resolved)
    assert severities == {}
    assert problems and "dropped" in problems[0]


def test_independent_ids_are_assigned_independently():
    resolved = {
        "00001": "00042", "18771": "00042", "37541": "00042",
        "00002": "00099", "18772": "00099", "37542": "00099",
    }
    severities, problems = _assign_severities(resolved)
    assert problems == []
    assert severities["00002"] == SEVERITIES[0]
    assert severities["37542"] == SEVERITIES[2]


def test_sources_by_id_inverts_the_mapping_keeping_the_subtree():
    report = RecoveryReport(
        mapping={
            "train/00001": {"image_id": "00042", "severity": "thin"},
            "train/18771": {"image_id": "00042", "severity": "moderate"},
            "val/00007": {"image_id": "00043", "severity": "thick"},
        }
    )
    out = sources_by_id(report)
    assert out["00042"]["thin"] == ("train", "00001")
    assert out["00042"]["moderate"] == ("train", "18771")
    # The subtree must survive inversion: it decides which directory to read.
    assert out["00043"]["thick"] == ("val", "00007")


def test_recovered_ids_deduplicates_across_severities():
    report = RecoveryReport(
        mapping={
            "train/00001": {"image_id": "00042", "severity": "thin"},
            "train/18771": {"image_id": "00042", "severity": "moderate"},
        }
    )
    assert report.recovered_ids == {"00042"}
