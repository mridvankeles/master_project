"""Structural invariants of the Hazy-DIOR release.

These need the actual data, so they skip when it is absent — the suite still
passes on a clean machine before anything is downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pairing import (  # noqa: E402
    ALIGNED_RESIDUE,
    ALIGNED_STRIDE,
    TOTAL_DIOR_IDS,
    aligned_ids,
    check_annotation_coverage,
    check_clear_is_severity_invariant,
    check_image_dimensions,
    check_inventory,
    check_split_membership,
    check_stride_rule,
    check_unaligned_are_renumbered,
    id_to_split,
    read_splits,
)
from src.utils.paths import DEFAULT_PATHS_YAML, load_paths  # noqa: E402


def _paths_or_skip():
    if not DEFAULT_PATHS_YAML.exists():
        pytest.skip("configs/paths.yaml not present")
    paths = load_paths()
    problems = paths.validate()
    if problems:
        pytest.skip(f"Hazy-DIOR release not available: {problems[0]}")
    return paths


@pytest.fixture(scope="module")
def paths():
    return _paths_or_skip()


def test_dior_detection_splits(paths):
    """ImageSets/Main is DIOR's published detection split. Sizes are fixed."""
    splits = read_splits(paths)
    assert len(splits["train"]) == 5_862
    assert len(splits["val"]) == 5_863
    assert len(splits["test"]) == 11_738
    assert sum(len(v) for v in splits.values()) == TOTAL_DIOR_IDS


def test_splits_are_disjoint(paths):
    splits = read_splits(paths)
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test


def test_aligned_ids_are_stride_9(paths):
    """The DIOR-ID-keyed subset is exactly {id : id % 9 == 8}."""
    check = check_stride_rule(paths)
    assert check.passed, check.detail
    ids = aligned_ids(paths)
    assert len(ids) == TOTAL_DIOR_IDS // ALIGNED_STRIDE == 2_607
    assert all(int(i) % ALIGNED_STRIDE == ALIGNED_RESIDUE for i in ids)


def test_inventory_partition_closes(paths):
    """aligned + train-unique + val-unique == every DIOR id, exactly."""
    check = check_inventory(paths)
    assert check.passed, check.detail
    assert check.evidence["aligned"] == 2_607
    assert check.evidence["train_unique"] == 18_770
    assert check.evidence["val_unique"] == 2_086
    assert check.evidence["partition_total"] == TOTAL_DIOR_IDS


def test_unaligned_directories_are_renumbered(paths):
    """train/ and val/ filenames are sequential indices, NOT DIOR ids.

    This is the finding that makes filename pairing unusable for 20,856 images.
    """
    check = check_unaligned_are_renumbered(paths)
    assert check.passed, check.detail
    assert check.evidence["train"]["period"] == 18_770
    assert check.evidence["val"]["period"] == 2_086


def test_clear_is_identical_across_severities(paths):
    """gt/ is severity-invariant; haze/ is not. Makes 'clear' well-defined."""
    check = check_clear_is_severity_invariant(paths)
    assert check.passed, check.detail


def test_every_aligned_id_has_an_annotation(paths):
    check = check_annotation_coverage(paths)
    assert check.passed, check.detail


def test_aligned_split_membership(paths):
    """651 / 651 / 1305 across DIOR's train / val / test."""
    check = check_split_membership(paths)
    assert check.passed, check.detail
    assert check.evidence["train"] == 651
    assert check.evidence["val"] == 651
    assert check.evidence["test"] == 1_305
    assert check.evidence["unknown"] == 0


def test_directory_split_would_leak(paths):
    """Quantifies why splits are keyed off ImageSets/Main and not directories."""
    check = check_split_membership(paths)
    assert check.evidence["would_leak"] == 10_433


def test_no_aligned_id_is_unassigned(paths):
    mapping = id_to_split(paths)
    assert all(i in mapping for i in aligned_ids(paths))


def test_image_dimension_anomalies_are_exactly_the_known_one(paths):
    """Pins the one known defect so a change in the data surfaces immediately.

    Image 15776 is 800x787 in every variant while its XML declares 800x800. It
    is the only such image. Asserting the known state rather than zero means
    this test fails if the anomaly is fixed, spreads, or moves — all of which
    are things worth being told about.
    """
    check = check_image_dimensions(paths)
    assert check.evidence["n_not_800x800"] == 1, check.detail
    assert check.evidence["n_xml_mismatch"] == 1, check.detail
    assert check.evidence["not_800x800"] == ["15776: 800x787"]
