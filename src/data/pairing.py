"""Pairing and inventory checks for the Hazy-DIOR release.

WHY THIS MODULE EXISTS
----------------------
The scaffold spec asked for a one-off confirmation that "Hazy-DIOR filenames map
1:1 onto clear DIOR filenames and that the splits match". They do not, and the
reasons are structural rather than a missing-files accident. This module encodes
every check that established that, so the finding is reproducible on the A6000
machine instead of living in a chat log.

WHAT THE RELEASE ACTUALLY CONTAINS
----------------------------------
23,463 DIOR ids, partitioned three ways by the *directory* layout:

  test/{gt,haze}/{thin,moderate,thick}/   2,607 ids   keyed by DIOR ID   .jpg
  train/{gt,haze}/                       18,770 ids   RENUMBERED 1..N    .png
  val/{gt,haze}/                          2,086 ids   RENUMBERED 1..N    .png
                                        ---------
                                         23,463

The 2,607 aligned ids are exactly the stride-9 subsample: every id congruent to
8 (mod 9). The train/ and val/ directories concatenate their ids three times
(once per haze severity) and renumber from 1, which destroys the DIOR id — so
`train/haze/00100.png` is NOT the hazy version of `Annotations/.../00100.xml`.

Separately, `ImageSets/Main/{train,val,test}.txt` is DIOR's *detection* split
(5,862 / 5,863 / 11,738), which is unrelated to the directory names above. This
project keys splits off `ImageSets/Main` and never off directory names: the
directory split places 10,433 DIOR *test* images into its train/ and val/
folders, which is the leak `05-experiment-plan.md` warns about first.

Consequence: only the 2,607 aligned ids can carry annotations, and those are
what the scaffold builds on.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..utils.paths import SourcePaths

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SEVERITIES = ("thin", "moderate", "thick")

# The aligned subset is every DIOR id where id % 9 == 8. Kept as named constants
# so the check reads as an assertion about the data rather than a magic number.
ALIGNED_STRIDE = 9
ALIGNED_RESIDUE = 8

TOTAL_DIOR_IDS = 23_463


@dataclass
class Check:
    """One named invariant, its verdict, and the evidence behind it."""

    name: str
    passed: bool
    detail: str
    evidence: dict = field(default_factory=dict)


def _stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        p.stem for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    }


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _find(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_splits(paths: SourcePaths) -> dict[str, list[str]]:
    """DIOR's detection splits from `ImageSets/Main/*.txt`."""
    splits: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        f = paths.imagesets_main / f"{name}.txt"
        if not f.exists():
            raise FileNotFoundError(f"missing split file: {f}")
        splits[name] = [
            line.strip() for line in f.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    return splits


def id_to_split(paths: SourcePaths) -> dict[str, str]:
    """Map every DIOR id to its detection split. The only split source we use."""
    mapping: dict[str, str] = {}
    for split, ids in read_splits(paths).items():
        for i in ids:
            mapping[i] = split
    return mapping


def aligned_ids(paths: SourcePaths) -> list[str]:
    """The DIOR-ID-keyed ids, read from disk (not computed from the stride rule).

    Reading them rather than generating them is deliberate: `check_stride_rule`
    then compares disk against the rule, which is a real check. Generating them
    would make that check vacuous.
    """
    return sorted(_stems(paths.aligned_clear_root / SEVERITIES[0]))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_inventory(paths: SourcePaths) -> Check:
    """Count every relevant directory and confirm the partition closes."""
    counts: dict[str, int] = {
        "annotations_hbb": len(list(paths.annotations_hbb.glob("*.xml"))),
        "annotations_obb": len(list(paths.annotations_obb.glob("*.xml"))),
    }
    for cond, root in (("clear", paths.aligned_clear_root), ("fog", paths.aligned_fog_root)):
        for sev in SEVERITIES:
            counts[f"test/{cond}/{sev}"] = len(_stems(root / sev))
    for split in ("train", "val"):
        for branch in ("gt", "haze"):
            counts[f"{split}/{branch}"] = len(_stems(paths.unaligned_root(split) / branch))

    n_aligned = counts["test/clear/thin"]
    n_train_unique = counts["train/gt"] // 3
    n_val_unique = counts["val/gt"] // 3
    total = n_aligned + n_train_unique + n_val_unique

    passed = (
        counts["annotations_hbb"] == TOTAL_DIOR_IDS
        and total == TOTAL_DIOR_IDS
        and counts["train/gt"] % 3 == 0
        and counts["val/gt"] % 3 == 0
    )
    return Check(
        name="inventory",
        passed=passed,
        detail=(
            f"{n_aligned:,} aligned + {n_train_unique:,} train-unique + "
            f"{n_val_unique:,} val-unique = {total:,} "
            f"(expected {TOTAL_DIOR_IDS:,}); "
            f"{counts['annotations_hbb']:,} HBB annotation files"
        ),
        evidence=counts | {
            "aligned": n_aligned,
            "train_unique": n_train_unique,
            "val_unique": n_val_unique,
            "partition_total": total,
        },
    )


def check_stride_rule(paths: SourcePaths) -> Check:
    """The aligned ids on disk are exactly {id : id % 9 == 8}."""
    ids = aligned_ids(paths)
    numeric = [int(i) for i in ids]
    violations = [i for i in numeric if i % ALIGNED_STRIDE != ALIGNED_RESIDUE]
    expected_count = TOTAL_DIOR_IDS // ALIGNED_STRIDE
    passed = not violations and len(ids) == expected_count
    return Check(
        name="aligned_ids_are_stride_9",
        passed=passed,
        detail=(
            f"{len(ids):,} aligned ids, {len(violations)} violating "
            f"id %% {ALIGNED_STRIDE} == {ALIGNED_RESIDUE}; "
            f"{len(ids):,} x {ALIGNED_STRIDE} = {len(ids) * ALIGNED_STRIDE:,} "
            f"vs {TOTAL_DIOR_IDS:,} total ids"
        ),
        evidence={
            "n_aligned": len(ids),
            "expected": expected_count,
            "violations": violations[:20],
            "first": ids[:5],
            "last": ids[-5:],
        },
    )


def check_severity_folders_share_ids(paths: SourcePaths) -> Check:
    """thin/moderate/thick cover the identical id set, for both clear and fog."""
    evidence: dict[str, int] = {}
    ok = True
    for cond, root in (("clear", paths.aligned_clear_root), ("fog", paths.aligned_fog_root)):
        sets = [_stems(root / sev) for sev in SEVERITIES]
        base = sets[0]
        for sev, s in zip(SEVERITIES, sets):
            evidence[f"{cond}/{sev}"] = len(s)
            if s != base:
                ok = False
    return Check(
        name="severity_folders_share_ids",
        passed=ok,
        detail="thin/moderate/thick carry the same ids in both gt/ and haze/"
        if ok
        else "SEVERITY FOLDERS DISAGREE on their id sets",
        evidence=evidence,
    )


def check_clear_is_severity_invariant(paths: SourcePaths, sample: int = 25) -> Check:
    """`gt/` is byte-identical across severity folders; `haze/` is not.

    This is what makes "clear = 2,607 unique images, fog = 7,821" true. If gt/
    ever differed per severity the clear condition would be ill-defined.
    """
    ids = aligned_ids(paths)
    step = max(1, len(ids) // sample)
    probe = ids[::step][:sample]

    gt_collapsed, haze_distinct, failures = 0, 0, []
    for i in probe:
        gt_hashes, haze_hashes = set(), set()
        for sev in SEVERITIES:
            g = _find(paths.aligned_clear_root / sev, i)
            z = _find(paths.aligned_fog_root / sev, i)
            if g:
                gt_hashes.add(_md5(g))
            if z:
                haze_hashes.add(_md5(z))
        if len(gt_hashes) == 1:
            gt_collapsed += 1
        else:
            failures.append(f"{i}: gt has {len(gt_hashes)} distinct hashes")
        if len(haze_hashes) == len(SEVERITIES):
            haze_distinct += 1
        else:
            failures.append(f"{i}: haze has {len(haze_hashes)} distinct hashes, expected 3")

    passed = not failures
    return Check(
        name="clear_is_severity_invariant",
        passed=passed,
        detail=(
            f"sampled {len(probe)} ids: gt identical across severities in "
            f"{gt_collapsed}/{len(probe)}, haze distinct in "
            f"{haze_distinct}/{len(probe)}"
        ),
        evidence={"sampled": len(probe), "failures": failures[:10]},
    )


def check_unaligned_are_renumbered(paths: SourcePaths) -> Check:
    """train/ and val/ repeat their images 3x under sequential renumbering.

    Proven by hash identity at the period, e.g. for train (period 18,770):
        gt/00001.png == gt/18771.png == gt/37541.png

    This is the finding that makes filename-based pairing unusable for 20,856
    of the 23,463 ids.
    """
    evidence: dict[str, object] = {}
    ok = True
    for split in ("train", "val"):
        gt = paths.unaligned_root(split) / "gt"
        stems = sorted(_stems(gt), key=int)
        if not stems:
            evidence[split] = "EMPTY"
            ok = False
            continue
        n = len(stems)
        if n % 3:
            evidence[split] = f"{n} files, not divisible by 3"
            ok = False
            continue
        period = n // 3
        matches, probes = 0, 0
        for offset in (1, period):  # first and last image of the first block
            group = []
            for k in range(3):
                p = _find(gt, f"{offset + k * period:05d}")
                if p:
                    group.append(_md5(p))
            if len(group) == 3:
                probes += 1
                if len(set(group)) == 1:
                    matches += 1
        renumbered = probes > 0 and matches == probes
        evidence[split] = {
            "files": n,
            "period": period,
            "unique_images": period,
            "probes": probes,
            "periodic_matches": matches,
            "renumbered": renumbered,
            "id_range": f"{stems[0]}..{stems[-1]}",
        }
        if not renumbered:
            ok = False

    return Check(
        name="unaligned_are_renumbered_3x",
        passed=ok,
        detail="train/ and val/ are 3x-repeated, sequentially renumbered; "
        "their filenames are NOT DIOR ids"
        if ok
        else "could not confirm the 3x renumbering — inspect manually",
        evidence=evidence,
    )


def check_annotation_coverage(paths: SourcePaths) -> Check:
    """Every aligned id has an HBB annotation file."""
    ids = aligned_ids(paths)
    missing = [i for i in ids if not (paths.annotations_hbb / f"{i}.xml").exists()]
    return Check(
        name="annotation_coverage",
        passed=not missing,
        detail=f"{len(ids) - len(missing):,}/{len(ids):,} aligned ids have an HBB xml",
        evidence={"missing": missing[:20], "n_missing": len(missing)},
    )


def check_split_membership(paths: SourcePaths) -> Check:
    """How the aligned ids distribute over DIOR's detection splits.

    Also quantifies the leak that the *directory* split would cause, which is
    the argument for keying off ImageSets/Main instead.
    """
    mapping = id_to_split(paths)
    ids = aligned_ids(paths)

    counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    for i in ids:
        counts[mapping.get(i, "unknown")] += 1

    all_test = sum(1 for v in mapping.values() if v == "test")
    leaked = all_test - counts["test"]

    return Check(
        name="split_membership",
        passed=counts["unknown"] == 0,
        detail=(
            f"aligned ids by DIOR split: train={counts['train']:,} "
            f"val={counts['val']:,} test={counts['test']:,} "
            f"unknown={counts['unknown']}. "
            f"Using the DIRECTORY split instead would put {leaked:,} DIOR-test "
            f"images into training."
        ),
        evidence=counts | {"dior_test_total": all_test, "would_leak": leaked},
    )


def check_image_dimensions(paths: SourcePaths) -> Check:
    """Every aligned image is 800x800 and agrees with its XML `<size>`.

    DIOR is uniform 800x800, which is why `03-datasets.md` says no tiling is
    required. Any image that is not 800x800 breaks that assumption, and — more
    importantly — a disagreement between the image and its XML makes the
    normalisation ambiguous: labels are normalised against the XML size, so if
    the render was *cropped* the boxes are wrong by the size ratio, while if it
    was *resized* they are correct. The two cases cannot be told apart without
    the clear DIOR original, which is not on disk.
    """
    from PIL import Image

    ids = aligned_ids(paths)
    mismatched: list[str] = []
    not_800: list[str] = []

    for image_id in ids:
        src = paths.aligned_clear_root / SEVERITIES[0] / f"{image_id}.jpg"
        if not src.exists():
            continue
        with Image.open(src) as im:
            size = im.size
        if size != (800, 800):
            not_800.append(f"{image_id}: {size[0]}x{size[1]}")

        xml = paths.annotations_hbb / f"{image_id}.xml"
        if xml.exists():
            import xml.etree.ElementTree as ET

            node = ET.parse(xml).getroot().find("size")
            if node is not None:
                declared = (
                    int(node.findtext("width")),  # type: ignore[arg-type]
                    int(node.findtext("height")),  # type: ignore[arg-type]
                )
                if declared != size:
                    mismatched.append(
                        f"{image_id}: xml {declared[0]}x{declared[1]}, "
                        f"image {size[0]}x{size[1]}"
                    )

    return Check(
        name="image_dimensions",
        passed=not mismatched and not not_800,
        detail=(
            f"{len(ids) - len(not_800):,}/{len(ids):,} aligned images are 800x800; "
            f"{len(mismatched)} disagree with their XML <size>"
        ),
        evidence={
            "n_not_800x800": len(not_800),
            "not_800x800": not_800[:20],
            "n_xml_mismatch": len(mismatched),
            "xml_mismatch": mismatched[:20],
        },
    )


ALL_CHECKS = (
    check_inventory,
    check_stride_rule,
    check_severity_folders_share_ids,
    check_clear_is_severity_invariant,
    check_unaligned_are_renumbered,
    check_annotation_coverage,
    check_split_membership,
    check_image_dimensions,
)


def run_all(paths: SourcePaths) -> list[Check]:
    return [fn(paths) for fn in ALL_CHECKS]


def checks_to_dict(checks: list[Check]) -> dict:
    return {
        "all_passed": all(c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
    }
