"""Materialise the Ultralytics `detect` corpus from the read-only release.

LAYOUT AND WHY IT IS SHAPED THIS WAY
------------------------------------
    data/dior_hbb/
      clear/images/{train,val,test}/<id>.jpg
      clear/labels/{train,val,test}/<id>.txt
      fog/images/{train,val,test}/<id>_<severity>.jpg
      fog/labels/{train,val,test}/<id>_<severity>.txt
      manifest.csv

The `images/` <-> `labels/` sibling structure is not cosmetic. Ultralytics finds
a label by string-replacing the last `/images/` in the image path with
`/labels/`. Point it at the release tree directly and that substitution finds no
`/images/` segment, so it would try to write labels *into* the read-only source
directories. Hence a materialised copy.

Images are HARDLINKED, not copied (C: is NTFS, verified). 10,428 hardlinks cost
essentially nothing on disk while still giving Ultralytics the directory shape
it needs. Copying is the automatic fallback if the link fails — for instance
when data/ and the release live on different volumes, which will be the case on
the A6000 machine.

SPLITS come from `ImageSets/Main`, never from the release's directory names.
See `pairing.py` for why: the directory split would put 10,433 DIOR-test images
into training.
"""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..utils.paths import SourcePaths, ensure_dir
from .pairing import SEVERITIES, aligned_ids, id_to_split
from .voc_hbb import format_label_file, parse_hbb_xml

SPLITS = ("train", "val", "test")
CONDITIONS = ("clear", "fog")

# `gt/` is byte-identical across the three severity folders (asserted by
# pairing.check_clear_is_severity_invariant), so the clear image is read from
# whichever one we name here.
CLEAR_SOURCE_SEVERITY = SEVERITIES[0]


@dataclass
class ManifestRow:
    image_id: str
    condition: str
    severity: str
    split: str
    dst_image: str
    src_image: str
    n_boxes: int
    n_issues: int


@dataclass
class BuildReport:
    rows: list[ManifestRow]
    issues: list  # list[voc_hbb.BoxIssue]
    linked: int
    copied: int
    missing_sources: list[str]

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {c: dict.fromkeys(SPLITS, 0) for c in CONDITIONS}
        for row in self.rows:
            out[row.condition][row.split] += 1
        return out


def _link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink `src` to `dst`, falling back to a copy. Returns 'link'|'copy'."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        # Different volume, or a filesystem without hardlink support.
        shutil.copy2(src, dst)
        return "copy"


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (width, height)


def build(
    paths: SourcePaths,
    out_root: Path,
    conditions: tuple[str, ...] = CONDITIONS,
    verify_image_size: bool = True,
) -> BuildReport:
    """Convert annotations and lay out the corpus. Idempotent."""
    ids = aligned_ids(paths)
    split_of = id_to_split(paths)

    for cond in conditions:
        for split in SPLITS:
            ensure_dir(out_root / cond / "images" / split)
            ensure_dir(out_root / cond / "labels" / split)

    rows: list[ManifestRow] = []
    all_issues: list = []
    linked = copied = 0
    missing: list[str] = []

    for image_id in ids:
        split = split_of.get(image_id)
        if split is None:
            missing.append(f"{image_id}: not present in any ImageSets/Main list")
            continue

        xml = paths.annotations_hbb / f"{image_id}.xml"
        if not xml.exists():
            missing.append(f"{image_id}: no HBB annotation")
            continue

        # Each (condition, severity) variant shares the same source annotation:
        # haze changes the pixels, never the object geometry.
        variants: list[tuple[str, str, Path]] = []
        if "clear" in conditions:
            src = paths.aligned_clear_root / CLEAR_SOURCE_SEVERITY / f"{image_id}.jpg"
            variants.append(("clear", "none", src))
        if "fog" in conditions:
            for sev in SEVERITIES:
                variants.append(
                    ("fog", sev, paths.aligned_fog_root / sev / f"{image_id}.jpg")
                )

        size = None
        if verify_image_size and variants:
            first = variants[0][2]
            if first.exists():
                size = _image_size(first)

        ann = parse_hbb_xml(xml, expected_size=size)
        all_issues.extend(ann.issues)
        label_text = format_label_file(ann)

        for cond, sev, src in variants:
            if not src.exists():
                missing.append(f"{image_id} ({cond}/{sev}): missing image {src}")
                continue

            stem = image_id if cond == "clear" else f"{image_id}_{sev}"
            dst_img = out_root / cond / "images" / split / f"{stem}{src.suffix}"
            dst_lbl = out_root / cond / "labels" / split / f"{stem}.txt"

            how = _link_or_copy(src, dst_img)
            linked += how == "link"
            copied += how == "copy"
            dst_lbl.write_text(label_text, encoding="utf-8")

            rows.append(
                ManifestRow(
                    image_id=image_id,
                    condition=cond,
                    severity=sev,
                    split=split,
                    dst_image=str(dst_img.relative_to(out_root)).replace("\\", "/"),
                    src_image=str(src),
                    n_boxes=len(label_text.splitlines()),
                    n_issues=len(ann.issues),
                )
            )

    return BuildReport(
        rows=rows, issues=all_issues, linked=linked, copied=copied, missing_sources=missing
    )


def write_manifest(report: BuildReport, out_root: Path) -> Path:
    """One row per materialised image. The provenance record for the corpus."""
    path = out_root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "image_id",
                "condition",
                "severity",
                "split",
                "dst_image",
                "src_image",
                "n_boxes",
                "n_issues",
            ]
        )
        for r in report.rows:
            writer.writerow(
                [
                    r.image_id,
                    r.condition,
                    r.severity,
                    r.split,
                    r.dst_image,
                    r.src_image,
                    r.n_boxes,
                    r.n_issues,
                ]
            )
    return path


def write_data_yaml(out_root: Path, condition: str, dst: Path) -> Path:
    """Write the Ultralytics `data.yaml` for one condition.

    Paths are absolute: Ultralytics resolves relative `path:` against its own
    settings datasets_dir, which is not this repo.
    """
    from .dior_classes import DIOR_CLASSES

    root = (out_root / condition).resolve()
    lines = [
        f"# Auto-generated by scripts/prepare_dataset.py — do not edit by hand.",
        f"# Condition: {condition}. Splits come from DIOR ImageSets/Main.",
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(DIOR_CLASSES)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst
