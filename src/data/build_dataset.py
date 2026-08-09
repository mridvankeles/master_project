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
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.paths import SourcePaths, ensure_dir
from .pairing import SEVERITIES, aligned_ids, id_to_split
from .voc_hbb import format_label_file, parse_hbb_xml
from .voc_obb import format_obb_label_file, parse_obb_xml

SPLITS = ("train", "val", "test")
CONDITIONS = ("clear", "fog")
TASKS = ("detect", "obb")

# `aligned` — the 2,607 DIOR-ID-keyed ids in the release's `test/` subtree. The
#   original scaffold behaviour, and the only option before the DIOR download.
# `full`    — every id in ImageSets/Main. Clear comes from the official DIOR
#   release; fog additionally draws on the renumbered train/ and val/ subtrees
#   via the recovered mapping (`recovery.py`), which lifts fog from 2,607 ids to
#   ~23,385 — 99.5%+ of every official split.
SCOPES = ("aligned", "full")

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
    # Which release subtree the pixels came from, and in what container. Both
    # are provenance the thesis needs: see BuildReport.format_counts.
    source_subtree: str = ""
    source_format: str = ""


@dataclass
class BuildReport:
    rows: list[ManifestRow]
    issues: list  # list[voc_hbb.BoxIssue]
    linked: int
    copied: int
    missing_sources: list[str]
    # Ids that CANNOT have a fog render, because their clear image is pixel
    # identical to another DIOR image and recovery refuses to guess between
    # them. Distinct from `missing_sources`: a missing source is a surprise and
    # should fail the build, whereas these are a known, bounded, documented
    # property of the release. Conflating them would make every full-scope
    # build exit non-zero and train people to ignore the exit code.
    unrecoverable: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {c: dict.fromkeys(SPLITS, 0) for c in CONDITIONS}
        for row in self.rows:
            out[row.condition][row.split] += 1
        return out

    def format_counts(self) -> dict[str, dict[str, int]]:
        """Image container per condition — a confound worth seeing, not hiding.

        In `full` scope clear comes from DIOR (JPEG) while most fog comes from
        the renumbered subtrees (PNG). The containers differ, and so does the
        compression history underneath: the PNG fog renders were never JPEG
        compressed, while every clear image was. A router keyed on high-frequency
        statistics — which is exactly what `04-method-open-questions.md` proposes
        for domain robustness — could separate the conditions on JPEG artefacts
        alone and never learn anything about haze.

        This cannot be fixed by re-encoding, because the difference is in the
        images' history rather than their storage. It has to be measured and
        stated, so the counts are surfaced here and recorded per row.
        """
        out: dict[str, dict[str, int]] = {}
        for row in self.rows:
            out.setdefault(row.condition, {}).setdefault(row.source_format, 0)
            out[row.condition][row.source_format] += 1
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


def _clear_source(paths: SourcePaths, image_id: str, scope: str) -> Path | None:
    """Where the clear image for `image_id` lives.

    In `full` scope every clear image comes from the official DIOR release, even
    for the 2,607 aligned ids that Hazy-DIOR also carries. Those were verified
    byte-identical, so preferring one source is free — and a single source means
    one provenance to state rather than two.
    """
    if scope == "aligned":
        return paths.aligned_clear_root / CLEAR_SOURCE_SEVERITY / f"{image_id}.jpg"
    return paths.dior_image(image_id)


def _fog_sources(
    paths: SourcePaths,
    image_id: str,
    aligned: set[str],
    recovered: dict[str, dict[str, tuple[str, str]]],
) -> list[tuple[str, Path, str]]:
    """(severity, path, subtree) for every haze render of `image_id`."""
    if image_id in aligned:
        return [
            (sev, paths.aligned_fog_root / sev / f"{image_id}.jpg", "test")
            for sev in SEVERITIES
        ]
    return [
        (sev, paths.renumbered_root(split, "haze") / f"{stem}.png", split)
        for sev, (split, stem) in sorted(recovered.get(image_id, {}).items())
    ]


def build(
    paths: SourcePaths,
    out_root: Path,
    conditions: tuple[str, ...] = CONDITIONS,
    verify_image_size: bool = True,
    task: str = "detect",
    scope: str = "aligned",
    id_map: dict[str, dict[str, tuple[str, str]]] | None = None,
) -> BuildReport:
    """Convert annotations and lay out the corpus. Idempotent.

    `task` selects which annotation set is read and which label format is
    written: "detect" -> HBB xml, `class cx cy w h`; "obb" -> oriented xml,
    `class x1 y1 ... x4 y4`. The two produce separate dataset roots so an OBB
    label can never be handed to a detect model, or the reverse — a mix-up that
    trains happily and reports nonsense.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    if scope == "full" and paths.dior_root is None:
        raise ValueError("scope='full' needs `dior_root` in configs/paths.yaml")

    ann_dir = paths.annotations_hbb if task == "detect" else paths.annotations_obb
    parse = parse_hbb_xml if task == "detect" else parse_obb_xml
    render = format_label_file if task == "detect" else format_obb_label_file

    split_of = id_to_split(paths)
    aligned = set(aligned_ids(paths))
    recovered = id_map or {}
    # `aligned` scope reads its ids off disk; `full` takes them from
    # ImageSets/Main, which is the only split definition this project trusts.
    ids = sorted(aligned) if scope == "aligned" else sorted(split_of)

    for cond in conditions:
        for split in SPLITS:
            ensure_dir(out_root / cond / "images" / split)
            ensure_dir(out_root / cond / "labels" / split)

    rows: list[ManifestRow] = []
    all_issues: list = []
    linked = copied = 0
    missing: list[str] = []
    unrecoverable: list[str] = []

    for image_id in ids:
        split = split_of.get(image_id)
        if split is None:
            missing.append(f"{image_id}: not present in any ImageSets/Main list")
            continue

        xml = ann_dir / f"{image_id}.xml"
        if not xml.exists():
            missing.append(f"{image_id}: no {task} annotation")
            continue

        # Each (condition, severity) variant shares the same source annotation:
        # haze changes the pixels, never the object geometry.
        variants: list[tuple[str, str, Path, str]] = []
        if "clear" in conditions:
            src = _clear_source(paths, image_id, scope)
            if src is None:
                missing.append(f"{image_id} (clear): no image in the DIOR release")
            else:
                variants.append(
                    ("clear", "none", src, "dior" if scope == "full" else "test")
                )
        if "fog" in conditions:
            fog = _fog_sources(paths, image_id, aligned, recovered)
            if not fog and scope == "full":
                # Known and bounded (78 ids on this release), not a surprise.
                unrecoverable.append(f"{image_id}: no recovered haze render")
            variants.extend(("fog", sev, path, subtree) for sev, path, subtree in fog)

        size = None
        if verify_image_size and variants:
            first = variants[0][2]
            if first.exists():
                size = _image_size(first)

        ann = parse(xml, expected_size=size)
        # Render BEFORE collecting issues: the OBB renderer appends a
        # `clipped_<method>` issue for every out-of-bounds polygon it refits, and
        # collecting first would drop exactly the records we most want to see.
        label_text = render(ann)
        all_issues.extend(ann.issues)

        for cond, sev, src, subtree in variants:
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
                    source_subtree=subtree,
                    source_format=src.suffix.lstrip(".").lower(),
                )
            )

    return BuildReport(
        rows=rows,
        issues=all_issues,
        linked=linked,
        copied=copied,
        missing_sources=missing,
        unrecoverable=unrecoverable,
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
                "source_subtree",
                "source_format",
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
                    r.source_subtree,
                    r.source_format,
                ]
            )
    return path


def write_data_yaml(out_root: Path, condition: str, dst: Path, task: str = "detect") -> Path:
    """Write the Ultralytics `data.yaml` for one condition.

    Paths are absolute: Ultralytics resolves relative `path:` against its own
    settings datasets_dir, which is not this repo.
    """
    from .dior_classes import DIOR_CLASSES

    root = (out_root / condition).resolve()
    lines = [
        f"# Auto-generated by scripts/prepare_dataset.py — do not edit by hand.",
        f"# Condition: {condition}. Task: {task}. Splits come from DIOR ImageSets/Main.",
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
