"""Recover DIOR ids and haze severities for the renumbered Hazy-DIOR subtrees.

THE PROBLEM
-----------
`pairing.py` establishes that only 2,607 of the release's 23,463 ids survive
with their filename intact. The other 20,856 live in `train/` and `val/`, whose
filenames are sequential indices (`00001.png`), so the DIOR id — and therefore
the annotation — is unknown. Without boxes those images cannot be used for
detection, which is why the corpus has been built on 651 training scenes.

THE RECOVERY
------------
Those subtrees are 800x800 *lossless PNG*, and their `gt` (clear) images are
re-encodings of DIOR originals. PNG preserves the decoded pixel array exactly,
so a clear image can be matched back to its DIOR id by hashing decoded pixels.
This only became possible once the full DIOR release was on disk.

Measured on this release: 56,097/56,310 of `train/gt` and 6,237/6,258 of
`val/gt` resolve uniquely, recovering 20,778 DIOR ids. With the 2,607 already
aligned that is 23,385 of 23,463 — 99.5%+ of every official split.

TWO THINGS THIS MODULE PROVES RATHER THAN ASSUMES
-------------------------------------------------
1. **gt <-> haze index correspondence.** The mapping is recovered from clear
   images; using it for the hazy ones assumes `haze/N.png` depicts the same
   scene as `gt/N.png`. Measured structural correlation: 0.756 for same-index
   pairs against -0.022 for strictly different scenes.
2. **Which severity each index is.** Every id appears exactly three times at a
   constant index stride, i.e. the subtree is three concatenated blocks, one per
   severity. Block order is thin -> moderate -> thick, confirmed by mean
   brightness (123.6 / 139.8 / 156.8) against the *named* severity folders in
   `test/haze/` (123.1 / 138.5 / 155.6) — haze adds airlight, so denser haze is
   brighter. `verify_recovery` re-checks both on every run rather than trusting
   this docstring.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
Splits still come from `ImageSets/Main`. The subtree an image sits in
(`train/` vs `val/`) is a *restoration* split and disagrees with DIOR's
detection split — `train/gt` alone contains 9,360 DIOR-test ids. Keying off it
would be the exact leak `05-experiment-plan.md` lists first.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.paths import SourcePaths
from .pairing import SEVERITIES

# Ids whose clear image is pixel-identical to another id's. DIOR genuinely
# contains a few duplicate scenes, so these cannot be resolved by pixel hash and
# are excluded rather than guessed at.
AMBIGUOUS = "ambiguous"
UNMATCHED = "unmatched"

RENUMBERED_SPLITS = ("train", "val")


@dataclass
class RecoveryReport:
    """One entry per renumbered subtree, plus the recovered mapping."""

    # (split, index_stem) -> {"image_id": str, "severity": str}
    mapping: dict[str, dict[str, str]] = field(default_factory=dict)
    stats: dict[str, dict[str, int]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def recovered_ids(self) -> set[str]:
        return {v["image_id"] for v in self.mapping.values()}

    def to_dict(self) -> dict:
        return {
            "schema": 1,
            "note": (
                "index stems keyed as '<split>/<stem>'; severity derived from "
                "block position, verified by brightness ordering"
            ),
            "stats": self.stats,
            "problems": self.problems,
            "mapping": self.mapping,
        }


def pixel_hash(path: Path) -> str:
    """sha256 over the decoded RGB pixel array.

    Not a file hash: the same image stored as JPEG and as PNG has different
    bytes but identical pixels, and matching across that boundary is the whole
    point.
    """
    from PIL import Image

    with Image.open(path) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


def build_dior_index(paths: SourcePaths, progress=None) -> tuple[dict[str, str], int]:
    """Pixel hash -> DIOR id, over the whole clear release.

    Hashes colliding across two ids are dropped, not resolved: a duplicate scene
    has no single correct id, and a wrong id means wrong boxes.
    """
    by_hash: dict[str, list[str]] = defaultdict(list)
    files = sorted(paths.dior_images_trainval.glob("*.jpg")) + sorted(
        paths.dior_images_test.glob("*.jpg")
    )
    for n, path in enumerate(files, 1):
        by_hash[pixel_hash(path)].append(path.stem)
        if progress and n % 4000 == 0:
            progress(n, len(files))
    index = {h: ids[0] for h, ids in by_hash.items() if len(ids) == 1}
    return index, len(by_hash) - len(index)


def recover(paths: SourcePaths, progress=None) -> RecoveryReport:
    """Resolve every renumbered clear image to a DIOR id and a severity."""
    report = RecoveryReport()
    index, n_colliding = build_dior_index(paths, progress=progress)
    report.stats["dior_index"] = {
        "distinct_hashes": len(index) + n_colliding,
        "unique": len(index),
        "colliding_dropped": n_colliding,
    }

    for split in RENUMBERED_SPLITS:
        gt_dir = paths.renumbered_root(split, "gt")
        files = sorted(p for p in gt_dir.iterdir() if p.is_file())
        resolved: dict[str, str] = {}
        n_unmatched = 0
        for n, path in enumerate(files, 1):
            image_id = index.get(pixel_hash(path))
            if image_id is None:
                n_unmatched += 1
            else:
                resolved[path.stem] = image_id
            if progress and n % 4000 == 0:
                progress(n, len(files), split)

        severities, issues = _assign_severities(resolved)
        report.problems.extend(issues)
        for stem, image_id in resolved.items():
            if stem in severities:
                report.mapping[f"{split}/{stem}"] = {
                    "image_id": image_id,
                    "severity": severities[stem],
                }

        report.stats[split] = {
            "files": len(files),
            "matched": len(resolved),
            "unmatched": n_unmatched,
            "with_severity": sum(1 for s in resolved if s in severities),
            "unique_ids": len(set(resolved.values())),
        }
    return report


def _assign_severities(resolved: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Assign thin/moderate/thick by block position within the subtree.

    Each DIOR id contributes exactly one image per severity, and the subtree is
    laid out as three concatenated blocks, so sorting an id's indices puts them
    in block order. Ids that did not resolve to exactly three indices are
    dropped: a partial group means one of its members failed to match, and
    guessing which severity the survivors are would be inventing data.
    """
    by_id: dict[str, list[str]] = defaultdict(list)
    for stem, image_id in resolved.items():
        by_id[image_id].append(stem)

    severities: dict[str, str] = {}
    incomplete = 0
    for image_id, stems in by_id.items():
        if len(stems) != len(SEVERITIES):
            incomplete += 1
            continue
        for position, stem in enumerate(sorted(stems)):
            severities[stem] = SEVERITIES[position]

    problems = []
    if incomplete:
        problems.append(
            f"{incomplete} id(s) resolved to != {len(SEVERITIES)} indices and were dropped"
        )
    return severities, problems


def verify_recovery(
    paths: SourcePaths, report: RecoveryReport, sample: int = 40, seed: int = 0
) -> list:
    """Re-check the two assumptions the mapping rests on. Returns Checks."""
    import random

    import numpy as np

    from .pairing import Check

    def thumb(path: Path):
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)

    def corr(a, b) -> float:
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float((a * b).mean())

    rng = random.Random(seed)
    checks = []

    # --- 1. gt <-> haze index correspondence ---------------------------
    keys = [k for k in report.mapping if k.startswith("train/")]
    picks = rng.sample(sorted(keys), min(sample, len(keys)))
    same, different = [], []
    for key in picks:
        stem = key.split("/", 1)[1]
        gt = thumb(paths.renumbered_root("train", "gt") / f"{stem}.png")
        haze = thumb(paths.renumbered_root("train", "haze") / f"{stem}.png")
        same.append(corr(gt, haze))
        # Control must never pair a stem with itself, or the "unrelated" sample
        # silently includes correct pairs and the contrast collapses.
        other = rng.choice([k for k in picks if k != key]).split("/", 1)[1]
        different.append(
            corr(thumb(paths.renumbered_root("train", "gt") / f"{other}.png"), haze)
        )

    mean_same, mean_diff = float(np.mean(same)), float(np.mean(different))
    checks.append(
        Check(
            name="haze_index_matches_gt_index",
            passed=mean_same > 0.4 and mean_same - mean_diff > 0.3,
            detail=(
                f"same-index gt/haze correlation {mean_same:.3f} vs "
                f"{mean_diff:.3f} for different scenes"
            ),
            evidence={"same_index": mean_same, "different_scene": mean_diff, "n": len(picks)},
        )
    )

    # --- 2. block order really is thin -> moderate -> thick ------------
    brightness: dict[str, list[float]] = {s: [] for s in SEVERITIES}
    for key in picks:
        split, stem = key.split("/", 1)
        brightness[report.mapping[key]["severity"]].append(
            float(thumb(paths.renumbered_root(split, "haze") / f"{stem}.png").mean())
        )
    means = {s: float(np.mean(v)) if v else float("nan") for s, v in brightness.items()}
    ordered = [means[s] for s in SEVERITIES]
    checks.append(
        Check(
            name="severity_order_matches_brightness",
            passed=all(a < b for a, b in zip(ordered, ordered[1:])),
            detail=(
                "mean brightness by assigned severity: "
                + ", ".join(f"{s}={means[s]:.1f}" for s in SEVERITIES)
                + " (haze adds airlight, so this must increase)"
            ),
            evidence=means,
        )
    )
    return checks


def write_mapping(report: RecoveryReport, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return dst


def load_mapping(path: Path) -> RecoveryReport:
    """Read a cached mapping. Recovery costs ~25 min, so it is an artefact."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RecoveryReport(
        mapping=raw["mapping"], stats=raw.get("stats", {}), problems=raw.get("problems", [])
    )


def sources_by_id(report: RecoveryReport) -> dict[str, dict[str, tuple[str, str]]]:
    """Invert the mapping: DIOR id -> severity -> (subtree split, index stem).

    This is the form `build_dataset` needs — it iterates ids and asks where the
    imagery for each condition lives.
    """
    out: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for key, entry in report.mapping.items():
        split, stem = key.split("/", 1)
        out[entry["image_id"]][entry["severity"]] = (split, stem)
    return dict(out)
