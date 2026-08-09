"""Materialise the `union` condition: clear + fog in one corpus.

    python scripts/make_union.py --scope full --task detect

WHY A REAL DIRECTORY RATHER THAN AN IMAGE LIST
----------------------------------------------
The union corpus is consumed by two different models that must be compared:
the dense baseline and the MoE. Anything that could let them see different data
invalidates the comparison, so the union is materialised once, as hardlinks,
and both read the same directory. `manifest_union.csv` records exactly what
went in.

BALANCE
-------
Fog carries three severities per id, so pooling everything would give a corpus
that is ~75% fog. A router trained on that has a strong prior before it has
learned anything, and the dense baseline sees a condition mix no one chose.
Train and val are therefore sampled 50/50 by condition, with fog's share spread
evenly across the three severities.

TEST IS NEVER SUBSAMPLED OR BALANCED. It is every clear test image plus every
fog test image, because a test set is supposed to be the population, not a
convenience sample.

SPLIT INTEGRITY
---------------
Images are drawn from the already-split clear/ and fog/ trees, which were keyed
off `ImageSets/Main`. No id can therefore cross a split boundary here, and
`--verify` re-checks that rather than assuming it.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_dataset import SPLITS, write_data_yaml  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import CONFIG_DIR, dataset_root, ensure_dir  # noqa: E402

log = get_logger("make_union")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Per-split budget for the union corpus. train matches the clear arm's 5,862 so
# every arm takes the same number of optimizer steps per epoch; val is capped so
# validation cost is identical across arms rather than three times larger for
# whichever condition happens to carry more renders.
BUDGET = {"train": 5862, "val": 2000, "test": None}  # None = take everything


def _labels_for(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _pick(images: list[Path], n: int, rng: random.Random) -> list[Path]:
    """Sample n images, spreading fog evenly over its three severities.

    Fog stems are `<id>_<severity>`; clear stems are bare ids, so the grouping
    collapses to a single bucket for clear and the same code path serves both.
    """
    if n >= len(images):
        return sorted(images)
    buckets: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        _, _, sev = p.stem.rpartition("_")
        buckets[sev if sev in {"thin", "moderate", "thick"} else ""].append(p)

    picked: list[Path] = []
    keys = sorted(buckets)
    per = n // len(keys)
    for k in keys:
        pool = sorted(buckets[k])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    # top up any rounding shortfall from whatever is left
    if len(picked) < n:
        rest = sorted(set(images) - set(picked))
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return sorted(picked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="full", choices=["aligned", "full"])
    parser.add_argument("--task", default="detect", choices=["detect", "obb"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-condition", default="union")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = dataset_root(args.task, args.scope)
    if not root.is_dir():
        log.error("no corpus at %s — run prepare_dataset.py --scope %s", root, args.scope)
        return 2

    union = root / args.out_condition
    rows: list[dict] = []

    for split in SPLITS:
        out_img = ensure_dir(union / "images" / split)
        out_lbl = ensure_dir(union / "labels" / split)
        for stale in list(out_img.iterdir()) + list(out_lbl.iterdir()):
            stale.unlink()

        budget = BUDGET[split]
        per_condition = None if budget is None else budget // 2

        for condition in ("clear", "fog"):
            src_dir = root / condition / "images" / split
            available = [p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            chosen = (
                sorted(available)
                if per_condition is None
                else _pick(available, per_condition, rng)
            )
            for img in chosen:
                # Prefix so a clear id and a fog id can never collide, and so the
                # condition of every union image is readable from its filename —
                # which is what the router's supervision reads.
                stem = f"{condition}_{img.stem}"
                _link_or_copy(img, out_img / f"{stem}{img.suffix}")
                _link_or_copy(_labels_for(img), out_lbl / f"{stem}.txt")
                rows.append(
                    {
                        "stem": stem,
                        "condition": condition,
                        "split": split,
                        "src_image": str(img),
                    }
                )
            log.info("%-5s %-5s %6d images", split, condition, len(chosen))

    manifest = union / "manifest_union.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem", "condition", "split", "src_image"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r["split"], r["condition"]) for r in rows)
    log.info("union corpus: %s", dict(counts))

    suffix = "" if args.task == "detect" else "_obb"
    scope = "" if args.scope == "aligned" else f"_{args.scope}"
    dst = write_data_yaml(
        root,
        args.out_condition,
        CONFIG_DIR / "data" / f"dior_{args.out_condition}{suffix}{scope}.yaml",
        task=args.task,
    )
    log.info("wrote %s", dst)
    log.info("wrote %s (%s rows)", manifest, f"{len(rows):,}")

    # An id must not appear in two splits. It cannot, given the inputs, which is
    # exactly why it is worth asserting: the check is cheap and the failure mode
    # is silent.
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["stem"].split("_")[1])
    ok = True
    for a in SPLITS:
        for b in SPLITS:
            if a < b and by_split[a] & by_split[b]:
                log.error("LEAK: %d ids shared between %s and %s", len(by_split[a] & by_split[b]), a, b)
                ok = False
    log.info("split disjointness: %s", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
