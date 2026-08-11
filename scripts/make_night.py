"""Synthesise the `night` condition from the clear corpus.

    python scripts/make_night.py --scope full --task detect

Reads `<corpus>/clear/images/<split>/` and writes `<corpus>/night/...` with the
SAME labels — darkness changes pixels, never object geometry, exactly as haze
does not. Parameters are sampled per image from the DIOR id, so the corpus is
reproducible without storing them (see `src/data/degradation.py`).

WHY IT REUSES THE CLEAR SPLIT DIRECTORIES
-----------------------------------------
The clear corpus was already materialised from `ImageSets/Main`, so generating
night from it inherits the split assignment rather than re-deriving it. An id
cannot land in a different split for night than it did for clear, which is the
kind of quiet leak that is otherwise very hard to notice.

Writes `manifest_night.csv` with per-image parameters and before/after
statistics, so the degradation can be audited rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_dataset import SPLITS, write_data_yaml  # noqa: E402
from src.data.degradation import (  # noqa: E402
    apply_night,
    night_statistics,
    sample_night_params,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import CONFIG_DIR, dataset_root, ensure_dir  # noqa: E402

log = get_logger("make_night")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _one(job: tuple[str, str, str, int]) -> dict | None:
    """Worker: read a clear image, darken it, write it, return its record."""
    import cv2

    src_s, dst_s, image_id, seed = job
    src, dst = Path(src_s), Path(dst_s)
    bgr = cv2.imread(str(src))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    params = sample_night_params(image_id, seed=seed)
    night = apply_night(rgb, params)
    stats = night_statistics(rgb, night)

    # JPEG at high quality: the clear source is already JPEG, so writing PNG
    # here would give the night condition a different compression history from
    # clear and hand a router a shortcut (the confound recorded for fog in
    # results-full-scale-and-moe.md).
    cv2.imwrite(str(dst), cv2.cvtColor(night, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"image_id": image_id, **asdict(params), **stats}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="full", choices=["aligned", "full"])
    parser.add_argument("--task", default="detect", choices=["detect", "obb"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap images per split")
    args = parser.parse_args()

    root = dataset_root(args.task, args.scope)
    clear = root / "clear"
    if not clear.is_dir():
        log.error("no clear corpus at %s — run prepare_dataset.py --scope %s", clear, args.scope)
        return 2

    jobs: list[tuple[str, str, str, int]] = []
    for split in SPLITS:
        src_dir = clear / "images" / split
        dst_dir = ensure_dir(root / "night" / "images" / split)
        lbl_src = clear / "labels" / split
        lbl_dst = ensure_dir(root / "night" / "labels" / split)
        images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if args.limit:
            images = images[: args.limit]
        for p in images:
            jobs.append((str(p), str(dst_dir / f"{p.stem}.jpg"), p.stem, args.seed))
            # Labels are copied verbatim: darkness does not move objects.
            (lbl_dst / f"{p.stem}.txt").write_text(
                (lbl_src / f"{p.stem}.txt").read_text(encoding="utf-8"), encoding="utf-8"
            )
        log.info("%-5s queued %s images", split, f"{len(images):,}")

    log.info("generating %s night images with %d workers", f"{len(jobs):,}", args.workers)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(_one, jobs, chunksize=32), 1):
            if rec:
                rows.append(rec)
            if i % 4000 == 0:
                log.info("  %s/%s", f"{i:,}", f"{len(jobs):,}")

    manifest = root / "night" / "manifest_night.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%s rows)", manifest, f"{len(rows):,}")

    import numpy as np

    ratio = np.array([r["brightness_ratio"] for r in rows])
    hf_c = np.array([r["highfreq_clear"] for r in rows])
    hf_n = np.array([r["highfreq_night"] for r in rows])
    log.info("brightness ratio: mean %.3f  p05 %.3f  p95 %.3f", ratio.mean(),
             np.percentile(ratio, 5), np.percentile(ratio, 95))
    log.info("high-freq energy: clear %.2f -> night %.2f (sensor noise retained)",
             hf_c.mean(), hf_n.mean())
    if ratio.mean() > 0.6:
        log.error("night images are not dark enough — check the exposure range")
        return 1

    suffix = "" if args.task == "detect" else "_obb"
    scope = "" if args.scope == "aligned" else f"_{args.scope}"
    dst = write_data_yaml(root, "night",
                          CONFIG_DIR / "data" / f"dior_night{suffix}{scope}.yaml", task=args.task)
    log.info("wrote %s", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
