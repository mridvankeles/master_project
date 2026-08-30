"""Audit what the model ACTUALLY sees during training, batch by batch.

Static EDA describes files on disk. This describes tensors reaching the loss,
which is a different thing once augmentation is involved -- and the gap between
them is where this project has repeatedly found its bugs.

THREE QUESTIONS
---------------
1. **Does mosaic mix conditions?** Ultralytics composes each training image from
   four randomly drawn images but records only the FIRST one in `im_file`
   (`augment.py:759`, documented as "File path of the first image in the
   mosaic"). Our gate supervision derives its condition label from `im_file`.
   If mosaic mixes conditions, the gate is being told "this image is fog" about
   a canvas that is three-quarters something else.
2. **Do photometric augmentations attack the conditions themselves?**
   `hsv_v` jitters brightness and `hsv_s` jitters saturation -- which are the
   exact signals that distinguish night and fog. An augmentation that randomises
   the label's evidence is not a neutral regulariser here.
3. **Do the labels survive?** Class histogram and box geometry measured on the
   augmented tensors, not on the source files.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.models.moe2 import CONDITION_ALIASES, CONDITION_ORDER  # noqa: E402
from src.utils.config import load_run_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("audit_batches")


def condition_of(path: str) -> str:
    stem = Path(str(path)).stem.lower()
    for tok in stem.split("_"):
        tok = CONDITION_ALIASES.get(tok, tok)
        if tok in CONDITION_ORDER:
            return tok
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train/cond3b_gated_yolo11n.yaml")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    import yaml as _yaml

    # The subset yaml the run actually consumed, not the corpus yaml.
    sub = cfg.dataset_root / "subsets" / f"{cfg.run_name}.yaml"
    data_yaml = sub if sub.exists() else cfg.data_yaml
    spec = _yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = spec.get("names", {})
    data = {"train": spec["train"], "val": spec.get("val"), "names": names,
            "nc": len(names), "path": Path(spec["path"]), "channels": 3}

    overrides = get_cfg(DEFAULT_CFG, dict(cfg.train))
    overrides.imgsz = args.imgsz
    overrides.mode = "train"

    ds = build_yolo_dataset(overrides, str(spec["train"]), 16, data, mode="train", rect=False)
    loader = build_dataloader(ds, batch=16, workers=0, shuffle=True, rank=-1)

    log.info("dataset: %d images | mosaic=%s close_mosaic=%s hsv_v=%s hsv_s=%s",
             len(ds), overrides.mosaic, overrides.close_mosaic,
             overrides.hsv_v, overrides.hsv_s)

    # --- 1. what is inside a mosaic, versus what im_file claims ------------
    mixes, reported, cls_hist = [], Counter(), Counter()
    wh = []
    n = 0
    for bi, batch in enumerate(loader):
        if bi >= args.batches:
            break
        for f in batch["im_file"]:
            reported[condition_of(f)] += 1
        for c in batch["cls"].flatten().tolist():
            cls_hist[int(c)] += 1
        b = batch["bboxes"].numpy()
        if len(b):
            wh.append(b[:, 2:4])
        n += len(batch["im_file"])

    # Probe the mosaic directly: ask the dataset for the buffer it samples from.
    log.info("")
    log.info("=== 1. MOSAIC CONDITION MIXING ===")
    # Mosaic sits inside a NESTED Compose (pre_transform), so a flat scan of
    # ds.transforms.transforms misses it and wrongly reports mosaic as absent.
    def find(t, name, out):
        if type(t).__name__ == name:
            out.append(t)
        for attr in ("transforms", "tlist"):
            for c in getattr(t, attr, []) or []:
                find(c, name, out)
        return out

    tr = find(ds.transforms, "Mosaic", [])
    if not tr:
        log.info("  mosaic is NOT in the transform pipeline")
    else:
        m = tr[0]
        log.info("  Mosaic active: p=%s, n=%s", m.p, m.n)
        rng = np.random.default_rng(0)
        mismatches = 0
        trials = 300
        for _ in range(trials):
            i = int(rng.integers(0, len(ds)))
            idxs = [i] + m.get_indexes()
            conds = [condition_of(ds.im_files[j]) for j in idxs]
            label = conds[0]
            frac_label = sum(c == label for c in conds) / len(conds)
            mixes.append(frac_label)
            if frac_label < 1.0:
                mismatches += 1
        log.info("  images composing each mosaic: %d", len(idxs))
        log.info("  mosaics whose tiles are NOT all one condition: %d/%d = %.1f%%",
                 mismatches, trials, 100 * mismatches / trials)
        log.info("  mean fraction of the canvas matching the REPORTED label: %.3f",
                 float(np.mean(mixes)))
        log.info("  -> the gate is supervised with a one-hot label on a canvas")
        log.info("     that is on average %.0f%% other conditions",
                 100 * (1 - float(np.mean(mixes))))

    log.info("")
    log.info("=== 2. PHOTOMETRIC AUGMENTATION vs THE CONDITION SIGNAL ===")
    log.info("  hsv_v=%.2f  -> brightness jitter +-%.0f%%  (night is DEFINED by brightness)",
             overrides.hsv_v, 100 * overrides.hsv_v)
    log.info("  hsv_s=%.2f  -> saturation jitter +-%.0f%%  (haze DESATURATES)",
             overrides.hsv_s, 100 * overrides.hsv_s)
    log.info("  erasing=%.2f fliplr=%.2f scale=%.2f translate=%.2f",
             overrides.erasing, overrides.fliplr, overrides.scale, overrides.translate)

    log.info("")
    log.info("=== 3. LABELS AS DELIVERED (augmented tensors, %d images) ===", n)
    inv = {v: k for k, v in names.items()} if isinstance(names, dict) else {}
    total = sum(cls_hist.values())
    for c, k in sorted(cls_hist.items(), key=lambda x: -x[1])[:8]:
        nm = names.get(c, c) if isinstance(names, dict) else c
        log.info("  %-24s %6d (%.1f%%)", nm, k, 100 * k / max(total, 1))
    log.info("  classes present: %d/%d", len(cls_hist), len(names))
    if wh:
        W = np.concatenate(wh)
        log.info("  normalised box w/h: median %.4f/%.4f  p05 %.4f  p95 %.4f",
                 float(np.median(W[:, 0])), float(np.median(W[:, 1])),
                 float(np.percentile(W, 5)), float(np.percentile(W, 95)))
        degenerate = int(((W[:, 0] <= 0) | (W[:, 1] <= 0)).sum())
        log.info("  degenerate boxes (w or h <= 0): %d", degenerate)
    log.info("  condition reported by im_file: %s", dict(reported))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
