"""EDA for HazyDet, with the questions this thesis actually needs answered.

    python scripts/eda_hazydet.py --root C:/Users/Ridvan/Desktop/tez/hazydet

Not a generic dataset summary. Three things decide whether HazyDet is usable
here, and each gets a section:

1. **Taxonomy compatibility.** HazyDet is car/truck/bus from a drone; DIOR is 20
   classes from satellites. If the class sets do not overlap usefully, HazyDet
   cannot be a second training corpus and can only ever be a separate arm.
2. **Object scale.** `01-scope-and-claim.md` proposes a tiny-object expert. That
   only makes sense if the objects really are tiny, so absolute and relative box
   sizes are measured against the COCO small/medium/large thresholds.
3. **Haze strength.** The release ships paired clear/hazy images, so the
   degradation can be measured directly rather than assumed — dark-channel and
   contrast statistics on matched pairs quantify how hard this haze actually is,
   and how it compares with Hazy-DIOR's three fixed severities.

Writes JSON + a markdown summary to outputs/eda_hazydet/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, ensure_dir  # noqa: E402

log = get_logger("eda_hazydet")

SPLITS = ("train", "val", "test")
# COCO's scale convention, so the numbers are comparable with published tables.
SMALL, MEDIUM = 32**2, 96**2


def load_split(root: Path, split: str) -> dict:
    return json.loads((root / split / split / f"{split}_coco.json").read_text(encoding="utf-8"))


def dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    """Min over colour channels, then a local min filter. Haze raises it."""
    import cv2

    return cv2.erode(img.min(axis=2), np.ones((patch, patch), np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="C:/Users/Ridvan/Desktop/tez/hazydet")
    parser.add_argument("--pairs", type=int, default=120, help="image pairs for haze stats")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    out = ensure_dir(OUTPUT_DIR / "eda_hazydet")
    report: dict = {"root": str(root)}
    rng = random.Random(args.seed)

    # ---------------- 1. inventory + taxonomy ----------------------------
    inv = {}
    all_boxes: dict[str, list] = defaultdict(list)
    per_image_counts: dict[str, list] = defaultdict(list)
    for split in SPLITS:
        d = load_split(root, split)
        cats = {c["id"]: c["name"] for c in d["categories"]}
        by_img = defaultdict(int)
        for a in d["annotations"]:
            name = cats[a["category_id"]]
            w, h = a["bbox"][2], a["bbox"][3]
            all_boxes[split].append((name, w, h, w * h))
            by_img[a["image_id"]] += 1
        sizes = Counter((i["width"], i["height"]) for i in d["images"])
        per_image_counts[split] = [by_img.get(i["id"], 0) for i in d["images"]]
        inv[split] = {
            "images": len(d["images"]),
            "annotations": len(d["annotations"]),
            "classes": dict(Counter(b[0] for b in all_boxes[split])),
            "distinct_resolutions": len(sizes),
            "top_resolutions": [f"{w}x{h}:{n}" for (w, h), n in sizes.most_common(4)],
            "objects_per_image_mean": float(np.mean(per_image_counts[split])),
            "objects_per_image_max": int(np.max(per_image_counts[split])),
            "empty_images": int(sum(1 for c in per_image_counts[split] if c == 0)),
        }
        log.info("%-5s %s", split, inv[split])
    report["inventory"] = inv

    # ---------------- 2. object scale ------------------------------------
    scale = {}
    for split in SPLITS:
        areas = np.array([b[3] for b in all_boxes[split]], dtype=np.float64)
        widths = np.array([b[1] for b in all_boxes[split]], dtype=np.float64)
        heights = np.array([b[2] for b in all_boxes[split]], dtype=np.float64)
        scale[split] = {
            "coco_small_pct": float(100 * (areas < SMALL).mean()),
            "coco_medium_pct": float(100 * ((areas >= SMALL) & (areas < MEDIUM)).mean()),
            "coco_large_pct": float(100 * (areas >= MEDIUM).mean()),
            "sqrt_area_median": float(np.median(np.sqrt(areas))),
            "sqrt_area_p05": float(np.percentile(np.sqrt(areas), 5)),
            "sqrt_area_p95": float(np.percentile(np.sqrt(areas), 95)),
            "median_w": float(np.median(widths)),
            "median_h": float(np.median(heights)),
        }
        log.info("%-5s scale %s", split, {k: round(v, 2) for k, v in scale[split].items()})
    report["object_scale"] = scale

    # per-class scale on train: is "tiny" a class property or a dataset property?
    by_class = defaultdict(list)
    for name, w, h, a in all_boxes["train"]:
        by_class[name].append(a)
    report["train_scale_by_class"] = {
        k: {
            "n": len(v),
            "sqrt_area_median": float(np.median(np.sqrt(v))),
            "coco_small_pct": float(100 * (np.array(v) < SMALL).mean()),
        }
        for k, v in by_class.items()
    }
    log.info("per-class scale: %s", report["train_scale_by_class"])

    # ---------------- 3. haze strength on matched pairs ------------------
    import cv2

    d = load_split(root, "test")
    names = [i["file_name"] for i in d["images"]]
    rng.shuffle(names)
    rows = []
    for fn in names[: args.pairs]:
        cp = root / "test" / "test" / "images" / fn
        hp = root / "test" / "test" / "hazy_images" / fn
        if not (cp.exists() and hp.exists()):
            continue
        ci = cv2.imread(str(cp))
        hi = cv2.imread(str(hp))
        if ci is None or hi is None:
            continue
        ci = cv2.resize(ci, (512, 512))
        hi = cv2.resize(hi, (512, 512))
        rows.append(
            {
                "dcp_clear": float(dark_channel(ci).mean()),
                "dcp_hazy": float(dark_channel(hi).mean()),
                "contrast_clear": float(cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY).std()),
                "contrast_hazy": float(cv2.cvtColor(hi, cv2.COLOR_BGR2GRAY).std()),
                "brightness_clear": float(ci.mean()),
                "brightness_hazy": float(hi.mean()),
            }
        )
    if rows:
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        agg["n_pairs"] = len(rows)
        agg["dcp_increase"] = agg["dcp_hazy"] - agg["dcp_clear"]
        agg["contrast_loss_pct"] = 100 * (1 - agg["contrast_hazy"] / agg["contrast_clear"])
        report["haze_strength"] = agg
        log.info("haze: %s", {k: round(v, 2) for k, v in agg.items()})

    (out / "eda_hazydet.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s", out / "eda_hazydet.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
