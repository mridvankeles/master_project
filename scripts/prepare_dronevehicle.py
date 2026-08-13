"""Convert DroneVehicle into our HBB pipeline, split into day and night.

    python scripts/prepare_dronevehicle.py --limit 200   # smoke
    python scripts/prepare_dronevehicle.py

WHY THIS DATASET
----------------
It is the standard real low-light drone benchmark (alongside VisDrone-Night),
covers "various urban areas from day to night", and — unlike every degraded
corpus we have built so far — its darkness is REAL, not synthesised. Our night
condition matches real night brightness closely (30.2 vs 32.4) but has 3.5x too
little contrast (12.8 vs 44.5), because real night is dark AND high-contrast:
streetlights and headlights blow out against black. That defect is only visible
against real data.

THREE CONVERSIONS, EACH A PLACE BUGS HIDE
-----------------------------------------
1. **De-letterbox.** The YOLO export pads every frame to a fixed aspect ratio
   with WHITE borders — 44% of the average frame. Left in, objects occupy 44%
   less of the input, which matters enormously for vehicles that are already
   tiny. It also corrupts any brightness statistic computed on the frame, which
   is how this dataset was initially and wrongly dismissed as "not really dark".
   Labels are normalised to the PADDED frame, so cropping means re-deriving
   every coordinate.
2. **OBB -> HBB.** The export is oriented (8 coordinates). We train horizontal,
   so each polygon becomes its axis-aligned enclosing box. That is a widening,
   never a shrink, and it is the standard conversion.
3. **dark / lit split** by content brightness. Deliberately NOT called
   "day/night": inspection shows the RGB channel of this release is
   predominantly night-time urban imagery, and the brighter half is well-lit
   night streets rather than daylight. Calling it day/night would put a false
   claim in every table that used it. `dark` is the genuinely low-light half
   (mean brightness 19) and `lit` is the illuminated half (mean 146).

The split is by IMAGE, and the source has no scene-level grouping, so a scene
photographed twice could in principle straddle it. Recorded rather than hidden.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import CONFIG_DIR, DATA_DIR, ensure_dir  # noqa: E402

log = get_logger("prepare_dronevehicle")

SRC = Path("C:/Users/Ridvan/Desktop/tez/dronevehicle/DroneVehiclesDatasetYOLO")
CLASSES = ("small-vehicle", "large-vehicle")
SPLITS = ("train", "val", "test")
# Measured on 700 content-cropped images: the distribution is bimodal and 60
# sits in the trough. 33% of the corpus falls below it.
NIGHT_BRIGHTNESS = 60.0  # trough of the bimodal brightness distribution
WHITE = 250


def content_box(img: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the non-white region, i.e. the real frame."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(grey < WHITE)
    if len(ys) < 100:
        return 0, 0, img.shape[1], img.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def convert_labels(lines: list[str], w: int, h: int, box: tuple[int, int, int, int]):
    """OBB (normalised to the padded frame) -> HBB (normalised to the crop)."""
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        cls = int(parts[0])
        xs = np.array([float(parts[i]) for i in range(1, 9, 2)]) * w - x0
        ys = np.array([float(parts[i]) for i in range(2, 9, 2)]) * h - y0
        # Axis-aligned enclosing box of the polygon, clipped to the crop.
        xmin, xmax = np.clip([xs.min(), xs.max()], 0, cw)
        ymin, ymax = np.clip([ys.min(), ys.max()], 0, ch)
        bw, bh = xmax - xmin, ymax - ymin
        if bw < 1 or bh < 1:
            continue  # fell entirely outside the crop
        out.append(
            f"{cls} {(xmin + bw / 2) / cw:.6f} {(ymin + bh / 2) / ch:.6f} "
            f"{bw / cw:.6f} {bh / ch:.6f}"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(DATA_DIR / "dronevehicle_hbb"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    rows: list[dict] = []

    for split in SPLITS:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.is_dir():
            log.warning("missing %s", img_dir)
            continue
        images = sorted(img_dir.iterdir())
        if args.limit:
            images = images[: args.limit]

        for cond in ("lit", "dark"):
            ensure_dir(out / cond / "images" / split)
            ensure_dir(out / cond / "labels" / split)

        kept = {"lit": 0, "dark": 0}
        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            box = content_box(img)
            crop = img[box[1]:box[3], box[0]:box[2]]
            if crop.size == 0:
                continue

            lbl = lbl_dir / f"{p.stem}.txt"
            lines = lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []
            converted = convert_labels([ln for ln in lines if ln.strip()], w, h, box)
            if not converted:
                continue  # no objects survive; nothing to learn from

            brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
            cond = "dark" if brightness < NIGHT_BRIGHTNESS else "lit"
            stem = p.stem.split("_jpg")[0]

            cv2.imwrite(str(out / cond / "images" / split / f"{stem}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / cond / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(converted) + "\n", encoding="utf-8"
            )
            kept[cond] += 1
            rows.append({
                "stem": stem, "split": split, "condition": cond,
                "brightness": round(brightness, 2), "n_boxes": len(converted),
                "pad_fraction": round(1 - (crop.shape[0] * crop.shape[1]) / (h * w), 3),
            })
        log.info("%-5s lit=%s dark=%s", split, f"{kept['lit']:,}", f"{kept['dark']:,}")

    if not rows:
        log.error("nothing converted")
        return 1

    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    b = np.array([r["brightness"] for r in rows])
    pad = np.array([r["pad_fraction"] for r in rows])
    night = np.array([r["condition"] == "dark" for r in rows])
    log.info("converted %s images (%s dark, %.0f%%)", f"{len(rows):,}",
             f"{night.sum():,}", 100 * night.mean())
    log.info("brightness: dark %.1f | lit %.1f   padding removed: %.0f%% of frame",
             b[night].mean(), b[~night].mean(), 100 * pad.mean())

    for cond in ("lit", "dark"):
        root = (out / cond).resolve()
        lines = [
            "# Generated by scripts/prepare_dronevehicle.py -- do not edit.",
            f"# DroneVehicle, {cond} half by measured content brightness (threshold 60).",
            f"path: {root.as_posix()}",
            "train: images/train", "val: images/val", "test: images/test",
            "", "names:", *[f"  {i}: {n}" for i, n in enumerate(CLASSES)],
        ]
        dst = CONFIG_DIR / "data" / f"dronevehicle_{cond}.yaml"
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("wrote %s", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
