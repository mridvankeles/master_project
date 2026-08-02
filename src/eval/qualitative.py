"""Side-by-side ground truth vs prediction panels.

The image set is FIXED — chosen by a seeded sample over the sorted split and
cached to disk on first use — so the same images are rendered every run and two
checkpoints can be compared by flicking between their outputs. A fresh random
sample each run would make visual comparison worthless.

Colour is by class, identical on both halves, so a class confusion shows up as a
colour change rather than requiring the label text to be read.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..data.dior_classes import DIOR_CLASSES
from ..data.voc_hbb import from_yolo

__all__ = ["fixed_sample", "render_pair", "render_all"]


def _palette() -> list[tuple[int, int, int]]:
    hues = ((np.arange(len(DIOR_CLASSES)) * 180 // len(DIOR_CLASSES)) % 180).astype("uint8")
    hsv = np.stack([hues, np.full_like(hues, 235), np.full_like(hues, 255)], axis=1)
    bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return [tuple(int(v) for v in row) for row in bgr]


PALETTE = _palette()


def labels_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def fixed_sample(images_dir: Path, n: int, seed: int, cache: Path) -> list[Path]:
    """A stable sample of `n` images. Written once, then reused verbatim.

    If the cache exists, it wins — even if `n` or `seed` changed. That is
    deliberate: the whole point is that the set never moves between runs. Delete
    the cache file to choose a new one.
    """
    if cache.exists():
        stems = json.loads(cache.read_text(encoding="utf-8"))
        found = [images_dir / s for s in stems if (images_dir / s).exists()]
        if found:
            return found

    import random

    images = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    # Prefer images that actually contain objects — an empty panel compares nothing.
    labelled = [p for p in images if labels_for(p).exists() and labels_for(p).stat().st_size > 0]
    pool = labelled or images
    picks = sorted(random.Random(seed).sample(pool, min(n, len(pool))))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([p.name for p in picks], indent=2), encoding="utf-8")
    return picks


def _draw(img, boxes: list[tuple[int, float, float, float, float, float | None]]):
    """boxes: (cls, x1, y1, x2, y2, conf|None) in absolute pixels."""
    for cls, x1, y1, x2, y2, conf in boxes:
        colour = PALETTE[cls % len(PALETTE)]
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, colour, 2)
        text = DIOR_CLASSES[cls] if conf is None else f"{DIOR_CLASSES[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), colour, -1)
        cv2.putText(
            img, text, (p1[0] + 2, ty - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return img


def _caption(img, text: str, height: int = 28):
    strip = np.zeros((height, img.shape[1], 3), dtype=img.dtype)
    cv2.putText(
        strip, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([strip, img])


def read_gt(image_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float, None]]:
    lbl = labels_for(image_path)
    if not lbl.exists():
        return []
    out = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        box = from_yolo(int(parts[0]), *(float(v) for v in parts[1:5]), w, h)
        out.append((box.cls, box.xmin, box.ymin, box.xmax, box.ymax, None))
    return out


def render_pair(image_path: Path, result, dst: Path) -> tuple[int, int]:
    """Write one GT | prediction panel. Returns (n_gt, n_pred)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    gt = read_gt(image_path, w, h)
    preds = []
    if result is not None and result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        preds = [
            (int(c), float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(cf))
            for b, c, cf in zip(xyxy, cls, conf)
        ]

    left = _caption(_draw(img.copy(), gt), f"GROUND TRUTH  {image_path.stem}  ({len(gt)} objects)")
    right = _caption(_draw(img.copy(), preds), f"PREDICTION  {image_path.stem}  ({len(preds)} detections)")

    divider = np.full((left.shape[0], 4, 3), 60, dtype=left.dtype)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), np.hstack([left, divider, right]))
    return len(gt), len(preds)


def render_all(image_paths: list[Path], results, out_dir: Path) -> list[dict]:
    """Render every (image, result) pair. Returns a small summary per image."""
    summary = []
    for path, result in zip(image_paths, results):
        n_gt, n_pred = render_pair(path, result, out_dir / f"predvsgt_{path.stem}.jpg")
        summary.append({"image": path.name, "n_gt": n_gt, "n_pred": n_pred})
    return summary
