"""Draw converted boxes onto images so the conversion can be checked by eye.

    python scripts/render_verification.py

This is the check that actually matters. The pairing arithmetic can be right
while the converter is wrong: a transposed axis, an off-by-one, a normalisation
against the wrong dimension all survive every automated check and show up
instantly as boxes that sit next to objects instead of on them.

Boxes are read back from the MATERIALISED LABEL FILES, not from the XML. That
means the round trip xml -> to_yolo -> label.txt -> from_yolo -> pixels is what
gets rendered, so an error anywhere in it is visible.

Writes to outputs/verification/:
  boxes_NN_<condition>_<split>_<id>.jpg  — 20 samples with boxes drawn
  pair_<id>.jpg                          — clear|thin|moderate|thick side by side
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_dataset import SPLITS  # noqa: E402
from src.data.dior_classes import DIOR_CLASSES  # noqa: E402
from src.data.pairing import SEVERITIES  # noqa: E402
from src.data.voc_hbb import from_yolo  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import VERIFICATION_DIR, dataset_root, ensure_dir  # noqa: E402
from src.utils.seed import DEFAULT_SEED, seed_everything  # noqa: E402

log = get_logger("render_verification")

N_SAMPLES = 20
N_PAIRS = 4

# Distinct BGR colours, one per class. Generated from a fixed hue sweep so the
# same class is always the same colour across every rendered image.
def _palette() -> list[tuple[int, int, int]]:
    import numpy as np

    hues = ((np.arange(len(DIOR_CLASSES)) * 180 // len(DIOR_CLASSES)) % 180).astype("uint8")
    hsv = np.stack([hues, np.full_like(hues, 235), np.full_like(hues, 255)], axis=1)
    bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return [tuple(int(v) for v in row) for row in bgr]


PALETTE = _palette()


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        rows.append((int(parts[0]), *(float(v) for v in parts[1:5])))
    return rows


def draw(image_path: Path, label_path: Path) -> "cv2.typing.MatLike":
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    h, w = img.shape[:2]

    for cls, cx, cy, bw, bh in read_label(label_path):
        box = from_yolo(cls, cx, cy, bw, bh, w, h)
        colour = PALETTE[cls % len(PALETTE)]
        p1 = (int(round(box.xmin)), int(round(box.ymin)))
        p2 = (int(round(box.xmax)), int(round(box.ymax)))
        cv2.rectangle(img, p1, p2, colour, 2)

        name = DIOR_CLASSES[cls]
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), colour, -1)
        cv2.putText(
            img, name, (p1[0] + 2, ty - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return img


def banner(img, text: str):
    """Prepend a caption strip so a saved render is self-describing.

    The strip is ADDED above the image rather than drawn onto it. Painting over
    the top rows would hide exactly the region where a normalisation error
    shows up first — a box that should touch y=0 but does not.
    """
    import numpy as np

    strip = np.zeros((26, img.shape[1], 3), dtype=img.dtype)
    cv2.putText(
        strip, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([strip, img])


def collect(root: Path, condition: str) -> dict[str, list[tuple[Path, Path]]]:
    """(image, label) pairs per split for one condition."""
    out: dict[str, list[tuple[Path, Path]]] = {}
    for split in SPLITS:
        img_dir = root / condition / "images" / split
        lbl_dir = root / condition / "labels" / split
        if not img_dir.is_dir():
            out[split] = []
            continue
        out[split] = [
            (p, lbl_dir / f"{p.stem}.txt")
            for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="materialised dataset root")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = random.Random(args.seed)

    root = Path(args.dataset) if args.dataset else dataset_root()
    if not root.is_dir():
        log.error("dataset root not found: %s — run scripts/prepare_dataset.py first", root)
        return 2

    out_dir = ensure_dir(Path(args.out) if args.out else VERIFICATION_DIR)

    clear = collect(root, "clear")
    fog = collect(root, "fog")

    # Sample the two conditions SEPARATELY and in equal numbers. Drawing from a
    # single pool would hand back roughly 3:1 fog, because fog carries three
    # severities per id — and a converter bug specific to the clear branch
    # could then hide behind two or three samples.
    def pool_for(collected) -> list[tuple[str, str, Path, Path]]:
        return [
            (cond, split, i, l)
            for split in SPLITS
            for cond, src in [(condition_name, collected)]
            for i, l in src.get(split, [])
        ]

    pools: dict[str, list[tuple[str, str, Path, Path]]] = {}
    for condition_name, collected in (("clear", clear), ("fog", fog)):
        pools[condition_name] = pool_for(collected)

    if not any(pools.values()):
        log.error("no images found under %s", root)
        return 2

    per_condition = max(1, args.n // 2)
    chosen: list[tuple[str, str, Path, Path]] = []
    for condition_name, entries in pools.items():
        # Prefer images that actually have boxes — an empty render verifies nothing.
        with_boxes = [e for e in entries if e[3].exists() and e[3].stat().st_size > 0]
        candidates = with_boxes or entries
        if not candidates:
            log.warning("no labelled images for condition %s", condition_name)
            continue
        chosen += rng.sample(candidates, min(per_condition, len(candidates)))
    chosen.sort(key=lambda r: (r[0], r[1], r[2].stem))

    written = 0
    for idx, (cond, split, img_path, lbl_path) in enumerate(chosen, start=1):
        img = draw(img_path, lbl_path)
        n = len(read_label(lbl_path))
        img = banner(img, f"{cond}/{split}  {img_path.stem}  ({n} boxes)")
        dst = out_dir / f"boxes_{idx:02d}_{cond}_{split}_{img_path.stem}.jpg"
        cv2.imwrite(str(dst), img)
        written += 1

    # Side-by-side panels: same DIOR id, clear next to all three fog severities.
    # This is the visual confirmation that the aligned pairing really holds —
    # the boxes must land on the same objects in all four frames.
    clear_by_id = {
        i.stem: (i, l) for split in SPLITS for i, l in clear.get(split, [])
    }
    pair_ids = rng.sample(sorted(clear_by_id), min(N_PAIRS, len(clear_by_id)))

    fog_lookup: dict[str, tuple[Path, Path]] = {}
    for split in SPLITS:
        for i, l in fog.get(split, []):
            fog_lookup[i.stem] = (i, l)

    import numpy as np

    for image_id in sorted(pair_ids):
        frames = []
        ci, cl = clear_by_id[image_id]
        frames.append(banner(draw(ci, cl), f"clear  {image_id}"))
        for sev in SEVERITIES:
            key = f"{image_id}_{sev}"
            if key in fog_lookup:
                fi, fl = fog_lookup[key]
                frames.append(banner(draw(fi, fl), f"fog/{sev}  {image_id}"))
        if len(frames) < 2:
            continue
        panel = np.hstack(frames)
        dst = out_dir / f"pair_{image_id}.jpg"
        cv2.imwrite(str(dst), panel)
        written += 1

    log.info("wrote %s images to %s", written, out_dir)
    log.info("Inspect them: boxes must sit ON the objects, not beside them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
