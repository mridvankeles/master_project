"""Audit the label files on disk, per class and per condition.

Three questions the training loss cannot answer for us:

1. **Are the labels consistent across conditions?** `clear`, `fog2` and `night`
   are three renderings of the SAME DIOR scenes, so their label files must be
   equivalent. Any divergence is a corpus bug that would teach the experts
   contradictory targets for identical geometry.
2. **Is any class malformed?** Out-of-range coordinates, zero-area boxes,
   duplicate boxes, class ids outside `nc`, images with no label, labels with no
   image.
3. **What does each class actually look like?** Crops the ground truth for a
   sample of every class into a per-class contact sheet, so a mislabelled class
   is visible rather than assumed absent.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, ensure_dir  # noqa: E402

log = get_logger("audit_labels")


def read_label(p: Path):
    rows = []
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        f = line.split()
        rows.append((ln, int(float(f[0])), tuple(round(float(v), 6) for v in f[1:5])))
    return rows


def load_names(root: Path) -> dict:
    for y in sorted(root.glob("*/*.yaml")):
        spec = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        if spec.get("names"):
            return spec["names"]
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/dior_hbb_full")
    ap.add_argument("--conditions", default="clear,fog2,night")
    ap.add_argument("--split", default="train")
    ap.add_argument("--crops", type=int, default=16, help="crops per class in the contact sheet")
    ap.add_argument("--no-crops", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    conds = args.conditions.split(",")
    names = load_names(root)

    per_cond_cls = {}
    problems: Counter = Counter()
    examples = defaultdict(list)

    for cond in conds:
        img_dir = root / cond / "images" / args.split
        lbl_dir = root / cond / "labels" / args.split
        if not img_dir.exists():
            log.warning("missing %s", img_dir)
            continue
        imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}}
        lbls = {p.stem: p for p in lbl_dir.iterdir() if p.suffix == ".txt"}

        cls: Counter = Counter()
        for stem, lp in lbls.items():
            if stem not in imgs:
                problems[f"{cond}: label with no image"] += 1
            seen = set()
            for ln, c, box in read_label(lp):
                cls[c] += 1
                cx, cy, w, h = box
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    problems[f"{cond}: coord out of range"] += 1
                    if len(examples[f"{cond}:coord"]) < 3:
                        examples[f"{cond}:coord"].append(f"{lp.name}:{ln} {box}")
                if names and not (0 <= c < len(names)):
                    problems[f"{cond}: class id out of range"] += 1
                if (c, box) in seen:
                    problems[f"{cond}: duplicate box"] += 1
                seen.add((c, box))
        for stem in imgs:
            if stem not in lbls:
                problems[f"{cond}: image with no label"] += 1
        per_cond_cls[cond] = cls
        log.info("%-6s %6d images, %6d labels, %7d objects, %d classes present",
                 cond, len(imgs), len(lbls), sum(cls.values()), len(cls))

    # --- 1. cross-condition label identity --------------------------------
    log.info("")
    log.info("=== 1. CROSS-CONDITION LABEL IDENTITY ===")
    log.info("  fog2/night render the same scenes as clear, so the labels must match.")
    base = conds[0]
    base_dir = root / base / "labels" / args.split
    base_lbls = {p.stem.split("_", 1)[-1]: p for p in base_dir.iterdir() if p.suffix == ".txt"}
    for cond in conds[1:]:
        d = root / cond / "labels" / args.split
        if not d.exists():
            continue
        n_cmp = n_diff = 0
        diffs = []
        for p in d.iterdir():
            if p.suffix != ".txt":
                continue
            bp = base_lbls.get(p.stem.split("_", 1)[-1])
            if bp is None:
                continue
            n_cmp += 1
            if sorted((c, b) for _, c, b in read_label(bp)) != \
               sorted((c, b) for _, c, b in read_label(p)):
                n_diff += 1
                if len(diffs) < 3:
                    diffs.append(p.name)
        log.info("  %-6s vs %-6s : %6d compared, %d differ%s",
                 cond, base, n_cmp, n_diff, ("  e.g. " + ", ".join(diffs)) if diffs else "")

    # --- 2. per-class table ------------------------------------------------
    log.info("")
    log.info("=== 2. PER-CLASS OBJECT COUNTS ===")
    present = [k for k in conds if k in per_cond_cls]
    allc = sorted({c for k in per_cond_cls.values() for c in k})
    log.info("  %-24s" % "class" + "".join("%10s" % c for c in present) + "   identical")
    for c in allc:
        vals = [per_cond_cls[k].get(c, 0) for k in present]
        nm = names.get(c, str(c)) if isinstance(names, dict) else str(c)
        log.info("  %-24s" % nm + "".join("%10d" % v for v in vals) +
                 ("   yes" if len(set(vals)) == 1 else "   *** NO ***"))
    if isinstance(names, dict):
        absent = [names[c] for c in sorted(names) if c not in allc]
        if absent:
            log.info("  classes with ZERO objects: %s", absent)

    log.info("")
    log.info("=== 3. MALFORMED LABELS ===")
    if not problems:
        log.info("  none")
    for k, v in problems.most_common():
        log.info("  %-40s %d", k, v)
    for k, v in examples.items():
        log.info("  e.g. %s: %s", k, v)

    # --- 4. per-class contact sheets ---------------------------------------
    if args.no_crops:
        return 0
    import cv2
    log.info("")
    log.info("=== 4. PER-CLASS CONTACT SHEETS ===")
    out = ensure_dir(OUTPUT_DIR / "analysis" / f"class_crops_{base}_{args.split}")
    lbl_dir = root / base / "labels" / args.split
    img_dir = root / base / "images" / args.split
    rng = np.random.default_rng(0)
    files = [p for p in lbl_dir.iterdir() if p.suffix == ".txt"]
    files = [files[i] for i in rng.permutation(len(files))]
    want = {c: args.crops for c in allc}
    crops = defaultdict(list)
    for p in files:
        if all(v <= 0 for v in want.values()):
            break
        rows = read_label(p)
        if not any(want.get(c, 0) > 0 for _, c, _ in rows):
            continue
        ip = next((img_dir / f"{p.stem}{e}" for e in (".jpg", ".png")
                   if (img_dir / f"{p.stem}{e}").exists()), None)
        if ip is None:
            continue
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        for _, c, (cx, cy, w, h) in rows:
            if want.get(c, 0) <= 0:
                continue
            pad = 0.15
            x1 = int(max((cx - w / 2 - pad * w) * W, 0)); x2 = int(min((cx + w / 2 + pad * w) * W, W))
            y1 = int(max((cy - h / 2 - pad * h) * H, 0)); y2 = int(min((cy + h / 2 + pad * h) * H, H))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crops[c].append(cv2.resize(im[y1:y2, x1:x2], (96, 96)))
            want[c] -= 1

    for c, cl in sorted(crops.items()):
        cols = 8
        nrows = int(np.ceil(len(cl) / cols))
        sheet = np.zeros((nrows * 96, cols * 96, 3), np.uint8)
        for i, cr in enumerate(cl):
            r, q = divmod(i, cols)
            sheet[r * 96:(r + 1) * 96, q * 96:(q + 1) * 96] = cr
        nm = str(names.get(c, c)).replace("/", "-") if isinstance(names, dict) else str(c)
        cv2.imwrite(str(out / f"{c:02d}_{nm}.jpg"), sheet)
        log.info("  %-24s %2d crops", nm, len(cl))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
