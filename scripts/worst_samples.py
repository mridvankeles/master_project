"""Rank test images by how badly the model does on them, and render the worst.

    python scripts/worst_samples.py --run cond3b_gated_yolo11n --condition union3b

Aggregate mAP says how much is wrong. It never says WHAT is wrong. This ranks
every image by per-image error and renders the tail, so the failure modes can be
looked at rather than guessed at -- missed tiny objects, duplicate boxes,
mislabelled ground truth, or a genuinely impossible image all look identical in
a single number.

PER-IMAGE SCORE
---------------
For each image, greedily match predictions to ground truth at IoU 0.5 (by
descending confidence, one prediction per target) and compute F1. Images are
then ranked by F1 ascending, weighted by how many objects they contain so that
a miss on a 40-object image ranks above a miss on a 1-object image.

Also aggregates WHY images fail: false negatives by object size and by class,
which is what distinguishes "the model cannot see small things" from "one class
is broken" from "the labels are wrong".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, ensure_dir  # noqa: E402

log = get_logger("worst_samples")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cond3b_gated_yolo11n")
    ap.add_argument("--condition", default="union3b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    from src.utils.paths import dataset_root

    ck = OUTPUT_DIR / "runs" / args.run / "weights" / "best.pt"
    model = YOLO(str(ck))
    names = model.model.names if hasattr(model.model, "names") else {}

    root = dataset_root("detect", "full") / args.condition
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    rng = np.random.default_rng(0)
    if len(images) > args.limit:
        images = [images[i] for i in rng.choice(len(images), args.limit, replace=False)]
    log.info("%s on %s/%s: scoring %d images", args.run, args.condition, args.split, len(images))

    rows = []
    fn_by_size: Counter = Counter()
    fn_by_class: Counter = Counter()
    gt_by_class: Counter = Counter()
    fp_by_class: Counter = Counter()
    # A missed object that a box DID cover, under the wrong class name, is a
    # classification failure, not a detection failure. They need different fixes,
    # and the miss rate alone cannot tell them apart.
    confusion: Counter = Counter()
    not_seen: Counter = Counter()
    gt_by_size: Counter = Counter()

    for i in range(0, len(images), 16):
        chunk = images[i : i + 16]
        preds = model.predict([str(p) for p in chunk], imgsz=args.imgsz, conf=args.conf,
                              verbose=False, device=args.device)
        for p, r in zip(chunk, preds):
            im_h, im_w = r.orig_shape
            lbl = lbl_dir / f"{p.stem}.txt"
            gt = []
            for line in (lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []):
                if not line.strip():
                    continue
                f = line.split()
                c = int(f[0]); cx, cy, bw, bh = (float(v) for v in f[1:5])
                gt.append([c, (cx - bw / 2) * im_w, (cy - bh / 2) * im_h,
                           (cx + bw / 2) * im_w, (cy + bh / 2) * im_h])
            gt = np.array(gt, dtype=np.float64) if gt else np.zeros((0, 5))

            pb = r.boxes
            pred = (np.concatenate([pb.cls.cpu().numpy()[:, None],
                                    pb.xyxy.cpu().numpy()], axis=1)
                    if pb is not None and len(pb) else np.zeros((0, 5)))
            order = np.argsort(-(pb.conf.cpu().numpy() if pb is not None and len(pb) else np.zeros(0)))
            pred = pred[order] if len(pred) else pred

            matched_gt, tp = set(), 0
            if len(gt) and len(pred):
                M = iou_matrix(pred[:, 1:], gt[:, 1:])
                for pi in range(len(pred)):
                    best, bj = 0.0, -1
                    for gj in range(len(gt)):
                        if gj in matched_gt or gt[gj, 0] != pred[pi, 0]:
                            continue
                        if M[pi, gj] > best:
                            best, bj = M[pi, gj], gj
                    if bj >= 0 and best >= 0.5:
                        matched_gt.add(bj); tp += 1

            fn = len(gt) - len(matched_gt)
            fp = len(pred) - tp
            prec = tp / max(len(pred), 1)
            rec = tp / max(len(gt), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)

            Many = iou_matrix(pred[:, 1:], gt[:, 1:]) if len(gt) and len(pred) else None
            for gj in range(len(gt)):
                cls = int(gt[gj, 0]); gt_by_class[cls] += 1
                side = float(np.sqrt(max((gt[gj, 3] - gt[gj, 1]) * (gt[gj, 4] - gt[gj, 2]), 0)))
                band = ("<8px" if side < 8 else "8-16px" if side < 16 else
                        "16-32px" if side < 32 else "32-64px" if side < 64 else ">=64px")
                gt_by_size[band] += 1
                if gj not in matched_gt:
                    fn_by_class[cls] += 1
                    fn_by_size[band] += 1
                    # was ANY box, of any class, sitting on this object?
                    if Many is not None and len(pred):
                        pi = int(np.argmax(Many[:, gj]))
                        if Many[pi, gj] >= 0.5:
                            confusion[(cls, int(pred[pi, 0]))] += 1
                            continue
                    not_seen[cls] += 1
            if len(pred) and tp < len(pred):
                for pi in range(len(pred)):
                    fp_by_class[int(pred[pi, 0])] += 0  # counted in aggregate below
            fp_by_class["_total"] = fp_by_class.get("_total", 0) + fp

            rows.append({"image": str(p), "n_gt": len(gt), "n_pred": len(pred),
                         "tp": tp, "fn": fn, "fp": fp, "f1": f1,
                         "condition": p.stem.split("_")[0]})

    rows.sort(key=lambda r: (r["f1"], -r["n_gt"]))
    out = ensure_dir(OUTPUT_DIR / "analysis" / f"worst_{args.run}_{args.condition}")

    log.info("")
    log.info("=== FAILURE MODES ===")
    tot_gt = sum(gt_by_class.values()); tot_fn = sum(fn_by_class.values())
    log.info("  ground-truth objects %d | missed %d (%.1f%%) | false positives %d",
             tot_gt, tot_fn, 100 * tot_fn / max(tot_gt, 1), fp_by_class.get("_total", 0))
    log.info("  MISSES BY OBJECT SIZE:")
    for band in ("<8px", "8-16px", "16-32px", "32-64px", ">=64px"):
        if fn_by_size[band]:
            log.info("    %-9s %5d of %5d objects missed = %5.1f%% miss rate  (%.1f%% of all misses)",
                     band, fn_by_size[band], gt_by_size[band],
                     100 * fn_by_size[band] / max(gt_by_size[band], 1),
                     100 * fn_by_size[band] / max(tot_fn, 1))
    log.info("  WORST CLASSES (miss rate, classes with >=30 objects):")
    for c, g in sorted(gt_by_class.items(), key=lambda x: -(fn_by_class[x[0]] / max(x[1], 1))):
        if g < 30:
            continue
        log.info("    %-24s %5d objects, %5.1f%% missed",
                 names.get(c, c), g, 100 * fn_by_class[c] / g)

    n_conf = sum(confusion.values()); n_blind = sum(not_seen.values())
    log.info("  WHY THE MISS: not detected at all %d (%.1f%%) | detected, wrong class %d (%.1f%%)",
             n_blind, 100 * n_blind / max(tot_fn, 1), n_conf, 100 * n_conf / max(tot_fn, 1))
    if confusion:
        log.info("  TOP CLASS CONFUSIONS (truth -> predicted):")
        for (a, b), k in confusion.most_common(10):
            log.info("    %-24s -> %-24s %4d", names.get(a, a), names.get(b, b), k)

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r["f1"])
    log.info("  MEAN PER-IMAGE F1 BY CONDITION:")
    for k, v in sorted(by_cond.items()):
        log.info("    %-8s %.3f  (n=%d)", k, float(np.mean(v)), len(v))

    log.info("")
    log.info("=== %d WORST IMAGES ===", args.top)
    for r in rows[: args.top]:
        log.info("  f1=%.3f gt=%3d pred=%3d fn=%3d fp=%3d  %s",
                 r["f1"], r["n_gt"], r["n_pred"], r["fn"], r["fp"], Path(r["image"]).name)

    # render the worst, ground truth in green, prediction in red
    for rank, r in enumerate(rows[: args.top], 1):
        p = Path(r["image"])
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        lbl = lbl_dir / f"{p.stem}.txt"
        for line in (lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []):
            if not line.strip():
                continue
            f = line.split(); cx, cy, bw, bh = (float(v) for v in f[1:5])
            cv2.rectangle(im, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 0), 2)
        pr = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False, device=args.device)[0]
        if pr.boxes is not None:
            for b, c in zip(pr.boxes.xyxy.cpu().numpy(), pr.boxes.cls.cpu().numpy()):
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.putText(im, f"GT=green pred=red  f1={r['f1']:.2f} gt={r['n_gt']} fn={r['fn']}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"{rank:02d}_{p.stem}.jpg"), im)

    (out / "ranking.json").write_text(json.dumps(rows[:200], indent=2), encoding="utf-8")
    log.info("")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
