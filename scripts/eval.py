"""Evaluate a checkpoint on a split: metrics JSON, results table, pred-vs-GT panels.

    python scripts/eval.py --checkpoint outputs/runs/smoke/weights/best.pt --split val
    python scripts/eval.py --config configs/train/fog_yolo11n.yaml --split test

Outputs land in outputs/eval/<name>/:
    metrics.json          overall + per-class, machine readable
    results.md            the same as a table
    predvsgt_*.jpg        fixed image set, ground truth | prediction side by side

The metrics come from Ultralytics' own validator — we reshape them, we do not
recompute them. A hand-rolled mAP that disagrees with the framework's is a
liability rather than a contribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.qualitative import fixed_sample, render_all  # noqa: E402
from src.eval.report import from_ultralytics  # noqa: E402
from src.utils.config import load_run_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, dataset_root, ensure_dir  # noqa: E402
from src.utils.seed import git_commit, seed_everything  # noqa: E402

log = get_logger("eval")

N_QUALITATIVE = 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="path to a .pt")
    parser.add_argument("--config", default=None, help="config to infer checkpoint/condition from")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence for the panels only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="output subdirectory name")
    parser.add_argument("--n-qualitative", type=int, default=N_QUALITATIVE)
    args = parser.parse_args()

    seed_everything(args.seed)

    cfg = load_run_config(args.config) if args.config else None
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        OUTPUT_DIR / "runs" / cfg.run_name / "weights" / "best.pt" if cfg else None
    )
    if checkpoint is None or not checkpoint.exists():
        log.error("checkpoint not found: %s", checkpoint)
        return 2

    condition = cfg.condition if cfg else "fog"
    data_yaml = cfg.data_yaml if cfg else (
        Path(__file__).resolve().parents[1] / "configs" / "data" / f"dior_{condition}.yaml"
    )
    if not data_yaml.exists():
        log.error("data yaml not found: %s — run scripts/prepare_dataset.py", data_yaml)
        return 2

    imgsz = args.imgsz or (cfg.train.get("imgsz", 640) if cfg else 640)
    name = args.name or f"{checkpoint.parent.parent.name}_{args.split}"
    out_dir = ensure_dir(OUTPUT_DIR / "eval" / name)

    os.environ.setdefault("MLFLOW_TRACKING_URI", (OUTPUT_DIR / "mlruns").resolve().as_uri())

    from ultralytics import YOLO

    log.info("checkpoint : %s", checkpoint)
    log.info("data       : %s  split=%s  imgsz=%s", data_yaml, args.split, imgsz)

    model = YOLO(str(checkpoint))

    # --- quantitative -----------------------------------------------------
    results = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=imgsz,
        device=args.device,
        plots=True,
        project=str(out_dir),
        name="val",
        exist_ok=True,
        verbose=False,
    )

    report = from_ultralytics(
        results,
        checkpoint=checkpoint,
        data_yaml=data_yaml,
        split=args.split,
        imgsz=imgsz,
        device=str(args.device),
        git_commit=git_commit(short=True),
    )
    report.write_json(out_dir / "metrics.json")
    report.write_markdown(out_dir / "results.md")

    log.info(
        "mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
        report.overall["mAP50"], report.overall["mAP50-95"],
        report.overall["precision"], report.overall["recall"],
    )

    # Cross-check: the numbers we tabulate must be the numbers Ultralytics printed.
    delta50 = abs(report.overall["mAP50"] - float(results.box.map50))
    delta = abs(report.overall["mAP50-95"] - float(results.box.map))
    log.info("delta vs Ultralytics' own values: mAP50 %.2e, mAP50-95 %.2e", delta50, delta)
    if max(delta50, delta) > 1e-9:
        log.error("METRICS DISAGREE with Ultralytics — do not trust this table")
        return 1

    # --- qualitative ------------------------------------------------------
    images_dir = dataset_root() / condition / "images" / args.split
    if images_dir.is_dir():
        picks = fixed_sample(
            images_dir,
            n=args.n_qualitative,
            seed=args.seed,
            cache=OUTPUT_DIR / "eval" / f"fixed_images_{condition}_{args.split}.json",
        )
        preds = model.predict(
            [str(p) for p in picks],
            imgsz=imgsz,
            device=args.device,
            conf=args.conf,
            verbose=False,
        )
        summary = render_all(picks, preds, out_dir)
        (out_dir / "qualitative.json").write_text(
            json.dumps({"conf": args.conf, "images": summary}, indent=2), encoding="utf-8"
        )
        log.info("rendered %d pred-vs-GT panels (conf=%.2f)", len(summary), args.conf)
    else:
        log.warning("no images at %s — skipping panels", images_dir)

    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
