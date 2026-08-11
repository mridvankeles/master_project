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

MLFLOW
------
Unlike training, evaluation metrics are logged FROM THIS SCRIPT. Ultralytics'
MLflow callback registers trainer hooks only, so a standalone `model.val()`
fires nothing — see `src/utils/tracking.py`. Each evaluation opens its own run
(tagged `kind=eval`, linked to the training run by `source_run`) carrying
overall metrics, per-class AP, VOC07 11-point mAP, speed and model complexity.
Pass `--no-mlflow` to write only the files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.qualitative import fixed_sample, labels_for, render_all  # noqa: E402
from src.eval.report import from_ultralytics  # noqa: E402
from src.eval.voc07 import Detection, GroundTruth, evaluate_voc07  # noqa: E402
from src.utils.config import load_run_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, dataset_root, ensure_dir  # noqa: E402
from src.utils.seed import git_commit, seed_everything  # noqa: E402
from src.utils.tracking import (  # noqa: E402
    dataset_fingerprint,
    evaluation_metrics,
    file_digest,
    log_evaluation,
    model_complexity,
    model_fingerprint,
)

log = get_logger("eval")

N_QUALITATIVE = 12

# Low enough that the precision/recall curve is not truncated, high enough that
# NMS stays tractable. Ultralytics' validator uses 0.001, but it batches NMS
# internally; calling predict() at 0.001 over a whole split builds an IoU matrix
# from ~8400 anchors x 20 classes and OOMs a 16 GB card outright. 0.01 keeps the
# tail of the curve that matters for AP.
VOC07_CONF = 0.01
VOC07_MAX_DET = 300  # Ultralytics' own default
VOC07_CHUNK = 64  # images per predict() call, so peak memory stays bounded


def collect_for_voc07(
    model,
    images_dir: Path,
    task: str,
    imgsz: int,
    device: str,
    conf: float = VOC07_CONF,
):
    """Run the model over a whole split and pair detections with ground truth.

    Returns (detections, ground_truths) in absolute pixel coordinates: 4 corner
    values for `detect`, 8 polygon values for `obb`.
    """
    images = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    detections: list[Detection] = []
    ground_truths: list[GroundTruth] = []

    for path in images:
        img_w = img_h = None
        lbl = labels_for(path)
        if lbl.exists():
            from PIL import Image

            with Image.open(path) as im:
                img_w, img_h = im.size
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                if task == "obb":
                    coords = tuple(
                        vals[i] * (img_w if i % 2 == 0 else img_h) for i in range(8)
                    )
                else:
                    cx, cy, w, h = vals[:4]
                    coords = (
                        (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                        (cx + w / 2) * img_w, (cy + h / 2) * img_h,
                    )
                ground_truths.append(GroundTruth(path.stem, cls, coords))

    for start in range(0, len(images), VOC07_CHUNK):
        chunk = images[start : start + VOC07_CHUNK]
        results = model.predict(
            [str(p) for p in chunk],
            imgsz=imgsz,
            device=device,
            conf=conf,
            max_det=VOC07_MAX_DET,
            stream=True,
            verbose=False,
        )
        _absorb(detections, chunk, results, task)

    return detections, ground_truths


def _absorb(detections, images, results, task: str) -> None:
    for path, res in zip(images, results):
        if task == "obb":
            obb = getattr(res, "obb", None)
            if obb is None or len(obb) == 0:
                continue
            polys = obb.xyxyxyxy.cpu().numpy().reshape(len(obb), 8)
            cls = obb.cls.cpu().numpy().astype(int)
            conf = obb.conf.cpu().numpy()
            for p, c, s in zip(polys, cls, conf):
                detections.append(Detection(path.stem, int(c), float(s), tuple(float(v) for v in p)))
        else:
            box = res.boxes
            if box is None or len(box) == 0:
                continue
            xyxy = box.xyxy.cpu().numpy()
            cls = box.cls.cpu().numpy().astype(int)
            conf = box.conf.cpu().numpy()
            for b, c, s in zip(xyxy, cls, conf):
                detections.append(Detection(path.stem, int(c), float(s), tuple(float(v) for v in b)))


def _model_yaml_for(checkpoint: Path, cfg) -> Path | None:
    """The architecture yaml that produced `checkpoint`.

    Ultralytics writes `args.yaml` beside the weights, recording the `model`
    the run was built from. That is authoritative; `--config` is only
    authoritative about the DATA when the two deliberately differ.
    """
    args_yaml = checkpoint.parent.parent / "args.yaml"
    if args_yaml.exists():
        try:
            import yaml as _yaml

            model = (_yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}).get("model")
            if model:
                path = Path(model)
                if path.exists():
                    return path
                # Ultralytics records the SCALED name (`yolo11n_dior_hbb.yaml`),
                # which is not a file on disk — it strips the scale letter when
                # resolving. Undo that to recover the real yaml.
                family, _, rest = path.stem.partition("_")
                descaled = path.with_name(f"{family[:-1]}_{rest}{path.suffix}")
                if descaled.exists():
                    return descaled
        except Exception:  # noqa: BLE001 - fall back rather than fail the eval
            pass
    return cfg.model_path if cfg else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="path to a .pt")
    parser.add_argument("--config", default=None, help="config to infer checkpoint/condition from")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16,
                        help="validation batch size; pinned because the speed protocol in "
                             "04-method-open-questions.md requires it to be stated")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence for the panels only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="output subdirectory name")
    parser.add_argument("--n-qualitative", type=int, default=N_QUALITATIVE)
    parser.add_argument("--voc07-conf", type=float, default=VOC07_CONF)

    parser.add_argument("--skip-voc07", action="store_true",
                        help="skip the VOC07 pass (it re-runs inference over the split)")
    parser.add_argument("--experiment", default=None,
                        help="MLflow experiment; defaults to the config's, else 'dior-eval'")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="write the files but do not open an MLflow run")
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
    task = cfg.task if cfg else "detect"
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

    model = YOLO(str(checkpoint), task=task)

    # --- quantitative -----------------------------------------------------
    results = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=imgsz,
        batch=args.batch,
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

    # --- VOC07 11-point mAP@0.5, the convention NIRNet reports ------------
    # Scope-aware: the aligned and full corpora live under different roots, so
    # reconstructing this from `task` alone would silently score the wrong one.
    root = cfg.dataset_root if cfg else dataset_root(task)
    images_dir = root / condition / "images" / args.split
    voc = None  # stays None when the pass is skipped; the tracker treats that as "not measured"
    if images_dir.is_dir() and not args.skip_voc07:
        log.info("computing VOC07 11-point mAP@0.5 over %s ...", args.split)
        dets, gts = collect_for_voc07(model, images_dir, task, imgsz, args.device, args.voc07_conf)
        voc = evaluate_voc07(dets, gts, task=task)
        log.info(
            "VOC07 mAP@0.5 = %.4f   (Ultralytics COCO mAP50 = %.4f, delta %+.4f)",
            voc.mean_ap, report.overall["mAP50"], voc.mean_ap - report.overall["mAP50"],
        )
        (out_dir / "voc07.json").write_text(
            json.dumps(
                {
                    "convention": "VOC07 11-point interpolation, IoU 0.50, "
                                  "matching nirnet mmrotate use_07_metric=True",
                    "task": task,
                    "split": args.split,
                    "mAP@0.5": voc.mean_ap,
                    "ultralytics_mAP50_coco101": report.overall["mAP50"],
                    "n_detections": len(dets),
                    "n_ground_truth": len(gts),
                    "per_class_ap": voc.per_class_ap,
                    "per_class_gt": voc.per_class_gt,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # --- qualitative ------------------------------------------------------
    if images_dir.is_dir():
        picks = fixed_sample(
            images_dir,
            n=args.n_qualitative,
            seed=args.seed,
            cache=OUTPUT_DIR / "eval" / f"fixed_images_{task}_{condition}_{args.split}.json",
        )
        preds = model.predict(
            [str(p) for p in picks],
            imgsz=imgsz,
            device=args.device,
            conf=args.conf,
            verbose=False,
        )
        summary = render_all(picks, preds, out_dir, task=task)
        (out_dir / "qualitative.json").write_text(
            json.dumps({"conf": args.conf, "images": summary}, indent=2), encoding="utf-8"
        )
        log.info("rendered %d pred-vs-GT panels (conf=%.2f)", len(summary), args.conf)
    else:
        log.warning("no images at %s — skipping panels", images_dir)

    # --- tracking ---------------------------------------------------------
    # Ultralytics' MLflow callback hooks the TRAINER only, so nothing above this
    # point has reached the tracker. See src/utils/tracking.py for the details.
    if not args.no_mlflow:
        import torch
        import ultralytics

        source_run = checkpoint.parent.parent.name
        experiment = args.experiment or (cfg.experiment if cfg else "dior-eval")
        n_images = (
            len([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            if images_dir.is_dir()
            else 0
        )

        metrics = evaluation_metrics(report, voc, args.split)
        metrics.update(model_complexity(model, imgsz=imgsz))

        run_id = log_evaluation(
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            experiment=experiment,
            run_name=f"eval-{name}-{report.git_commit}",
            tags={
                "kind": "eval",
                "source_run": source_run,
                "eval_condition": condition,
                # Heuristic on the run-name convention (`<condition>_<model>`),
                # recorded so a cross-condition evaluation — a clear-trained
                # model scored on fog — is visible in the run table rather than
                # inferable only from the two tags above.
                "cross_condition": str(not source_run.startswith(condition)).lower(),
                "split": args.split,
                "task": task,
                "box_format": "hbb" if task == "detect" else "obb",
                "git_commit": report.git_commit,
                "metric_convention": report.metric_convention,
                "voc07_measured": str(voc is not None).lower(),
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            params={
                "checkpoint": checkpoint,
                # Which weights, exactly. Two checkpoints at the same path after
                # a re-run are otherwise indistinguishable in the run table.
                "checkpoint_sha": file_digest(checkpoint),
                "split": args.split,
                "imgsz": imgsz,
                # Batch composition changes measured throughput — 06 §3.4 warns
                # that a condition-sorted batch and a mixed batch give different
                # numbers once routing is sparse. Record the conditions of the
                # measurement, not just its result.
                "batch": args.batch,
                "device": args.device,
                "panel_conf": args.conf,
                "voc07_conf": args.voc07_conf,
                f"n_images_{args.split}": n_images,
                **dataset_fingerprint(data_yaml),
                # Architecture identity must come from the CHECKPOINT, not from
                # `--config`. Cross-condition evaluations deliberately pair a
                # checkpoint with another arm's data config (an MoE model scored
                # on clear, say), and taking the yaml from the config would
                # label an MoE run with the stock architecture.
                **model_fingerprint(_model_yaml_for(checkpoint, cfg)),
            },
            metrics=metrics,
            artifacts=[out_dir],
        )
        log.info("mlflow run %s  experiment=%s", run_id, experiment)

    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


