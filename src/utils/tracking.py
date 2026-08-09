"""MLflow logging for the parts Ultralytics' own callback does not cover.

WHAT ULTRALYTICS LOGS, AND WHAT IT DOES NOT
-------------------------------------------
Ultralytics ships an MLflow callback, and `scripts/train.py` relies on it. But
the callback registers TRAINER hooks only — `on_pretrain_routine_end`,
`on_train_epoch_end`, `on_fit_epoch_end`, `on_train_end` (verified against
ultralytics 8.4.115, `utils/callbacks/mlflow.py`). Three consequences:

1. A standalone `model.val()` — which is the whole of `scripts/eval.py` — fires
   NO callback and logs NOTHING. Every test-split number this project has
   produced lived only as JSON on disk.
2. Only aggregate metrics are logged. Per-class AP never reaches the tracker,
   although the per-class table is what the results docs actually argue from.
3. Nothing knows about VOC07 11-point AP, which is the convention NIRNet
   reports and therefore the number the thesis comparison rests on.

This module fills those three gaps, plus model complexity, which
`06-moe-design-guide.md` §5 requires to be reported alongside accuracy and which
becomes mandatory once experts make total and activated parameters diverge.

WHY EVALUATION GETS ITS OWN RUN
-------------------------------
An evaluation is not a property of a training run. The same checkpoint is
evaluated on several splits, and — the case that settles it — on a condition it
was not trained on (`clearOBB_on_fog` in `results-obb-and-nirnet.md`). That
result cannot belong to the clear model's training run without lying about what
was measured. So each evaluation opens its own run and links back via the
`source_run` tag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Metric keys are namespaced by split (`test/mAP50`, `val/mAP50`) so a checkpoint
# evaluated on several splits stays unambiguous in one MLflow table.
_SLASH = "/"


def model_complexity(model: Any, imgsz: int = 640) -> dict[str, float]:
    """Layer / parameter / FLOP counts.

    Counted directly rather than via `model.info()`: that method RETURNS None
    unless `verbose=True` (it prints instead of returning), so unpacking its
    result silently fails and costs you the GFLOPs number. Verified against
    ultralytics 8.4.115.

    Both counts matter for the MoE. `n_parameters` is the check that an expert
    block actually landed — if adding experts does not move it, the edit went
    into the wrong Ultralytics copy, which `06-moe-design-guide.md` §3.2 names as
    the most common failure. GFLOPs is the *dense-equivalent* cost: thop traces
    every branch, so once routing is sparse this becomes total, not activated,
    compute. Activated cost has to be measured separately.
    """
    inner = getattr(model, "model", model)
    out: dict[str, float] = {}
    try:
        out["n_parameters"] = float(sum(p.numel() for p in inner.parameters()))
        out["n_gradients"] = float(sum(p.numel() for p in inner.parameters() if p.requires_grad))
        out["n_layers"] = float(len(list(inner.modules())))
    except Exception:  # noqa: BLE001 - diagnostic only, never load-bearing
        pass
    try:
        from ultralytics.utils.torch_utils import get_flops

        out["gflops"] = float(get_flops(inner, imgsz))
    except Exception:  # noqa: BLE001 - thop is optional and fails on custom modules
        pass
    return out


def file_digest(path: Path, length: int = 12) -> str:
    """Short sha256 of a file's contents, or '' if it is not readable."""
    import hashlib

    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:length]
    except OSError:
        return ""


def split_images(data_yaml: Path) -> dict[str, list[str]]:
    """Image stems per split, as the data yaml actually resolves them.

    Handles both forms Ultralytics accepts: a directory, or a .txt list of image
    paths (which is what `train.py` writes for a subset run).
    """
    import yaml as _yaml

    spec = _yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(spec["path"])
    exts = {".jpg", ".jpeg", ".png"}
    out: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        entry = spec.get(split)
        if not entry:
            continue
        p = Path(entry)
        p = p if p.is_absolute() else root / p
        if p.suffix == ".txt" and p.exists():
            out[split] = sorted(
                Path(ln).stem for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()
            )
        elif p.is_dir():
            out[split] = sorted(q.stem for q in p.iterdir() if q.suffix.lower() in exts)
    return out


def dataset_fingerprint(data_yaml: Path) -> dict[str, Any]:
    """Identity of the corpus a run consumed — contents, not just a path.

    Ultralytics already logs `data` as a PATH. A path is not identity: rebuild
    `data/dior_hbb` from a different source and every logged path stays byte for
    byte the same while the numbers change underneath. That stops being
    hypothetical the moment the aligned 2,607-id corpus and the full 23,463-id
    DIOR corpus both exist.

    So this records, per split, the count and a digest over the sorted image
    stems. Two runs share a `data_sha_train` if and only if they trained on the
    same set of images.
    """
    import hashlib

    data_yaml = Path(data_yaml)
    out: dict[str, Any] = {
        "data_yaml": data_yaml.as_posix(),
        "data_yaml_sha": file_digest(data_yaml),
    }
    try:
        import yaml as _yaml

        spec = _yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        root = Path(spec["path"])
        out["data_root"] = root.as_posix()
        # manifest.csv is written by prepare_dataset one level above the
        # condition directory, and is the provenance record for the whole build.
        manifest = root.parent / "manifest.csv"
        if manifest.exists():
            out["data_manifest_sha"] = file_digest(manifest)

        for split, stems in split_images(data_yaml).items():
            out[f"data_n_{split}"] = len(stems)
            out[f"data_sha_{split}"] = hashlib.sha256(
                "\n".join(stems).encode("utf-8")
            ).hexdigest()[:12]
    except Exception:  # noqa: BLE001 - a missing fingerprint must not kill a run
        out["data_fingerprint_error"] = "unreadable"
    return out


def model_fingerprint(model_yaml: Path | None) -> dict[str, Any]:
    """Identity of the architecture: the yaml path and a digest of its contents.

    The digest is the part that matters once the MoE arrives. Ultralytics logs
    `model` as a path and this project tags `model_cfg` with the filename, but
    neither changes when a `C3k2` line in that yaml is swapped for an
    `MoEBlock` line. The digest does.

    Complexity (parameters, GFLOPs) is deliberately NOT included here: it is
    logged as a METRIC rather than a param, because MLflow stores params as
    strings and cannot plot them. Recording it as a metric is what makes the
    capacity-control curve — mAP against parameter count across runs — a chart
    in the UI instead of a spreadsheet.
    """
    out: dict[str, Any] = {}
    if model_yaml is not None:
        model_yaml = Path(model_yaml)
        out["model_yaml"] = model_yaml.as_posix()
        out["model_yaml_sha"] = file_digest(model_yaml)
    return out


def log_evaluation(
    *,
    tracking_uri: str,
    experiment: str,
    run_name: str,
    tags: dict[str, str],
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
) -> str | None:
    """Open one MLflow run for an evaluation and log it. Returns the run id.

    Deliberately takes plain dicts rather than the report objects: this module
    should not have to know the shape of `EvalReport` or `VOC07Report`, and the
    caller already has both open.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        # Values are stringified by MLflow anyway; do it here so a Path or a bool
        # reads the same in the UI as it does in metrics.json.
        mlflow.log_params({k: str(v) for k, v in params.items()})
        # step=0: an evaluation is a single point, not a series. Leaving the step
        # unset would make repeated evals of the same checkpoint overwrite in the
        # UI rather than appear as separate runs.
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=0)
        for path in artifacts:
            if not path.exists():
                continue
            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=path.name)
            else:
                mlflow.log_artifact(str(path))
        return run.info.run_id


def evaluation_metrics(report: Any, voc: Any, split: str) -> dict[str, float]:
    """Flatten an EvalReport (+ optional VOC07Report) into MLflow metric keys.

    Per-class AP is included because the thesis argues from it — the
    `storagetank` P 0.905 / R 0.093 finding in `results-fog-yolo11n.md` is a
    per-class observation, and it should be queryable rather than buried in a
    JSON file.
    """
    m: dict[str, float] = {}

    for key, value in report.overall.items():
        m[f"{split}{_SLASH}{key}"] = value

    # Speed. `04-method-open-questions.md` § Inference-speed measurement protocol
    # requires BOTH throughput and latency, at a stated batch size and
    # resolution, alongside FLOPs and parameter count — and warns that omitting
    # any component of the cost is easy for a reviewer to spot.
    speed = report.speed_ms or {}
    for stage, ms in speed.items():
        m[f"{split}{_SLASH}speed_{stage}_ms"] = ms

    # `loss` is a validation-only stage and is not part of deployed inference,
    # so it is excluded from the end-to-end figure but still logged above.
    stages = {k: v for k, v in speed.items() if k != "loss"}
    if stages:
        total = sum(stages.values())
        m[f"{split}{_SLASH}latency_ms"] = total
        if total > 0:
            m[f"{split}{_SLASH}fps"] = 1000.0 / total
        inference = speed.get("inference")
        if inference:
            # Model-only throughput. Always >= fps, and the number that gets
            # quoted misleadingly if pre/postprocess are dropped — both are
            # recorded so the gap is visible rather than hidden.
            m[f"{split}{_SLASH}fps_inference_only"] = 1000.0 / inference

    for c in report.per_class:
        m[f"{split}{_SLASH}AP50{_SLASH}{c.name}"] = c.ap50
        m[f"{split}{_SLASH}AP50-95{_SLASH}{c.name}"] = c.ap50_95
        # Instance counts explain an AP of zero: absent class vs. genuine miss.
        m[f"{split}{_SLASH}instances{_SLASH}{c.name}"] = float(c.instances)

    if voc is not None:
        m[f"{split}{_SLASH}voc07_mAP50"] = voc.mean_ap
        for name, ap in voc.per_class_ap.items():
            m[f"{split}{_SLASH}voc07_AP50{_SLASH}{name}"] = ap

    return m
