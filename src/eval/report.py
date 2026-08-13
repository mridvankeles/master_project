"""Turn Ultralytics validation results into the tables the thesis needs.

Ultralytics computes mAP50, mAP50-95 and per-class AP already; there is no
reason to recompute them and every reason not to — a hand-rolled mAP that
disagrees with the framework's is a liability, not a contribution. This module
only *reshapes* what the validator returns into a metrics JSON and a markdown
table, and records which metric convention produced them.

METRIC CONVENTION — matters for comparability
---------------------------------------------
Ultralytics reports COCO-style AP: 101-point interpolation, mAP50 at IoU 0.50
and mAP50-95 averaged over IoU 0.50:0.05:0.95. Most DIOR papers report VOC-style
mAP@0.5 (11-point or all-point interpolation). mAP50 here is therefore close to,
but NOT identical with, a published DIOR "mAP" number. Any table that puts them
side by side has to say so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..data.dior_classes import DIOR_CLASSES

METRIC_CONVENTION = (
    "Ultralytics/COCO 101-point interpolation; mAP50 at IoU=0.50, "
    "mAP50-95 averaged over IoU 0.50:0.05:0.95"
)


@dataclass
class ClassMetrics:
    name: str
    class_id: int
    instances: int
    precision: float
    recall: float
    ap50: float
    ap50_95: float


@dataclass
class EvalReport:
    checkpoint: str
    data_yaml: str
    split: str
    imgsz: int
    device: str
    git_commit: str
    metric_convention: str = METRIC_CONVENTION
    overall: dict[str, float] = field(default_factory=dict)
    per_class: list[ClassMetrics] = field(default_factory=list)
    speed_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def write_json(self, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dst

    def write_markdown(self, dst: Path) -> Path:
        lines = [
            f"# Evaluation — {Path(self.checkpoint).name}",
            "",
            f"- Checkpoint: `{self.checkpoint}`",
            f"- Data: `{self.data_yaml}`  split: **{self.split}**",
            f"- Image size: {self.imgsz}   device: {self.device}",
            f"- Repo commit: `{self.git_commit}`",
            f"- Metric convention: {self.metric_convention}",
            "",
            "## Overall",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        for k, v in self.overall.items():
            lines.append(f"| {k} | {v:.4f} |")

        if self.speed_ms:
            lines += [
                "",
                "## Speed (ms/image)",
                "",
                "| Stage | ms |",
                "|---|---|",
                *[f"| {k} | {v:.2f} |" for k, v in self.speed_ms.items()],
            ]

        lines += [
            "",
            "## Per class",
            "",
            "| Class | Instances | P | R | AP50 | AP50-95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for c in sorted(self.per_class, key=lambda x: x.ap50, reverse=True):
            lines.append(
                f"| {c.name} | {c.instances} | {c.precision:.3f} | {c.recall:.3f} "
                f"| {c.ap50:.3f} | {c.ap50_95:.3f} |"
            )

        absent = [c.name for c in self.per_class if c.instances == 0]
        if absent:
            lines += [
                "",
                f"> {len(absent)} class(es) have zero instances in this split and their AP is "
                f"meaningless, not zero-performance: {', '.join(absent)}.",
            ]

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dst


def class_names_from(data_yaml: Path) -> list[str]:
    """Class names from the data yaml, falling back to DIOR's.

    Naming per-class rows with DIOR_CLASSES regardless of corpus reports the two
    DroneVehicle classes as "airplane" and "airport". The aggregate metrics stay
    correct, so the error is invisible in the headline number and wrong in
    exactly the table the analysis argues from.
    """
    import yaml as _yaml

    try:
        spec = _yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
        names = spec.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list) and names:
            return list(names)
    except Exception:  # noqa: BLE001
        pass
    return list(DIOR_CLASSES)


def from_ultralytics(
    results,
    checkpoint: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    device: str,
    git_commit: str,
) -> EvalReport:
    """Build an EvalReport from an Ultralytics DetMetrics object."""
    box = results.box
    names = class_names_from(data_yaml)

    report = EvalReport(
        checkpoint=str(checkpoint),
        data_yaml=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        git_commit=git_commit,
        overall={
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "fitness": float(results.fitness),
        },
        speed_ms={k: float(v) for k, v in (results.speed or {}).items()},
    )

    # `ap_class_index` lists only the classes actually present in the split, so
    # the per-class arrays are indexed positionally, not by class id.
    present = list(getattr(box, "ap_class_index", []))
    counts = _instance_counts(results)

    for pos, class_id in enumerate(present):
        report.per_class.append(
            ClassMetrics(
                name=names[int(class_id)] if int(class_id) < len(names) else f"class_{int(class_id)}",
                class_id=int(class_id),
                instances=int(counts.get(int(class_id), 0)),
                precision=float(box.p[pos]),
                recall=float(box.r[pos]),
                ap50=float(box.ap50[pos]),
                ap50_95=float(box.ap[pos]),
            )
        )

    # Classes with no instances never appear in ap_class_index; record them
    # explicitly so a reader can tell "absent" from "scored zero".
    for class_id, name in enumerate(names):
        if class_id not in [int(c) for c in present]:
            report.per_class.append(
                ClassMetrics(
                    name=name,
                    class_id=class_id,
                    instances=int(counts.get(class_id, 0)),
                    precision=0.0,
                    recall=0.0,
                    ap50=0.0,
                    ap50_95=0.0,
                )
            )

    return report


def _instance_counts(results) -> dict[int, int]:
    """Ground-truth instances per class, if the validator exposed them."""
    nt = getattr(results, "nt_per_class", None)
    if nt is None:
        return {}
    return {i: int(n) for i, n in enumerate(nt)}
