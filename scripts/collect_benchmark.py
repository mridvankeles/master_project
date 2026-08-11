"""Assemble the benchmark table from evaluation artefacts.

    python scripts/collect_benchmark.py --out docs/benchmarks.md

Every number in the output is read from a file on disk -- `metrics.json`,
`voc07.json`, `speed.json` -- never transcribed by hand. `05-experiment-plan.md`
requires that one script regenerates every reported table from checkpoints; this
is that script for the benchmark table.

Rows are keyed by evaluation NAME (`<run>_<split>`), because one checkpoint can
be evaluated on several conditions -- a clear-trained model scored on fog is a
different row from the same model scored on clear, and collapsing them would
destroy the only comparison that matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR  # noqa: E402

log = get_logger("collect_benchmark")


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def collect_evals(eval_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        metrics = _load(d / "metrics.json")
        if not metrics:
            continue
        voc = _load(d / "voc07.json")
        row = {
            "name": d.name,
            "split": metrics.get("split"),
            "checkpoint": metrics.get("checkpoint", ""),
            "mAP50": metrics["overall"].get("mAP50"),
            "mAP50_95": metrics["overall"].get("mAP50-95"),
            "precision": metrics["overall"].get("precision"),
            "recall": metrics["overall"].get("recall"),
            "voc07_mAP50": (voc or {}).get("mAP@0.5"),
            "per_class": metrics.get("per_class", []),
        }
        rows.append(row)
    return rows


def collect_speed(path: Path) -> dict[str, dict]:
    data = _load(path) or {}
    return {r["run"]: r for r in data.get("results", [])}


def _fmt(v, digits=4):
    return "—" if v is None else f"{v:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--speed", default=None)
    parser.add_argument("--out", default="docs/benchmarks.md")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir) if args.eval_dir else OUTPUT_DIR / "eval"
    speed_path = Path(args.speed) if args.speed else OUTPUT_DIR / "benchmark" / "speed.json"

    rows = collect_evals(eval_dir)
    speed = collect_speed(speed_path)
    if not rows:
        log.error("no metrics.json under %s", eval_dir)
        return 1

    def run_of(row):
        # `.../outputs/runs/<run>/weights/best.pt`
        parts = Path(row["checkpoint"]).parts
        return parts[-3] if len(parts) >= 3 else ""

    lines = [
        "| Evaluation | Split | COCO mAP50 | COCO mAP50-95 | VOC07 mAP@0.5 | P | R |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['split']} | {_fmt(r['mAP50'])} | {_fmt(r['mAP50_95'])} "
            f"| {_fmt(r['voc07_mAP50'])} | {_fmt(r['precision'], 3)} | {_fmt(r['recall'], 3)} |"
        )

    speed_lines = [
        "",
        "| Model | Params | GFLOPs | b1 latency (ms) | b1 FPS | b16 ms/img |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in sorted(speed.items()):
        speed_lines.append(
            f"| `{name}` | {int(s.get('n_parameters', 0)):,} | {s.get('gflops', 0):.2f} "
            f"| {s.get('latency_b1_ms', 0):.2f} | {s.get('fps_b1', 0):.1f} "
            f"| {s.get('ms_per_image_b16', 0):.3f} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines + speed_lines) + "\n", encoding="utf-8")
    log.info("wrote %s (%d eval rows, %d speed rows)", out, len(rows), len(speed))
    print("\n".join(lines + speed_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
