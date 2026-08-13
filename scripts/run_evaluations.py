"""Run every evaluation the benchmark table needs, in one pass.

    python scripts/run_evaluations.py             # all of them
    python scripts/run_evaluations.py --dry-run   # print the plan only

WHAT GETS EVALUATED, AND WHY EACH CELL EXISTS
---------------------------------------------
Each checkpoint is scored on its OWN condition and on the OTHER one. The
off-diagonal is not padding:

* `clear -> fog` is **rPC**, the robustness-retention ratio NIRNet (64.3) and
  MPRNet (64.1) both report. It is the only quantity of theirs that is
  legitimately comparable to ours, because a ratio cancels the box-format and
  metric-convention differences that make their absolute mAP uncomparable
  (`comparison-baselines.md` § How to use it honestly).
* specialist-vs-union on the same condition is the **interference test**. If the
  union model matches both specialists, routing has nothing to recover and the
  MoE is unmotivated no matter how well it trains. That is claim C2, and it is
  cheaper to answer here than in month 8.

VOC07 IS ONLY COMPUTED WHERE IT IS NEEDED
-----------------------------------------
The VOC07 pass re-runs inference over the whole split at conf 0.01. On clear
(11,738 images) that is worth it, because 57.1 is a VOC-style number and the
comparison is the point of Gate 1. On fog (35,112) and union (46,850) it costs
over an hour of GPU for a number no published result is waiting on, so it is
skipped and the COCO metrics carry those rows. Skipped cells are recorded as
such rather than left blank.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, REPO_ROOT  # noqa: E402

log = get_logger("run_evaluations")

CFG = "configs/train"

# (checkpoint run, data config supplying the condition, output name, run voc07)
PLAN: list[tuple[str, str, str, bool]] = [
    # --- diagonal: each model on the condition it was trained for ----------
    ("clear_full_yolo11n", "clear_full_yolo11n", "clear_on_clear", True),
    ("fog_full_yolo11n", "fog_full_yolo11n", "fog_on_fog", False),
    ("union_full_yolo11n", "union_full_yolo11n", "union_on_union", False),
    # --- union model split by condition: the interference test -------------
    ("union_full_yolo11n", "clear_full_yolo11n", "union_on_clear", False),
    ("union_full_yolo11n", "fog_full_yolo11n", "union_on_fog", False),
    # --- cross-condition: rPC for the clear specialist ---------------------
    ("clear_full_yolo11n", "fog_full_yolo11n", "clear_on_fog", False),
    ("fog_full_yolo11n", "clear_full_yolo11n", "fog_on_clear", False),
    # --- MoE arms, same cells as the union model so they are comparable ----
    ("moe_backbone_yolo11n", "union_full_yolo11n", "moe_backbone_on_union", False),
    ("moe_backbone_yolo11n", "clear_full_yolo11n", "moe_backbone_on_clear", False),
    ("moe_backbone_yolo11n", "fog_full_yolo11n", "moe_backbone_on_fog", False),
    ("moe_neck_yolo11n", "union_full_yolo11n", "moe_neck_on_union", False),
    ("moe_neck_yolo11n", "clear_full_yolo11n", "moe_neck_on_clear", False),
    ("moe_neck_yolo11n", "fog_full_yolo11n", "moe_neck_on_fog", False),
    # --- three-condition arms (clear / fog / night) ------------------------
    ("union3_full_yolo11n", "union3_full_yolo11n", "union3_on_union3", False),
    ("union3_full_yolo11n", "clear_full_yolo11n", "union3_on_clear", False),
    ("union3_full_yolo11n", "fog_full_yolo11n", "union3_on_fog", False),
    ("union3_full_yolo11n", "night_full_yolo11n", "union3_on_night", False),
    ("moe3_neck_yolo11n", "union3_full_yolo11n", "moe3_on_union3", False),
    ("moe3_neck_yolo11n", "clear_full_yolo11n", "moe3_on_clear", False),
    ("moe3_neck_yolo11n", "fog_full_yolo11n", "moe3_on_fog", False),
    ("moe3_neck_yolo11n", "night_full_yolo11n", "moe3_on_night", False),
    ("night_full_yolo11n", "night_full_yolo11n", "night_on_night", False),
    ("night_full_yolo11n", "clear_full_yolo11n", "night_on_clear", False),
    # --- NWD box loss: isolated, then composed with routing ----------------
    ("union_nwd_yolo11n", "union_full_yolo11n", "union_nwd_on_union", False),
    ("union_nwd_yolo11n", "clear_full_yolo11n", "union_nwd_on_clear", False),
    ("union_nwd_yolo11n", "fog_full_yolo11n", "union_nwd_on_fog", False),
    ("moe_neck_nwd_yolo11n", "union_full_yolo11n", "moe_neck_nwd_on_union", False),
    ("moe_neck_nwd_yolo11n", "clear_full_yolo11n", "moe_neck_nwd_on_clear", False),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--only", nargs="*", default=None, help="output names to run")
    args = parser.parse_args()

    python = sys.executable
    done, skipped, failed = [], [], []

    for ckpt_run, data_run, name, voc in PLAN:
        if args.only and name not in args.only:
            continue
        ckpt = OUTPUT_DIR / "runs" / ckpt_run / "weights" / "best.pt"
        cfg = REPO_ROOT / CFG / f"{data_run}.yaml"
        if not cfg.exists():
            # Never fall back to another arm's data config. A missing config
            # silently evaluated against the wrong corpus once already, and the
            # result was two "different" cells reporting byte-identical numbers.
            log.error("skip %-24s (no data config: %s)", name, cfg.name)
            failed.append(name)
            continue
        if not ckpt.exists():
            log.warning("skip %-24s (no checkpoint: %s)", name, ckpt_run)
            skipped.append(name)
            continue

        cmd = [
            python, "scripts/eval.py",
            "--checkpoint", str(ckpt),
            "--config", str(cfg),
            "--split", args.split,
            "--name", name,
        ]
        if not voc:
            cmd.append("--skip-voc07")

        log.info("[%s] %s on %s%s", name, ckpt_run, data_run, "  (+voc07)" if voc else "")
        if args.dry_run:
            continue

        start = time.time()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error("FAILED %s\n%s", name, proc.stdout[-1500:] + proc.stderr[-1500:])
            failed.append(name)
        else:
            done.append(name)
            log.info("  ok in %.1f min", (time.time() - start) / 60)

    log.info("done=%d skipped=%d failed=%d", len(done), len(skipped), len(failed))
    if skipped:
        log.warning("skipped (no checkpoint): %s", ", ".join(skipped))
    if failed:
        log.error("failed: %s", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
