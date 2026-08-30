"""Do the experts actually DO anything? Force each one and measure the damage.

    python scripts/expert_intervention.py --run cond3d_nomosaic_yolo11n

WHY REPRESENTATION METRICS WERE THE WRONG TOOL
----------------------------------------------
This project has judged expert diversity with linear CKA, and the two metrics it
reports disagree completely: learned expert outputs score CKA 0.82-0.98 (nearly
redundant) and cosine +0.02 to +0.06 (nearly orthogonal). A search over 36
candidate priors then found that the best achievable worst-pair CKA is 0.950
against the current 0.988 -- and that an earlier "0.30" reading came from a
divisive operator exploding to 2.4e7 on near-zero local means, not from
diversity.

The reason is structural: linear CKA on N samples measures whether two
representations carry the same INFORMATION, and any well-conditioned linear map
of a common input does. It cannot answer the question that actually matters.

That question is functional, and it is an intervention, not a correlation:
**if you send every image through the wrong expert, does the detector get
worse?** If it does not, the experts are interchangeable no matter what any
similarity metric says. If it does, they are specialised no matter how high CKA
is.

MODES
-----
    gate     the model as trained
    none     shared branch only, every expert off  (what 57% of cond3b's test
             images already received)
    all      every expert on for every image
    force:i  expert i on for every image, the others off

A specialised block should show: `force:fog` best on fog images, `force:night`
best on night, and `none` clearly worst. A clone block shows a flat table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, ensure_dir  # noqa: E402

log = get_logger("intervention")


def patched_forward(blk, mode: str):
    """Replace the block's routing with a fixed policy, leaving weights alone."""
    def fwd(x: torch.Tensor) -> torch.Tensor:
        logits = blk.gate(x.mean((2, 3)))
        probs = logits.sigmoid()
        blk.last_logits, blk.last_gate = logits, probs
        blk.last_index = logits.argmax(1)

        if mode == "gate":
            active = (probs > blk.threshold).float()
            weight = probs
        elif mode == "none":
            active = torch.zeros_like(probs)
            weight = probs
        elif mode == "all":
            active = torch.ones_like(probs)
            # Weight 1.0, not p: forcing a branch on at p=0.02 would "force" it
            # to contribute 2% of itself and measure almost nothing.
            weight = torch.ones_like(probs)
        else:  # force:i
            i = int(mode.split(":")[1])
            active = torch.zeros_like(probs)
            active[:, i] = 1.0
            weight = torch.ones_like(probs)
        blk.last_active = active

        out = blk.proj(x)
        if blk.shared is not None:
            out = out + blk.shared(x)
        for i, expert in enumerate(blk.experts):
            m = active[:, i] > 0
            if m.any():
                c = expert(x[m]) * weight[m, i].view(-1, 1, 1, 1)
                out = out.index_add(0, m.nonzero(as_tuple=True)[0], c.to(out.dtype))
        return out
    return fwd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cond3d_nomosaic_yolo11n")
    ap.add_argument("--conditions", default="union3b,fog2,night,clear")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    from src.models.moe2 import cond_moe_blocks
    from src.utils.config import REPO_ROOT

    results: dict[str, dict[str, float]] = {}
    conds = args.conditions.split(",")

    for cond in conds:
        yaml = REPO_ROOT / "configs" / "data" / f"dior_{cond}_full.yaml"
        if not yaml.exists():
            log.warning("no data config for %s (%s) - skipped", cond, yaml.name)
            continue
        for mode in ["gate", "none", "all", "force:0", "force:1", "force:2"]:
            model = YOLO(str(OUTPUT_DIR / "runs" / args.run / "weights" / "best.pt"))
            blk = cond_moe_blocks(model.model)[0]
            if mode.startswith("force:") and int(mode.split(":")[1]) >= blk.n_experts:
                continue
            blk.forward = patched_forward(blk, mode)
            m = model.val(data=str(yaml), split=args.split, imgsz=args.imgsz,
                          device=0, verbose=False, plots=False, augment=False)
            label = mode
            if mode.startswith("force:"):
                label = f"force:{blk.expert_kinds[int(mode.split(':')[1])]}"
            results.setdefault(cond, {})[label] = float(m.box.map50)
            results.setdefault(cond + "_5095", {})[label] = float(m.box.map)
            log.info("  %-8s %-14s mAP50 %.4f  mAP50-95 %.4f",
                     cond, label, m.box.map50, m.box.map)
            del model
            torch.cuda.empty_cache()

    log.info("")
    log.info("=== mAP50 UNDER FORCED ROUTING (%s) ===", args.run)
    modes = ["gate", "none", "all", "force:clear", "force:fog", "force:night"]
    log.info("  %-10s" % "data" + "".join(f"{m:>14s}" for m in modes))
    for cond in conds:
        if cond not in results:
            continue
        row = "  %-10s" % cond
        for m in modes:
            v = results[cond].get(m)
            row += f"{v:14.4f}" if v is not None else f"{'-':>14s}"
        log.info(row)
        vals = [v for v in results[cond].values()]
        log.info("  %-10s spread %.4f  (max - min across routing policies)",
                 "", max(vals) - min(vals))

    out = ensure_dir(OUTPUT_DIR / "analysis") / f"intervention_{args.run}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("")
    log.info("A flat row means the experts are interchangeable: routing cannot")
    log.info("matter if every destination behaves the same. wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
