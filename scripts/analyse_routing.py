"""Does the gate route by CONDITION, and do the experts actually differ?

    python scripts/analyse_routing.py --run moe_neck_yolo11n --condition union

Accuracy was never the point of the MoE. The point was that one branch handles
haze and another handles low light, selected by the gate. That claim is not
tested by mAP at all — it is tested by two measurements, and neither has been
made until now:

1. **Router confusion.** The union corpus encodes the true condition in every
   filename (`clear_00042.jpg`), so the gate's choice can be compared directly
   against ground truth. Reported as a confusion matrix, per-condition purity,
   and normalised mutual information between route and condition. NMI is the
   headline: 1.0 means the route IS the condition, 0.0 means the gate is
   splitting the data on something unrelated.

2. **Expert divergence.** Specialisation requires the experts to compute
   DIFFERENT functions. Each expert is run on the same inputs and their outputs
   compared (cosine similarity and CKA). If the experts converge to near
   identical functions then routing is cosmetic — every route gives the same
   answer — and that alone would explain a null accuracy result without saying
   anything about whether routing is a good idea.

Together these separate three very different failure modes that mAP collapses
into one: the gate is random, the experts are clones, or the design is sound but
the task does not reward it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.moe import moe_blocks  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, dataset_root, ensure_dir  # noqa: E402

log = get_logger("analyse_routing")

CONDITIONS = ("clear", "fog", "night")


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """CKA between two (N, D) activation matrices. 1.0 = same representation."""
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    hsic = (x.T @ y).norm("fro") ** 2
    nx = (x.T @ x).norm("fro")
    ny = (y.T @ y).norm("fro")
    return float(hsic / (nx * ny + 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--condition", default="union", help="corpus condition dir to read")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=900)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from ultralytics import YOLO

    ckpt = OUTPUT_DIR / "runs" / args.run / "weights" / "best.pt"
    if not ckpt.exists():
        log.error("no checkpoint: %s", ckpt)
        return 2
    model = YOLO(str(ckpt))
    net = model.model.to(args.device).eval().float()
    blocks = moe_blocks(net)
    if not blocks:
        log.error("%s has no MoE block", args.run)
        return 2
    block = blocks[0]
    n_experts = block.n_experts
    log.info("%s: %d expert(s), kernels=%s", args.run, n_experts, block.kernels)

    img_dir = dataset_root("detect", "full") / args.condition / "images" / args.split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    rng = np.random.default_rng(0)
    if len(images) > args.limit:
        images = [images[i] for i in rng.choice(len(images), args.limit, replace=False)]
    log.info("scoring %d images from %s", len(images), img_dir)

    # --- capture the block's input so experts can be compared on it --------
    captured: list[torch.Tensor] = []

    def hook(_m, inp, _out):
        captured.append(inp[0].detach())

    handle = block.register_forward_hook(hook)

    routes: list[int] = []
    truth: list[str] = []
    import cv2

    with torch.no_grad():
        for i in range(0, len(images), 16):
            chunk = images[i : i + 16]
            batch = []
            for p in chunk:
                im = cv2.imread(str(p))
                im = cv2.resize(im, (args.imgsz, args.imgsz))
                batch.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(batch)).float().div(255).to(args.device)
            captured.clear()
            net(x)
            routes.extend(block.last_index.cpu().tolist())
            # `clear_00042` -> clear ; a bare id means the dir is single-condition
            truth.extend(
                [p.stem.split("_")[0] if p.stem.split("_")[0] in CONDITIONS else args.condition
                 for p in chunk]
            )
    handle.remove()

    # --- 1. routing vs condition ------------------------------------------
    conds = sorted(set(truth))
    matrix = np.zeros((len(conds), n_experts), dtype=int)
    for r, t in zip(routes, truth):
        matrix[conds.index(t), r] += 1

    log.info("ROUTER CONFUSION (rows = true condition, cols = expert):")
    header = "  " + " ".join(f"{'e'+str(e):>8s}" for e in range(n_experts))
    log.info("%-10s%s%9s", "", header, "purity")
    purities = []
    for i, c in enumerate(conds):
        row = matrix[i]
        purity = row.max() / max(row.sum(), 1)
        purities.append(purity)
        log.info("%-10s  %s%9.3f", c, " ".join(f"{v:8d}" for v in row), purity)

    # Normalised mutual information: 1.0 = the route IS the condition.
    joint = matrix / max(matrix.sum(), 1)
    p_route = joint.sum(0)          # (n_experts,)
    p_cond = joint.sum(1)           # (n_conditions,)
    independent = np.outer(p_cond, p_route)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / independent[nz])).sum())
    h_c = float(-(p_cond[p_cond > 0] * np.log(p_cond[p_cond > 0])).sum())
    h_r = float(-(p_route[p_route > 0] * np.log(p_route[p_route > 0])).sum())
    # h_r == 0 when every image takes the same route: the gate carries no
    # information at all, so NMI is 0 by definition rather than undefined.
    nmi = 0.0 if h_r < 1e-12 or h_c < 1e-12 else mi / (h_c * h_r) ** 0.5
    log.info("route distribution: %s", {f"e{i}": round(float(p), 3) for i, p in enumerate(p_route)})
    log.info("normalised mutual information(route ; condition) = %.4f", nmi)
    log.info("  (1.0 = route is the condition; 0.0 = gate splits on something else)")

    # --- 2. do the experts compute different functions? --------------------
    feat = captured[-1] if captured else None
    sims: dict[str, float] = {}
    if feat is not None:
        with torch.no_grad():
            # Flattened expert outputs are C*H*W wide, which makes the CKA Gram
            # matrices enormous. Spatially average-pool to a fixed grid first:
            # specialisation is a question about what the channels encode, not
            # about pixel-level detail.
            outs = [
                torch.nn.functional.adaptive_avg_pool2d(e(feat), 4).flatten(1)
                for e in block.experts
            ]
        for a in range(n_experts):
            for b in range(a + 1, n_experts):
                cos = float(
                    torch.nn.functional.cosine_similarity(outs[a], outs[b], dim=1).mean()
                )
                sims[f"cosine_e{a}_e{b}"] = cos
                sims[f"cka_e{a}_e{b}"] = linear_cka(outs[a], outs[b])
        norms = [float(o.norm(dim=1).mean()) for o in outs]
        log.info("EXPERT DIVERGENCE on identical inputs:")
        for k, v in sims.items():
            log.info("  %-16s %.4f", k, v)
        log.info("  output norms per expert: %s", [round(n, 3) for n in norms])
        log.info("  (cosine/CKA near 1.0 => the experts are clones and routing is cosmetic)")

    out = ensure_dir(OUTPUT_DIR / "analysis")
    (out / f"routing_{args.run}_{args.condition}.json").write_text(
        json.dumps(
            {
                "run": args.run,
                "condition_dir": args.condition,
                "split": args.split,
                "n_images": len(images),
                "conditions": conds,
                "confusion": matrix.tolist(),
                "purity": {c: float(p) for c, p in zip(conds, purities)},
                "nmi_route_condition": nmi,
                "expert_similarity": sims,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", out / f"routing_{args.run}_{args.condition}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
