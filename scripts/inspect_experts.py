"""Look inside the experts: did the fog branch learn to dehaze, and the night
branch to see in the dark?

    python scripts/inspect_experts.py --run cond3h_hetero_yolo11n

THE MEASUREMENT
---------------
The corpus gives us the answer for free. `clear_00042`, `fog2_00042` and
`night_00042` are the same scene, so for any degraded image there is a *target*:
the block output the model produces for its clear twin. Dehazing then has a
definition, not an opinion:

    gap_before = || (proj + shared)(f_degraded)  -  block_out(f_clear) ||
    gap_after  = || block_out(f_degraded)        -  block_out(f_clear) ||
    gain       = 1 - gap_after / gap_before

`gap_before` is the block WITHOUT any expert -- the always-on path alone.
`gap_after` is the block as it runs. A positive gain means the experts moved the
degraded features towards their clear twin, which is exactly what "removing fog
before the prediction head" means. A gain of zero means they did nothing, which
is what `expert_intervention.py` found for design 2.

SPECIALISATION
--------------
The same number is computed with each expert FORCED. If the fog branch closes
the fog gap better than the night branch does, and vice versa, the branches are
specialised -- measured on the thing they were built for, rather than inferred
from a representation-similarity score.

RENDERS
-------
Per-image panels: the degraded input, its clear twin, each expert's output
(channel mean), and the night branch's contrast-attention map, so the claim
"it enhances high-contrast regions" can be looked at rather than asserted.
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

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, ensure_dir  # noqa: E402

log = get_logger("inspect_experts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cond3h_hetero_yolo11n")
    ap.add_argument("--condition", default="union3b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--images", type=int, default=96)
    ap.add_argument("--render", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    from src.models.moe2 import cond_moe_blocks
    from src.models.paired import clear_twin_path, condition_token
    from src.utils.paths import dataset_root

    net = YOLO(str(OUTPUT_DIR / "runs" / args.run / "weights" / "best.pt")).model.cuda().eval()
    blk = cond_moe_blocks(net)[0]
    hetero = getattr(blk, "arch", "static") == "hetero"
    kinds = list(blk.expert_kinds)

    cap: dict[str, torch.Tensor] = {}
    blk.register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0].detach()))

    d = dataset_root("detect", "full") / args.condition / "images" / args.split
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    deg = [p for p in files if condition_token(p) in ("fog", "night")]
    rng = np.random.default_rng(0)
    deg = [deg[i] for i in rng.choice(len(deg), min(args.images, len(deg)), replace=False)]

    def load(p):
        im = cv2.imread(str(p))
        im = cv2.resize(im, (args.imgsz, args.imgsz))
        return torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)))

    def features(paths):
        x = torch.stack([load(p) for p in paths]).cuda().float().div_(255)
        with torch.no_grad():
            net(x)
        return cap["x"].clone()

    gains: dict[str, list[float]] = defaultdict(list)
    forced: dict[tuple[str, str], list[float]] = defaultdict(list)
    render_rows = []

    for i in range(0, len(deg), 8):
        chunk = [p for p in deg[i:i + 8] if clear_twin_path(p) is not None]
        if not chunk:
            continue
        twins = [clear_twin_path(p) for p in chunk]
        f_deg = features(chunk)
        f_cl = features(twins)

        with torch.no_grad():
            ctx_c = blk.proj(f_cl)
            sh_c = blk.shared(f_cl, None) if hetero else blk.shared(f_cl)
            target = ctx_c + sh_c                     # block output for the clear twin,
            # with no expert -- clear images route to the clear branch, whose job
            # is by construction "do nothing special".

            ctx = blk.proj(f_deg)
            base = ctx + (blk.shared(f_deg, None) if hetero else blk.shared(f_deg))
            gap_before = (base - target).flatten(1).norm(dim=1)

            probs = blk.gate(f_deg.mean((2, 3))).sigmoid()
            active = (probs > blk.threshold).float()
            w = active if getattr(blk, "hard_mask", False) else probs
            out = base.clone()
            per_expert = {}
            for e_i, expert in enumerate(blk.experts):
                ei = expert(f_deg, ctx) if hetero else expert(f_deg)
                per_expert[kinds[e_i]] = ei
                out = out + ei * w[:, e_i].view(-1, 1, 1, 1)
            gap_after = (out - target).flatten(1).norm(dim=1)

            for j, p in enumerate(chunk):
                c = condition_token(p)
                gains[c].append(float(1 - gap_after[j] / gap_before[j].clamp_min(1e-6)))
                for k in kinds:
                    g = ((base[j] + per_expert[k][j]) - target[j]).norm()
                    forced[(c, k)].append(float(1 - g / gap_before[j].clamp_min(1e-6)))

            if len(render_rows) < args.render:
                render_rows.append((chunk[0], twins[0],
                                    {k: v[0].cpu() for k, v in per_expert.items()},
                                    base[0].cpu(), target[0].cpu()))

    log.info("")
    log.info("=== DID THE EXPERTS MOVE DEGRADED FEATURES TOWARDS THEIR CLEAR TWIN? ===")
    log.info("  gain = 1 - ||block_out(deg) - block_out(clear)|| / ||no-expert(deg) - block_out(clear)||")
    log.info("  positive = the experts close the gap; 0 = they do nothing")
    for c in ("fog", "night"):
        if gains[c]:
            v = np.array(gains[c])
            log.info("  %-6s n=%3d   gain %+.4f  (median %+.4f, %.0f%% of images improved)",
                     c, len(v), v.mean(), float(np.median(v)), 100 * (v > 0).mean())

    log.info("")
    log.info("=== SPECIALISATION: gain when each branch is FORCED alone ===")
    log.info("  %-8s" % "data" + "".join(f"{k:>12s}" for k in kinds) + "     winner")
    for c in ("fog", "night"):
        row = "  %-8s" % c
        best, bestv = None, -9
        for k in kinds:
            v = float(np.mean(forced[(c, k)])) if forced[(c, k)] else float("nan")
            row += f"{v:12.4f}"
            if v > bestv:
                best, bestv = k, v
        log.info(row + f"   {best}")
    log.info("  A specialised block puts `fog` first on fog and `night` first on night.")

    # ------------------------------------------------------------------ render
    out_dir = ensure_dir(OUTPUT_DIR / "analysis" / f"experts_{args.run}")

    def heat(t: torch.Tensor) -> np.ndarray:
        a = t.float().mean(0).numpy()
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
        return cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    for n, (dp, cp, experts, base, target) in enumerate(render_rows, 1):
        tiles = [cv2.resize(cv2.imread(str(dp)), (256, 256)),
                 cv2.resize(cv2.imread(str(cp)), (256, 256)),
                 cv2.resize(heat(base), (256, 256)),
                 cv2.resize(heat(target), (256, 256))]
        labels = ["degraded input", "clear twin", "no expert", "target (clear)"]
        for k in kinds:
            tiles.append(cv2.resize(heat(experts[k]), (256, 256)))
            labels.append(f"expert: {k}")
        attn = getattr(getattr(blk.experts[kinds.index("night")], "attn", None),
                       "last_contrast", None) if "night" in kinds else None
        if attn is not None:
            tiles.append(cv2.resize(heat(attn[0].cpu()), (256, 256)))
            labels.append("night contrast attn")
        sheet = np.zeros((286, 256 * len(tiles), 3), np.uint8)
        for j, (t, lab) in enumerate(zip(tiles, labels)):
            sheet[30:286, j * 256:(j + 1) * 256] = t
            cv2.putText(sheet, lab, (j * 256 + 6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"{n:02d}_{Path(dp).stem}.jpg"), sheet)

    (out_dir / "gains.json").write_text(json.dumps(
        {"gain": {k: float(np.mean(v)) for k, v in gains.items()},
         "forced": {f"{c}|{k}": float(np.mean(v)) for (c, k), v in forced.items()}},
        indent=2), encoding="utf-8")
    log.info("")
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
