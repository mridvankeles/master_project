"""Inference cost at test time: parameters, FLOPs, activated FLOPs, and wall clock.

    python scripts/benchmark_speed.py --runs union3d_nomosaic_yolo11n cond3d_nomosaic_yolo11n

NO test-time augmentation anywhere: `augment=False`, single scale, no flips.
Numbers measured with a TTA path enabled are not comparable to published
single-pass figures and are not what a deployed model would do.

WHY THIS NEEDS ITS OWN SCRIPT
-----------------------------
`get_flops` traces every branch, so for a conditional model it reports the
**dense-equivalent** cost -- what the block would take if every expert ran. The
whole claim of conditional routing is that they do not. The activated cost has
to be measured, and it depends on the data, because the gate decides per image.

So this reports three different things that are all called "cost":

1. **static GFLOPs** -- every branch traced. What `model.info()` prints.
2. **activated GFLOPs** -- static, minus every expert, plus each expert weighted
   by its measured activation rate on real test images. What the design claims.
3. **wall clock** -- what actually happens on the GPU. Fewer FLOPs on a narrow
   branch does not imply less time: masking, `index_add`, and a variable batch
   per expert each cost kernel launches that a dense conv does not pay.

The gap between (2) and (3) is the point of the measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, ensure_dir  # noqa: E402

log = get_logger("benchmark_speed")


def module_gflops(mod: torch.nn.Module, shape: tuple[int, ...]) -> float:
    """GFLOPs of one submodule at a given input shape, via thop."""
    try:
        import thop
    except ImportError:
        return float("nan")
    x = torch.zeros(shape, device=next(mod.parameters()).device,
                    dtype=next(mod.parameters()).dtype)
    with torch.no_grad():
        macs, _ = thop.profile(mod, inputs=(x,), verbose=False)
    return macs * 2 / 1e9  # MACs -> FLOPs, and thop counts one image


def timed(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    """Return (mean_ms, p50_ms, p95_ms) with the GPU actually synchronised."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--condition", default="union3b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--route-images", type=int, default=300)
    ap.add_argument("--half", action="store_true", help="fp16; default is fp32")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    from src.models.moe2 import cond_moe_blocks
    from src.utils.paths import dataset_root

    dev = torch.device("cuda:0")
    dtype = torch.half if args.half else torch.float

    # Real test images: the gate decides per image, so timing a conditional
    # model on random noise measures a routing pattern that never occurs.
    img_dir = dataset_root("detect", "full") / args.condition / "images" / args.split
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.choice(len(files), min(args.route_images, len(files)),
                                          replace=False)]
    log.info("timing on %d real %s/%s images, imgsz=%d, dtype=%s, augment=False",
             len(files), args.condition, args.split, args.imgsz, dtype)

    def load(n):
        ims = []
        for p in files[:n]:
            im = cv2.imread(str(p))
            im = cv2.resize(im, (args.imgsz, args.imgsz))
            ims.append(im[:, :, ::-1].transpose(2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(np.stack(ims))).to(dev, dtype).div_(255)

    report = {}
    for run in args.runs:
        ck = OUTPUT_DIR / "runs" / run / "weights" / "best.pt"
        if not ck.exists():
            log.warning("missing %s", ck)
            continue
        yolo = YOLO(str(ck))
        net = yolo.model.to(dev).to(dtype).eval()
        for m in net.modules():
            m.training = False

        r: dict[str, float] = {}
        r["params"] = float(sum(p.numel() for p in net.parameters()))
        r["gflops_static"] = float(get_flops(net, args.imgsz) or float("nan"))

        blocks = cond_moe_blocks(net)
        # --- routing on real data, and the per-branch cost it gates -------
        if blocks:
            b = blocks[0]
            acts = []
            all_x = load(len(files))          # load once, not once per chunk
            with torch.no_grad():
                for i in range(0, len(all_x), 16):
                    net(all_x[i:i + 16])
                    if b.last_active is not None:
                        acts.append(b.last_active.float().cpu())
            del all_x
            act = torch.cat(acts) if acts else torch.zeros(1, b.n_experts)
            rates = act.mean(0)
            r["experts_per_image"] = float(act.sum(1).mean())
            r["shortcut_only"] = float((act.sum(1) == 0).float().mean())

            # Block input is P3: c1 channels at imgsz/8 square.
            side = args.imgsz // 8
            shape = (1, b.c1, side, side)
            costs = [module_gflops(e, shape) for e in b.experts]
            r["gflops_gate"] = module_gflops(b.gate, (1, b.c1))
            r["gflops_experts_all"] = float(sum(costs))
            r["gflops_experts_active"] = float(sum(c * float(rates[i])
                                                   for i, c in enumerate(costs)))
            for i, kind in enumerate(b.expert_kinds):
                r[f"rate_{kind}"] = float(rates[i])
                r[f"gflops_{kind}"] = costs[i]
            # get_flops traces every branch, so subtract what does not run.
            r["gflops_activated"] = (r["gflops_static"]
                                     - r["gflops_experts_all"]
                                     + r["gflops_experts_active"])
        else:
            r["gflops_activated"] = r["gflops_static"]

        # --- wall clock, network forward only, no pre/post, no TTA -------
        for bs in args.batches:
            x = load(bs)
            with torch.no_grad():
                mean, p50, p95 = timed(lambda: net(x), args.warmup, args.iters)
            r[f"ms_b{bs}"] = mean
            r[f"ms_b{bs}_p95"] = p95
            r[f"fps_b{bs}"] = 1000.0 * bs / mean

        # --- end to end, as a user would call it -------------------------
        paths = [str(p) for p in files[:16]]
        with torch.no_grad():
            mean_e, _, _ = timed(
                lambda: yolo.predict(paths, imgsz=args.imgsz, device=0,
                                     verbose=False, augment=False), 3, 15)
        r["ms_e2e_per_image"] = mean_e / 16
        r["fps_e2e"] = 16000.0 / mean_e

        report[run] = r
        log.info("%s: %.0f params, static %.2f GF, activated %.2f GF, "
                 "%.3f ms/img @b1, %.1f FPS @b16",
                 run, r["params"], r["gflops_static"], r["gflops_activated"],
                 r["ms_b1"], r["fps_b16"])
        del net, yolo
        torch.cuda.empty_cache()

    out = ensure_dir(OUTPUT_DIR / "analysis") / "speed_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- table
    log.info("")
    log.info("=== COST AT TEST TIME (no TTA, imgsz %d, %s) ===", args.imgsz, dtype)
    hdr = f"  {'run':30s}{'params':>10s}{'GF static':>11s}{'GF active':>11s}"
    for bs in args.batches:
        hdr += f"{'ms@b' + str(bs):>10s}{'FPS@b' + str(bs):>11s}"
    log.info(hdr)
    for run, r in report.items():
        line = (f"  {run:30s}{r['params'] / 1e6:9.3f}M{r['gflops_static']:11.2f}"
                f"{r['gflops_activated']:11.2f}")
        for bs in args.batches:
            line += f"{r[f'ms_b{bs}']:10.3f}{r[f'fps_b{bs}']:11.1f}"
        log.info(line)

    moe = {k: v for k, v in report.items() if "experts_per_image" in v}
    if moe:
        log.info("")
        log.info("=== WHERE THE CONDITIONAL COMPUTE GOES ===")
        for run, r in moe.items():
            log.info("  %s", run)
            log.info("    experts active per image   %.3f  (shortcut-only %.3f)",
                     r["experts_per_image"], r["shortcut_only"])
            for k in [k for k in r if k.startswith("rate_")]:
                kind = k[5:]
                log.info("      %-8s fires %.3f, costs %.3f GF",
                         kind, r[k], r[f"gflops_{kind}"])
            log.info("    all experts %.3f GF -> activated %.3f GF (saves %.3f)",
                     r["gflops_experts_all"], r["gflops_experts_active"],
                     r["gflops_experts_all"] - r["gflops_experts_active"])
            log.info("    gate itself %.5f GF", r["gflops_gate"])
    log.info("")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
