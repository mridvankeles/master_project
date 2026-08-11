"""Measure inference cost for every trained model, on an idle GPU.

    python scripts/benchmark_speed.py --out outputs/benchmark/speed.json

WHY THIS IS SEPARATE FROM eval.py
---------------------------------
`eval.py` reports the speed Ultralytics measured while it was validating, which
is correct but only trustworthy if nothing else was using the GPU at the time.
Accuracy evaluations can run alongside training; latency measurements cannot.
This script exists so every speed number in the benchmark table comes from the
same idle machine under the same protocol.

WHAT IS MEASURED, AND WHY EACH PART
-----------------------------------
`04-method-open-questions.md` § Inference-speed measurement protocol requires
throughput AND latency at a stated batch size and resolution, alongside FLOPs
and parameter count, on a named GPU, with the router cost included. So:

* **params / GFLOPs** -- capacity and dense-equivalent compute.
* **batch 1 latency** -- the deployment-shaped number.
* **batch 16 throughput** -- the number that hides dispatch overhead, reported
  so the gap to batch 1 is visible rather than chosen.
* **mixed vs sorted batches (MoE only)** -- §3.4 warns the per-expert loop only
  saves wall-clock when a batch routes uniformly. A condition-sorted batch and a
  mixed batch give different numbers and the honest one is mixed, so both are
  measured and reported side by side.

The router's cost is inside the model's forward pass and is therefore included
in every number here by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.models.moe import moe_blocks, routing_stats  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, ensure_dir  # noqa: E402
from src.utils.tracking import model_complexity  # noqa: E402

log = get_logger("benchmark_speed")

WARMUP = 20
ITERS = 60


@torch.no_grad()
def _time_forward(model, x: torch.Tensor, iters: int = ITERS) -> float:
    """Median ms per forward. Median, not mean: one scheduler hiccup should not
    define the number, and these distributions have a long right tail."""
    for _ in range(WARMUP):
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def benchmark(checkpoint: Path, device: str, imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    net = model.model.to(device).eval()
    if device.startswith("cuda"):
        net = net.float()

    out: dict = {"checkpoint": str(checkpoint), "imgsz": imgsz}
    out.update(model_complexity(model, imgsz=imgsz))

    for batch in (1, 16):
        x = torch.randn(batch, 3, imgsz, imgsz, device=device)
        ms = _time_forward(net, x)
        out[f"latency_b{batch}_ms"] = ms
        out[f"fps_b{batch}"] = 1000.0 * batch / ms
        out[f"ms_per_image_b{batch}"] = ms / batch

    blocks = moe_blocks(net)
    out["n_moe_blocks"] = len(blocks)
    if blocks:
        # Batch composition matters once routing is sparse (06 §3.4). A batch of
        # identical inputs routes uniformly (best case for the per-expert loop);
        # a batch of varied inputs generally does not.
        uniform = torch.randn(1, 3, imgsz, imgsz, device=device).repeat(16, 1, 1, 1)
        ms_uniform = _time_forward(net, uniform)
        out["latency_b16_uniform_ms"] = ms_uniform
        out["fps_b16_uniform"] = 1000.0 * 16 / ms_uniform
        with torch.no_grad():
            net(torch.randn(16, 3, imgsz, imgsz, device=device))
        out["routing_on_random_batch"] = routing_stats(net)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=None, help="run names under outputs/runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs_dir = OUTPUT_DIR / "runs"
    names = args.runs or sorted(
        p.name for p in runs_dir.iterdir() if (p / "weights" / "best.pt").exists()
    )
    device = args.device if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    log.info("device: %s   imgsz: %s", gpu, args.imgsz)

    results = []
    for name in names:
        ckpt = runs_dir / name / "weights" / "best.pt"
        if not ckpt.exists():
            log.warning("skip %s (no best.pt)", name)
            continue
        row = benchmark(ckpt, device, args.imgsz)
        row["run"] = name
        row["gpu"] = gpu
        results.append(row)
        log.info(
            "%-28s %9s params  %5.2f GFLOPs  b1 %6.2f ms (%6.1f fps)  b16 %5.1f ms/img",
            name, f"{int(row.get('n_parameters', 0)):,}", row.get("gflops", 0),
            row["latency_b1_ms"], row["fps_b1"], row["ms_per_image_b16"],
        )
        if row["n_moe_blocks"]:
            log.info("%-28s mixed b16 %.2f ms vs uniform-route b16 %.2f ms  routing=%s",
                     "", row["latency_b16_ms"], row["latency_b16_uniform_ms"],
                     row["routing_on_random_batch"])

    dst = Path(args.out) if args.out else ensure_dir(OUTPUT_DIR / "benchmark") / "speed.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"gpu": gpu, "imgsz": args.imgsz, "results": results}, indent=2),
                   encoding="utf-8")
    log.info("wrote %s", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
