"""Are the 'heterogeneous static priors' actually different operators?

    python scripts/probe_expert_priors.py --run cond3d_nomosaic_yolo11n

THE CLAIM BEING TESTED
----------------------
`src/models/experts.py` argues that fixed, non-learned priors keep the experts
apart: "the fixed part cannot be trained away, so two experts cannot converge to
the same function". Inter-expert CKA nonetheless reached 0.956-0.975 in
`cond3d_nomosaic`, and rose as routing improved.

Two explanations fit that: either the priors diverge and training pulls the
branches back together (a late-training collapse), or the priors were never
different to begin with. They call for opposite fixes -- a repulsion term versus
a redesign -- so guessing is expensive.

This measures the priors DIRECTLY, on the real P3 features the block receives,
with no learned weights involved. If prior-to-prior CKA already matches
expert-to-expert CKA, nothing collapsed: the branches started together.

A CAVEAT ON THE METRIC
----------------------
Linear CKA is generous to linear maps: any full-rank linear operator largely
preserves sample-space similarity structure, so two different linear filters
score high. Both `TransmissionPrior` (x - avgpool(x)) and `IlluminationInvariant`
(log-domain high-pass, which is near-affine on positive features) are in that
family. Cosine similarity between the same tensors is near zero. The two metrics
disagree, and this script prints both -- which measure is right is itself a
finding, since the project has been steering on CKA.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR  # noqa: E402

log = get_logger("probe_priors")


def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA via sample-space Gram matrices.

    tr(KL) / sqrt(tr(KK) tr(LL)) with K = XX^T. Computed this way because the
    feature-space form needs a D x D matrix, and D = 256*80*80 here -- the
    direct version asks for 21 TB.
    """
    X = X.reshape(X.shape[0], -1).double(); Y = Y.reshape(Y.shape[0], -1).double()
    X = X - X.mean(0); Y = Y - Y.mean(0)
    K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / ((K * K).sum().sqrt() * (L * L).sum().sqrt()))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(a.shape[0], -1),
                                     b.reshape(b.shape[0], -1), dim=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cond3d_nomosaic_yolo11n")
    ap.add_argument("--condition", default="union3b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--images", type=int, default=48)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    from src.models.moe2 import cond_moe_blocks
    from src.utils.paths import dataset_root

    net = YOLO(str(OUTPUT_DIR / "runs" / args.run / "weights" / "best.pt")).model.cuda().eval()
    blk = cond_moe_blocks(net)[0]

    cap: dict[str, torch.Tensor] = {}
    blk.register_forward_pre_hook(lambda m, inp: cap.__setitem__("x", inp[0].detach()))

    d = dataset_root("detect", "full") / args.condition / "images" / args.split
    fs = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    rng = np.random.default_rng(0)
    fs = [fs[i] for i in rng.choice(len(fs), min(args.images, len(fs)), replace=False)]
    ims = np.stack([cv2.resize(cv2.imread(str(p)), (args.imgsz, args.imgsz))[:, :, ::-1]
                    .transpose(2, 0, 1) for p in fs])
    with torch.no_grad():
        net(torch.from_numpy(np.ascontiguousarray(ims)).cuda().float().div_(255))
    x = cap["x"].float().cpu()
    log.info("%s: real P3 features into the block %s", args.run, tuple(x.shape))

    kinds = list(blk.expert_kinds)
    priors = {k: e.prior(x)[:, : blk.c1] for k, e in zip(kinds, blk.experts)}
    with torch.no_grad():
        outs = {k: e(x.cuda()).cpu() for k, e in zip(kinds, blk.experts)}

    log.info("")
    log.info("=== 1. THE FIXED PRIORS, no learned weights involved ===")
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            a, b = priors[kinds[i]], priors[kinds[j]]
            log.info("  %-6s vs %-6s   CKA %.4f   cosine %+.4f",
                     kinds[i], kinds[j], cka(a, b), cosine(a, b))
    log.info("  %-6s vs %-6s   CKA %.4f   (a prior against its own raw input)",
             "input", "fog", cka(x, priors["fog"]) if "fog" in priors else float("nan"))

    log.info("")
    log.info("=== 2. THE LEARNED EXPERT OUTPUTS on the same features ===")
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            a, b = outs[kinds[i]], outs[kinds[j]]
            log.info("  %-6s vs %-6s   CKA %.4f   cosine %+.4f",
                     kinds[i], kinds[j], cka(a, b), cosine(a, b))

    log.info("")
    log.info("=== 3. WHAT A GENUINELY DIFFERENT OPERATOR FAMILY LOOKS LIKE ===")
    mu = F.avg_pool2d(x, 7, 1, 3)
    var = F.avg_pool2d(x * x, 7, 1, 3) - mu * mu
    alts = {
        "divisive  x/mu": x / (mu + 1e-3),
        "contrast (x-mu)/sd": (x - mu) / ((var + 1e-6).sqrt() + 1e-3),
        "highpass x-mu": x - mu,
    }
    for name, v in alts.items():
        row = "  %-20s" % name
        for k in kinds:
            row += f"  vs {k}: {cka(v, priors[k]):.4f}"
        log.info(row)
    log.info("")
    log.info("  A prior that scores ~1.0 against the others adds no inductive bias")
    log.info("  the learnable conv could not have reached on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
