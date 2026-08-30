"""Search the expert-prior design space WITHOUT training anything.

    python scripts/search_expert_priors.py --run cond3d_nomosaic_yolo11n

WHY THIS IS NOT A TRAINING SEARCH
---------------------------------
`probe_expert_priors.py` showed the learned inter-expert CKA is inherited almost
exactly from the priors: 0.946/0.932/0.983 before training, 0.828/0.821/0.977
after, same ordering. So prior diversity can be scored **before a single
gradient step**, on real features, deterministically -- no seed noise, no
schedule, seconds per candidate instead of half an hour.

That matters because the alternative is unaffordable. Every accuracy difference
in this project so far is inside the ~2-point single-seed noise floor, so a
random search ranked on mAP would mostly be fitting noise, at ~20 minutes a
sample. Here the effect size is a factor of three (CKA 0.98 vs 0.30) and the
measurement is free.

WHAT IS SEARCHED
----------------
Operator family x kernel width x dilation, over the real P3 feature maps the
block actually receives. `clear` is pinned to identity -- it is the control
branch by design -- and the fog/night pair is searched.

SCORING
-------
Objective is the WORST pairwise CKA in the triple, minimised. A triple is only
as diverse as its most similar pair, and fog-vs-night has been the offender
every time. Physical suitability is not something CKA can judge, so candidates
are annotated, not auto-selected: the shortlist is for a human to choose from.

MEMORY
------
CKA needs only the N x N sample Gram matrix, so each operator's output is
reduced to N x N and discarded. The feature-space form would need a
1.6M x 1.6M matrix.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import OUTPUT_DIR, ensure_dir  # noqa: E402

log = get_logger("search_priors")
EPS = 1e-3


# --------------------------------------------------------------- operators
def op_identity(x, k, d):
    return x


def _mu(x, k, d):
    if d == 1:
        return F.avg_pool2d(x, k, 1, k // 2)
    # dilated mean: average-pool a strided view, then resize back
    pad = (k // 2) * d
    w = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    w = w.expand(x.shape[1], 1, k, k)
    return F.conv2d(F.pad(x, (pad,) * 4, mode="replicate"), w,
                    groups=x.shape[1], dilation=d)


def op_sub_mu(x, k, d):
    """Subtractive high-pass. Removes an ADDITIVE local offset -- ASM airlight."""
    return x - _mu(x, k, d)


def op_div_mu(x, k, d):
    """Divisive. Removes a MULTIPLICATIVE local gain -- illumination."""
    return x / (_mu(x, k, d).abs() + EPS)


def op_contrast(x, k, d):
    """(x-mu)/sd. Removes offset AND gain -- so it subsumes both, which is a risk."""
    m = _mu(x, k, d)
    v = (_mu(x * x, k, d) - m * m).clamp_min(0)
    return (x - m) / (v.sqrt() + EPS)


def op_log_sub(x, k, d):
    """The current night prior: log then subtract local mean."""
    s = x - x.amin(dim=(1, 2, 3), keepdim=True) + EPS
    lg = torch.log(s)
    return lg - _mu(lg, k, d)


def op_range(x, k, d):
    """Local max minus local min: a contrast magnitude, not a residual."""
    hi = F.max_pool2d(x, k, 1, k // 2)
    lo = -F.max_pool2d(-x, k, 1, k // 2)
    return hi - lo


def op_dark(x, k, d):
    """Dark-channel analogue: per-location min across channels, broadcast back."""
    dark = x.amin(dim=1, keepdim=True)
    return x - dark


def op_sobel(x, k, d):
    """Gradient magnitude -- structure only, invariant to any local offset."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    c = x.shape[1]
    gx = F.conv2d(x, kx.expand(c, 1, 3, 3), padding=1, groups=c)
    gy = F.conv2d(x, ky.expand(c, 1, 3, 3), padding=1, groups=c)
    return (gx * gx + gy * gy + EPS).sqrt()


def op_phase(x, k, d):
    """Phase-only reconstruction: discard the Fourier amplitude entirely.

    Haze and illumination perturb the amplitude spectrum far more than the
    phase, which is the premise behind Fourier-domain domain-generalisation
    (FDA, FACT). A genuinely different family from any spatial filter.
    """
    f = torch.fft.rfft2(x.float())
    return torch.fft.irfft2(f / (f.abs() + EPS), s=x.shape[-2:]).to(x.dtype)


def op_lowfreq_drop(x, k, d):
    """Zero the lowest Fourier amplitudes -- a global, not local, high-pass."""
    f = torch.fft.rfft2(x.float())
    h, w = f.shape[-2:]
    r = max(1, min(h, w) // (2 * k))
    f[..., :r, :r] = 0
    f[..., -r:, :r] = 0
    return torch.fft.irfft2(f, s=x.shape[-2:]).to(x.dtype)


OPS = {
    "identity": (op_identity, [0], [1]),
    "sub_mu": (op_sub_mu, [3, 7, 11, 15, 21], [1, 2]),
    "div_mu": (op_div_mu, [3, 7, 11, 15, 21], [1, 2]),
    "contrast": (op_contrast, [3, 7, 15], [1]),
    "log_sub": (op_log_sub, [3, 7, 15], [1]),
    "range": (op_range, [3, 7, 15], [1]),
    "dark": (op_dark, [0], [1]),
    "sobel": (op_sobel, [0], [1]),
    "phase": (op_phase, [0], [1]),
    "lowfreq_drop": (op_lowfreq_drop, [2, 4, 8], [1]),
}

# What each family is physically motivated FOR. CKA cannot judge this.
SUITS = {
    "identity": "clear (control)",
    "sub_mu": "fog - removes the ASM additive airlight A(1-t)",
    "div_mu": "night - cancels a multiplicative illumination gain",
    "contrast": "both - removes offset AND gain, so it can subsume its neighbour",
    "log_sub": "night (current) - degenerates to a linear high-pass",
    "range": "fog - contrast magnitude, which haze attenuates",
    "dark": "fog - dark-channel analogue",
    "sobel": "structure - degradation-agnostic",
    "phase": "both - amplitude-invariant, a different family entirely",
    "lowfreq_drop": "fog - the veil is low frequency, globally",
}


def gram(x: torch.Tensor) -> torch.Tensor:
    """Centred sample Gram matrix, float64, N x N."""
    v = x.reshape(x.shape[0], -1).double()
    v = v - v.mean(0)
    return v @ v.T


def cka_from_grams(K: torch.Tensor, L: torch.Tensor) -> float:
    return float((K * L).sum() / ((K * K).sum().sqrt() * (L * L).sum().sqrt()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cond3d_nomosaic_yolo11n")
    ap.add_argument("--condition", default="union3b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--images", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--top", type=int, default=15)
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
    x = cap["x"].float()
    log.info("real P3 features %s from %s/%s", tuple(x.shape), args.condition, args.split)

    # --- one Gram matrix per candidate; outputs are never kept ------------
    grams: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, (fn, ks, ds) in OPS.items():
            for k in ks:
                for dl in ds:
                    if dl > 1 and k == 0:
                        continue
                    tag = name if k == 0 else f"{name}_k{k}" + (f"_d{dl}" if dl > 1 else "")
                    try:
                        grams[tag] = gram(fn(x, k, dl).cpu())
                    except Exception as e:  # noqa: BLE001 - a bad candidate is not fatal
                        log.warning("  %s failed: %s", tag, e)
    log.info("scored %d candidate priors", len(grams))

    tags = list(grams)
    C = {(a, b): cka_from_grams(grams[a], grams[b])
         for a, b in itertools.combinations(tags, 2)}
    C.update({(b, a): v for (a, b), v in list(C.items())})

    def pair(a, b):
        return 1.0 if a == b else C[(a, b)]

    # --- 1. every candidate against the current three --------------------
    cur = ["identity", "sub_mu_k15", "log_sub_k7"]
    log.info("")
    log.info("=== CURRENT DESIGN ===")
    for a, b in itertools.combinations(cur, 2):
        log.info("  %-14s vs %-14s  CKA %.4f", a, b, pair(a, b))
    log.info("  worst pair: %.4f", max(pair(a, b) for a, b in itertools.combinations(cur, 2)))

    # --- 2. search fog/night with clear pinned to identity ---------------
    pool = [t for t in tags if t != "identity"]
    cands = []
    for e1, e2 in itertools.combinations(pool, 2):
        worst = max(pair("identity", e1), pair("identity", e2), pair(e1, e2))
        cands.append((worst, e1, e2))
    cands.sort()

    log.info("")
    log.info("=== BEST TRIPLES (clear = identity, searching fog/night) ===")
    log.info("  %-8s %-16s %-16s  %-8s %-8s %-8s", "worst", "expert B", "expert C",
             "id-B", "id-C", "B-C")
    for worst, e1, e2 in cands[: args.top]:
        log.info("  %-8.4f %-16s %-16s  %-8.4f %-8.4f %-8.4f", worst, e1, e2,
                 pair("identity", e1), pair("identity", e2), pair(e1, e2))

    log.info("")
    log.info("=== PHYSICALLY MOTIVATED SHORTLIST ===")
    log.info("  (fog wants an ADDITIVE-offset remover, night a MULTIPLICATIVE-gain remover)")
    fog_ok = ("sub_mu", "range", "dark", "lowfreq_drop", "phase")
    night_ok = ("div_mu", "log_sub", "contrast", "phase")
    shown = 0
    for worst, e1, e2 in cands:
        for f, n in ((e1, e2), (e2, e1)):
            if f.split("_k")[0].rsplit("_d", 1)[0].startswith(fog_ok) and \
               n.split("_k")[0].rsplit("_d", 1)[0].startswith(night_ok):
                log.info("  worst %.4f   fog=%-16s night=%-16s", worst, f, n)
                log.info("      fog  : %s", SUITS[f.split("_k")[0].rsplit("_d", 1)[0]])
                log.info("      night: %s", SUITS[n.split("_k")[0].rsplit("_d", 1)[0]])
                shown += 1
                break
        if shown >= 6:
            break

    out = ensure_dir(OUTPUT_DIR / "analysis") / "prior_search.json"
    out.write_text(json.dumps(
        {"current": {f"{a}|{b}": pair(a, b) for a, b in itertools.combinations(cur, 2)},
         "top": [{"worst": w, "b": a, "c": c} for w, a, c in cands[:50]]},
        indent=2), encoding="utf-8")
    log.info("")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
