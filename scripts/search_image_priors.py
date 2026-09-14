"""Are CLASSICAL operators more distinct when computed on the IMAGE than on P3?

    python scripts/search_image_priors.py

WHY ASK
-------
`search_expert_priors.py` searched 36 classical operators on the P3 feature map
and found the whole space buys 0.04 of CKA: best worst-pair 0.9504 against the
current design's 0.9883, and every candidate scoring >= 0.95 against the raw
input. The conclusion recorded then was that the priors are not badly chosen,
they are in the wrong PLACE -- by P3 the backbone's BatchNorms have removed the
first- and second-order statistics that define fog and night, and a channel-min
over 256 SiLU activations is not a dark channel.

This tests that claim directly: the same FAMILIES of operator, computed on the
RGB image at full resolution where each one means what it is supposed to mean,
then average-pooled to the 80x80 grid an expert would receive.

Two numbers decide whether image-space priors are worth building:

1. **Mutual CKA.** If image-space operators are as mutually redundant as the
   feature-space ones (~0.95+), the idea adds nothing and the place was never
   the problem.
2. **Condition separation.** A prior is only useful to a *conditional* expert if
   its response differs by condition. Measured as the between-condition spread
   of each operator's mean response, in units of its within-condition spread --
   an effect size, so operators on different scales stay comparable.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import dataset_root  # noqa: E402

log = get_logger("image_priors")
EPS = 1e-6


# ------------------------------------------------------------- operators
def op_identity(bgr, gray):
    return gray


def op_dark_channel(bgr, gray):
    """He et al. dark channel: min over colour channels, then a min-filter.

    THE real one -- a minimum across R, G and B, which only exists on an image.
    Haze raises it, which is the entire basis of dark-channel dehazing.
    """
    import cv2
    m = bgr.min(axis=2)
    return cv2.erode(m, np.ones((15, 15), np.uint8))


def op_transmission(bgr, gray):
    """DCP transmission estimate t = 1 - omega * dark/A, the physical quantity."""
    dark = op_dark_channel(bgr, gray)
    a = np.percentile(bgr.reshape(-1, 3).max(axis=1), 99.9) + EPS
    return 1.0 - 0.95 * dark / a


def op_clahe(bgr, gray):
    """Contrast-limited adaptive histogram equalisation -- 'increase contrast'."""
    import cv2
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        gray.astype(np.uint8)).astype(np.float32)


def op_lcn(bgr, gray):
    """Local contrast normalisation (x-mu)/sd."""
    import cv2
    mu = cv2.blur(gray, (15, 15))
    sd = np.sqrt(np.maximum(cv2.blur(gray * gray, (15, 15)) - mu * mu, 0)) + 1.0
    return (gray - mu) / sd


def op_sobel(bgr, gray):
    """Edge magnitude -- 'highlight edges'."""
    import cv2
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def op_harris(bgr, gray):
    """Harris corner response -- 'highlight corners'."""
    import cv2
    return cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)


def op_saturation(bgr, gray):
    """HSV saturation. Haze desaturates; this is its cleanest single cue."""
    import cv2
    return cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)


def op_value(bgr, gray):
    """HSV value. Darkness IS this channel."""
    import cv2
    return cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)


def op_retinex(bgr, gray):
    """Single-scale Retinex: log(I) - log(blur(I)). The classical low-light method."""
    import cv2
    g = gray + 1.0
    return np.log(g) - np.log(cv2.GaussianBlur(g, (0, 0), 15) + EPS)


def op_laplacian(bgr, gray):
    """Laplacian -- a second-derivative sharpness / 'cleaning' cue."""
    import cv2
    return cv2.Laplacian(gray, cv2.CV_32F, ksize=3)


def op_dcp_dehazed(bgr, gray):
    """Actually invert the ASM: J = (I - A)/max(t, t0) + A. A 'cleaning filter'."""
    t = np.clip(op_transmission(bgr, gray), 0.1, 1.0)
    a = np.percentile(bgr.reshape(-1, 3).max(axis=1), 99.9)
    j = (gray - a) / t + a
    return j


OPS = {
    "identity": op_identity,
    "dark_channel": op_dark_channel,
    "transmission": op_transmission,
    "dcp_dehazed": op_dcp_dehazed,
    "clahe": op_clahe,
    "lcn": op_lcn,
    "retinex": op_retinex,
    "sobel": op_sobel,
    "harris": op_harris,
    "laplacian": op_laplacian,
    "saturation": op_saturation,
    "value": op_value,
}

SUITS = {
    "dark_channel": "fog", "transmission": "fog", "dcp_dehazed": "fog",
    "saturation": "fog", "clahe": "night", "retinex": "night", "value": "night",
    "lcn": "night", "sobel": "tiny/edges", "harris": "tiny/corners",
    "laplacian": "tiny/edges", "identity": "clear (control)",
}


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via sample Gram matrices, matching search_expert_priors.py."""
    X = X.reshape(X.shape[0], -1).astype(np.float64)
    Y = Y.reshape(Y.shape[0], -1).astype(np.float64)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / (np.sqrt((K * K).sum()) * np.sqrt((L * L).sum()) + EPS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-condition", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--grid", type=int, default=80, help="the P3 grid an expert sees")
    args = ap.parse_args()

    import cv2

    root = dataset_root("detect", "full")
    conds = ["clear", "fog2", "night"]
    rng = np.random.default_rng(0)

    maps: dict[str, list[np.ndarray]] = {k: [] for k in OPS}
    labels: list[int] = []
    for ci, cond in enumerate(conds):
        d = root / cond / "images" / "val"
        fs = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png"})
        fs = [fs[i] for i in rng.choice(len(fs), min(args.per_condition, len(fs)),
                                        replace=False)]
        for p in fs:
            bgr = cv2.resize(cv2.imread(str(p)), (args.imgsz, args.imgsz)).astype(np.float32)
            gray = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for name, fn in OPS.items():
                m = np.asarray(fn(bgr, gray), dtype=np.float32)
                # average-pool to the grid an expert would actually receive
                maps[name].append(cv2.resize(m, (args.grid, args.grid),
                                             interpolation=cv2.INTER_AREA))
            labels.append(ci)
    y = np.array(labels)
    stacked = {k: np.stack(v) for k, v in maps.items()}
    log.info("%d images (%d per condition), operators computed at %d px, pooled to %d",
             len(y), args.per_condition, args.imgsz, args.grid)

    # ---- 1. are they mutually distinct? --------------------------------
    names = list(OPS)
    C = {}
    for a, b in itertools.combinations(names, 2):
        C[(a, b)] = C[(b, a)] = cka(stacked[a], stacked[b])
    log.info("")
    log.info("=== 1. MUTUAL CKA (image space) ===")
    log.info("  feature-space reference: current design 0.9883, best of 36 = 0.9504")
    log.info("  %-14s" % "" + "".join(f"{n[:9]:>10s}" for n in names))
    for a in names:
        log.info("  %-14s" % a + "".join(
            f"{(1.0 if a == b else C[(a, b)]):10.3f}" for b in names))

    # ---- 2. does each operator respond to the CONDITION? ---------------
    log.info("")
    log.info("=== 2. CONDITION SEPARATION (effect size) ===")
    log.info("  between-condition spread of the mean response, in within-condition SDs.")
    log.info("  A prior only helps a CONDITIONAL expert if this is large.")
    log.info("  %-14s%10s%10s%10s%12s", "operator", "clear", "fog2", "night", "effect")
    eff = {}
    for n in names:
        per = stacked[n].reshape(len(y), -1).mean(1)
        mus = [per[y == c].mean() for c in range(3)]
        within = np.mean([per[y == c].std() for c in range(3)]) + EPS
        e = float(np.std(mus) / within)
        eff[n] = e
        log.info("  %-14s%10.2f%10.2f%10.2f%12.2f", n, mus[0], mus[1], mus[2], e)

    # ---- 3. the triple a designer would actually pick -------------------
    log.info("")
    log.info("=== 3. BEST TRIPLE (identity pinned as `clear`) ===")
    pool = [n for n in names if n != "identity"]
    cands = sorted((max(C[("identity", a)], C[("identity", b)], C[(a, b)]), a, b)
                   for a, b in itertools.combinations(pool, 2))
    log.info("  %-8s %-14s %-14s   suits", "worst", "expert B", "expert C")
    for w, a, b in cands[:8]:
        log.info("  %-8.4f %-14s %-14s   %s / %s", w, a, b,
                 SUITS.get(a, "?"), SUITS.get(b, "?"))

    best = cands[0][0]
    log.info("")
    log.info("  image-space best worst-pair : %.4f", best)
    log.info("  feature-space best worst-pair: 0.9504   (current design 0.9883)")
    if best < 0.80:
        log.info("  -> image space IS materially more diverse. Worth building.")
    else:
        log.info("  -> image space is no more diverse. The place was not the problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
