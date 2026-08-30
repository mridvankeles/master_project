"""Heterogeneous expert ARCHITECTURES — design 3.

WHY A THIRD DESIGN
------------------
Design 2's experts were the same architecture behind different fixed prefixes.
`scripts/expert_intervention.py` measured what that produced: switching **every
expert off** changes mAP by 0.001 on every condition, and forcing the wrong
expert costs nothing. The branches were inert.

Two causes were measured, and both are addressed here:

1. **Gradient starvation.** Expert output RMS is 10–21% of the always-on path,
   and is then multiplied by the gate probability. `RichProj` is fed *into* every
   expert as `ctx`, so each branch sits one 1x1 away from the block output
   instead of behind a bottleneck.
2. **No architectural difference.** A search over 36 fixed priors
   (`scripts/search_expert_priors.py`) showed the whole prior space buys 0.04 of
   CKA, because every candidate is a near-linear filter of the same input. So
   the branches now differ in *architecture*, shaped by what each degradation
   physically does:

   | expert | shape | why |
   |---|---|---|
   | clear | plain 3x3 -> 1x1 | nothing is wrong with the image; do ordinary work |
   | fog | encoder–decoder + global vector | airlight is scene-wide; needs a receptive field far beyond 5x5 |
   | night | local contrast attention | darkness compresses contrast rather than destroying it; find where it survives and amplify |

Whether each branch learned its job is not left to argument: `fog` and `night`
are additionally supervised against their *clear twin's* features
(`src/models/paired.py`), and `scripts/inspect_experts.py` reads the answer out.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import IlluminationInvariant, TransmissionPrior

__all__ = ["ContrastAttention", "ClearExpert", "DehazeExpert", "LowLightExpert",
           "RichProj", "EXPERT_ARCH", "ExpertBase"]


class ContrastAttention(nn.Module):
    """Spatial gate proportional to LOCAL CONTRAST of the incoming features.

    Low light compresses dynamic range: the structure is still present, it is
    just small. Local standard deviation is where it survives, so this measures
    it, normalises per sample — so a globally dark image is not globally
    down-weighted, which would defeat the purpose — and amplifies accordingly.

    `gamma` is learnable and starts at 1.0, so the branch begins as a plain
    contrast boost and can learn its way out if that turns out to be wrong.
    """

    def __init__(self, kernel: int = 5, eps: float = 1e-4):
        super().__init__()
        self.kernel, self.eps = kernel, eps
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, feats: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        k, p = self.kernel, self.kernel // 2
        mu = F.avg_pool2d(ref, k, 1, p)
        var = (F.avg_pool2d(ref * ref, k, 1, p) - mu * mu).clamp_min(0)
        sd = var.sqrt().mean(dim=1, keepdim=True)                   # (B,1,H,W)
        lo = sd.amin(dim=(2, 3), keepdim=True)
        hi = sd.amax(dim=(2, 3), keepdim=True)
        contrast = (sd - lo) / (hi - lo + self.eps)                 # per sample, [0,1]
        self.last_contrast = contrast.detach()                      # for inspection
        return feats * (1.0 + self.gamma * contrast)


def _cbs(c_in: int, c_out: int, k: int = 3, s: int = 1, d: int = 1) -> nn.Sequential:
    p = ((k - 1) // 2) * d
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, dilation=d, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True),
    )


class ExpertBase(nn.Module):
    """Contract: `(x, ctx) -> c2 channels at x's resolution`."""

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def zero_init_output(self) -> None:
        """Zero only the final conv, so the branch starts as a no-op.

        Kept for continuity with design 2, but note the cost it carries: with an
        always-on shared branch that is NOT zeroed, the shared path satisfies the
        detection loss on step 1 and no pressure ever builds for the experts to
        grow. That is half of why they ended up inert. `scale_init` is the
        alternative, and is what the design-3 configs use.
        """
        for m in reversed(list(self.modules())):
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                return

    def scale_init(self, gain: float = 0.1) -> None:
        """Start small but NOT at zero.

        A branch initialised at exactly zero has zero output and receives
        gradient only through its (zero) output weight, so it climbs out slowly
        while a fully-functional bypass is already minimising the loss. Starting
        at a small non-zero gain keeps the block near-identity at step 0 while
        leaving the branch a real gradient from the first step.
        """
        for m in reversed(list(self.modules())):
            if isinstance(m, nn.Conv2d):
                with torch.no_grad():
                    m.weight.mul_(gain)
                return


class ClearExpert(ExpertBase):
    """Plain 3x3 -> 1x1. The control branch, and the cheapest of the three."""

    def __init__(self, c1: int, c2: int, c_mid: int, use_ctx: bool = True):
        super().__init__()
        self.use_ctx = use_ctx
        self.stem = _cbs(c1, c_mid, 3)
        self.out = nn.Conv2d(c_mid + (c2 if use_ctx else 0), c2, 1, bias=False)

    def forward(self, x, ctx=None):
        h = self.stem(x)
        if self.use_ctx and ctx is not None:
            h = torch.cat([h, ctx], 1)
        return self.out(h)


class DehazeExpert(ExpertBase):
    """Encoder–decoder for haze: wide receptive field plus a global airlight cue.

    The atmospheric scattering model is `I = J*t + A*(1-t)`. `A` is a scene-wide
    constant and `t` varies smoothly, so recovering `J` needs information from
    far outside any local window — a 3x3 conv at stride 8 sees 24 input pixels.
    This branch sees the whole image, three ways: a stride-2 stage, a dilated
    bottleneck, and an explicit global-average vector that plays the part of the
    airlight estimate and modulates the bottleneck FiLM-style.

        prior (veil removal + dark-channel cue)
          -> enc  5x5                  80x80    kept for the skip
          -> down 3x3 stride 2         40x40
          -> bottleneck 3x3 dilation 2 40x40    scaled/shifted by a global vector
          -> bilinear up + 3x3         80x80
          -> + enc                     skip, so the branch can sit near identity
          -> concat ctx -> 1x1 -> c2
    """

    def __init__(self, c1: int, c2: int, c_mid: int, use_ctx: bool = True):
        super().__init__()
        self.use_ctx = use_ctx
        self.prior = TransmissionPrior()
        cp = c1 + 1                                   # prior concatenates a dark-channel row
        self.enc = _cbs(cp, c_mid, 5)
        self.down = _cbs(c_mid, c_mid * 2, 3, s=2)
        self.bott = _cbs(c_mid * 2, c_mid * 2, 3, d=2)
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(cp, c_mid * 4, 1), nn.SiLU(inplace=True))
        self.up = _cbs(c_mid * 2, c_mid, 3)
        self.out = nn.Conv2d(c_mid + (c2 if use_ctx else 0), c2, 1, bias=False)

    def forward(self, x, ctx=None):
        p = self.prior(x)
        e = self.enc(p)
        d = self.bott(self.down(e))
        scale, shift = self.glob(p).chunk(2, dim=1)
        d = d * (1.0 + scale) + shift
        u = self.up(F.interpolate(d, size=e.shape[-2:], mode="bilinear", align_corners=False))
        h = u + e
        if self.use_ctx and ctx is not None:
            h = torch.cat([h, ctx], 1)
        return self.out(h)


class LowLightExpert(ExpertBase):
    """Illumination-invariant features, then amplification where contrast survives.

    Two stages doing two different jobs. `IlluminationInvariant` removes the
    multiplicative gain — the thing that separates a night view of a scene from
    a day view of it. `ContrastAttention` then boosts the locations that still
    carry contrast, which in a dark image are the few high-signal regions and
    exactly what the detector needs.

    Deliberately LOCAL (3x3, 5x5). Unlike haze, darkness needs no scene-wide
    reasoning, and giving this branch a wide receptive field too would push it
    straight back towards the fog branch — which is how design 2 failed.
    """

    def __init__(self, c1: int, c2: int, c_mid: int, use_ctx: bool = True):
        super().__init__()
        self.use_ctx = use_ctx
        self.prior = IlluminationInvariant()
        self.stem = _cbs(c1, c_mid, 3)
        self.attn = ContrastAttention(kernel=5)
        self.refine = _cbs(c_mid, c_mid, 3)
        self.out = nn.Conv2d(c_mid + (c2 if use_ctx else 0), c2, 1, bias=False)

    def forward(self, x, ctx=None):
        h = self.stem(self.prior(x))
        # Contrast is measured on the RAW features, not the log-normalised ones:
        # the prior has already flattened exactly the statistic we want to find.
        h = self.refine(self.attn(h, x))
        if self.use_ctx and ctx is not None:
            h = torch.cat([h, ctx], 1)
        return self.out(h)


class RichProj(nn.Module):
    """The always-on projection, widened into a real context branch.

    Design 2 used a bare 1x1. The block's input at neck index 16 is
    `Concat[upsampled P4-head, backbone P3]`, so half its channels are already
    the deeper, coarser, semantically richer top-down path — a 1x1 barely reads
    it.

    The output is used TWICE: added to the block output as before, and passed
    into every expert as `ctx`. The second use is the point. It puts each expert
    one 1x1 away from the block output instead of behind a bottleneck, which is
    the shortest available fix for the gradient starvation the intervention
    measured.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.body = nn.Sequential(_cbs(c1, c2, 3), nn.Conv2d(c2, c2, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


EXPERT_ARCH = {"clear": ClearExpert, "fog": DehazeExpert,
               "night": LowLightExpert, "plain": ClearExpert}
