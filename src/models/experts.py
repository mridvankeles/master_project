"""Static expert branches — different FUNCTIONS, not just different kernel sizes.

WHY THIS FILE EXISTS
--------------------
The first MoE collapsed: inter-expert CKA 0.968-0.996, i.e. the branches learned
the same function, so which one the gate picked could not matter
(`finding-router-never-specialised.md`). The cause was in the design, not the
optimiser — the experts differed only by convolution kernel size (3x3 vs 5x5)
with identical architecture and identical zero-initialisation. Nothing in that
setup makes one branch a dehazer and another a low-light branch, so gradient
descent had no reason to produce them.

Each expert here begins with a FIXED, non-learned transform chosen for the
degradation it targets, followed by a learnable convolution. The fixed part
cannot be trained away, so two experts cannot converge to the same function no
matter how the gate behaves. That is the property the previous design lacked.

A NOTE ON PHYSICS AT THE NECK
-----------------------------
These blocks sit on feature maps, not on images, so the operations are
structural analogues of the image-space physics rather than the physics itself.
Illumination acts multiplicatively on early features much as it does on pixels,
so log-compression plus local-mean removal still suppresses it; haze acts as a
low-frequency additive veil, so low-frequency removal still targets it. The
claim being made is the weaker and defensible one: these are *distinct fixed
inductive biases aimed at distinct degradations*, and their distinctness is what
prevents collapse. Whether each is optimal for its condition is exactly what the
experiments are for.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "IlluminationInvariant",
    "TransmissionPrior",
    "IdentityPrior",
    "StaticExpert",
    "EXPERT_TYPES",
]


class IlluminationInvariant(nn.Module):
    """Removes multiplicative gain — the low-light handle.

    A change in illumination scales features by roughly a constant factor. In
    log space that becomes an additive offset, and subtracting a local mean
    removes it. What survives is local *contrast structure*, which is what
    stays constant between a daylight and a night view of the same scene.

    This is homomorphic filtering / single-scale Retinex, and it is the
    feature-level version of the thing FeatEnHancer found works — its Table 5
    shows image-level enhancement collapsing detection from 32.8 to ~7.5 mAP
    while feature-level enhancement raised it to 34.6.
    """

    def __init__(self, kernel: int = 7, eps: float = 1e-3):
        super().__init__()
        self.kernel = kernel
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Features are post-SiLU and can be negative; shift to a positive range
        # per-sample before the log rather than clamping, which would discard
        # the negative lobe entirely.
        shifted = x - x.amin(dim=(1, 2, 3), keepdim=True) + self.eps
        log_x = torch.log(shifted)
        local_mean = F.avg_pool2d(log_x, self.kernel, stride=1, padding=self.kernel // 2)
        return log_x - local_mean


class TransmissionPrior(nn.Module):
    """Isolates the low-frequency veil — the haze handle.

    The atmospheric scattering model makes haze a spatially smooth additive
    term plus a contrast attenuation. Estimating that smooth component with a
    wide average pool and subtracting it leaves the scene structure, which is
    the same reasoning behind dark-channel dehazing but applied to features.

    The channel-min branch is the dark-channel analogue: haze raises the
    per-location minimum across channels, so that minimum is informative about
    how much veil is present, and it is concatenated back as an extra cue
    rather than used to divide (division by a small transmission estimate is
    numerically hostile inside a network).
    """

    def __init__(self, kernel: int = 15):
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        veil = F.avg_pool2d(x, self.kernel, stride=1, padding=self.kernel // 2)
        detail = x - veil
        dark = x.amin(dim=1, keepdim=True).expand_as(veil[:, :1]).contiguous()
        return torch.cat([detail, dark], dim=1)


class IdentityPrior(nn.Module):
    """No transform — the clear-weather branch, and the control.

    Present so "static heterogeneity helps" can be ablated against a branch
    that has no prior at all.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


EXPERT_TYPES = {
    "clear": IdentityPrior,
    "fog": TransmissionPrior,
    "night": IlluminationInvariant,
    "plain": IdentityPrior,
}

# How many channels each prior emits, given c_in.
_CHANNEL_DELTA = {"fog": 1}


class StaticExpert(nn.Module):
    """A fixed prior followed by a learnable convolution.

    STATIC AND DYNAMIC TOGETHER, which is the requested design: the prior fixes
    *what the branch is looking at* and cannot be trained away, while the
    convolution learns *what to do about it*. Two experts with different priors
    therefore cannot converge to the same function, but both remain trainable.
    """

    def __init__(self, c1: int, c2: int, kind: str = "plain", bottleneck: float = 0.5):
        super().__init__()
        if kind not in EXPERT_TYPES:
            raise ValueError(f"unknown expert kind {kind!r}; have {sorted(EXPERT_TYPES)}")
        self.kind = kind
        self.prior = EXPERT_TYPES[kind]()
        c_mid = max(8, int(c2 * bottleneck))
        c_prior = c1 + _CHANNEL_DELTA.get(kind, 0)
        # Kernel follows the prior's scale: the haze branch reasons over a wide
        # neighbourhood, the illumination branch over a local one.
        k = 5 if kind == "fog" else 3
        self.body = nn.Sequential(
            nn.Conv2d(c_prior, c_mid, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.prior(x))

    def zero_init_output(self) -> None:
        """Make this branch a no-op at step 0 without making it identical to others.

        Only the final 1x1 is zeroed. The priors still differ, so the branches
        diverge the moment the output weights leave zero — unlike the previous
        design, where zero-init made every expert the same function AND the same
        gradient, which is what let them collapse together.
        """
        nn.init.zeros_(self.body[-1].weight)
