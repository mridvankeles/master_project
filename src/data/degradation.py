"""Synthetic low-illumination degradation for the third condition.

Fog is supplied ready-made by the Hazy-DIOR release. Night must be synthesised,
and — this is the point, not a detail — synthesised on **the same DIOR images**.
`03-datasets.md` Risk 2: if fog came from one corpus and darkness from another,
a router could separate the conditions on platform statistics alone (ground
sampling distance, object density, viewing geometry) and never learn anything
about illumination. Same source images makes that impossible.

THE MODEL
---------
`03-datasets.md` is explicit that this must be a citable physical model rather
than a gamma hack, because an examiner who knows imaging will notice. So the
pipeline runs backwards through the camera and forwards again:

    sRGB  --inverse gamma-->  linear radiance
          --x exposure----->  fewer photons
          --Poisson--------->  shot noise (signal dependent)
          --+Gaussian------->  read noise (signal independent)
          --quantise-------->  sensor bit depth
          --forward gamma-->  sRGB

Two properties fall out of doing it this way rather than scaling sRGB directly:

* **Noise grows as the signal shrinks.** Photon shot noise is Poisson, so its
  relative magnitude scales as 1/sqrt(signal). Darkening in sRGB space keeps the
  noise floor fixed and produces an image that is merely dim — visually similar,
  but a far easier detection problem, and wrong in a way that would flatter the
  results.
* **Colour behaves correctly.** Per-channel gains applied in linear space model
  the white-balance shift of low light; applied in sRGB they distort hue.

PARAMETERS ARE RANDOMISED PER IMAGE
-----------------------------------
`04-method-open-questions.md` calls this "the single highest-value line of code
in the project": a router trained on one fixed simulator setting learns the
setting, not the degradation. Every image samples its own exposure, read-noise
level, ISO gain and white-balance shift, deterministically seeded by image id so
the corpus is reproducible without storing the parameters.

This also fixes a known weakness of the fog condition, which ships only three
fixed severities (`results-full-scale-and-moe.md` § limitations).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["NightParams", "sample_night_params", "apply_night", "SRGB_GAMMA"]

SRGB_GAMMA = 2.2
# 12-bit sensor. Quantising here rather than at 8 bits keeps the shot noise from
# being swamped by rounding before the forward pipeline runs.
SENSOR_MAX = 4095.0


@dataclass(frozen=True)
class NightParams:
    """One image's illumination parameters. Reproducible from the image id."""

    exposure: float          # linear scale factor on radiance (<1 darkens)
    read_noise: float        # Gaussian sigma, in sensor DN
    iso_gain: float          # analogue gain applied after the sensor
    wb_gain_r: float         # linear per-channel gains: low light is blue-shifted
    wb_gain_b: float
    gamma: float = SRGB_GAMMA


def sample_night_params(image_id: str, seed: int = 0) -> NightParams:
    """Deterministic per-image parameters.

    Keyed on the image id rather than on a running counter so that a rebuild, a
    reordering, or building only part of the corpus all produce the same
    degradation for the same image.
    """
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))

    # Exposure spans roughly civil twilight to night. Log-uniform because
    # perceived brightness is logarithmic, so uniform sampling would crowd the
    # bright end and leave the hard cases under-represented.
    exposure = float(np.exp(rng.uniform(np.log(0.02), np.log(0.25))))
    return NightParams(
        exposure=exposure,
        read_noise=float(rng.uniform(1.0, 6.0)),
        iso_gain=float(rng.uniform(1.0, 4.0)),
        wb_gain_r=float(rng.uniform(0.85, 1.05)),
        wb_gain_b=float(rng.uniform(1.0, 1.35)),
    )


def apply_night(img: np.ndarray, params: NightParams) -> np.ndarray:
    """Apply the low-light pipeline to an 8-bit RGB image.

    Args:
        img: uint8 array, HxWx3, RGB order.
        params: from `sample_night_params`.

    Returns:
        uint8 array of the same shape.
    """
    if img.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {img.dtype}")

    # --- inverse ISP: sRGB -> linear radiance -------------------------
    linear = (img.astype(np.float32) / 255.0) ** params.gamma

    # --- fewer photons, and a white-balance shift ---------------------
    linear = linear * params.exposure
    linear[..., 0] *= params.wb_gain_r
    linear[..., 2] *= params.wb_gain_b

    # --- sensor: shot noise, then read noise --------------------------
    # ISO gain models a SHORTER shutter amplified afterwards, not extra light.
    # So the photon count is divided by the gain before Poisson sampling and
    # multiplied back after: output brightness is set by `exposure` alone, while
    # `iso_gain` controls how noisy that brightness is. Applying the gain only
    # on the way out (the obvious mistake) brightens the image back up and can
    # cancel the darkening entirely — measured: 11% darker at the p95 of the
    # parameter range, which is not night by any definition.
    electrons = np.clip(linear, 0, None) * SENSOR_MAX / params.iso_gain
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(str(asdict(params)).encode()).digest()[:8], "big")
    )
    noisy = rng.poisson(electrons).astype(np.float32)
    noisy += rng.normal(0.0, params.read_noise, size=noisy.shape).astype(np.float32)

    # --- analogue gain, quantise, back to sRGB ------------------------
    noisy = np.clip(noisy * params.iso_gain, 0, SENSOR_MAX)
    linear_out = np.round(noisy) / SENSOR_MAX
    srgb = np.clip(linear_out, 0, 1) ** (1.0 / params.gamma)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def night_statistics(clear: np.ndarray, night: np.ndarray) -> dict[str, float]:
    """Diagnostics for verifying a generated corpus.

    Reported per image so a build can assert the degradation actually happened
    and lands in a sane range, rather than trusting that it did.
    """
    c = clear.astype(np.float32)
    n = night.astype(np.float32)
    grey_c = c.mean(axis=2)
    grey_n = n.mean(axis=2)
    return {
        "brightness_clear": float(grey_c.mean()),
        "brightness_night": float(grey_n.mean()),
        "brightness_ratio": float(grey_n.mean() / max(grey_c.mean(), 1e-6)),
        "contrast_clear": float(grey_c.std()),
        "contrast_night": float(grey_n.std()),
        # High-frequency energy rises with sensor noise even as contrast falls;
        # the pair separates "dark" from "dark and noisy".
        "highfreq_clear": float(np.abs(np.diff(grey_c, axis=1)).mean()),
        "highfreq_night": float(np.abs(np.diff(grey_n, axis=1)).mean()),
    }
