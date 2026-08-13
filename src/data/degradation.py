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

__all__ = [
    "NightParams",
    "sample_night_params",
    "apply_night",
    "SRGB_GAMMA",
    "FogParams",
    "sample_fog_params",
    "apply_fog",
    "fog_statistics",
]

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


# ---------------------------------------------------------------------------
# Fog / haze
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS WHEN THE RELEASE ALREADY SHIPS FOG
# --------------------------------------------------
# Measured against RRSHID (real paired clear/hazy remote sensing), the
# Hazy-DIOR release is much too weak: dark-channel increase over the matching
# clear image is +36.8 / +62.3 / +93.0 for its thin / moderate / thick tiers,
# against +100.5 for real MODERATE fog and +142.8 for real thick. Our thickest
# setting is weaker than real moderate haze, two thirds of the corpus is barely
# degraded, and that alone explains why the router reaches only 38.5% on fog
# while scoring 98-100% on real haze (`research-expert-design.md` 0.1-0.2).
#
# The release also ships exactly three fixed severities, which is the fixed
# simulator setting `04-method-open-questions.md` warns a router will simply
# memorise.
#
# So fog is synthesised here too: same atmospheric scattering model the
# literature uses, calibrated to the RRSHID statistics, with per-image random
# parameters.


@dataclass(frozen=True)
class FogParams:
    """One image's haze parameters. Reproducible from the image id."""

    beta: float          # scattering coefficient; controls density
    airlight: float      # atmospheric light in [0, 1], near-white for haze
    depth_scale: float   # mean of the pseudo-depth field
    depth_relief: float  # how non-uniform the haze is (0 = flat veil)
    depth_freq: float    # spatial frequency of the depth field


def sample_fog_params(image_id: str, seed: int = 0) -> FogParams:
    digest = hashlib.sha256(f"fog:{seed}:{image_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return FogParams(
        # Calibrated against RRSHID so the dark-channel increase spans roughly
        # +55..+150 -- from a little under real MODERATE haze (+100.5) up to
        # real THICK (+142.8), rather than clustering at one severity. An
        # earlier range of log-uniform(0.9, 3.2) put the mean at +143.8 and the
        # p90 at +188.8, i.e. denser than any real sample measured.
        beta=float(np.exp(rng.uniform(np.log(0.45), np.log(1.9)))),
        airlight=float(rng.uniform(0.80, 0.98)),
        depth_scale=float(rng.uniform(0.5, 1.0)),
        # Real haze is not a flat veil. A little relief makes the density vary
        # across the scene, which is what the dehazing literature reports and
        # what a spatially-aware fog expert would need in order to have
        # anything to be spatially aware of.
        depth_relief=float(rng.uniform(0.15, 0.55)),
        depth_freq=float(rng.uniform(1.5, 4.0)),
    )


def _smooth_depth(h: int, w: int, params: FogParams, rng: np.random.Generator) -> np.ndarray:
    """A low-frequency pseudo-depth field in [0, 1].

    Nadir remote sensing has almost no depth range, so a physical depth map
    would give a perfectly flat veil. Real haze still varies across a scene
    because the haze layer itself is not uniform, so the field models the haze
    layer rather than terrain: a few low-frequency components, smoothed.
    """
    import cv2

    k = max(2, int(params.depth_freq))
    coarse = rng.random((k, k)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(h, w) / 16.0)
    lo, hi = float(field.min()), float(field.max())
    field = (field - lo) / max(hi - lo, 1e-6)
    return params.depth_scale * (1.0 - params.depth_relief + params.depth_relief * field)


def apply_fog(img: np.ndarray, params: FogParams) -> np.ndarray:
    """Atmospheric scattering model: I = J*t + A*(1 - t), t = exp(-beta*d).

    Applied in LINEAR radiance, not in sRGB. Scattering is a physical mixing of
    radiances, so compositing in gamma-encoded space would produce the wrong
    contrast falloff -- the same class of error as darkening in sRGB.
    """
    if img.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {img.dtype}")

    h, w = img.shape[:2]
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(str(asdict(params)).encode()).digest()[:8], "big")
    )
    linear = (img.astype(np.float32) / 255.0) ** SRGB_GAMMA
    depth = _smooth_depth(h, w, params, rng)
    t = np.exp(-params.beta * depth)[..., None]
    a = params.airlight ** SRGB_GAMMA
    hazy = linear * t + a * (1.0 - t)
    srgb = np.clip(hazy, 0, 1) ** (1.0 / SRGB_GAMMA)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def fog_statistics(clear: np.ndarray, hazy: np.ndarray, patch: int = 15) -> dict[str, float]:
    """Dark-channel and contrast diagnostics, matching the RRSHID measurement."""
    import cv2

    def dark(x: np.ndarray) -> float:
        return float(cv2.erode(x.min(axis=2), np.ones((patch, patch), np.uint8)).mean())

    gc = cv2.cvtColor(clear, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gh = cv2.cvtColor(hazy, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return {
        "darkchannel_clear": dark(clear),
        "darkchannel_hazy": dark(hazy),
        "darkchannel_delta": dark(hazy) - dark(clear),
        "contrast_clear": float(gc.std()),
        "contrast_hazy": float(gh.std()),
        "contrast_delta": float(gh.std() - gc.std()),
        "brightness_ratio": float(gh.mean() / max(gc.mean(), 1e-6)),
    }
