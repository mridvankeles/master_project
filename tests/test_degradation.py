"""Tests for the low-illumination synthesis.

These assert the properties that make the condition a fair one. A darkening that
merely scales pixels produces a dim but clean image — an easier detection problem
that would flatter every result computed on it — so the tests check the physics,
not just the output shape.
"""

from __future__ import annotations

import numpy as np

from src.data.degradation import (
    NightParams,
    apply_night,
    night_statistics,
    sample_night_params,
)


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Structured rather than pure noise, so contrast statistics mean something.
    base = np.linspace(20, 220, 64, dtype=np.float32)
    img = np.stack(np.meshgrid(base, base), axis=-1).mean(axis=-1)
    img = np.repeat(img[:, :, None], 3, axis=2)
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_parameters_are_deterministic_per_image_id():
    """A rebuild must reproduce the corpus exactly."""
    assert sample_night_params("00042") == sample_night_params("00042")
    assert sample_night_params("00042") != sample_night_params("00043")


def test_output_is_darker():
    img = _image()
    out = apply_night(img, sample_night_params("00001"))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert night_statistics(img, out)["brightness_ratio"] < 0.7


def test_brightness_is_set_by_exposure_not_iso_gain():
    """ISO models a shorter shutter amplified afterwards, not extra light.

    Applying the gain only on the way out brightens the image back up and can
    cancel the darkening entirely — measured at 11% darker before this was
    fixed, which is not night by any definition.
    """
    img = _image()
    ratios = []
    for iso in (1.0, 2.0, 4.0):
        p = NightParams(exposure=0.08, read_noise=3.0, iso_gain=iso,
                        wb_gain_r=0.9, wb_gain_b=1.2)
        ratios.append(night_statistics(img, apply_night(img, p))["brightness_ratio"])
    assert max(ratios) - min(ratios) < 0.05, f"ISO changed brightness: {ratios}"


def test_iso_gain_raises_noise_at_fixed_brightness():
    """The whole point of the sensor model: darker means noisier, not just dimmer."""
    img = _image()
    hf = []
    for iso in (1.0, 4.0):
        p = NightParams(exposure=0.08, read_noise=3.0, iso_gain=iso,
                        wb_gain_r=1.0, wb_gain_b=1.0)
        hf.append(night_statistics(img, apply_night(img, p))["highfreq_night"])
    assert hf[1] > hf[0] * 1.2, f"higher ISO must be noisier: {hf}"


def test_beats_a_gamma_hack_on_noise_retention():
    """A plain exposure scale keeps the image clean; the sensor model does not.

    03-datasets.md requires a citable physical model precisely because the naive
    version is visually similar and quantitatively much easier.
    """
    img = _image()
    p = sample_night_params("00001")
    full = apply_night(img, p)

    linear = (img.astype(np.float32) / 255.0) ** p.gamma * p.exposure
    gamma_only = (np.clip(linear, 0, 1) ** (1 / p.gamma) * 255).astype(np.uint8)

    hf_full = night_statistics(img, full)["highfreq_night"]
    hf_gamma = night_statistics(img, gamma_only)["highfreq_night"]
    assert hf_full > hf_gamma * 1.5, f"sensor model must retain noise: {hf_full} vs {hf_gamma}"


def test_exposure_range_spans_a_useful_spread():
    """One fixed setting would let a router learn the setting, not the condition.

    04-method-open-questions.md calls per-sample randomisation the single
    highest-value line in the project; this asserts it actually varies.
    """
    exps = [sample_night_params(f"{i:05d}").exposure for i in range(400)]
    assert min(exps) < 0.03 and max(exps) > 0.2
    # Essentially every image gets its own exposure. Rounded to 6 places rather
    # than 4: the distribution is log-uniform, so it concentrates near 0.02 and
    # 4-place rounding collides there without the sampling being any less varied.
    assert len(set(round(e, 6) for e in exps)) > 390
    # And the spread is wide, not a tight cluster with two outliers.
    assert float(np.std(exps)) / float(np.mean(exps)) > 0.5


def test_rejects_non_uint8_input():
    try:
        apply_night(np.zeros((8, 8, 3), dtype=np.float32), sample_night_params("x"))
    except TypeError:
        return
    raise AssertionError("expected TypeError on float input")
