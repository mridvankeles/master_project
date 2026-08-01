"""Synthetic degradation. STUB — nothing is implemented here yet.

Per the standing rules for this scaffold: no degradation synthesis. This module
exists so the shape of the eventual work is recorded next to the code it will
live beside, not so it can be called.

WHAT GOES HERE
--------------
The low-illumination condition of the three-condition grid in `03-datasets.md`.
The Fog condition is supplied ready-made by the Hazy-DIOR release; Dark must be
synthesised on the *same* DIOR images, which is the whole point — it is what
controls the platform confound described in `03-datasets.md` Risk 2. If fog came
from satellite imagery and darkness from drone imagery, a router could separate
the two conditions on ground-sampling-distance statistics alone and never learn
anything about illumination.

`03-datasets.md` is explicit that this must be a citable physical model rather
than a gamma hack:

  1. Inverse ISP. Undo the camera pipeline back to a linear sensor-space signal
     (inverse tone curve, inverse gamma, inverse white balance), scale exposure
     down there, then re-apply the forward pipeline. Darkening in sRGB space
     instead produces the wrong noise and colour behaviour, and an examiner who
     knows imaging will notice.
  2. Poisson-Gaussian sensor noise on the darkened linear signal: Poisson for
     photon shot noise (signal-dependent), Gaussian for read noise
     (signal-independent). This is the standard sensor model in the low-light
     literature, and it is what makes the darkened images genuinely harder
     rather than merely dimmer.

PARAMETERS MUST BE RANDOMISED PER SAMPLE
----------------------------------------
Not per-dataset, not per-split — per image. `04-method-open-questions.md`
§ Router domain robustness calls this "the single highest-value line of code in
the project": a router trained on one fixed simulator setting learns that
setting, whereas a router trained across a distribution of exposure and noise
parameters has to learn the degradation itself.

`05-experiment-plan.md` § Leakage checklist adds the constraint that the
parameter distribution must be sampled per-split and never fitted on the full
dataset.

STILL UNDECIDED — do not resolve these in code
----------------------------------------------
`03-datasets.md` open question 2 asks which parameterisation is used and whether
the parameters are held fixed between train and test. That is a design decision,
not an implementation detail, and it is unanswered.

The compound fog+dark condition (`03-datasets.md`, "the money experiment") is
also generated here eventually, and is TEST-ONLY. It must never reach a training
split.
"""

from __future__ import annotations

import numpy as np

__all__ = ["synthesize_low_light"]


def synthesize_low_light(
    image: np.ndarray,
    rng: np.random.Generator,
    **params,
) -> np.ndarray:
    """Darken one RGB image with an inverse-ISP + sensor-noise model.

    Args:
        image: HxWx3 uint8 RGB.
        rng: seeded generator; parameters are drawn per call, per the
            domain-randomisation requirement above.
        **params: bounds of the parameter distribution — undecided, see
            `03-datasets.md` open question 2.

    Returns:
        HxWx3 uint8 RGB, darkened.

    Raises:
        NotImplementedError: always. Synthesis is out of scope for the scaffold.
    """
    raise NotImplementedError(
        "Low-light synthesis is deliberately unimplemented. It is blocked on "
        "03-datasets.md open question 2 (which parameterisation, and are "
        "parameters held fixed across train and test?). See this module's "
        "docstring for what goes here."
    )
