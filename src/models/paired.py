"""Paired degraded/clear supervision — how we find out what an expert learned.

THE OPPORTUNITY THE CORPUS ALREADY GIVES US
-------------------------------------------
`clear_00042`, `fog2_00042` and `night_00042` are the SAME DIOR scene rendered
three ways, and `scripts/audit_corpus_labels.py` verified their labels are
identical (5,862 compared, 0 differ). That makes this a **paired restoration
dataset**, which is rare and which nothing in the project has used.

So the fog expert does not have to be argued about. It can be told directly:

    take the foggy features, and produce what the block would have produced
    for the clear twin of this exact scene.

    L_restore = SmoothL1( block_out(f_degraded),  block_out(f_clear).detach() )

applied only to degraded samples. That is dehazing, defined in feature space,
with ground truth. The same term asks the night branch for light invariance.

It also fixes the problem `expert_intervention.py` exposed. The experts were
inert because an always-on shared branch satisfied the detection loss first, so
no gradient pressure ever reached them. This term gives the branches an
objective of their own that the shared path cannot absorb.

WHY THE TWIN NEEDS THE SAME AUGMENTATION
----------------------------------------
Feature maps only align if both images got the same crop, scale and flip. The
transforms draw from the global RNG, so this saves the RNG state, transforms the
degraded sample, restores the state, and transforms the twin. Identical labels
mean identical control flow through the pipeline, so identical draws.

**This requires `mosaic: 0.0`.** With mosaic on, the twin's transform would draw
three further random images and the alignment argument collapses. That is fine —
mosaic is already off for exactly the reason described in
`docs/pipeline-and-data-audit.md`.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.data.dataset import YOLODataset

from src.models.moe2 import CONDITION_ALIASES

__all__ = ["PairedYOLODataset", "restoration_loss", "scene_id", "condition_token",
           "clear_twin_path", "attach_twins"]


def _tokens(path) -> list[str]:
    return Path(str(path)).stem.lower().split("_")


def condition_token(path) -> str:
    """Canonical condition for a corpus filename (`fog2` -> `fog`)."""
    for t in _tokens(path):
        c = CONDITION_ALIASES.get(t, t)
        if c in ("clear", "fog", "night"):
            return c
    return "?"


def scene_id(path) -> str:
    """The DIOR id shared by a scene's three renderings."""
    toks = _tokens(path)
    return toks[-1] if len(toks) > 1 else toks[0]


def clear_twin_path(f) -> Path | None:
    """Where the clear rendering of this scene lives, or None.

    Two corpus layouts, both supported and both real here:

    * per-condition corpora use bare ids -- `dior_hbb_full/clear/images/train/00042.jpg`
    * the union corpus prefixes them   -- `union3b/images/train/fog2_00042.jpg`

    and the union corpus PARTITIONS scenes across conditions rather than holding
    all three renderings of each, so a union image's twin is almost never beside
    it. Measured: 91 of 400 found in-place, versus 3908 of 3908 found in the
    sibling `clear/` corpus. So the sibling is tried first.
    """
    f = Path(str(f))
    sid = scene_id(f)
    split = f.parent.name                    # train / val / test
    root = f.parents[3] if len(f.parents) > 3 else None
    if root is not None:
        sibling = root / "clear" / "images" / split / f"{sid}{f.suffix}"
        if sibling.exists():
            return sibling
    beside = f.with_name(f"clear_{sid}{f.suffix}")
    return beside if beside.exists() else None


def attach_twins(ds) -> int:
    """Give `ds` a clear twin for every degraded sample, and return how many.

    THE SUBSET PROBLEM
    ------------------
    `stratified_subset` samples each condition independently, so a fog image's
    clear twin is almost never in the same subset -- measured, 6 of 400. Looking
    the twin up inside the dataset therefore finds nothing.

    Rebuilding the subset as scene triplets would fix it and cost too much: the
    budget is 5,862 images per epoch, so triplets would mean 1,954 unique scenes
    instead of 5,862, a 3x loss of scene diversity, and every comparison against
    an existing arm would break.

    Twins are resolved by `clear_twin_path`, which looks in the sibling
    per-condition `clear/` corpus first. The missing ones are APPENDED as extra
    entries that
    `__len__` hides, so the sampler never draws them and the training budget is
    untouched -- they exist only to be fetched as restoration targets. Their
    label dicts are copied from the degraded sample, which is exact: the corpus
    audit verified all three renderings carry identical labels.
    """
    import copy

    n_sampled = len(ds.labels)
    by_path = {str(f): i for i, f in enumerate(ds.im_files)}
    twin_index: dict[int, int] = {}
    for i in range(n_sampled):
        f = Path(str(ds.im_files[i]))
        if condition_token(f) not in ("fog", "night"):
            twin_index[i] = i
            continue
        clear = clear_twin_path(f)
        if clear is None:
            twin_index[i] = i
            continue
        j = by_path.get(str(clear))
        if j is None:
            lab = copy.deepcopy(ds.labels[i])
            lab["im_file"] = str(clear)
            ds.labels.append(lab)
            ds.im_files.append(str(clear))
            j = len(ds.labels) - 1
            by_path[str(clear)] = j
        twin_index[i] = j

    ds.twin_index = twin_index
    ds.n_sampled = n_sampled
    # The cache arrays are sized by self.ni, so they have to grow with it.
    ds.ni = len(ds.labels)
    ds.ims = [None] * ds.ni
    ds.im_hw0 = [None] * ds.ni
    ds.im_hw = [None] * ds.ni
    ds.npy_files = [Path(f).with_suffix(".npy") for f in ds.im_files]
    ds.buffer = []
    return sum(1 for i, j in twin_index.items() if j != i)


class PairedYOLODataset(YOLODataset):
    """`YOLODataset` that also returns each degraded sample's clear twin.

    Adds one key, `twin_img`, holding the clear rendering of the same scene under
    the same geometric augmentation. Clear samples get their own image back, and
    are masked out of the loss by the trainer rather than here, so batch shapes
    stay uniform.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        attach_twins(self)

    def __len__(self) -> int:
        """Hide the appended twins from the sampler.

        They are targets, not training samples. Without this the epoch would
        silently grow and the matched-budget rule -- 5,862 images per epoch for
        every arm in the project -- would be broken.
        """
        return getattr(self, "n_sampled", len(self.labels))

    def __getitem__(self, index):
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        out = super().__getitem__(index)
        twin = self.twin_index.get(index, index)
        if twin == index:
            out["twin_img"] = out["img"]
        else:
            # Same RNG state in, same geometric transform out. Only valid because
            # the twin carries identical labels, so the pipeline branches the
            # same way and consumes the same number of draws.
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
            out["twin_img"] = super().__getitem__(twin)["img"]
        return out

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        twins = [b.pop("twin_img") for b in batch]
        new = YOLODataset.collate_fn(batch)
        new["twin_img"] = torch.stack(twins, 0)
        return new


def restoration_loss(student: torch.Tensor, target: torch.Tensor,
                     degraded: torch.Tensor, beta: float = 0.1) -> torch.Tensor | float:
    """SmoothL1 between the block's output on a degraded image and on its twin.

    Normalised by the target's own RMS so the term does not depend on how large
    the block's activations happen to be, and so it stays comparable between
    runs. Applied only where `degraded` is True: asking a clear image to
    reconstruct itself is free and would only dilute the average.
    """
    if degraded.sum() == 0:
        return 0.0
    s = student[degraded]
    t = target[degraded].detach()
    scale = t.pow(2).mean().sqrt().clamp_min(1e-3)
    return F.smooth_l1_loss(s / scale, t / scale, beta=beta)
