"""Condition-gated MoE block, second design.

WHAT CHANGED AND WHY
--------------------
The first block failed both acceptance criteria: NMI(route ; condition) = 0.0000
and inter-expert CKA up to 0.996 (`finding-router-never-specialised.md`). Four
changes, each aimed at one cause.

1. **Heterogeneous STATIC experts.** Branches now start from different fixed
   priors (`src/models/experts.py`) rather than differing only by kernel size,
   so they cannot converge to the same function.

2. **A supervised, MULTI-LABEL gate.** The condition is known — it is in the
   filename — so the gate is trained against it directly instead of being asked
   only to stay balanced. Sigmoid rather than softmax, because an image can be
   foggy *and* dark at once and a softmax forces those to compete for one
   probability budget.

3. **An always-on shared expert.** It runs for every image regardless of the
   gate, so there is always a competent path. Router error then degrades
   performance instead of destroying it, and the routed branches only have to
   learn a *correction* on top of a working representation.

4. **Clean and noisy routes recorded separately.** The previous block stored the
   noise-perturbed argmax in `last_index`, so the utilisation logs described the
   injected noise and reported a collapsed router as healthy. `last_gate` now
   holds the clean gate probabilities and `last_active` the clean activation
   mask; the noisy selection is used for training only and never logged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .experts import StaticExpert

__all__ = ["CondMoEBlock", "gate_supervision_loss", "condition_from_paths"]

# Order is load-bearing: it fixes which output unit means which condition, and
# the supervision target is built from it.
CONDITION_ORDER = ("clear", "fog", "night")

# Corpus directory names that denote the same physical condition as a branch.
# `fog2` is the calibrated ASM synthesis; it is still fog, and without this the
# filename token would match no branch, every sample would be masked out of the
# supervision, and the gate would train on nothing while reporting a healthy
# loss. Aliases are resolved to the branch they belong to.
CONDITION_ALIASES = {"fog2": "fog", "haze": "fog", "dark": "night", "lowlight": "night"}


def condition_from_paths(paths, order=CONDITION_ORDER) -> torch.Tensor:
    """Multi-hot condition targets from union-corpus filenames.

    Stems look like `fog_00042_thin` or `clear_00042`, so the condition is the
    prefix. Multi-hot rather than one-hot so a future `fog_night_*` compound
    condition needs no code change: any recognised token present sets its bit.
    """
    import os

    rows = []
    for p in paths:
        stem = os.path.basename(str(p)).rsplit(".", 1)[0].lower()
        tokens = {CONDITION_ALIASES.get(t, t) for t in stem.split("_")}
        row = [1.0 if c in tokens else 0.0 for c in order]
        if not any(row):
            # Single-condition corpus: no prefix to read, so supervision is
            # unavailable for this sample and it is masked out downstream.
            row = [0.0] * len(order)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


class CondMoEBlock(nn.Module):
    """Shared always-on expert + condition-gated static experts.

    Args:
        c1, c2: channels, injected by Ultralytics' parse_model.
        experts: condition names, one branch each. Order must match
            CONDITION_ORDER for supervision to line up.
        shared: include the always-on branch.
        noise_std: exploration noise on gate logits during training only.
        threshold: a routed branch runs when its clean sigmoid exceeds this.
            Multi-label, so several may run; `top1=True` forces exactly one.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        experts: tuple[str, ...] = ("clear", "fog", "night"),
        shared: bool = True,
        noise_std: float = 0.5,
        threshold: float = 0.5,
        top1: bool = False,
        bottleneck: float = 0.5,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c1, self.c2 = c1, c2
        self.expert_kinds = tuple(experts)
        self.n_experts = len(self.expert_kinds)
        self.noise_std = float(noise_std)
        self.threshold = float(threshold)
        self.top1 = bool(top1)

        self.experts = nn.ModuleList(
            StaticExpert(c1, c2, kind=k, bottleneck=bottleneck) for k in self.expert_kinds
        )
        for e in self.experts:
            e.zero_init_output()

        # Always-on path. Not zero-initialised: it is meant to carry the model
        # from step 0, so the routed branches learn corrections to something
        # that already works.
        self.shared = StaticExpert(c1, c2, kind="plain", bottleneck=bottleneck) if shared else None
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

        self.gate = nn.Linear(c1, self.n_experts)

        # Diagnostics, read by the trainer. Deliberately not buffers.
        self.last_logits: torch.Tensor | None = None   # clean, for supervision
        self.last_gate: torch.Tensor | None = None     # clean sigmoid
        self.last_active: torch.Tensor | None = None   # clean activation mask
        self.last_index: torch.Tensor | None = None    # clean argmax, for reporting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x.mean((2, 3)))
        self.last_logits = logits
        probs = logits.sigmoid()
        self.last_gate = probs
        self.last_index = logits.argmax(1)

        if self.top1:
            sel = logits
            if self.training and self.noise_std > 0:
                sel = logits + torch.randn_like(logits) * self.noise_std
            active = torch.zeros_like(probs)
            active.scatter_(1, sel.argmax(1, keepdim=True), 1.0)
        else:
            gate_for_selection = probs
            if self.training and self.noise_std > 0:
                gate_for_selection = (logits + torch.randn_like(logits) * self.noise_std).sigmoid()
            active = (gate_for_selection > self.threshold).float()
        # Report the CLEAN activations, never the noisy ones — reporting the
        # noisy mask is exactly what hid the previous collapse.
        self.last_active = (probs > self.threshold).float()

        out = self.proj(x)
        if self.shared is not None:
            out = out + self.shared(x)

        for i, expert in enumerate(self.experts):
            mask = active[:, i] > 0
            if mask.any():
                # Weighted by the clean probability so gradient reaches the gate.
                contribution = expert(x[mask]) * probs[mask, i].view(-1, 1, 1, 1)
                out = out.index_add(
                    0, mask.nonzero(as_tuple=True)[0], contribution.to(out.dtype)
                )
        return out


def cond_moe_blocks(model: nn.Module) -> list[CondMoEBlock]:
    return [m for m in model.modules() if isinstance(m, CondMoEBlock)]


def gate_supervision_loss(model: nn.Module, targets: torch.Tensor) -> torch.Tensor | float:
    """Binary cross-entropy between clean gate logits and condition labels.

    This is what makes branch i *mean* condition i. The entropy auxiliary used
    before asked only for balance, which a gate splitting on JPEG quantisation
    satisfies just as well as one splitting on weather.

    Samples with an all-zero target carry no condition information (a
    single-condition corpus has no filename prefix) and are masked out rather
    than taught that every condition is absent.
    """
    blocks = cond_moe_blocks(model)
    logits = [b.last_logits for b in blocks if b.last_logits is not None]
    if not logits:
        return 0.0
    total = 0.0
    for lg in logits:
        t = targets.to(lg.device, lg.dtype)
        valid = t.sum(1) > 0
        if not valid.any():
            continue
        total = total + nn.functional.binary_cross_entropy_with_logits(lg[valid], t[valid])
    return total / max(len(logits), 1)


def routing_report(model: nn.Module) -> dict[str, float]:
    """Clean per-expert activation rates plus mean gate confidence."""
    out: dict[str, float] = {}
    for b_i, block in enumerate(cond_moe_blocks(model)):
        if block.last_active is None:
            continue
        act = block.last_active
        for e, kind in enumerate(block.expert_kinds):
            out[f"moe{b_i}/{kind}_active"] = float(act[:, e].mean())
        out[f"moe{b_i}/experts_per_image"] = float(act.sum(1).mean())
    return out
