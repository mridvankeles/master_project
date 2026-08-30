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

__all__ = ["CondMoEBlock", "gate_supervision_loss", "routing_cost",
           "expert_floor_loss", "condition_from_paths"]

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
        arch: str = "static",
        hard_mask: bool = False,
        rich_proj: bool = False,
        init_gain: float = 0.0,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c1, self.c2 = c1, c2
        self.expert_kinds = tuple(experts)
        self.n_experts = len(self.expert_kinds)
        self.noise_std = float(noise_std)
        self.threshold = float(threshold)
        self.top1 = bool(top1)

        # arch="static" is design 2: one architecture behind different fixed
        # priors. `expert_intervention.py` measured that design as inert --
        # switching every expert off moved mAP by 0.001. arch="hetero" is
        # design 3: genuinely different architectures per degradation.
        self.arch = arch
        self.hard_mask = bool(hard_mask)
        c_mid = max(8, int(c2 * bottleneck))
        if arch == "hetero":
            from .experts3 import EXPERT_ARCH, RichProj

            self.experts = nn.ModuleList(
                EXPERT_ARCH[k](c1, c2, c_mid, use_ctx=rich_proj) for k in self.expert_kinds)
            self.shared = EXPERT_ARCH["plain"](c1, c2, c_mid, use_ctx=False) if shared else None
            self.proj = RichProj(c1, c2) if rich_proj else (
                nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False))
        else:
            self.experts = nn.ModuleList(
                StaticExpert(c1, c2, kind=k, bottleneck=bottleneck) for k in self.expert_kinds)
            self.shared = StaticExpert(c1, c2, kind="plain",
                                       bottleneck=bottleneck) if shared else None
            self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

        # init_gain 0 reproduces design 2 exactly (branch starts at zero). A
        # small non-zero gain leaves the block near-identity at step 0 while
        # giving each branch a real gradient from the first step -- a branch
        # pinned at zero climbs out slowly while a working bypass is already
        # minimising the loss, which is half of why design 2's experts died.
        for e in self.experts:
            if init_gain > 0 and hasattr(e, "scale_init"):
                e.scale_init(init_gain)
            else:
                e.zero_init_output()

        # Whether the always-on path is fed INTO the experts as context.
        self.rich_proj = bool(rich_proj)

        self.gate = nn.Linear(c1, self.n_experts)

        # Diagnostics, read by the trainer. Deliberately not buffers.
        self.last_logits: torch.Tensor | None = None   # clean, for supervision
        self.last_gate: torch.Tensor | None = None     # clean sigmoid
        self.last_active: torch.Tensor | None = None   # clean activation mask
        self.last_index: torch.Tensor | None = None    # clean argmax, for reporting
        self.last_out: torch.Tensor | None = None      # block output, for restoration
        self.last_expert_out: dict[str, torch.Tensor] = {}  # per-branch, for inspection

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

        ctx = self.proj(x)
        out = ctx
        if self.shared is not None:
            sh = self.shared(x) if self.arch != "hetero" else self.shared(x, None)
            out = out + sh
            self.last_expert_out["shared"] = sh

        # Straight-through: the branch runs at full strength when selected, but
        # the gate still receives gradient through p. Design 2 multiplied the
        # output by p, which tied MAGNITUDE to SELECTION -- a branch could only
        # reach full strength if the gate was fully confident, and BCE caps that.
        weight = probs + (active - probs).detach() if self.hard_mask else probs

        for i, expert in enumerate(self.experts):
            mask = active[:, i] > 0
            if mask.any():
                xi = x[mask]
                ei = expert(xi, ctx[mask]) if self.arch == "hetero" else expert(xi)
                contribution = ei * weight[mask, i].view(-1, 1, 1, 1)
                out = out.index_add(
                    0, mask.nonzero(as_tuple=True)[0], contribution.to(out.dtype)
                )
        self.last_out = out
        return out


def cond_moe_blocks(model: nn.Module) -> list[CondMoEBlock]:
    return [m for m in model.modules() if isinstance(m, CondMoEBlock)]


def gate_supervision_loss(
    model: nn.Module, targets: torch.Tensor, pos_weight: float = 1.0
) -> torch.Tensor | float:
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
        pw = None
        if pos_weight != 1.0:
            # With 3 outputs and typically 1 positive, two thirds of every
            # target is zero and BCE is minimised by predicting low. Weighting
            # positives by (n_experts - 1) restores the balance between the
            # positive and negative halves of the objective.
            pw = torch.full((lg.shape[1],), float(pos_weight), device=lg.device, dtype=lg.dtype)
        total = total + nn.functional.binary_cross_entropy_with_logits(
            lg[valid], t[valid], pos_weight=pw
        )
    return total / max(len(logits), 1)


def routing_cost(
    model: nn.Module,
    targets: torch.Tensor,
    weight_count: float = 1.0,
) -> torch.Tensor | float:
    """Charge the gate for activating the wrong NUMBER of experts.

    THE PROBLEM THIS SOLVES
    -----------------------
    The supervised BCE says *which* branch should fire; nothing says *how many*.
    With three outputs and typically one positive, two thirds of every target is
    zero, so BCE is minimised by predicting uniformly low. Measured on
    `cond3b_gated`: probabilities peak on the correct branch but reach only
    0.336, so at a 0.5 threshold **0.466 experts activate per image** and most
    images reach no specialist at all. The gate is miscalibrated, not wrong.

    THE TERM
    --------
        L_count = | sum_i p_i  -  n_true |

    `n_true` is the number of conditions actually present, read from the same
    free filename labels. So it is one term doing two jobs, which is exactly the
    pair of costs asked for:

    * a **coverage** cost when too few experts fire (sum below n_true) --
      "choosing no expert" is charged for;
    * a **sparsity** cost when too many fire (sum above n_true) -- routing
      through experts the image does not need is charged for.

    Using the label rather than a constant matters: a compound `fog_night`
    image has n_true = 2, so the term asks for two experts there and one
    elsewhere, instead of forcing every image to a single route. That is what
    keeps the multi-label design intact.

    Relation to the literature: this is a supervised form of the per-sample
    sparsity / commitment losses used in sparse MoE, where the target count is
    normally a fixed k. Load-balancing terms (Switch, CV-based) constrain usage
    across a BATCH and cannot fix a per-image calibration error, which is why
    they were not the right tool here.
    """
    blocks = cond_moe_blocks(model)
    gates = [b.last_gate for b in blocks if b.last_gate is not None]
    if not gates or weight_count == 0:
        return 0.0
    total = 0.0
    for g in gates:
        t = targets.to(g.device, g.dtype)
        if t.shape != g.shape:
            continue
        valid = t.sum(1) > 0
        if not valid.any():
            continue
        n_true = t[valid].sum(1)
        n_active = g[valid].sum(1)
        total = total + (n_active - n_true).abs().mean()
    return weight_count * total / max(len(gates), 1)


def expert_floor_loss(model: nn.Module, tau: float = 0.6) -> torch.Tensor | float:
    """Charge the model for routing an image through the shared branch ALONE.

    THE ARGUMENT
    ------------
    The shared branch exists so that a router mistake degrades the prediction
    instead of destroying it. It is a safety net, not a route. But nothing in
    the objective ever said so: an image whose gate probabilities all sit below
    the threshold takes the shortcut, pays nothing for it, and the specialists
    it was supposed to reach never see the sample or its gradient. Measured on
    `cond3b_gated`, the gate's mean maximum probability was 0.500 -- sitting
    exactly on the threshold, so roughly half of all images took the shortcut.

    THE TERM
    --------
        L_floor = mean_b  relu( tau - max_i p_i(b) )

    A hinge, not a penalty: once some expert reaches `tau` the cost is zero and
    the term stops pulling, so it sets a floor on routing without competing with
    the supervised BCE about *which* expert should win. `tau` should sit above
    the block's firing threshold (0.6 against a 0.5 threshold) so a satisfied
    hinge means the expert genuinely fires rather than hovering at the cut.

    Deliberately NOT masked to labelled samples. `routing_cost` needs `n_true`
    and so only applies where a condition label exists; this one applies to
    every image, which is the case that matters for out-of-distribution input --
    an unlabelled, unfamiliar sample must still be routed somewhere rather than
    silently falling through to the shortcut.

    Relation to the literature: this is the per-sample commitment / utilisation
    floor used in sparse MoE, in its supervised-free form. Batch-level load
    balancing (Switch, CV) cannot express it -- a batch can be perfectly
    balanced while every individual image routes nowhere.
    """
    blocks = cond_moe_blocks(model)
    gates = [b.last_gate for b in blocks if b.last_gate is not None]
    if not gates or tau <= 0:
        return 0.0
    total = 0.0
    for g in gates:
        total = total + torch.relu(tau - g.max(1).values).mean()
    return total / max(len(gates), 1)


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
        # The number the expert floor exists to drive to zero: images that
        # reached no specialist and were carried by the shared branch alone.
        out[f"moe{b_i}/shortcut_only"] = float((act.sum(1) == 0).float().mean())
        if block.last_gate is not None:
            # Expected count is threshold-free: it shows whether the gate is
            # confident, independently of where the cut is placed.
            out[f"moe{b_i}/expected_experts"] = float(block.last_gate.sum(1).mean())
            out[f"moe{b_i}/gate_max_prob"] = float(block.last_gate.max(1).values.mean())
    return out
