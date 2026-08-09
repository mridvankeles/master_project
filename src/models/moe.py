"""A two-expert, top-1 routed Mixture-of-Experts block for YOLO11.

SCOPE
-----
Two experts: one per condition in the current corpus (clear, fog). Low
illumination is out of scope, so there is no third expert and no third route.

WHY THIS SHAPE
--------------
`06-moe-design-guide.md` supplies the constraints this design has to satisfy:

* **Image-level gate (§2.3).** Haze is a global property of an image, not of a
  region, so one routing decision per image is the granularity the condition
  actually has. Per-token routing would be answering a question nobody asked.
* **Top-1 hard routing (§2.1).** Only one expert runs per image, which is what
  makes the FLOPs argument true rather than rhetorical. Soft routing would run
  both and forfeit the efficiency claim.
* **Heterogeneous receptive fields (§2.6).** The single largest effect in
  MFG-HMoE's entire ablation was kernel-size heterogeneity (a 1.5 dB swing)
  rather than routing cleverness (0.03 dB). The two experts therefore differ in
  kernel size: 3x3 and 5x5. There is also a physical reason to expect this to
  matter here — haze is a low-frequency veiling effect whose local airlight is
  better estimated over a wider receptive field, while clear imagery needs
  sharper, more local features.
* **Residual output.** Each expert learns a *correction* to the incoming
  features rather than a replacement. Two consequences that matter: the block
  can be dropped into a pretrained network without destroying it at
  initialisation, and the residual's magnitude is directly measurable, which
  turns "did the experts specialise?" into a number instead of a t-SNE plot.

COLLAPSE
--------
The documented failure mode is the gate collapsing onto one expert while the
other dies (§2.5). `last_logits` is stashed on every forward so the trainer can
add an auxiliary balance loss and so utilisation can be logged per epoch. At
N=2 collapse is less likely than at scale, but it is free to guard against and
impossible to diagnose after the fact without the statistics.

EFFICIENCY, HONESTLY
--------------------
The per-expert loop below only saves wall-clock when a batch routes uniformly.
With mixed conditions in a batch each expert runs on a sub-batch, so FLOPs drop
while wall-clock largely does not (§3.4). Both numbers get reported; the mixed
batch is the honest one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MoEBlock", "routing_stats", "routing_aux_loss"]


def _conv_bn_act(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True),
    )


class MoEBlock(nn.Module):
    """Top-1 routed MoE over `n_experts` heterogeneous-kernel expert branches.

    Args (as they arrive from Ultralytics' `parse_model`):
        c1: input channels, injected by parse_model
        c2: output channels after width scaling, injected by parse_model
        n_experts: number of routed experts (2 for clear/fog)
        kernels: one kernel size per expert; cycled if shorter than n_experts
        bottleneck: expert width as a fraction of c2, so two experts cost far
            less than two full-width convolutions
        shared: add an always-on branch alongside the routed ones. Off by
            default: this design is exactly two routed experts.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        n_experts: int = 2,
        kernels: tuple[int, ...] = (3, 5),
        bottleneck: float = 0.5,
        shared: bool = False,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c1, self.c2 = c1, c2
        self.n_experts = int(n_experts)
        hidden = max(8, int(c2 * bottleneck))

        ks = [kernels[i % len(kernels)] for i in range(self.n_experts)]
        self.kernels = tuple(ks)
        self.experts = nn.ModuleList(
            nn.Sequential(_conv_bn_act(c1, hidden, k), nn.Conv2d(hidden, c2, 1, bias=False))
            for k in ks
        )
        self.shared = (
            nn.Sequential(_conv_bn_act(c1, hidden, 3), nn.Conv2d(hidden, c2, 1, bias=False))
            if shared
            else None
        )
        # Residual path. Identity when the block preserves channels, which is
        # the intended placement; a 1x1 projection otherwise so the block can
        # still be dropped where the width changes.
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

        self.gate = nn.Linear(c1, self.n_experts)
        # Zero-init the last conv of each expert so the block starts as an exact
        # identity. A pretrained backbone therefore survives insertion intact and
        # the experts differentiate from a working model rather than from noise.
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
        if self.shared is not None:
            nn.init.zeros_(self.shared[-1].weight)

        # Read by the trainer for the auxiliary loss and by the eval hooks for
        # utilisation. Never a buffer: it must not enter the state dict.
        self.last_logits: torch.Tensor | None = None
        self.last_index: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x.mean((2, 3)))  # image-level: one decision per sample
        self.last_logits = logits
        weights = logits.softmax(1)
        index = logits.argmax(1)
        self.last_index = index

        out = self.proj(x)
        if self.shared is not None:
            out = out + self.shared(x)

        # Top-1: each sample passes through exactly one expert. The loop runs at
        # most n_experts times regardless of batch size.
        for i, expert in enumerate(self.experts):
            mask = index == i
            if mask.any():
                # Multiply by the gate weight so gradients reach the router.
                # Without this the argmax is not differentiable and the gate
                # would never learn anything.
                contribution = expert(x[mask]) * weights[mask, i].view(-1, 1, 1, 1)
                # `.to(out.dtype)`: under AMP autocast, BatchNorm runs in fp32
                # while the residual path stays fp16, and index_add_ refuses a
                # dtype mismatch.
                out = out.index_add(
                    0, mask.nonzero(as_tuple=True)[0], contribution.to(out.dtype)
                )
        return out


def moe_blocks(model: nn.Module) -> list[MoEBlock]:
    return [m for m in model.modules() if isinstance(m, MoEBlock)]


def routing_stats(model: nn.Module) -> dict[str, float]:
    """Per-expert utilisation over the last forward. Empty if no MoE present."""
    blocks = moe_blocks(model)
    out: dict[str, float] = {}
    for b_i, block in enumerate(blocks):
        idx = block.last_index
        if idx is None:
            continue
        total = idx.numel()
        for e in range(block.n_experts):
            share = float((idx == e).sum()) / max(total, 1)
            out[f"moe{b_i}/expert{e}_share"] = share
    return out


def routing_aux_loss(model: nn.Module, mode: str = "entropy") -> torch.Tensor | float:
    """Auxiliary loss discouraging gate collapse.

    `entropy` is the default on evidence, not taste: the switch/load-balance
    loss failed to prevent dying experts in sparse-MoE CNNs under adversarial
    training, where an entropy term kept multiple experts alive
    (`04-method-open-questions.md` § Auxiliary losses).

    The term maximises the entropy of the BATCH-MEAN routing distribution, not
    of each sample's distribution. Maximising per-sample entropy would push the
    gate toward routing every image 50/50, which is the opposite of
    specialisation; maximising the batch mean asks only that both experts are
    used across the batch, leaving individual decisions free to be confident.
    """
    blocks = moe_blocks(model)
    logits = [b.last_logits for b in blocks if b.last_logits is not None]
    if not logits:
        return 0.0

    total = 0.0
    for lg in logits:
        p = lg.softmax(-1)
        u = p.mean(0).clamp_min(1e-9)  # per-expert importance over the batch
        n = u.numel()
        if mode == "cv":
            total = total + (u.std() / (u.mean() + 1e-6)) ** 2
        elif mode == "switch":
            # Switch-style: N * sum_i f_i * u_i, with f the hard dispatch
            # fraction and u the mean gate probability. Minimised at 1.0 when
            # both are uniform.
            #
            # The gradient is the point. f comes from argmax and is therefore
            # constant, so all gradient flows through u, weighted by how much
            # each expert was ACTUALLY used. An over-dispatched expert has large
            # f, so minimising f*u pushes its gate probability down. Entropy on
            # u alone has no such weighting and treats a 96/4 dispatch split
            # identically to a 50/50 one whenever u happens to look uniform.
            #
            # Honest limitation: at exactly uniform u the term is 1.0 regardless
            # of f, because sum_i f_i = 1. It responds to realistic collapse,
            # where f and u concentrate together, not to that degenerate point.
            f = torch.zeros_like(u).index_add_(
                0, lg.argmax(1), torch.ones(lg.shape[0], device=lg.device)
            ) / max(lg.shape[0], 1)
            total = total + n * (f * u).sum()
        else:
            entropy = -(u * u.log()).sum()
            total = total + (torch.log(torch.tensor(float(n), device=u.device)) - entropy)
    return total / len(logits)
