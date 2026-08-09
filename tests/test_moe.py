"""Tests for the two-expert MoE block.

These cover the properties the thesis actually leans on, not just "it runs":
identity at initialisation (so insertion into a pretrained network is lossless),
top-1 exclusivity (so the FLOPs argument is true), gradient flow to the gate (so
the router can learn at all), and the auxiliary loss's ability to see collapse.
"""

from __future__ import annotations

import torch

from src.models.moe import MoEBlock, routing_aux_loss, routing_stats


def _block(**kw) -> MoEBlock:
    return MoEBlock(c1=32, c2=32, n_experts=2, **kw)


def test_identity_at_initialisation():
    """Zero-initialised expert outputs must leave the input untouched.

    This is what allows an MoE block to replace a layer in a COCO-pretrained
    network without destroying it: at step 0 the network computes exactly what
    it did before, and the experts differentiate from a working model.
    """
    block = _block().eval()
    x = torch.randn(4, 32, 16, 16)
    with torch.no_grad():
        assert torch.allclose(block(x), x, atol=1e-6)


def test_output_shape_preserved():
    block = _block().eval()
    x = torch.randn(3, 32, 20, 20)
    with torch.no_grad():
        assert block(x).shape == x.shape


def test_projection_when_channels_change():
    block = MoEBlock(c1=32, c2=64, n_experts=2).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        assert block(x).shape == (2, 64, 8, 8)


def test_routing_is_top1_exclusive():
    """Every sample must be assigned to exactly one expert.

    The efficiency claim rests on this: if samples reached both experts the
    block would cost two experts per image and the FLOPs argument would be
    false.
    """
    block = _block().eval()
    x = torch.randn(8, 32, 8, 8)
    with torch.no_grad():
        block(x)
    idx = block.last_index
    assert idx.shape == (8,)
    assert set(idx.tolist()) <= {0, 1}


def test_utilisation_shares_sum_to_one():
    block = _block().eval()
    with torch.no_grad():
        block(torch.randn(10, 32, 8, 8))
    stats = routing_stats(block)
    assert abs(sum(stats.values()) - 1.0) < 1e-6


def test_gate_receives_gradient():
    """The gate must be trainable.

    argmax is not differentiable, so the gradient reaches the router only
    through the softmax weight multiplying the selected expert's output. If that
    multiplication were dropped the router would never learn and the MoE would
    be a random switch.
    """
    block = _block()
    # Experts start zero-initialised, which also zeroes the gate gradient, so
    # give the last layer some signal first.
    for expert in block.experts:
        torch.nn.init.normal_(expert[-1].weight, std=0.05)
    out = block(torch.randn(4, 32, 8, 8))
    out.sum().backward()
    assert block.gate.weight.grad is not None
    assert block.gate.weight.grad.abs().sum() > 0


def test_both_aux_losses_are_minimal_at_balance_and_rise_under_collapse():
    block = _block()

    block.last_logits = torch.tensor([[0.0, 0.0]] * 16)  # perfectly balanced
    balanced = {m: float(routing_aux_loss(block, mode=m)) for m in ("entropy", "switch")}

    block.last_logits = torch.tensor([[3.0, 0.0]] * 16)  # gate strongly prefers expert 0
    collapsed = {m: float(routing_aux_loss(block, mode=m)) for m in ("entropy", "switch")}

    assert balanced["entropy"] < 1e-6
    assert abs(balanced["switch"] - 1.0) < 1e-6  # N * sum(f*u) minimum is 1.0
    for mode in ("entropy", "switch"):
        assert collapsed[mode] > balanced[mode], f"{mode} must penalise collapse"


def test_switch_is_insensitive_at_exactly_uniform_gate():
    """A documented blind spot, asserted so it cannot regress silently.

    N * sum_i f_i * u_i collapses to 1.0 whenever u is exactly uniform, because
    sum_i f_i = 1 -- the hard dispatch split becomes invisible at that point.
    This is why utilisation is logged every epoch rather than being inferred
    from the auxiliary loss alone: neither term can be trusted to reveal a dead
    expert on its own.
    """
    block = _block()
    block.last_logits = torch.tensor([[0.0, 0.0]] * 16)  # uniform u, unanimous argmax
    assert abs(float(routing_aux_loss(block, mode="switch")) - 1.0) < 1e-6


def test_aux_loss_is_zero_without_moe_blocks():
    plain = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1))
    assert routing_aux_loss(plain) == 0.0
    assert routing_stats(plain) == {}


def test_experts_have_heterogeneous_kernels():
    """Kernel heterogeneity was the single largest effect in MFG-HMoE's ablation."""
    block = _block()
    assert block.kernels == (3, 5)
    sizes = {e[0][0].kernel_size for e in block.experts}
    assert len(sizes) == 2
