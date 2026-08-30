"""Normalized Wasserstein Distance box loss, size-gated for tiny objects.

WHY IoU FAILS ON TINY OBJECTS
-----------------------------
IoU is scale-sensitive in a way that hurts exactly where DIOR is weakest. For a
4x4 box a one-pixel centre shift drops IoU from 1.00 to about 0.36; for a 40x40
box the same shift costs almost nothing. So the gradient a tiny object produces
is dominated by quantisation rather than by localisation quality, and the label
assignment it feeds is unstable from step to step.

Wang et al. (2021), *A Normalized Gaussian Wasserstein Distance for Tiny Object
Detection*, replace the overlap with a distribution distance. Each box becomes a
2-D Gaussian — mean at the centre, covariance diag((w/2)^2, (h/2)^2) — and the
similarity is a normalised Wasserstein distance:

    W2^2 = ||(cx_a, cy_a, wa/2, ha/2) - (cx_b, cy_b, wb/2, hb/2)||^2
    NWD  = exp(-sqrt(W2^2) / C)

This is smooth and finite even when two boxes do not overlap at all, which is
the common case early in training for small objects, and its gradient magnitude
does not collapse with box size.

WHY IT IS GATED BY SIZE RATHER THAN APPLIED EVERYWHERE
------------------------------------------------------
NWD's scale-invariance is a liability on large objects: it stops distinguishing
a good large box from a slightly worse one, because the normalising constant C
makes big differences look uniformly small. The published results reflect this —
NWD helps on tiny-object benchmarks and does not help on COCO-scale ones.

So this implementation blends per box, by box size:

    loss = (1 - a) * (1 - CIoU) + a * (1 - NWD)

with `a` a smooth function of the object's area: ~1 for boxes well below the
tiny threshold, ~0 for boxes well above it. That is the "different loss for
different object regimes" idea, expressed as a continuous gate rather than a
hard switch — a hard switch puts a discontinuity in the loss surface at the
threshold, and objects sitting near it oscillate between two objectives.

The gate is a *static* routing decision: it depends on the target box, which is
known, not on a learned gate. That distinguishes it from the MoE's dynamic
routing and lets the two be studied independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

__all__ = ["nwd_similarity", "size_gate", "NWDBboxLoss"]


def nwd_similarity(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor, c: float = 12.8):
    """NWD in [0, 1]; 1 is a perfect match.

    Args:
        pred_xyxy, target_xyxy: (N, 4) boxes in xyxy, IN PIXELS at model scale.
           The caller is responsible for converting out of stride units.
        c: normalising constant. Should sit near the dataset's typical object
           size — the loss is insensitive to it within a factor of ~2, but a
           badly wrong C flattens the gradient. 12.8 is the value used for
           AI-TOD-scale objects and suits DIOR's `vehicle`/`storagetank`.
    """
    pcx = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) * 0.5
    pcy = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) * 0.5
    pw = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp_min(1e-6)
    ph = (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp_min(1e-6)

    tcx = (target_xyxy[..., 0] + target_xyxy[..., 2]) * 0.5
    tcy = (target_xyxy[..., 1] + target_xyxy[..., 3]) * 0.5
    tw = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp_min(1e-6)
    th = (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp_min(1e-6)

    # Squared 2-Wasserstein between the two axis-aligned Gaussians.
    w2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2 + ((pw - tw) ** 2 + (ph - th) ** 2) * 0.25
    return torch.exp(-torch.sqrt(w2.clamp_min(1e-12)) / c)


def size_gate(target_xyxy: torch.Tensor, tiny_area: float = 32.0**2, sharpness: float = 2.0):
    """Weight in [0, 1]: 1 for tiny boxes, 0 for large ones.

    Uses a logistic in log-area so the transition is smooth and symmetric in
    scale — a factor-of-two-smaller box moves the same distance along the gate
    wherever it starts. `tiny_area` defaults to COCO's small-object threshold so
    the split matches the convention published tables report against.
    """
    w = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp_min(1e-6)
    h = (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp_min(1e-6)
    log_ratio = torch.log((w * h).clamp_min(1e-6)) - torch.log(
        torch.tensor(tiny_area, device=w.device, dtype=w.dtype)
    )
    return torch.sigmoid(-sharpness * log_ratio)


class NWDBboxLoss(BboxLoss):
    """`BboxLoss` with the IoU term replaced by a size-gated CIoU/NWD blend.

    The DFL term is inherited unchanged: it regresses a distance distribution and
    is not the part that misbehaves on small objects.

    UNITS -- the bug this class shipped with
    ----------------------------------------
    Ultralytics passes `pred_bboxes` and `target_bboxes / stride_tensor` into
    `bbox_loss`, so boxes arrive in FEATURE-MAP CELLS, not pixels. CIoU does not
    care -- IoU is scale invariant -- but NWD's constant `c` and the size gate's
    `tiny_area` are absolute, so both were being fed numbers 8-32x too small.
    Measured consequence: `size_gate` returned ~1.0 for every box in DIOR, from a
    16 px vehicle to a 320 px stadium, and `mode="gated"` silently behaved as
    `mode="always"` -- NWD applied to everything, which is precisely the regime
    the paper says hurts large objects. That is a better explanation of the
    original NWD arm (+0.008 aggregate, **-0.006 on the small classes it
    targeted**) than anything about the loss itself.

    Boxes are now multiplied back by their anchor's stride before NWD and the
    size gate see them.

    LEVELS
    ------
    `levels="p3"` restricts the NWD blend to anchors on the finest pyramid level
    (stride 8) -- the branch the MoE block feeds and the only one that carries
    tiny objects. P4/P5 keep plain CIoU, where it works. `p3_weight` scales that
    level's regression loss so gradient is spent where the misses are: measured
    on `cond3b_gated`, 100% of objects under 8 px and 63.7% under 16 px are
    missed, against ~10% above 32 px.
    """

    def __init__(self, reg_max: int = 16, c: float = 12.8,
                 tiny_area: float = 32.0**2, mode: str = "gated",
                 levels: str = "all", p3_weight: float = 1.0):
        super().__init__(reg_max)
        self.c = float(c)
        self.tiny_area = float(tiny_area)
        # "gated" blends by size; "always" applies NWD to every box (the
        # ablation that shows why gating is needed); "off" is stock CIoU.
        self.mode = mode
        # "all" | "p3": which pyramid levels the NWD blend applies to.
        self.levels = levels
        # Multiplier on the P3 regression loss (IoU term and DFL alike).
        self.p3_weight = float(p3_weight)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        pred_fg = pred_bboxes[fg_mask]
        tgt_fg = target_bboxes[fg_mask]

        # Per-foreground-anchor stride, so cells can be converted to pixels and
        # the finest level can be identified. `stride` is (A, 1) over anchors and
        # `fg_mask` is (B, A), so it has to be broadcast over the batch first.
        s = stride.view(1, -1).expand(fg_mask.shape)[fg_mask].unsqueeze(-1)
        on_p3 = s <= stride.min()

        iou = bbox_iou(pred_fg, tgt_fg, xywh=False, CIoU=True)
        ciou_term = 1.0 - iou

        if self.mode == "off":
            blended = ciou_term
        else:
            nwd_term = (1.0 - nwd_similarity(pred_fg * s, tgt_fg * s, self.c)).unsqueeze(-1)
            if self.mode == "always":
                blended = nwd_term
            else:
                a = size_gate(tgt_fg * s, self.tiny_area).unsqueeze(-1)
                blended = (1.0 - a) * ciou_term + a * nwd_term
            if self.levels == "p3":
                blended = torch.where(on_p3, blended, ciou_term)

        # Extra weight on the tiny-object branch. Deliberately NOT renormalised:
        # the point is to spend more gradient there, so the box loss magnitude
        # rises with p3_weight and the run is not comparable to one without it.
        w = weight
        if self.p3_weight != 1.0:
            w = weight * (1.0 + (self.p3_weight - 1.0) * on_p3.to(weight.dtype))

        loss_iou = (blended * w).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * w
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


def install_nwd(model: nn.Module, c: float = 12.8, tiny_area: float = 32.0**2,
                mode: str = "gated", levels: str = "all",
                p3_weight: float = 1.0) -> bool:
    """Swap the criterion's box loss in place. Returns True if it was applied.

    Called after the criterion exists, because Ultralytics builds it lazily on
    the first loss computation and reassigns `model.criterion` during setup.
    """
    criterion = getattr(model, "criterion", None)
    if criterion is None or not hasattr(criterion, "bbox_loss"):
        return False
    old = criterion.bbox_loss
    reg_max = getattr(getattr(old, "dfl_loss", None), "reg_max", 16)
    new = NWDBboxLoss(reg_max=reg_max, c=c, tiny_area=tiny_area, mode=mode,
                      levels=levels, p3_weight=p3_weight)
    new.to(next(model.parameters()).device)
    criterion.bbox_loss = new
    return True
