# Loss functions for the experts — what the literature offers, read against our diagnosis

> **Date:** 2026-08-31. Sources: the papers in `tez/other-papers/` and `tez/main-papers/`,
> plus a web sweep. Every candidate below is scored against a *measured* failure
> of ours, not against a general description of MoE.

---

## The diagnosis these have to answer

Not "our mAP is low". Three specific measurements:

| measurement | value | what it means |
|---|---|---|
| intervention spread across 6 routing policies | **0.0011–0.0017 mAP** | switching **every expert off** costs nothing — the branches are inert |
| expert output RMS / always-on path | **10–21%**, then × p | the experts are a rounding error on the block output |
| prior-to-prior CKA, before any training | **0.946 / 0.932 / 0.983** | they never diverged; they started together |
| best worst-pair CKA over 36 candidate priors | **0.9504** vs current 0.9883 | the whole prior space buys 0.04 — not the lever |

So a useful loss must do one of two things: **make an expert's output matter to
the detection loss**, or **make two experts compute different things**. Balance
losses do neither, which is why ours have never paid.

---

## 1. Joint restoration + detection — the strongest match, and already implemented

**TogetherNet, DSNet, FriendNet, MTW-DETR** all share one idea: pair the
detector with a restoration decoder and optimise both, so degraded features are
pulled towards their clean counterparts by a term the detector cannot ignore.
MTW-DETR states it plainly — a pixel-level restoration subnetwork with a joint
loss, restoration and detection trained together.

**Why it fits us better than it fits them.** Those papers need synthetic pairs
or a separate clean dataset. Our corpus *is* the pair: `clear_00042`,
`fog2_00042`, `night_00042` are one scene, and the corpus audit verified all
three carry identical labels. So we get the supervision for free, at feature
level:

```
L_restore = SmoothL1( block_out(f_degraded), block_out(f_clear_twin).detach() )
```

**Status: implemented** (`src/models/paired.py`, `restore.lambda: 1.0`). It also
attacks inertness directly — this is an objective the always-on branch *cannot*
absorb, because the term is computed on the block output that the experts are
the only conditional contributor to.

**Feature level, not pixel level, is the right choice** and FeatEnHancer (ICCV
2023, in `main-papers/`) is the evidence: its Table 5 shows image-level
enhancement collapsing detection from 32.8 to ~7.5 mAP while feature-level
enhancement raised it to 34.6. Restoring pixels optimises for human viewing;
restoring features optimises for the detector.

**Weights to try:** `lambda ∈ {0.5, 1.0, 2.0}`. Currently 1.0. Watch for the
failure mode where a large lambda makes every expert an identity map towards the
clear features, which would *reduce* specialisation.

---

## 2. AW-MoE — the paper that describes our failure, and fixes it structurally

`other-papers/AW-MoE_ All-Weather Mixture of Experts...pdf` (Lin et al.). Same
problem statement as ours: weather-specific experts, a routing module, a shared
trunk. Their Algorithm 1 differs from ours in four ways, and **all four target
inertness**:

| | AW-MoE | ours |
|---|---|---|
| expert init | **copy a fully pretrained expert into all branches** | zero-init |
| shared trunk during MoE training | **frozen** | trained jointly |
| router | trained **separately**, stage 2 | joint, from step 0 |
| expert output | its **own head and own full detection loss** | a residual added to a shared sum |

The last row is the important one. In AW-MoE each expert produces its own
predictions and pays its own loss:

```
L_CW = Σ_{w ∈ S}  P_w · L_w(WSE_w)          (their Eq. 8)
```

— the **Confidence-Weighted MoE Loss**, each expert's loss scaled by its routing
probability. An expert cannot be ignored, because nothing else computes its
output. In our design the shared branch computes a perfectly good output and the
experts are optional additions; that is precisely the difference between a
mixture and a residual.

They report ~15% improvement in adverse weather at "negligible inference
overhead".

**What to take.** Two things, in order of cost:

- **cheap:** initialise the experts from a trained branch instead of from zero.
  Our own measurement says why this matters — at exactly zero,
  `d(out)/d(p_i) = expert_i(x) = 0`, so the gate receives *no gradient at all*
  from the detection loss (measured: 0.0).
- **expensive but principled:** give each expert its own detection head and use
  the confidence-weighted loss. This is a real architecture change and it makes
  the experts non-optional by construction.

---

## 3. Orthogonality and variance losses — the direct anti-collapse pair

*Advancing Expert Specialization for Better MoE* (NeurIPS 2025, arXiv 2505.22323)
adds two terms to the usual objective:

```
L = L_task + α·L_aux + β·L_o + γ·L_v
```

- **Orthogonality loss `L_o`** — penalises similarity between the outputs of
  different experts *on the same input*. Exactly our CKA/cosine problem, stated
  as an objective instead of a diagnostic.
- **Variance loss `L_v`** — rewards routing scores that spread out:
  `L_v = −Σ_i Σ_j (1/n)·(s_ij − s̄_j)²`, i.e. *maximise* the variance of each
  expert's routing score across the batch, making routing decisions more
  discriminative.

**Read against our numbers.** `L_v` is aimed at a problem we no longer have —
`cond3d` routes at NMI 0.8748 with 0.95–0.99 purity. `L_o` is aimed at one we
demonstrably do have. So: **implement `L_o`, skip `L_v`.**

One caveat worth recording: a 2026 analysis
([Geometric Regularization in MoE](https://arxiv.org/pdf/2601.00457)) finds
orthogonality regularisation can *increase* weight-space overlap by up to 114%
with inconsistent effects on loss. So `L_o` should be applied to **outputs**, not
weights, and it should be measured by the intervention rather than by CKA.

**Weights to try:** `β ∈ {0.05, 0.1, 0.5}`. Start low — an orthogonality term
strong enough to dominate will push experts apart in ways that have nothing to
do with the conditions.

---

## 4. Heterogeneous experts and expert groups

`other-papers/Heterogeneous Mixture of Experts.pdf` (Chen et al., RS
super-resolution). Experts are organised into **groups**: homogeneous within a
group, heterogeneous across groups, plus a **multi-level feature aggregation**
to guide routing and a **dual-routing** mechanism selecting per pixel.

Two transferable ideas:

- **Multi-level features guide the router.** Our gate sees only a
  global-average-pooled P3 vector — 771 parameters reading one scale. AW-MoE
  makes the same point from the other direction, routing on *image* features
  because they are "invariant to scene variations". Both suggest the router
  should look at something other than the feature it is gating.
- **Per-pixel routing.** Ours is per-image. Fog is rarely uniform across a
  scene — our own synthesis uses a smooth pseudo-depth field, so transmission
  varies spatially. A spatial gate is a natural extension, and it is the one
  place where "more routing" might still be worth something.

---

## 5. What NOT to add, with our own evidence

| candidate | why not |
|---|---|
| load balancing (Switch, CV, z-loss) | constrains usage across a **batch**; our failure is per-image and per-branch. Measured: activation is already 0.33/0.30/0.36 — balanced and useless. |
| entropy / noisy top-k | design 1 had both; NMI was 0.0000. Supervision replaced them. |
| routing/count cost | tried (`cond3c`). Improved every routing metric, made accuracy worse. It treated a symptom of the mosaic label bug. |
| NWD box loss | tried twice, once with a units bug and once without. Both traded small-object recall for aggregate mAP. |
| variance loss `L_v` | aimed at indecisive routing; ours is already at NMI 0.875. |

---

## Ranked plan

1. **Restoration loss** (done) — the only term that gives an expert an objective
   the bypass cannot absorb. Sweep `lambda ∈ {0.5, 1, 2}`.
2. **Non-zero expert init** (done, `init_gain: 0.1`) — AW-MoE copies a trained
   branch; we at least start off zero, which restores the gate's gradient from
   the detection loss.
3. **Orthogonality loss `L_o` on outputs**, `β ≈ 0.1`. Cheap, and the one
   remaining term aimed squarely at a measured problem.
4. **Per-expert heads + confidence-weighted loss** (AW-MoE Eq. 8). The real fix
   for inertness, and a genuine architecture change — worth doing only if 1–3
   leave the intervention spread flat.
5. **Spatial (per-pixel) gating** — last, and only once an expert demonstrably
   matters somewhere.

Judge every one of these with `scripts/expert_intervention.py`, not with CKA.
The representation metrics disagree with each other (CKA 0.82–0.98 versus cosine
+0.02–0.06 on the same tensors) and neither answers the question that matters.

---

## Sources

- [AW-MoE: All-Weather Mixture of Experts for Robust Multi-Modal 3D Object Detection](https://arxiv.org/abs/2508.06452) — local copy in `tez/other-papers/`
- [Heterogeneous Mixture of Experts for Remote Sensing Image Super-Resolution](https://arxiv.org/abs/2502.09654) — local copy in `tez/other-papers/`
- [Advancing Expert Specialization for Better MoE](https://arxiv.org/pdf/2505.22323) (NeurIPS 2025)
- [Geometric Regularization in Mixture-of-Experts: The Disconnect Between Weights and Activations](https://arxiv.org/pdf/2601.00457)
- [A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications](https://arxiv.org/html/2503.07137v1)
- [FriendNet: Detection-Friendly Dehazing Network](https://arxiv.org/pdf/2403.04443)
- [MTW-DETR: multi-task collaborative optimization for adverse weather object detection](https://www.sciencedirect.com/science/article/abs/pii/S0167865525003459)
- [Degradation Type-Aware Image Restoration for Object Detection in Adverse Weather](https://pmc.ncbi.nlm.nih.gov/articles/PMC11478636/)
- [FeatEnHancer: Enhancing Hierarchical Features for Object Detection and Beyond Under Low-Light Vision](https://openaccess.thecvf.com/content/ICCV2023/papers/Hashmi_FeatEnHancer_Enhancing_Hierarchical_Features_for_Object_Detection_and_Beyond_Under_ICCV_2023_paper.pdf) — local copy in `tez/main-papers/`
