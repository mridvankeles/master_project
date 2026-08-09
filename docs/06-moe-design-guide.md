# 06 — MoE Design Guide

> **Status:** reference
> **Sources:** Cui, Liu & Chen (2026) *Mixture-of-Experts in Remote Sensing: A Survey*, JGEO 1(1):4–38;
> Chen et al. (2025) *Heterogeneous Mixture of Experts for RS Image Super-Resolution* (MFG-HMoE), IEEE GRSL 22
> **Open questions:** 3 | **Conflicts:** 0

Bracketed numbers `[42]`, `[122]` etc. are the survey's own reference numbers, kept so they can be
resolved against its bibliography.

---

## 1. Why add MoE — the case, stated fairly

Four arguments, in descending order of how well they apply to this thesis.

**1.1 Conditional computation.** Sparsely-gated layers decouple total parameter count from
per-example compute: only the top-k experts activate per input. Capacity grows without FLOPs
growing proportionally. This is the argument the efficiency claim in `01-scope-and-claim.md` rests
on, and it is the strongest one available here.

**1.2 Reduced interference between regimes.** A dense network trained on a heterogeneous union must
find one parameter set that serves all conditions. MoE allocates different parameter subsets to
different regions of the input space. M³ViT [30] uses this explicitly to reduce inter-task
interference; Mod-Squad [32] treats experts as reusable modules shared or specialized across tasks.
For this thesis, the claim would be that fog, low illumination and clear imagery demand mutually
interfering features. **That interference must be demonstrated, not assumed** — see claim C2.

**1.3 Handling RS heterogeneity.** The survey's framing: RS data vary in sensor physics, acquisition
geometry, spatial scale, seasonality and regional domain shift, which provides a plausible basis for
conditional specialization.

**1.4 Interpretability via routing.** Routing decisions are a direct handle on model inspection —
expert-assignment maps plus usage statistics can reveal whether specialization aligns with
meaningful regimes. This is the mechanism behind the "examine experts as latent space" goal.
Caveat the survey attaches: routing-based interpretability is not guaranteed; routers may respond
to correlations unrelated to the task, and routing patterns can shift across retraining.

### 1.5 The counter-case — read this before designing anything

From the survey's own Discussion (§4.1), paraphrased:

- A rigorous assessment of MoE requires comparison to simpler alternatives **under matched capacity
  and matched compute**, because a substantial portion of reported gains can also be obtained by
  scaling a dense backbone, adding explicit multi-branch or multi-scale pathways, or using ensembles
  at inference.
- For object-centric RS tasks specifically, multi-expert detection and proposal mechanisms often
  **resemble structured multi-branch designs** that already encourage specialization across scales.
- **"MoE is most appropriate when the regimes that benefit from specialization are difficult to
  predefine."**

That last sentence is the one to sit with. This thesis predefines its regimes — fog, dark, clear —
which is precisely the setting where the survey says the MoE motivation is *weakest*. This does not
kill the project, but it dictates the framing: the contribution cannot be "MoE helps." It has to be
either the efficiency argument (§1.1, measured against a named baseline) or the router-failure
analysis (claim C3), and the thesis must state why predefined regimes are still worth routing over.

Also from §4.1, both binding on `05-experiment-plan.md`:
- MoE models can show **higher run-to-run variance** than comparable dense models, because routing
  stochasticity and uneven expert training add instability. Repeated runs with confidence intervals
  are therefore not optional.
- **Random splits in geospatial data can be overly optimistic due to spatial autocorrelation.**
  Spatially or environmentally separated validation is better aligned with deployment and reduces
  leakage between folds. → For DIOR this means checking whether train/test share geography, and
  reporting that they don't (or that they do).

---

## 2. Mechanics — how MoE is actually built in vision models

### 2.1 The core equation

Given experts `E_1..E_n` and gate `G(x)` producing weights `g_1..g_n`:

```
y = Σ_i g_i(x) · E_i(x)          # dense/soft: all experts active, Σg_i = 1
y = Σ_{i ∈ S_k(x)} g_i(x)·E_i(x)  # sparse: g_i = 0 for i ∉ S_k(x)
```

Top-1 (`k=1`) is the cheapest and gives the clean FLOPs argument. Top-2 is the common default in
LLM work. Soft routing is more accurate on ambiguous inputs but forfeits the efficiency claim.

### 2.2 Where to put the MoE layer — published placements

| Placement | Example | Notes |
|---|---|---|
| Feed-forward sublayer of every Transformer block | Switch [27], V-MoE [54], Swin2-MoSE [140] | The standard modern position |
| After each backbone block | M²CD [43] | Gate selects top-k by similarity between input features and expert embeddings |
| Grid cells of the backbone feature map | SM3Det [122] | Dispatcher splits feature map into grids, routes each, collector reassembles |
| Each level of the feature pyramid | SAFPN [119] | A group of experts per FPN resolution, gate fuses across levels |
| Upsampling / reconstruction head only | MFG-HMoE [42] | Backbone left untouched — cheapest possible integration |
| LoRA adapters inside a frozen backbone | Land-MoE [107], MSLoRA-Net [109], MaMOL [118] | Parameter-efficient; router picks which adapter to apply |

**Relevant empirical note (external to this survey):** adversarial-training work on sparse-MoE CNNs
found deeper placement (second residual block) gave more meaningful routing and more stable
specialization than early placement. Consistent with the table above — most successful placements
are mid-to-late.

### 2.3 Routing granularity — the axis that defines the architecture

| Granularity | What the gate sees | Example |
|---|---|---|
| **Per-image / per-example** | One routing decision for the whole image | HotMoE [127] — hard routing, exactly one ViT expert per inference; Swin2-MoSE [140] — all tokens of an example go to the same experts |
| **Per-grid / per-region** | Local feature blocks routed independently | SM3Det [122] |
| **Per-token / per-patch** | Standard sparse-MoE granularity | Switch [27], Mensah et al. [112] |
| **Per-pixel** | Finest; each spatial location routed | MFG-HMoE [42] |

Image-level routing is what a *degradation* expert set implies, since fog and illumination are
global image properties. Note that HotMoE [127] is the closest published precedent for that
choice — hard routing to one expert per inference — and it reports 43.7 FPS with AUC 0.704 on
HOT2022, i.e. the efficiency argument working in practice.

### 2.4 Gating strategies

- **Softmax linear gate** on pooled features. The default. `p_i = exp(W_i·x) / Σ exp(W_j·x)`.
- **Noisy top-k** — noise added to logits before selection, to encourage exploration and balance.
- **Expert-choice routing** [80] — inverts the usual direction: experts pick a fixed number of
  tokens rather than tokens picking experts. Better load balance by construction.
- **Hash / static routing** [66, 67] — deterministic assignment, no learned gate. BASE layers show
  fixed random assignment can still work with the right regularization. **A cheap and brutal
  ablation: if hash routing matches your learned router, your router learned nothing.**
- **Dual / hierarchical routing** [42] — first select expert *group*, then expert within group.
- **Adaptive-k** [81] — more experts for harder examples.
- **Metadata-conditioned routing** — MAPEX [131] routes on learnable modality embeddings;
  MoE-MAE [128] encodes lat/lon/week-of-year/hour-of-day as sinusoidal pairs and feeds them as
  router input. **Directly relevant:** acquisition metadata is a legitimate router input, so if
  DIOR carried acquisition time this would be a free signal. Check whether it does.

### 2.5 Auxiliary losses — the collapse problem

The failure mode: the gate collapses onto a subset of experts, the rest die, and the MoE degenerates
into a dense model with wasted parameters.

**Load-balancing loss** (the survey's formulation). With `u_i = E_{x∼B}[g_i(x)]` the expected routing
importance of expert `i` over minibatch `B`:

```
L_balance = CV(u) = sqrt(Var_i(u)) / E_i[u]
```

The squared coefficient of variation is often used instead, for training stability. Add to the task
loss with a small weight.

Other options worth knowing:
- **Switch auxiliary loss** [27] — penalizes uneven usage. Note: external work found it *failed* to
  prevent dying experts under adversarial training, where an entropy loss succeeded.
- **z-loss** (ST-MoE [79]) — router logit stability.
- **SimSMoE** [88] — explicitly minimizes similarity between expert *representations* to attack
  representational collapse. Relevant to the CKA analysis in `04-method-open-questions.md`: if
  experts converge to near-identical features, this is the loss that targets it.
- **Auxiliary-loss-free balancing** [85] — maintains per-expert bias terms updated from recent
  routing statistics, so loads balance without injecting extra gradients into the main objective.
- **Expert dropout** — forcing robustness to missing experts. Named in the survey's future
  directions; a plausible cheap addition to claim C3's robustness story.

The survey also notes small changes in gating losses, capacity factors and noise can strongly affect
training stability and final performance — so these are hyperparameters to log, not defaults to
ignore.

### 2.6 Numbers worth stealing from MFG-HMoE

The one paper here with a full ablation table, on ×4 SR, UCMerced, single RTX 4090, 100k iterations:

| Finding | Evidence |
|---|---|
| MoE helps vs. single layer | 1 expert 29.10 dB → 16 experts 29.19 dB |
| Expert count has an optimum, not a monotone trend | 8 → 29.26, 16 → 29.26, **32 → 29.21 (worse)** |
| **Receptive-field heterogeneity dominates everything else** | all 1×1: **27.78** / all 3×3: **29.26** / all 5×5: **27.98** |
| Mixing kernel sizes beats uniform | 8×(1×1) + 8×(3×3) → 29.29, the best configuration |
| Dual routing adds a little | 29.26 → 29.29 |
| Config used | N=2 groups, M=8 per group, **K=1** |

Three lessons for this thesis: (a) **K=1 is a published, working choice**; (b) more experts is not
better — 32 was worse than 8, so a 3-expert design is not obviously under-scaled; (c) the largest
effect in the whole table came from *heterogeneity in kernel size*, not from the routing mechanism —
a 1.5 dB swing versus 0.03 dB for dual routing. Expert *architecture* may matter more than expert
*assignment*.

---

## 3. Implementation in Ultralytics

Per `04-method-open-questions.md`, experts live as a block inside one shared network.

### 3.1 The module

```python
# ultralytics/nn/modules/moe.py
import torch, torch.nn as nn

class MoEBlock(nn.Module):
    """Top-1 routed MoE with heterogeneous kernel sizes + one always-on shared expert."""
    def __init__(self, c1, c2, kernels=(1, 3, 5), shared=True):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c1, c2, k, padding=k // 2, bias=False),
                          nn.BatchNorm2d(c2), nn.SiLU())
            for k in kernels
        ])
        self.shared = (nn.Sequential(nn.Conv2d(c1, c2, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(c2), nn.SiLU()) if shared else None)
        self.gate = nn.Linear(c1, len(kernels))
        self.last_logits = None          # stashed for the auxiliary loss

    def forward(self, x):
        logits = self.gate(x.mean((2, 3)))          # image-level gate
        self.last_logits = logits                    # loss reaches in via named_modules()
        idx = logits.argmax(1)                       # top-1, hard
        w = logits.softmax(1)
        out = x.new_zeros(x.shape[0], self.experts[0][0].out_channels, *x.shape[2:])
        for i, e in enumerate(self.experts):         # loop, not batched — see 3.4
            m = idx == i
            if m.any():
                out[m] = e(x[m]) * w[m, i].view(-1, 1, 1, 1)
        return out + self.shared(x) if self.shared is not None else out
```

### 3.2 Registration

1. Export `MoEBlock` from `ultralytics/nn/modules/__init__.py`.
2. Add a branch in `parse_model()` in `ultralytics/nn/tasks.py` reading input channels from `ch[f]`
   and output from `args[0]`.
3. Reference it by name in a model YAML.
4. **Verify with `model.info()` that parameter count changed.** The most common failure reported by
   others is editing the pip-installed copy instead of an editable clone, so the original
   architecture silently keeps training.

### 3.3 The auxiliary loss

```python
def routing_aux(model, mode="cv"):
    logits = [m.last_logits for m in model.modules()
              if hasattr(m, "last_logits") and m.last_logits is not None]
    if not logits:
        return 0.0
    g = torch.cat(logits).softmax(-1)
    u = g.mean(0)                                     # per-expert importance over the batch
    if mode == "cv":
        return (u.std() / (u.mean() + 1e-6)) ** 2     # squared CV, §2.5
    return -(-(g * g.clamp_min(1e-9).log()).sum(-1)).mean()   # entropy alternative
```

Wire it in by subclassing `DetectionTrainer`, overriding the loss computation to add
`λ · routing_aux(self.model)`, and passing `trainer=MyTrainer` to `model.train()`. Start with
λ ≈ 0.01 and **log expert utilization every epoch** — a utilization histogram is the only way to see
collapse happening.

### 3.4 Efficiency warning

The per-expert loop above gives real FLOPs savings only when the batch is homogeneous. With mixed
conditions in a batch, each expert runs on a sub-batch and wall-clock gain largely disappears even
though FLOPs drop. The survey makes the general version of this point: routing introduces dispatch
and combination overhead, and accuracy comparisons are only interpretable alongside throughput, peak
memory, and wall-clock cost under consistent hardware.

**Consequence for the speed claim: report FLOPs *and* measured latency, and state batch composition.**
A condition-sorted batch and a mixed batch will give different numbers, and the honest one is mixed.

---

## 4. Expert types — the design space for fog / dark / clear

Eight candidates. Assessment is for this thesis's specific constraints: one year, 16 GB personal
card, 2× A6000, DIOR-derived data, HBB detection.

### 4.1 Degradation-type experts — *the current plan*
One expert per condition; image-level gate classifies the condition.
Precedent: **PhyDAE [141]** does exactly this for restoration (dehazing / denoising / deblurring /
low-light experts, gate routes each degraded image to the best-suited expert); **AW-MoE** for 3D
detection.
*For:* interpretable, matches the "examine experts" goal, condition labels free on synthetic data.
*Against:* regimes are predefined (§1.5); a 3-way gate has no correct answer for compound
degradation; router error is a single point of failure. **Assessment: viable as the main design, but
it is the design most exposed to the survey's §4.1 critique.**

### 4.2 Heterogeneous receptive-field experts
Experts differ by convolution kernel size (1×1 / 3×3 / 5×5), not by semantic condition.
Precedent: **MFG-HMoE [42]** — and per §2.6, this was the single largest effect in its entire
ablation.
*For:* no condition labels needed at all; cheap; the strongest empirical evidence in the sources.
*Against:* experts are no longer nameable as "the fog expert," which weakens the interpretability
story. **Assessment: adopt as a component *inside* whichever design wins. Nearly free, and the
evidence says it matters more than routing cleverness.**

### 4.3 Frequency-band experts
Split into low- and high-frequency paths; low-frequency handles global haze removal, high-frequency
handles detail reconstruction; a mixture of fusion experts recombines.
Precedent: **Shen et al. [142]**, spatial-frequency adaptive RS dehazing with MoE — mixture of
modulation experts in the spatial domain plus a decoupled frequency learning block. Also He et
al. [143] for pan-sharpening (separate low-freq and high-freq MoE components).
*For:* physically motivated — haze is a low-frequency multiplicative effect, so the decomposition
matches the degradation. Connects to the frequency-normalization idea already in
`04-method-open-questions.md` § Router domain robustness.
*Against:* a second published RS-dehazing MoE to differentiate from.
**Assessment: the most physically principled option, and underexplored for detection rather than
restoration.**

### 4.4 Physics-guided experts
Experts built around explicit physical models: atmospheric scattering (ASM) for haze, Retinex for
low-light, sensor noise models and PSFs as constraints.
Precedent: **PhyDAE [141]**.
*For:* strong narrative, gives the router something real to key on, and the synthesis pipeline
already uses ASM.
*Against:* **this is PhyDAE's contribution.** Adopting it wholesale for detection is a port, not a
contribution, unless the detection coupling is the novelty. **Assessment: read PhyDAE before
choosing this.**

### 4.5 Scale / FPN-level experts
A group of experts per pyramid level; gate fuses across levels.
Precedent: **SAFPN [119]** — reports detection mAP 71.3 → 82.7 and instance segmentation 62.4 → 71.1
on Airbus Ship.
*For:* the largest reported detection gain in the survey; orthogonal to degradation routing, so it
could compose with it.
*Against:* those gains are on scale, not degradation — so a reviewer may ask whether your gain is
just SAFPN's gain. **Assessment: this is the tiny-object expert idea from the dropped task set,
resurfacing in its proper form. Keep as future work, not phase one.**

### 4.6 Loss-clustered experts — *the dangerous one*
Assign training samples to experts by clustering on loss distances, unsupervised, no manual expert
labels.
Precedent: **MEDNet [123]** — multiple feature pyramids plus multiple detection experts with a loss
distance-based k-experts clustering (LD-kEC) strategy.
*Assessment:* **this is a required baseline, not an option.** MEDNet discovers its own expert
partition from data. If unsupervised loss-clustering matches or beats hand-designed
degradation experts, the central premise of this thesis — that fog/dark/clear is the right
partition — is falsified. Running it is how you find out before the committee does. It also gives a
strong answer if you win: "we compared against learned partitioning and hand-designed degradation
routing was better, here is why."

### 4.7 Shared expert + routed experts — *recommended*
One always-active expert captures what is common across all conditions; routed experts capture
condition-specific behaviour.
Precedent: **DeepSeekMoE [57]** (shared experts always active for general skills, routed experts for
domain-specific); **RingMoE [47]** (modal-specialized + collaborative + shared expert);
**MambaMoE [116]** (spectral *shared* expert module + spatial *routed* expert module);
**MaMOL [118]** (shared and modality-specific static experts + task-aware dynamic experts).
*For:* solves the "what happens on a clear image" problem elegantly — there is always a competent
path, so router error degrades performance instead of destroying it. That is a **mitigation for
claim C3's failure mode, built into the architecture.** Also reduces the number of
condition-specific experts needed, which matters on 16 GB.
**Assessment: strongest fit. One shared expert + fog + dark, with the shared path handling clear
imagery. Adopt §4.2's kernel heterogeneity within it.**

### 4.8 LoRA / adapter experts
Experts are low-rank adapters injected into a frozen backbone; router picks which adapter applies.
Precedent: **Land-MoE [107]** (frequency-aware mixture of low-rank token experts, for cross-sensor
and cross-geospatial domain shift), **MSLoRA-Net [109]**, **MaMOL [118]**, TT-LoRA MoE [76].
*For:* very cheap in memory — the 16 GB card becomes usable for real experiments. Land-MoE's target
problem is *domain generalization*, which is the axis in `01-scope-and-claim.md` § Problem 3.
*Against:* frozen backbone caps achievable accuracy; adds a pretrained-weights dependency.
**Assessment: the fallback if 3 full experts don't fit in memory. Also the natural vehicle if the
domain-generalization arm becomes a chapter.**

### 4.9 Recommended composition

> **One shared always-on expert + one fog expert + one low-light expert. Image-level top-1 hard
> routing over the two routed experts. Heterogeneous kernel sizes within the expert set. Squared-CV
> or entropy auxiliary loss, λ≈0.01, utilization logged every epoch. Placed at the neck.**

Rationale: shared expert removes the clear-condition gap and softens router failure (§4.7); top-1
preserves the efficiency claim (§2.3); kernel heterogeneity is the highest-evidence design choice
available (§2.6); K=1 with few experts is published practice, and 32 experts was *worse* than 8.

---

## 5. Reporting requirements this survey imposes

From §4.1 and §4.3, all now folded into `05-experiment-plan.md`:

1. **Matched capacity and matched compute** dense baselines. Non-negotiable.
2. **Total parameters *and* activated parameters per input**, reported separately.
3. **Throughput, peak memory, wall-clock training cost** alongside accuracy, on named hardware.
4. **Repeated runs with confidence intervals** — MoE variance exceeds dense variance.
5. **Expert utilization, load balance, assignment entropy, per-expert counts.**
6. **Scaling curves over expert count and top-k**, not a single configuration.
7. **Spatially separated validation**, not random splits, because spatial autocorrelation makes
   random geospatial splits optimistic.
8. **Full routing configuration specified** — capacity factors, top-k, auxiliary objectives, batching
   — since small changes shift both convergence and specialization.

---

## 6. Open questions

1. Shared-expert design adopted (§4.7), or pure degradation routing (§4.1)?
2. Is MEDNet [123] loss-clustering run as a baseline in month 6, or dropped with a stated reason?
3. Does DIOR carry acquisition metadata usable as router input (§2.4)?
