# 04 — Method: Open Design Questions

> **Status:** skeleton
> **Sources:** chat 2026-07-28, advisor analysis 2026-07-28
> **Open questions:** 7  |  **Conflicts:** 0

Nothing in this file is decided. It exists so the decisions get made deliberately rather than
by whatever the first working script happens to do.

## Routing granularity — the blocking decision

Restating the problem from `01-scope-and-claim.md`: fog and low illumination are image-level
properties; tiny-object-ness is a per-instance property. One image-level gate cannot route
three experts when one of them is defined at a different level.

### Resolution A — Drop tiny objects from the expert set

Two degradation experts (fog, low light) plus a clear-weather expert. Small-object handling
moves into the neck/FPN and the label-assignment strategy, where the tiny-object literature
already puts it.

- *For:* Clean, honest, matches how routing actually works. Two condition experts plus a
  baseline expert is a coherent MoE.
- *Against:* "Task-specific" then means only degradation-specific, and the tiny-object
  contribution disappears from the novelty story.
- *Advisor's view:* the most defensible and the cheapest. Small-object detection is a scale
  problem, and MoE is not the natural tool for a scale problem — feature pyramids are.

### Resolution B — Hierarchical / two-level routing

An image-level degradation gate selects a degradation expert; within the detector, a scale-level
or region-level gate selects a scale expert.

- *For:* Keeps all three tasks. Genuinely more novel than a flat 3-way gate.
- *Against:* Two gates, two auxiliary losses, two collapse modes, a combinatorial ablation
  matrix. Substantially more engineering and more ways to fail.
- *Advisor's view:* the most interesting thesis and the highest risk. Only choose this if
  Resolution A's pipeline is already working, i.e. as a second-phase extension.

### Resolution C — Degradation as conditioning, not routing

Route per-token or per-region over a homogeneous expert pool, and inject the degradation
estimate as a *conditioning signal* (FiLM-style modulation) rather than as a routing key.

- *For:* Sidesteps the granularity mismatch entirely. Closer to how modern sparse MoE actually
  works, so the "this is a 1991-style mixture of specialists, not a modern MoE" objection
  disappears.
- *Against:* Experts are no longer interpretable as "the fog expert," which directly undercuts
  the declared plan to examine experts in latent space.
- *Advisor's view:* technically strongest, but it conflicts with your own stated analysis goal.

**Recommendation: A for the core thesis, B as the declared future extension.** That also matches
your own roadmap — confirm the architecture works first, then add complexity.

## The "is this a modern MoE?" objection

With 3 dense experts and top-1 image-level routing, the architecture is closer to a classical
mixture of specialists with a gating network than to a modern sparse MoE layer. An examiner
familiar with the MoE literature will say so, and the response cannot be defensive.

Two honest answers, pick one:
1. **Own it.** The contribution is not the MoE mechanism; it is condition-specialized
   conditional computation for RS detection, and the classical formulation is appropriate at
   N=3. State this in the method chapter rather than letting it be discovered.
2. **Move to token/region-level routing** (Resolution C), which makes it a modern sparse MoE
   and makes the sparsity question meaningful.

Either is defensible. Being caught without a position is not.

## Router design

### Supervision
- **Supervised gate.** Free labels on synthetic data (you know what you added). Trains fast,
  routes accurately, and is honest. Problem: no labels on real degraded data, so router
  performance on RDDTS is unknown until measured.
- **End-to-end learned gate.** No labels needed; risks collapse and may learn the platform
  confound (`03-datasets.md` § Risk 2) instead of the condition.
- **Hybrid.** Supervised pretraining on synthetic, then end-to-end fine-tuning. Probably the
  right default.

### Auxiliary losses
Directly actionable from the literature `[advisor's addition]`: the switch/load-balance
auxiliary loss failed to prevent dying experts under adversarial training in sparse-MoE CNNs,
while an **entropy loss** kept multiple experts active. The same work found MoE placement
**deeper in the network** gave more meaningful routing and more stable specialization.
See `02-related-work.md` § Priority 3.

→ Default: entropy loss, MoE placed deep in the backbone or at the neck rather than early.
At N=3 collapse is less likely than at scale, but this costs nothing to guard against and
"we monitored expert utilization and used an entropy regularizer" is a one-sentence answer to
a question you will otherwise fumble.

### Hard vs. soft routing
Top-1 hard routing gives the FLOPs argument (only one expert activates). Soft routing may be
more accurate on ambiguous inputs — twilight fog, thin cloud — but forfeits the efficiency claim
and introduces a calibration risk: a soft-routed MoE can be miscalibrated under distribution
shift even when every individual expert is well calibrated `[advisor's addition, see 02 § Priority 3]`.

**Compound conditions are the real test here.** Fog *and* low illumination together, or thin
cloud over tiny objects, is where a 3-way hard gate has no correct answer. This is the interesting
failure case and should be an explicit experiment, not an afterthought.

## Latent-space analysis — making the declared plan concrete

You wrote that you would confirm the architecture by "examining experts as latent space"
`[from: chat 2026-07-28]`. Concrete instrumentation, so this becomes a measurable result rather
than a figure:

1. **Expert utilization histograms** per condition. Does the fog expert actually receive the
   foggy images? Any dead experts?
2. **Router confusion matrix** on held-out synthetic, then on real (RDDTS). The gap between the
   two is a headline number.
3. **Oracle routing upper bound.** Force ground-truth routing; measure the gap to learned
   routing. This separates "the experts are weak" from "the router is weak" and it is the single
   most informative ablation you can run.
4. **Inter-expert feature similarity** (CKA or similar) between expert representations. If experts
   converge to near-identical features, specialization did not happen and the architecture is
   redundant — a real finding, positive or negative.
5. **t-SNE/UMAP of features by expert assignment and by true condition.** The qualitative figure.
   Fine as illustration; do not let it carry an argument alone.
6. **Router failure severity curve.** Detection mAP as a function of router accuracy. Answers
   "what happens when routing is wrong," which is committee question #2.

## Inference-speed measurement protocol

Per `01-scope-and-claim.md`, the speed claim needs a named baseline. Whatever is chosen, report:
throughput (img/s) and latency (ms/img) at fixed batch size and input resolution, **plus** FLOPs
and parameter count, on a single named GPU, with the router cost included. Router cost is often
omitted and its omission is easy for a reviewer to spot.

## Framework decision (DECIDED 2026-08-01)

**Ultralytics YOLO11 (HBB detect), installed as `git clone` + `pip install -e .`.**

This is an engineering choice, not a research decision, and needs no defense in the thesis. Record
it in the implementation chapter as a one-line statement of what was used. Reasons, for our own
reference: horizontal boxes remove the reason most RS work uses MMRotate; Ultralytics supports
custom modules officially (define the block, expose it in `ultralytics/nn/modules/__init__.py`, add
a branch in `parse_model()`, reference from a model YAML; custom losses via a `DetectionTrainer`
subclass passed as `trainer=`); and it runs on current PyTorch/CUDA without version conflicts.

### The real issue: comparability with published numbers

Separate from the framework, and it does need handling. **NIRNet published results on Hazy-DIOR** —
the dataset this thesis trains on. Readers will compare directly. Quoting published numbers is
normal practice, but only valid if the protocol matches. Requirements on the data pipeline:

- **Same splits** — use the division shipped with the Hazy-DIOR release; do not re-split.
- **Same metric** — DIOR convention is typically mAP@0.5. Ultralytics reports mAP50 and mAP50-95.
  Report whichever matches the source and state which.
- **Same class set** — all 20, no dropping rare classes.

If any of these diverge, the number is not comparable and the thesis must say so rather than let a
table imply otherwise. This applies to DHC-Net and CM-YOLO too, whose numbers can be quoted but not
re-run.

### Consequence for the architecture

Ultralytics builds the network as a sequential list with a save-index. An MoE *block* inside one
shared network (expert branches at a neck or backbone stage, gate computed from the incoming
feature map) fits naturally — one custom module, one YAML line. Three *separate* full expert
detectors with an image-level router does not fit: it bypasses the YAML system entirely, costs 3×
memory, and is the reading an examiner calls "an ensemble with a switch."
**Recommendation: experts as blocks inside one shared network.** Still formally open — see open
question 2.

## Router domain robustness

`[advisor's addition, from: chat 2026-07-28]` — this replaces the "domain-independent expert"
idea; see `01-scope-and-claim.md` § Problem 3 for why that could not be a module.

The router is the component that breaks under domain shift, so it is the component to harden.
Three cheap, citable measures:

1. **Randomize the degradation parameters per sample** — fog density and depth assumptions, and
   the low-light exposure/noise parameters. A router trained on one fixed simulator setting learns
   that setting. A router trained across a distribution of settings has to learn the degradation.
   This is the single highest-value line of code in the project.
2. **Photometric augmentation** decoupled from the condition labels, so brightness and contrast
   alone cannot identify the condition.
3. **Frequency-domain normalization** on the router's input — fog and low light have distinct
   spectral signatures that survive across platforms better than raw intensity statistics.

Measure the effect directly: router confusion matrix on synthetic → on real (RDDTS, DroneVehicle
night) → on cross-platform (Hazy-DIOR). Three numbers, one table, and it is the core result of
the analysis chapter.

## Open questions

1. Resolution A, B, or C for routing granularity?
2. Classical mixture-of-specialists framing, or move to token-level sparse MoE?
3. Router supervision: supervised, end-to-end, or hybrid?
4. Hard top-1 or soft routing?
5. What happens on compound degradations, and is that an experiment or a limitation?
6. Which detector is the backbone/base? (Nothing chosen yet; must be OBB-capable if OBB is the
   chosen box representation — see `03-datasets.md` § Risk 3.)
7. "Static and dynamic experts" — definition still needed from the user.
