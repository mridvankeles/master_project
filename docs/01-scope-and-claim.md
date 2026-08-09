# 01 — Scope and Claim

> **Status:** skeleton
> **Sources:** papers-datasets__2_.xlsx, chat 2026-07-28
> **Open questions:** 6  |  **Conflicts:** 0

## Working title

Robust Remote Sensing Object Detection in Adverse Conditions via Task-Specific Mixture of Experts
`[from: chat 2026-07-28]`

## Task set (DECIDED 2026-07-28)

Three conditions, one expert each, on a single source corpus:

1. **Clear** (control)
2. **Fog**
3. **Low illumination**

Tiny objects **dropped as an expert** — scale is handled in the neck/FPN, per the
routing-granularity argument below. Note the dimension is not lost: at DOTA's ground sampling
distances, `small vehicle` and `ship` instances are small objects already.

Superseded: an earlier plan listed low illumination, fog, and tiny objects as the three tasks,
with two experts and no clear route `[from: chat 2026-07-28]`.

## Declared roadmap

`[from: chat 2026-07-28]` — preserved as written:

- First confirm the architecture works, by (a) examining the experts as latent space and (b) increase of inference speed.
- After confirmation, possible extensions: more experts, more losses, static and dynamic experts, and other strategies.

## Hardware

- Personal: RTX 5070 Ti (16 GB VRAM)
- Work: 2× A6000 (48 GB each), available for long runs
`[from: chat 2026-07-28]`

## Timeline

**One year to completion**, from 2026-07-28 `[from: chat 2026-07-28]`. Effectively ten working
months plus two for writing. See `05-experiment-plan.md` for the month-by-month plan and the two
go/no-go gates. This is a hard constraint on scope: one training corpus, one arm, three experts.

## [advisor's addition] Problem 3: domain independence is a property, not an expert

Raised as a possible fourth expert `[from: chat 2026-07-28]`. It cannot be one, for two reasons:

1. **Nothing can route to it.** A "domain-independent expert" would need the router to recognize
   "this input is from an unfamiliar domain" — which is out-of-distribution detection, the hard
   part of the problem. If you could do that reliably you would not need the expert.
2. **It pulls against the architecture.** Specialization and domain independence are in tension.
   The harder three experts specialize on one corpus, the *worse* the system behaves on an unseen
   domain, because routing an OOD input is the documented failure mode — the RS MoE survey warns
   that OOD samples get routed to experts never trained for that regime, yielding confident but
   incorrect predictions. Adding an expert increases specialization; it cannot confer generality.

**Ruling: domain independence is an evaluation axis, not a module.** It is measured by the
test-only sets in `03-datasets.md` (real fog, real night, cross-platform), and it is a *claim*
the thesis either supports or honestly fails to support.

**But the instinct redirects to something better.** The component that actually breaks under
domain shift is the **router**, not the experts. Making the router domain-robust — via
domain-randomized degradation parameters rather than one fixed simulator setting, photometric
augmentation, and frequency-domain normalization — is a real, cheap, defensible contribution, and
it belongs in the method chapter. See `04-method-open-questions.md` § Router domain robustness.

The honest limitation, to be written into the thesis rather than defended against: a
condition-routed MoE trained on one corpus is expected to degrade on unseen domains. Measuring
*how much*, and showing which component fails first, is the contribution.

Fog and low illumination are **input degradations** — properties of the whole image, caused by the atmosphere or the illumination conditions. They can be inferred from global image statistics.

Tiny-object detection is **not a degradation**. It is an intrinsic property of individual objects, present in perfectly clear imagery. Critically: a single DOTA or AI-TOD-v2 image contains tiny objects *and* large objects at the same time.

Consequence: an image-level router cannot route to a "tiny object expert," because "tiny" is not a property of the image. The three experts as currently defined live at **two different routing granularities**:

| Expert | Natural routing level |
|---|---|
| Fog | image / global |
| Low illumination | image / global |
| Tiny objects | region, instance, or FPN scale level |

This is the most important unresolved design issue in the thesis. It is not fatal — it has at least three clean resolutions (see `04-method-open-questions.md` § Routing granularity) — but it must be resolved before any code is written, because each resolution implies a different model.

## [advisor's addition] Problem 2: the inference-speed claim needs a named comparison

With 3 dense experts and top-1 image-level routing, parameter count roughly triples while FLOPs per image stay approximately constant relative to a single expert. Against a **single unified model of one expert's size**, that is FLOPs parity and 3× memory — not a speedup.

An MoE speed claim is only meaningful against a named alternative. Candidates:
- vs. a **cascade** (dehaze → detect, or enhance → detect): here MoE plausibly wins, since restoration preprocessing is skipped entirely.
- vs. a **dense model of equal parameter count**: MoE wins on FLOPs (only 1 of 3 experts activates).
- vs. a **dense model of equal FLOPs**: MoE has no speed advantage; the claim must be accuracy.
- vs. **running all three specialists and fusing**: MoE wins ~3×.

Pick one and state it in the claim. Confidence: certain on the arithmetic, since it follows directly from top-1 routing over equal-sized experts.

## Candidate claims (choose one, narrow it, defend it)

**C1 — Efficiency.** A task-routed MoE matches or beats per-condition specialist models at ~1/N the inference cost, and beats a restoration-then-detection cascade at lower latency.

**C2 — Interference.** Fog, low illumination, and small scale demand mutually interfering feature representations; a single dense detector trained on their union loses accuracy that expert separation recovers. *Requires showing the interference exists — see `05-experiment-plan.md`.*

**C3 — Router robustness.** Condition-routed MoE fails confidently when the router is wrong, and the router's input is degraded exactly in the conditions the router must classify. This thesis characterizes that failure mode for remote sensing detection and mitigates it.

**Advisor's assessment:** C2 is the strongest scientific claim but the most expensive to prove and the easiest to fail — if the dense union model matches the MoE, the thesis has no result. C3 is the most defensible and the least contested in the literature (see `02-related-work.md` § Routing failure). C1 alone is an engineering contribution and thin for a thesis unless paired.

A defensible combination: **C1 as the headline, C3 as the substantive analysis chapter.** C2 becomes an ablation that either supports the design or is reported honestly as a null result.

## Fallback claim

If the MoE does not beat the dense union baseline: the thesis still holds as a **characterization** — "condition-specialized routing does not improve degraded-condition RS detection under matched compute, and here is why, with router-failure analysis." Negative results are publishable in this specific form. Plan for this now so the thesis has a floor.

## Scope boundaries (to be written explicitly into the proposal)

- **In:** optical remote sensing / aerial imagery; fog and cloud degradation; low illumination; small and tiny objects.
- **Out (proposed):** SAR (HRSID is in the dataset list but no SAR task is declared); 3D / LiDAR / radar detection; ground-level photography.
- **Undecided:** whether drone-view (VisDrone, DroneVehicle, HazyDet) and satellite-view (DOTA, DIOR, xView) are treated as one domain or two. This is not cosmetic — see `03-datasets.md` § Platform confound.

## Loose ends

- "static and dynamic experts" `[from: chat 2026-07-28]` — the intended meaning is not recorded. Static = frozen pretrained specialists vs. dynamic = jointly trained? Or static = fixed expert count vs. dynamic = experts instantiated on drift detection? Needs the user's definition.
- "examining experts as latent space" `[from: chat 2026-07-28]` — the intended analysis is not specified. Concrete instrumentation proposed in `04-method-open-questions.md` § Latent-space probes.
- Program, institution, submission deadline, expected thesis length, and committee composition are still unrecorded.
- No baseline detector has been chosen (one-stage vs. two-stage, OBB vs. HBB head).

## Open questions

1. Which of C1/C2/C3 is the headline claim?
2. Which resolution of the routing-granularity problem?
3. Is "tiny objects" an expert, or a neck/FPN design choice handled outside the MoE?
4. What is the speed claim's named comparison?
5. Is drone-view in scope alongside satellite-view?
6. What does "static and dynamic experts" mean?
