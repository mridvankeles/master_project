# Conditional gating: the router now specialises

> **Status:** three arms, single seed, 100 epochs each · 2026-08-13
> **Corpus:** `union3` (clear / fog / night, 1,954 each per epoch), identical
> hyperparameters across arms — only the block and the gate supervision differ.

The previous MoE failed both acceptance criteria: NMI(route ; condition) =
0.0000 and inter-expert CKA up to 0.996
(`finding-router-never-specialised.md`). This round tests the redesign, and
checks routing **before** accuracy.

---

## 1. Headline: routing works now

| Arm | gate supervised | expert priors | **NMI(argmax ; cond)** | inter-expert CKA |
|---|---|---|---:|---|
| `cond3_gated` | **yes** | heterogeneous static | **0.473** | **0.789 / 0.794 / 0.927** |
| `cond3_nogate` | no | heterogeneous static | **0.000** | 0.774 / 0.846 / 0.944 |
| `plain3_gated` | yes | identical | 0.485 | 0.936 / 0.970 / 0.938 |
| *(previous design)* | no | kernel size only | 0.000 | 0.968 – 0.996 |

Two clean, separable ablation results:

**Gate supervision is what creates routing.** NMI goes 0.000 → 0.473 when the
gate is trained against the condition labels, and the unsupervised arm collapses
to a single argmax route exactly as before. The entropy auxiliary never produced
routing because it only ever asked for *balance*; nothing told the gate what a
condition is. **This is the single change that made the mechanism work.**

**Static heterogeneous priors are what make the experts different.** With
identical priors the branches sit at CKA 0.936–0.970; with different fixed
priors they drop to 0.789–0.794. Same supervision, same everything else — the
priors are doing that. They cannot be trained away, which is the point.

Note the two factors are independent: supervision alone (`plain3_gated`) gives
routing over near-clone experts, which is routing without specialisation.
Both are needed.

---

## 2. Router confusion — and fog is the hard one

`cond3_gated`, 900 val images, argmax route against true condition:

| true ↓ / route → | clear | fog | night | purity |
|---|---:|---:|---:|---:|
| clear | **276** | 2 | 12 | 0.952 |
| fog | 164 | **137** | 13 | 0.522 |
| night | 34 | 4 | **258** | 0.872 |

Argmax accuracy 73.2% overall: **clear 93.5%, night 87.9%, fog 38.5%.**

Fog is not merely weaker, it is confused *specifically with clear* (164 of 314).
That is physically sensible rather than a bug — the corpus mixes thin, moderate
and thick haze, and thin haze on a high-contrast scene is genuinely close to
clear. It also matches the detection results: fog was the one condition showing
no interference against a union model, i.e. the one that least needs its own
parameters.

The obvious follow-up is to check routing accuracy per haze severity. If thin
fog is the whole error, the gate is right and the label is too coarse.

---

## 3. The gate is miscalibrated, not wrong

Mean gate probability by true condition:

| true ↓ / branch → | clear | fog | night |
|---|---:|---:|---:|
| clear | **0.467** | 0.292 | 0.252 |
| fog | 0.367 | **0.346** | 0.279 |
| night | 0.259 | 0.251 | **0.511** |

The maximum is on the diagonal everywhere, but the absolute values are low —
overall mean 0.336, maximum 0.828. At the default 0.5 activation threshold this
means **most images activate no routed expert at all**:

| threshold | experts active / image | correct branch fires |
|---:|---:|---:|
| 0.5 | 0.29 | 26.3% |
| 0.4 | 0.76 | 60.8% |
| **0.3** | **1.59** | **91.2%** |
| 0.2 | 2.81 | 100% |

**0.3 is the operating point.** There the block activates ~1.6 experts per image
— genuinely multi-label, as intended — and the branch matching the true
condition fires 91% of the time.

The cause is the objective, not the gate: with 3 outputs and one positive per
image, two thirds of every BCE target is zero, so the loss is minimised by
predicting low everywhere. Fixes, in order of preference:

1. `pos_weight ≈ 2` in the BCE, which is exactly the 1:2 positive:negative ratio;
2. calibrate the threshold on val (already configurable in the model YAML);
3. a per-branch learned bias initialised to the prior.

This was measured rather than assumed, and it means the *routing* result above
(argmax-based, threshold-free) stands independently of the calibration issue.

---

## 4. Acceptance verdict

| criterion | required | `cond3_gated` | verdict |
|---|---|---|---|
| NMI(route ; condition) | > 0 | 0.473 | **pass** |
| inter-expert CKA | < 1 | 0.789–0.927 | **pass** |
| route not concentrated | — | 0.527 / 0.159 / 0.314 | **pass** |

**For the first time the architecture does what it was designed to do**: a gate
that selects by condition, over branches that compute different functions. The
mechanism the thesis is about now exists and is measurable.

Accuracy is deliberately not the headline here, per the design goal. It is
recorded in `docs/benchmarks.md` for completeness, and the earlier finding
stands: in-distribution mAP is not where this design is expected to pay.

---

## 5. What this does and does not show

**Does:** condition-conditional computation is achievable in this network, with
free supervision from labels the corpus already carries, and the specialisation
is measurable in two independent ways.

**Does not:** show that it helps anything yet. The claims it was built for —
inference efficiency and robustness under drift — are untested. Both are now
*testable*, which they were not before, because a router that never routed could
not have demonstrated either.

### Immediate next steps

1. **Calibrate** (`pos_weight`) and re-measure at threshold 0.5, so the
   deployed configuration matches the analysed one.
2. **Routing accuracy per fog severity**, to see whether the fog confusion is a
   label-granularity artefact.
3. **Compound conditions.** `condition_from_paths` already emits `[0,1,1]` for a
   `fog_night_*` filename, so a held-out compound corpus needs no code change —
   and a multi-label gate is exactly the design that can express it. That is the
   drift test, and it is the first experiment where routing has a mechanism to
   win that the dense model structurally lacks.
4. **Efficiency.** With ~1.6 of 3 experts active, activated cost is now
   meaningfully below total. Worth measuring once the block is sized below the
   `C3k2` it replaces (currently it is not — 8.96 vs 6.52 GFLOPs).

### Limitations

Single seed. Fog routing at 38.5% is weak enough that the mean NMI is carried by
clear and night. The priors are feature-space analogues of image-space physics,
not the physics itself. And the accuracy consequences of routing at threshold
0.3 have not been evaluated — only the routing behaviour has.
