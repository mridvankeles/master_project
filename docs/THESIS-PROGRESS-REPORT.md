# Degradation-Conditioned Mixture-of-Experts for Remote Sensing Object Detection

**Progress report — step by step, with measured results**

M. Rıdvan Keleş · 2026-08-13 · code and full logs:
`github.com/mridvankeles/master_project`

---

## Summary

The thesis asks whether a detector can devote separate branches to separate
degradations — one for haze, one for low light — selected by a learned gate.
Twelve full training runs and 40+ evaluations were completed. The main results:

1. **Gate 1 passed.** Stock YOLO11n reaches **65.2 VOC07 mAP** on clear DIOR
   against a published reference of 57.1, with a healthy per-class spread.
2. **A first MoE showed no gain — and the reason turned out to be that it never
   routed.** Direct measurement (NMI = 0.000, inter-expert CKA = 0.996) showed
   the gate sent every image to one branch and the branches were near-identical.
   The result was withdrawn.
3. **A redesigned gate does route**, NMI **0.000 → 0.473**, with supervision and
   heterogeneous static priors isolated as the two necessary factors.
4. **The remaining routing error was traced to the data, not the model.** Our
   synthetic haze was weaker than real haze; strengthening it raised routing
   from 36.5% to 66.4% **with no retraining**.

Every number below is regenerable from a script, and every run is logged to
MLflow with dataset and model content fingerprints.

---

## Step 1 — Data recovery: 651 → 5,862 training scenes

**Problem.** Only 2,607 of DIOR's 23,463 images kept their identifier in the
Hazy-DIOR release, so early experiments trained on 651 unique scenes.

**Finding.** The "unusable" images are lossless PNG re-encodings of DIOR
originals. Matching them by decoded-pixel hash recovers the identity.

| | before | after |
|---|---:|---:|
| fog train ids | 651 | **5,835** (99.5% of the official split) |
| fog val / test ids | 651 / 1,305 | 5,846 / 11,704 |

Two assumptions were verified rather than assumed: image↔haze pairing
(correlation 0.756 vs −0.022 for unrelated scenes) and severity ordering
(brightness monotone thin→moderate→thick, matched against the release's own
named folders). Splits were audited: zero train/val/test identifier overlap.

---

## Step 2 — Gate 1: baseline reproduction ✅

The reference was committed to git **before** training so the target could not
be chosen after the fact (`docs/gate1-reference.md`).

**Reference:** Li et al., *ISPRS J. Photogramm. Remote Sens.* 159 (2020) —
the paper that introduced DIOR. YOLOv3/Darknet-53 = **57.1 mAP**; best published
= 66.1 (RetinaNet-R101, PANet-R101).

**Result — YOLO11n, clear DIOR, full 11,738-image test split:**

| metric | value |
|---|---:|
| **VOC07 mAP@0.5** (comparable convention) | **65.2** |
| COCO mAP50 | 69.8 |
| COCO mAP50-95 | 49.5 |
| Precision / Recall | 0.836 / 0.639 |

**+8.1 over the reference, 0.9 below the published best** — from a 2.6 M
parameter model trained on half the data those numbers used (val is held back
for checkpoint selection). 19 of 20 classes exceed AP50 0.5; none fall below 0.2.
The gap is attributed to six years of detector progress and COCO pretraining,
not to a better protocol.

---

## Step 3 — Is condition-specialisation even needed?

Before building a router, we tested its premise: does a single model trained on
all conditions lose accuracy relative to per-condition specialists?

| condition | specialist | union (dense) | deficit |
|---|---:|---:|---:|
| clear | 0.6984 | 0.6760 | +0.0225 |
| fog | 0.6594 | 0.6589 | **+0.0005** |
| night | 0.6612 | 0.6405 | +0.0208 |

**Nearly no interference on fog**, and a follow-up showed most of the apparent
deficit is a *data-volume* effect: clear-condition accuracy falls 0.6984 →
0.6760 → 0.6728 as clear images per epoch fall 5,862 → 2,931 → 1,954. Halving
the data costs 0.0224; adding a whole extra condition costs 0.0032.

The matrix is also strongly asymmetric — a fog-trained model loses nothing on
clear (0.6646), while a clear-trained model loses 15.5 points under fog (0.5430).

---

## Step 4 — First MoE: no gain, and a withdrawn conclusion

Two and three experts, two placements, matched data and hyperparameters:

| tested on | dense | MoE ×2 neck | MoE ×3 neck |
|---|---:|---:|---:|
| union | 0.6626 | 0.6609 | — |
| union3 | — | — | 0.6530 vs 0.6569 dense |

No configuration beat the dense baseline. **The natural conclusion — "routing
does not help" — turned out to be unsupported**, because a direct measurement of
the router showed it was not routing at all:

| | measured |
|---|---|
| NMI(route ; condition) | **0.0000** |
| inter-expert CKA | **0.968 – 0.996** |
| eval-time route distribution | 100% to one expert |

Every image took the same route, and the branches computed nearly the same
function, so the comparison was a dense model against a dense model carrying an
unused branch. **The MoE conclusions were withdrawn**
(`docs/finding-router-never-specialised.md`).

A reporting error contributed: training logs recorded the *noise-perturbed*
route, which made a collapsed router look balanced.

---

## Step 5 — Redesign: the gate now specialises

Four changes, each targeting one identified cause: heterogeneous **static**
priors per branch (illumination-invariance for night, transmission for haze),
a **supervised multi-label gate** using condition labels the corpus already
carries, an **always-on shared branch**, and clean/noisy route separation.

| arm | supervised | priors | **NMI** | inter-expert CKA |
|---|---|---|---:|---|
| **cond3_gated** | **yes** | heterogeneous | **0.473** | **0.789 / 0.794 / 0.927** |
| cond3_nogate | no | heterogeneous | 0.000 | 0.774 / 0.846 / 0.944 |
| plain3_gated | yes | identical | 0.485 | 0.936 / 0.970 / 0.938 |

**Two factors, cleanly separated by ablation:**
- **Supervision creates routing** — 0.000 → 0.473. Without it the gate collapses.
- **Static heterogeneous priors create specialisation** — CKA 0.79 vs 0.94 with
  identical priors, same supervision.

Router confusion: clear 93.5%, night 87.9%, **fog 38.5%** (confused with clear).

---

## Step 6 — The fog error was the data, not the model

Measured against **RRSHID** (real paired clear/hazy remote sensing imagery):

| | dark-channel increase |
|---|---:|
| REAL moderate fog | **+100.5** |
| REAL thick fog | **+142.8** |
| our corpus (thin / moderate / thick) | +36.8 / +62.3 / **+93.0** |

**Our thickest haze was weaker than real *moderate* haze.** Two thirds of the
fog corpus was barely degraded, so the gate had almost nothing to detect.

Confirming evidence, requiring no retraining — the *same* gate run on real haze
and on a newly calibrated synthesis:

| corpus | routed to fog branch |
|---|---:|
| clear | 0.7% |
| our OLD fog | 36.5% |
| **our NEW calibrated fog** (Δ dark channel +110) | **66.4%** |
| **REAL haze (RRSHID)** | **98 – 100%** |
| night | 1.0% |

**The router was correct; the corpus was too easy.** The 38.5% figure was
measuring the data.

---

## Step 7 — A loss-side result, positive but for the wrong reason

A size-gated Wasserstein (NWD) box loss for tiny objects:

| arm | CIoU | + NWD | Δ |
|---|---:|---:|---:|
| dense | 0.6626 | 0.6702 | **+0.0076** |
| MoE | 0.6609 | **0.6704** | **+0.0095** |

Consistent across two architectures, at zero inference cost. But the per-class
breakdown **inverts** the interpretation:

| group | mean Δ AP50 |
|---|---:|
| small classes (< 40 px median) | **−0.0062** |
| large classes | **+0.0129** |

`vehicle` — the smallest class and a target of the method — **lost 1.7 points**.
Cause: `1 − NWD` is much smaller than `1 − CIoU` for tiny boxes (0.27× at 8 px),
so substituting it *reduces* tiny-object gradient and reallocates capacity to
large objects. Aggregate mAP would have recorded this as a clean win.

---

## Step 8 — Real low-light data (in progress)

DroneVehicle converted into the pipeline: de-letterboxed (44% of every frame was
white padding), oriented→horizontal boxes, split by measured brightness.
**17,325 images, 6,096 genuinely dark** (mean brightness 32.5 vs 118.9).

This also exposed a defect in our night synthesis: brightness matches real night
almost exactly (30.2 vs 32.4) but **contrast is 3.5× too low** (12.8 vs 44.5) —
real night is dark *and* high-contrast because light sources saturate, whereas
ours is uniformly dimmed.

---

## Methodological notes

- **Single seed.** All differences discussed are below the ~2-point noise floor
  the experiment plan sets. Three seeds are required before quotation.
- **Matched budgets.** Every arm sees 5,862 training images per epoch under an
  identical schedule; a larger fog arm would score higher and be uncomparable.
- **Provenance.** Every run logs dataset and model *content* fingerprints, not
  just paths, so two runs are provably comparable or provably not.
- **Known confound.** Clear imagery is 100% JPEG and fog 89% PNG; the difference
  is compression *history*, so re-encoding cannot remove it. Recorded per image.
- **Corrections made during the work:** an ISO-gain error that left night images
  only 11% darker; a scope bug that would have trained on 651 images while
  logging 5,862; a claim that DroneVehicle was unusable, disproved by looking at
  the images after the statistics were corrupted by padding.

## What the evidence supports today

Condition-conditional computation **is achievable** in this network, using
supervision that costs nothing, and it is **measurable** in two independent ways.
Whether it *helps* — in efficiency or in robustness under drift — is still open,
but is now testable for the first time: a router that never routed could not
have demonstrated either.
