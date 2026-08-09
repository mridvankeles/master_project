# 05 — Experiment Plan and Engineering Protocol

> **Status:** skeleton
> **Sources:** chat 2026-07-28, advisor analysis 2026-07-28
> **Open questions:** 3  |  **Conflicts:** 0

## Baselines — the part that decides whether the thesis stands

There is no published MoE for 2D degraded RS detection (`02-related-work.md` § The gap), which
means **there is no baseline to download. You must build every comparison yourself.** Budget for
this; it is the largest hidden cost in the project.

Required, in priority order:

| # | Baseline | Why an examiner requires it |
|---|---|---|
| B1 | **Dense model, equal parameters** to the full MoE, trained on the union of all conditions | Isolates whether the gain comes from routing or from parameter count. **This is the one that can sink the thesis.** |
| B2 | **Dense model, equal FLOPs** (= one expert's size), trained on the union | The fair efficiency comparison |
| B3 | **Per-condition specialists, oracle-selected** | Upper bound; tells you what perfect routing would buy |
| B4 | **Restoration-then-detection cascade** (dehaze/enhance → detect) | The standard alternative in the RS literature; the comparison your efficiency claim needs |
| B5 | **Published competitors on shared benchmarks** — DHC-Net, NIRNet, CM-YOLO (all have public repos), plus RShDet's reported numbers on DOTA-derived haze | Situates you against the field rather than against your own baselines only |

Note on B5: RShDet reports +7.1% mAP50 over baseline on its Hazy-DOTA, and +2.4% mAP50 over
prior SOTA `[from: advisor web search 2026-07-28]`. If your gains are smaller than the field's
existing margins, say so plainly rather than choosing a weaker comparison.

**An untuned baseline is not a baseline.** Every baseline gets the same tuning budget, the same
data, the same augmentation, the same schedule, and the same number of seeds as your method.
Record the budget so you can state it.

## Ablations

1. **Router removed** — average all experts. Isolates the routing contribution.
2. **Expert count** — 1 (dense) / 2 (fog, dark) / 3 (+ tiny or + clear).
3. **Oracle vs. learned routing** — the single most informative run you will do.
4. **Auxiliary loss** — none / switch / entropy. Report expert utilization for each, not just mAP.
5. **MoE placement** — early backbone / late backbone / neck.
6. **Real vs. synthetic evaluation** — same model, synthetic test set and RDDTS. The gap is a
   result in itself.
7. **Compound degradation** — fog + low light simultaneously, held out of training entirely.
8. **Unseen degradation** — a condition never trained on (e.g. motion blur, or a sandstorm-like
   corruption). UniDet-D claims generalization to unseen conditions including rain-fog mixtures,
   so this comparison will be expected of you.

## Variance and reporting — non-negotiable

- **Minimum 3 seeds** for every headline number. Report mean ± standard deviation.
- A gain under ~2 points on a single seed will be treated as noise, and correctly so.
- If you cannot afford 3 seeds on everything, run 3 seeds on the **main comparison** (MoE vs. B1)
  and 1 seed on ablations, and state that split explicitly in the thesis.
- Report the same metric set the competing papers report (mAP50 and mAP, matching each dataset's
  convention) so numbers are comparable without translation.

## Leakage checklist — run before each experiment, not after

- Tiling overlap: DOTA-style tiling with overlap can place the **same object** in a train tile
  and a test tile. Split by **source image**, never by tile.
- Hazy-DIOR, DOTA-v2.0Haze, Foggy-DOTA and DOTA-Cloud are all derived from **DOTA/DIOR source
  imagery**. If clear DOTA is in training and hazy DOTA is in testing, the test images are the
  training images with fog on top. This is the most likely leakage path in your entire setup —
  check it first.
- Router training must not see test-split images, even for its condition labels.
- Synthetic degradation parameters must be sampled per-split, not fitted on the full dataset.
- No tuning on the test split. Hold out a validation split and touch the test split once.

## Reproducibility

Pin library versions, fix and record seeds, log all hyperparameters, and keep one script per
reported table that regenerates its numbers from a checkpoint. Every number in the thesis should
be traceable to a command. This is also what makes the "does a script regenerate every reported
number?" question a one-word answer.

## Month-by-month plan (12 months from 2026-07-28)

RTX 5070 Ti (16 GB) personal; 2× A6000 (48 GB) at work `[from: chat 2026-07-28]`.
The 16 GB card cannot train three experts plus backbone at full tile resolution — use it for the
data pipeline, router prototyping and single-expert debugging. All seeded final numbers come from
the A6000s.

Useful calibration: RShDet trained a haze-robust detector on DOTA-v2.0-derived data on a **single
RTX 4090, batch size 4, 100 epochs** `[from: advisor web search 2026-07-28]`. The fog leg is
feasible on hardware comparable to yours.

| Month | Phase | Deliverable | GPU |
|---|---|---|---|
| 1 | Design freeze | PMFN read; routing granularity, box format and claim decided | none |
| 2 | Proposal | Written proposal with the delta against PMFN stated explicitly | none |
| 3 | Data | Condition grid generated; splits verified against leakage checklist; dataset card | 5070 Ti |
| 4 | Baseline reproduction | Stock YOLO11 (HBB) reproduced at a published mAP on clear DIOR | A6000 |
| — | **GATE 1** | **If the baseline is not reproduced, stop and fix. Nothing downstream is measurable without it.** Pick a specific paper and a specific DIOR mAP number, and write it down *before* training. | |
| 5 | Single experts | One detector per condition; per-run wall-clock measured for budgeting | A6000 |
| 6 | Dense baselines | B1 equal-parameter, B2 equal-FLOPs, both fully tuned | A6000 |
| 7–8 | MoE main comparison | MoE vs. B1, 3 seeds, mean ± std | A6000 |
| — | **GATE 2** | **If B1 matches the MoE, switch to the fallback claim in `01` and report it. Do not spend months 9–10 trying to rescue the number.** | |
| 9 | Analysis I | Oracle routing, router confusion synthetic vs. real, expert utilization, CKA | A6000 + 5070 Ti |
| 10 | Analysis II | Compound (fog+dark), unseen (DOTA-Cloud), cross-platform (Hazy-DIOR) | A6000 |
| 11 | Writing | Full draft, all tables regenerable from scripts | none |
| 12 | Defense | Revisions, defense prep against the committee questions below | none |

### What this schedule excludes

One training arm. No second training corpus, no hierarchical routing, no static/dynamic expert
study, no extra losses beyond the entropy regularizer. Those are the "future work" section.
Attempting any of them is how a one-year thesis becomes an eighteen-month thesis.

### Slack

There is roughly three weeks of slack in this plan, concentrated in months 7–8. Data acquisition
overruns (VPN issues on DOTA-v2.0Haze, the 27 GB and 7 GB downloads) are the most likely source of
delay, which is why months 3–4 come before any modelling.

## Committee questions to be able to answer, with what a good answer contains

1. **What does the MoE do that a dense model of equal parameter count cannot?**
   Needs B1 at matched parameters and matched tuning budget, across 3 seeds, plus the
   interference evidence. "Capacity" is not an answer.
2. **What happens when the router is wrong?**
   Needs the router-accuracy-vs-mAP curve, the oracle upper bound, and results on compound
   degradation absent from training. If wrong routing is worse than no routing, say so and show
   the mitigation.
3. **Your degraded data is synthetic. What have you shown about real conditions?**
   Needs headline numbers on RDDTS or equivalent real imagery, and the synthetic-vs-real gap
   reported rather than hidden.
4. **Why three experts and not one model with three attention branches?**
   Needs the ablation with the router removed, and a position on the classical-vs-modern MoE
   framing (`04-method-open-questions.md`).
5. **Is "tiny objects" a task or a scale problem?**
   Needs whichever routing-granularity resolution you chose, stated as a design decision with
   its reasoning — not discovered by the committee.

## Open questions

1. Which detector architecture as the base?
2. Full 3-seed coverage, or 3 seeds on the main comparison only?
3. Which corruption serves as the "unseen degradation" test?
