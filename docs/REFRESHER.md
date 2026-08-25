# Refresher — where the project is, in one page

> Read this first after time away. Everything here is measured, not remembered.
> Deeper detail: `MODEL-AND-TRAINING-CONFIG.md` (architecture),
> `THESIS-PROGRESS-REPORT.md` (supervisor-facing narrative).

---

## The question

Can a detector give **separate branches to separate degradations** — one for
haze, one for low light — chosen by a learned gate? The target was never a
higher mAP. It was specialisation, and through it, efficiency and robustness.

## The one-line answer today

**The mechanism works; the payoff has not arrived yet.** The router routes by
condition, the experts compute genuinely different functions, and the
interference the design targets does exist — but the block still scores slightly
*below* a plain dense model on the same data.

---

## Five numbers that define the current state

| | value | meaning |
|---|---:|---|
| **65.2** | VOC07 mAP, clear DIOR | Gate 1 passed (+8.1 over the 57.1 reference) |
| **0.542** | NMI(route ; condition) | the gate routes by condition (was 0.000) |
| **0.83** | inter-expert CKA | the experts differ (was 0.996 — clones) |
| **+0.0235** | fog interference deficit | there IS something to recover (was +0.0016) |
| **−0.0044** | MoE vs dense | ...and it is not being recovered |

---

## What is built

**Data.** Three conditions on the *same* 23,463 DIOR images, so the router
cannot cheat on platform statistics:

- `clear` — DIOR originals
- `fog2` — our ASM synthesis, calibrated against **real** haze (RRSHID)
- `night` — inverse-ISP + Poisson-Gaussian sensor model

Plus **DroneVehicle** converted into the pipeline as *real* low light (6,096
genuinely dark images), and **RRSHID** as real haze for router probing.

**Model.** `cond3b_gated_yolo11n` — stock YOLO11n with the P3 neck block
replaced by a `CondMoEBlock`: an always-on shared branch plus three static
experts (identity / transmission / illumination-invariant), selected by a
**supervised multi-label sigmoid gate**. 3,014,031 params (+16% over dense); the
gate itself is 771 params.

**Training.** Joint, end to end, single optimiser. `L_det + 1.0 * L_gate(BCE)`.
Gate labels come free from the corpus filenames. **No separate router training
stage.**

---

## The four things that turned out to be true

1. **Supervision is what creates routing.** Unsupervised, the gate collapses to
   one branch (NMI 0.000) no matter what else is done. Supervised, NMI 0.473.
2. **Static physical priors are what create specialisation.** With identical
   priors, experts converge (CKA 0.94). With different fixed priors, they stay
   apart (CKA 0.79). Both factors are needed; neither alone suffices.
3. **Weak data can masquerade as a broken model.** Our fog was weaker than real
   *moderate* haze at its thickest setting. Fixing that raised routing 36.5% ->
   66.4% **with no retraining**, and grew the fog interference deficit 15x.
4. **Aggregate metrics hid every single error.** NWD looked like a +0.008 win and
   was −0.006 on the classes it targeted. A collapsed router looked balanced
   because the logs recorded the noisy route. Two eval cells reported identical
   numbers because a config fell back silently. In every case the headline number
   looked fine.

## The one thing that is still wrong

**Routing quality and accuracy have decoupled.** Three interventions have now
raised NMI (0.000 -> 0.473 -> 0.542 -> 0.596) and not one has raised accuracy;
the last moved it down. Better routing is not translating into better detection.

An earlier diagnosis here -- "the gate predicts uniformly low" -- was measured
and found **wrong**: `sum(p)` is already 1.012. The gate spreads its mass
(0.50 / 0.29 / 0.22) instead of concentrating it, so the maximum barely clears
the 0.5 threshold. A routing cost aimed at the sum therefore addressed a
non-problem.

**Next experiments** target the experts rather than the router: decouple
selection from magnitude (experts currently run at ~half strength because their
output is scaled by the gate probability), sweep the gate weight down from 1.0,
and widen the experts, since inter-expert CKA is still 0.82-0.96.

---

## Ground rules that have earned their place

- **Check routing before accuracy.** NMI > 0, CKA < 1, route not concentrated.
  A model failing these says nothing about whether routing helps.
- **Matched budgets.** Every arm sees 5,862 training images per epoch under the
  same schedule and seed. A bigger arm would score higher and mean nothing.
- **Look at the disaggregated view.** Per-class, per-condition, per-scale. Every
  error so far was invisible in the aggregate.
- **Single seed.** Every difference discussed is below the ~2-point noise floor.
  Three seeds are required before any of it is quotable.

---

## Where things live

| | |
|---|---|
| final model | `configs/train/cond3b_gated_yolo11n.yaml` |
| MoE block | `src/models/moe2.py`, priors in `src/models/experts.py` |
| joint loss wiring | `src/models/moe_trainer.py` |
| degradation synthesis | `src/data/degradation.py` |
| routing diagnostics | `scripts/analyse_routing.py` |
| all scores | `docs/benchmarks.md` (regenerated by `scripts/collect_benchmark.py`) |
| runs + metrics | `outputs/runs/`, MLflow in `outputs/mlruns` (experiment `dior-hbb-full`) |

Every run logs dataset and model **content fingerprints**, not just paths, so any
two runs are provably comparable or provably not.
