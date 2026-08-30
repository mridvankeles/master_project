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

**The mechanism works; the payoff has not arrived — and we now know why not.**
The router routes by condition, the experts compute different functions, and the
interference the design targets does exist. But an audit of what the model
*actually consumes* found that mosaic augmentation was mixing four conditions
into every training image while labelling it with only the first, so the gate
and the experts spent 90 of 100 epochs on a half-noise label. Separately, the
detector's dominant error — a 100% miss rate under 8 px — has nothing to do with
routing at all.

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

**The gate was trained on a corrupted label for 90 of 100 epochs.** Ultralytics'
mosaic builds every training image from four random images and records only the
*first* one in `im_file` -- which is exactly where our gate supervision reads its
condition from. Measured: **97.3% of mosaics mix conditions**, and the canvas
matches its reported label only **50.6%** of the time.

The gate's own max probability was **0.500**. The canvas fraction was **0.506**.
The gate was not miscalibrated -- it was correctly fitting a half-noise label,
and so were the experts. See `pipeline-and-data-audit.md`.

This retires the previous entry here ("the gate spreads its mass") as a symptom
rather than a cause, and puts every routing result in the project up for
re-reading. `cond3d_nomosaic` is the first arm trained on labels that match
their images.

## The larger thing the audit found

**The accuracy ceiling is a small-object ceiling, not a routing one.** Measured
miss rate by object size on the best model:

| < 8 px | 8-16 px | 16-32 px | 32-64 px | >= 64 px |
|---:|---:|---:|---:|---:|
| **100.0%** | **63.7%** | 16.8% | 12.9% | 9.6% |

Every object under 8 px is missed. Two of three under 16 px. And the cause is
representational, not a threshold: dropping conf from 0.25 to 0.01 costs 13x the
false positives and recovers only 147 of 471 misses, while `airplane`,
`basketballcourt` and `expressway-service-area` do not move at all. Only 1.9% of
misses are class confusion -- 98.1% are objects that produce no box at any score.

A neck-level expert block cannot recover that. The levers are P2/stride-4 heads
and training resolution -- **not** NWD, which was tried twice and traded
small-object recall for aggregate mAP both times (`results-cond3e-experiment.md`).

## Where the problem actually sits now

After the mosaic fix, `cond3d_nomosaic` routes nearly perfectly:

```
NMI(route ; condition) = 0.8748     purity clear .986 / fog .949 / night .973
inter-expert CKA        0.956 / 0.964 / 0.975
```

**Routing is solved and the experts are near-clones.** CKA rose from 0.831-0.941
as routing improved. Perfect routing into identical experts is worth nothing,
and that is the whole explanation for four interventions improving routing while
none improved accuracy.

The next work is expert **diversity**, not the router. And the cause is now
measured, not guessed: the experts never diverged. Prior-to-prior CKA, with no
learned weights involved, is already **0.946 / 0.932 / 0.983** -- the learned
values (0.828 / 0.821 / 0.977) are inherited from it, ordering included. Both
"different physics" priors are subtractive high-passes differing only in kernel
width, which is the same non-difference the FIRST MoE had. A divisive prior
(`x / mu`) scores 0.30 instead of 0.98. See
`results-speed-and-expert-diversity.md`.

**Cost, measured without TTA:** the MoE runs 6.10 activated GFLOPs against the
dense model's 6.52 -- 6% less arithmetic -- and is **18% slower at batch 1 and
39% slower at batch 16**. Masking, gather and `index_add` cost more than the
convolution they skip. One expert is 15% of the whole model; the fog expert is
41%. The gate is 5 kFLOPs.

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
| data + pipeline audit | `docs/pipeline-and-data-audit.md` |
| cond3e experiment | `docs/results-cond3e-experiment.md` |
| speed + expert diversity | `docs/results-speed-and-expert-diversity.md` |
| what the loss really sees | `scripts/audit_training_batches.py` |
| corpus label / per-class check | `scripts/audit_corpus_labels.py` |
| error analysis on worst images | `scripts/worst_samples.py` |
| runs + metrics | `outputs/runs/`, MLflow in `outputs/mlruns` (experiment `dior-hbb-full`) |

Every run logs dataset and model **content fingerprints**, not just paths, so any
two runs are provably comparable or provably not.
