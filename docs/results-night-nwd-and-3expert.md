# Night condition, three experts, and the NWD box loss

> **Status:** complete, single seed
> **Date:** 2026-08-12 · RTX 5070 Ti · continues `results-full-scale-and-moe.md`
> **Raw numbers:** `docs/benchmarks.md`

Same protocol as the previous round — 100 epochs, 5,862 training images per
epoch, batch 16, 640 px, cosine LR, seed 0, COCO-pretrained. Only the corpus, the
architecture and the box loss vary.

New this round: a synthetic **night** condition, a **three-expert** MoE, and a
size-gated **NWD** box loss aimed at tiny objects.

---

## 1. Headline

| | result |
|---|---|
| Three-expert MoE vs dense, three conditions | **no gain** (−0.004 mAP50, all four cells) |
| NWD box loss vs CIoU | **+0.008 mAP50 dense, +0.010 with MoE** |
| …but NWD's gain is on **large** classes | small −0.006, large **+0.013** |
| Night condition | interferes like clear (+0.021), unlike fog (+0.002) |
| Router collapse | none: 3 experts held 0.240 / 0.438 / 0.323 |

The MoE result reproduces at N=3. The NWD result is the first positive
architecture-side finding — and it is positive for the wrong reason, which is
the more useful part.

---

## 2. Three conditions, three experts

All on the full test splits, COCO mAP50, identical training data
(`union3`: 1,954 clear + 1,954 fog + 1,954 night per epoch).

| Tested on | Dense (union3) | MoE ×3 (neck) | Δ |
|---|---:|---:|---:|
| union3 | 0.6569 | 0.6530 | −0.0039 |
| clear | 0.6728 | 0.6691 | −0.0037 |
| fog | 0.6578 | 0.6538 | −0.0040 |
| night | 0.6405 | 0.6362 | −0.0043 |

**Adding a third expert changes nothing.** The deficit is −0.004 in every cell,
the same magnitude as the two-expert result (−0.002 to −0.006). Whatever routing
is doing, it is not recovering accuracy, and it does not begin to at N=3.

Routing itself is healthy: expert shares settled at 0.240 / 0.438 / 0.323 and
never collapsed, so this is a measurement of routing, not of a dead branch.

### The interference result, and a confound in it

| Condition | Specialist | union3 | Δ |
|---|---:|---:|---:|
| clear | 0.6984 | 0.6728 | +0.0256 |
| fog | 0.6594 | 0.6578 | +0.0016 |
| night | 0.6612 | 0.6405 | +0.0208 |

Night behaves like clear (≈2 points) and fog remains the outlier at ≈0. But the
comparison is not clean, and the numbers themselves show why:

| Clear images seen per epoch | mAP50 on clear |
|---|---:|
| 5,862 (specialist) | 0.6984 |
| 2,931 (2-condition union) | 0.6760 |
| 1,954 (3-condition union3) | 0.6728 |

Halving the clear data costs 0.0224. Cutting it by a further third costs only
0.0032. **The apparent "interference" is mostly a data-volume effect**, and the
marginal cost of adding a whole extra condition is almost nothing — which is the
opposite of what representational interference predicts. If conditions genuinely
fought over capacity, heterogeneity would hurt; instead only data volume does.

That strengthens last round's conclusion rather than complicating it: there is
little for condition-routing to recover, because the conditions are not
competing.

---

## 3. The NWD box loss — a positive result, for the wrong reason

`src/models/nwd.py` blends `1 − NWD` with `1 − CIoU` through a logistic gate on
log-area centred at COCO's 32² threshold, so tiny boxes get the scale-free
objective and large boxes keep CIoU.

| Arm | CIoU | + NWD | Δ |
|---|---:|---:|---:|
| dense on union | 0.6626 | 0.6702 | **+0.0076** |
| MoE neck on union | 0.6609 | 0.6704 | **+0.0095** |
| dense on clear | 0.6760 | 0.6831 | +0.0071 |

Consistent in direction and magnitude across two independent architectures. It
also **composes with routing**: the MoE+NWD arm is the best union-corpus result
of any run so far (0.6704). Still short of the 2-point single-seed noise floor,
but a consistent +0.8 across separate architectures is worth more than one cell.

### Then the per-class breakdown inverts the story

Per-class AP50 change on clear, ordered by measured median object size:

| Class | median px | instances | CIoU | +NWD | Δ |
|---|---:|---:|---:|---:|---:|
| vehicle | 12.0 | 26,639 | 0.486 | 0.469 | **−0.0172** |
| storagetank | 22.0 | 23,361 | 0.720 | 0.707 | **−0.0124** |
| ship | 25.3 | 35,184 | 0.890 | 0.881 | −0.0088 |
| airplane | 37.7 | 8,212 | 0.774 | 0.767 | −0.0063 |
| … | | | | | |
| baseballfield | 92.2 | 3,434 | 0.731 | 0.764 | +0.0334 |
| stadium | 117.4 | 672 | 0.600 | 0.661 | **+0.0610** |

| group | mean Δ |
|---|---:|
| small classes (<40 px median, n=6) | **−0.0062** |
| large classes (≥40 px, n=14) | **+0.0129** |

**NWD helped exactly the classes it was not meant to help, and hurt the ones it
was.** `vehicle` — the smallest and one of the two weakest classes in DIOR — lost
1.7 points.

### Why: the gate reweights scales instead of reshaping gradients

Measured loss magnitude at a realistic 10 %-of-side localisation error:

| box size | 1 − CIoU | 1 − NWD | gate | blended | vs CIoU |
|---:|---:|---:|---:|---:|---:|
| 8 px | 0.392 | 0.105 | 0.996 | 0.106 | **0.27×** |
| 12 px | 0.328 | 0.124 | 0.981 | 0.128 | **0.39×** |
| 32 px | 0.328 | 0.298 | 0.500 | 0.313 | 0.95× |
| 128 px | 0.328 | 0.757 | 0.004 | 0.329 | 1.01× |

`1 − NWD` is far *smaller* than `1 − CIoU` for tiny boxes. Substituting it
shrinks the tiny-object loss term to about a quarter of its previous magnitude,
so those objects receive proportionally **less** gradient and the optimiser
reallocates capacity to large ones. The aggregate mAP rises because 14 of 20
classes are large; the intended beneficiaries pay for it.

This is a scale-reweighting effect masquerading as a localisation improvement,
and only the per-class table distinguishes them. Aggregate mAP would have
recorded this as a clean win.

### What this implies for NWD-RKA

The original method is **NWD-RKA** — a Ranking-based *Assigner*. Its contribution
is using NWD to decide *which anchors are positive for which target*, where
scale-invariance fixes a real problem: IoU-based assignment starves tiny objects
of positive samples. Using NWD as a regression loss, as implemented here, is a
different intervention, and this experiment is evidence that the loss-side
version is not the part that matters.

**Next step, if pursued:** either normalise the NWD term so its magnitude matches
CIoU's before blending (making it a gradient-shape change rather than a
reweighting), or implement the assigner instead. The second is closer to the
published method.

---

## 4. Cost

Parameters and GFLOPs are deterministic and trustworthy. **Batch-1 latency on
this machine is not** — two runs of the identical `union` checkpoint measured
8.82 ms and 11.77 ms, and architecturally identical arms differ by 30 % within a
single pass. Batch-16 throughput is stable and is what is reported.

| Model | Params | GFLOPs | b16 ms/img |
|---|---:|---:|---:|
| dense | 2,593,740 | 6.52 | 0.73 |
| MoE ×2 @ backbone | 3,524,686 | 6.10 | 0.76 |
| MoE ×2 @ neck | 2,861,294 | 8.96 | 0.96 |
| MoE ×3 @ neck | 2,937,391 | 8.96 | 0.94 |
| dense + NWD | 2,593,740 | 6.52 | 0.73 |

Two things worth noting. The backbone MoE has *lower* GFLOPs than dense (6.10 vs
6.52) because it replaced `C2PSA`, which is more expensive than the block that
replaced it — yet it was also the least accurate arm. And **NWD is free**: it
changes the loss only, so it costs nothing at inference. Of everything tried
across both rounds, it is the only change with a positive accuracy delta and zero
inference cost.

---

## 5. Where this leaves the thesis

- **C2 (interference) remains unsupported**, now at N=3 and with a third
  degradation. The data-volume analysis in §2 makes the negative result
  stronger, not weaker: heterogeneity is nearly free, so there is nothing for
  routing to separate.
- **C1 (efficiency) remains unsupported.** Every MoE arm costs more per image
  than the dense model it fails to beat.
- **The most promising direction found so far is not the MoE.** It is the
  loss/assignment side, where a free change moved the number — and where the
  per-class analysis says the published method's actual mechanism (the assigner)
  has not been tried yet.

### Limitations

1. **Single seed throughout.** All deltas discussed are below the 2-point noise
   floor `05-experiment-plan.md` sets. The NWD direction is supported by
   consistency across two architectures, not by significance.
2. **Night is synthetic and unvalidated against real night imagery.** The model
   is physically motivated and per-sample randomised, but no real low-light
   remote-sensing test set was evaluated.
3. **`C = 12.8` in NWD was not tuned.** It is the published AI-TOD value. The
   magnitude mismatch in §3 may be partly a bad constant rather than an
   intrinsic property of the blend.
4. **No RDDTS.** The HazyDet copy on disk lacks the real-hazy split, so the
   synthetic-to-real gap — 52.0 → 38.7 in the published table — remains
   unmeasured for our models.

---

## 6. Reproducing

```bash
python scripts/make_night.py --scope full                       # night corpus
python scripts/make_union.py --scope full \
       --conditions clear fog night --out-condition union3      # 3-condition corpus

python scripts/train.py --config configs/train/union3_full_yolo11n.yaml
python scripts/train.py --config configs/train/moe3_neck_yolo11n.yaml
python scripts/train.py --config configs/train/union_nwd_yolo11n.yaml
python scripts/train.py --config configs/train/moe_neck_nwd_yolo11n.yaml
python scripts/train.py --config configs/train/night_full_yolo11n.yaml

python scripts/run_evaluations.py
python scripts/benchmark_speed.py        # idle GPU; b1 latency is noisy, use b16
python scripts/collect_benchmark.py --out docs/benchmarks.md
```
