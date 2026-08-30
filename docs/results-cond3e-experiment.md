# cond3e: no photometric augmentation, an expert floor, and NWD on P3

> **Date:** 2026-08-30 · experimental confirmation run, 40 epochs, seed 0.
> `configs/train/cond3e_p3nwd_yolo11n.yaml` and its ablation `cond3e_nonwd_yolo11n.yaml`.
> Short schedule deliberately — nothing here is a final number. The two cond3e
> arms are matched to each other in everything but the box loss, and *that*
> comparison is sound.

---

## The three questions, answered

| | question | answer |
|---|---|---|
| 1 | does zeroing HSV help the gate separate the conditions? | **inconclusive** — confounded by the 40-epoch schedule |
| 2 | does the expert floor stop images taking the shortcut? | **yes, and it was already true** |
| 3 | does NWD on the P3 branch move the small-object miss rate? | **no — it makes it worse** |

---

## 3. NWD on P3: a clean negative, and the same signature as before

Small-object miss rate, identical measurement (450 union3b val images, conf 0.25):

| run | epochs | box loss | **8–16 px missed** | < 8 px |
|---|---:|---|---:|---:|
| `cond3b_gated` | 100 | CIoU | 63.7% | 100% |
| `cond3d_nomosaic` | 100 | CIoU | 64.1% | 100% |
| **`cond3e_nonwd`** | 40 | CIoU | **63.3%** | 100% |
| **`cond3e_p3nwd`** | 40 | **+NWD on P3, ×2** | **68.0%** | 100% |

The two cond3e arms differ *only* in the box loss. NWD on the tiny-object branch
costs **4.7 points of recall on exactly the objects it was added to rescue.**

And it does so while raising the aggregate:

| run | val mAP50 | val mAP50-95 | 8–16 px missed |
|---|---:|---:|---:|
| `cond3e_nonwd` | 0.7266 | 0.5463 | 63.3% |
| `cond3e_p3nwd` | **0.7327** | **0.5491** | **68.0%** |

**This is the same signature the first NWD arm produced** — +0.008 aggregate,
−0.006 on the small classes it targeted — reproduced independently, after the
units bug was fixed and after the loss was restricted to the one pyramid level
that carries tiny objects. Two different implementations, same trade.

That is now strong enough to close the question: on this corpus, NWD buys
aggregate mAP by spending small-object recall. It is not the tiny-object fix.

### The units bug it was worth finding anyway

Ultralytics passes `target_bboxes / stride_tensor` into `bbox_loss`, so boxes
arrive in **feature-map cells, not pixels**. CIoU does not care — IoU is
scale-invariant — but NWD's `c` and the size gate's `tiny_area` are absolute:

| true box | old `size_gate` | fixed `size_gate` |
|---|---:|---:|
| 16 px | **1.000** | 0.941 |
| 64 px | **0.996** | 0.059 |
| 320 px | 0.291 | 0.000 |

The gate returned ~1.0 for everything, so `mode="gated"` had been running as
`mode="always"` in every NWD experiment this project has done. The fix is in
`src/models/nwd.py`; the conclusion above is measured *after* it.

---

## 2. The expert floor works, and its premise had already evaporated

```
L_floor = mean_b  relu( tau - max_i p_i(b) )      tau = 0.6, lambda = 0.5
```

| | epoch 1 | epoch 40 |
|---|---:|---:|
| `floor_loss` | 0.001 | 0.023 |
| `gate_max_prob` | 0.601 | 0.651 |
| **`shortcut_only`** | **0.000** | **0.000** |

The hinge is satisfied within one epoch and then stops pulling, which is what a
hinge should do. But `cond3d` — no floor at all — already ended with
`experts_per_image = 1.000` and `shortcut_only = 0.000`. **The mosaic fix had
already solved the shortcut problem.** The floor was sized from `cond3b`'s
`gate_max_prob = 0.500`, a number that only existed because the gate label was
half noise.

It is not harmful, it costs nothing, and it still earns its place for the case
it was actually wanted for — an unlabelled out-of-distribution image, where
`routing_cost` cannot apply because there is no `n_true` to read. But it is not
a lever on in-distribution accuracy, and it should not be credited with anything
cond3e scores.

**One risk, observed and resolved.** At epoch 1 the gate collapsed to
`night_active = 1.000` — the cheapest way to satisfy the hinge is to push one
unit up for every image. By epoch 40 activation was balanced again
(clear 0.389 / fog 0.286 / night 0.254): the BCE pulled it apart, as intended by
setting `floor_lambda` at half the BCE weight. Worth watching in any longer run.

---

## 1. Photometric augmentation off: not answerable at 40 epochs

| run | epochs | mosaic | hsv (h,s,v) | NMI | gate_max_prob |
|---|---:|---|---|---:|---:|
| `cond3b_gated` | 100 | 1.0 | .015/.7/.4 | 0.542 | 0.500 |
| `cond3d_nomosaic` | 100 | 0 | .015/.3/.15 | **0.875** | **0.773** |
| `cond3e_nonwd` | 40 | 0 | **0/0/0** | 0.768 | 0.688 |
| `cond3e_p3nwd` | 40 | 0 | **0/0/0** | 0.678 | 0.651 |

cond3e's NMI is *below* cond3d's, but cond3e ran 40 epochs against cond3d's 100
and routing was still improving in both. **The comparison is not valid** and
this table should not be read as "zeroing HSV hurt routing". Settling it needs a
100-epoch `hsv: 0` run against `cond3d`.

What *is* valid, between the two matched cond3e arms: the P3-NWD arm has worse
routing too (NMI 0.678 vs 0.768, fog branch fires 0.812 vs 0.869). Changing the
box loss degraded the gate — a second cost of the same intervention.

---

## The finding that outranks all three

`cond3d`, at 100 epochs, with the best router this project has produced:

```
NMI(route ; condition) = 0.8748
  clear 286/290 correct (purity 0.986)
  fog   298/314 correct (purity 0.949)
  night 288/296 correct (purity 0.973)

inter-expert CKA:  0.9556 / 0.9641 / 0.9748
```

Routing is essentially solved. **And the experts are near-clones** — CKA rose
from `cond3b`'s 0.831–0.941 as routing improved.

Perfect routing into identical experts is worth nothing. That is a complete
explanation of why four consecutive interventions have improved routing and none
has improved accuracy, and it moves the work off the router entirely.

### A hypothesis worth testing

| run | epochs | cka e0–e1 | e0–e2 | e1–e2 |
|---|---:|---:|---:|---:|
| `cond3e_p3nwd` | 40 | 0.742 | 0.848 | 0.968 |
| `cond3e_nonwd` | 40 | 0.824 | 0.815 | 0.988 |
| `cond3d_nomosaic` | 100 | 0.956 | 0.964 | 0.975 |

The 40-epoch arms are *less* collapsed than the 100-epoch one on two of three
pairs. If that holds, **expert collapse is a late-training phenomenon**: the
static priors start the branches apart and training pulls them back together.
That would be a different problem from the one the priors were designed to
solve, and it would need a term that keeps them apart rather than a better
initialisation.

Confounded by schedule, so it is a hypothesis, not a result. The cheap way to
settle it is to log inter-expert CKA every epoch rather than once at the end.

---

## Accuracy, for the record

| run | epochs | augmentation | val mAP50 | val mAP50-95 |
|---|---:|---|---:|---:|
| `union3d_nomosaic` dense | 100 | mosaic off | 0.7397 | 0.5611 |
| `cond3d_nomosaic` MoE | 100 | mosaic off | 0.7402 | 0.5590 |
| `cond3e_p3nwd` MoE | 40 | + hsv off, floor, P3-NWD | 0.7327 | 0.5491 |
| `cond3e_nonwd` MoE | 40 | + hsv off, floor | 0.7266 | 0.5463 |

On **test**, the number this project quotes, removing mosaic cost ~3.2 points of
mAP50 for MoE and dense alike (0.6627 → 0.6305 dense, 0.6583 → 0.6239 MoE), and
the MoE's deficit against its control widened slightly, −0.0044 → −0.0066. An
earlier note claiming the deficit had closed was reading the validation split;
it does not hold on test.

## What this run changes

- **NWD is finished as a tiny-object strategy here.** Two implementations, one
  with a real bug and one without, both trade small-object recall for aggregate
  mAP. `p3_weight` and `levels: p3` stay in the code as options; they are not
  the default and should not be in the final model.
- **The floor stays**, cheap and correct, scoped to the OOD case rather than
  sold as an accuracy lever.
- **The next experiment is about expert diversity, not routing.** Either a
  repulsion term on the expert outputs, or a per-epoch CKA log to find out when
  the collapse happens before trying to prevent it.
