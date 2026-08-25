# Final model and training configuration

> **Date:** 2026-08-24 · every number verified against the checkpoints and
> configs on disk, not from notes.
> **Final model:** `cond3b_gated_yolo11n`
> **Reproduce:** `python scripts/train.py --config configs/train/cond3b_gated_yolo11n.yaml`

---

## 1. The final architecture

Stock YOLO11n with **one block replaced**: the `C3k2` at neck index 16 — the
P3/8 stage that feeds the small-object detection head — becomes a
`CondMoEBlock`. Nothing else in the network changes, and no layer indices shift,
so `Concat` and `Detect` still reference the same numbers.

```
input 640x640
  |
  +-- backbone  (Conv, C3k2 x4, SPPF, C2PSA)          <- unchanged
  |
  +-- neck (PAN-FPN)
        +-- index 13  C3k2 [512]                       <- unchanged
        +-- index 16  ## CondMoEBlock [256 -> 64] ##   <- THE ONLY CHANGE
        +-- index 19  C3k2 [512]                       <- unchanged
        +-- index 22  C3k2 [1024]                      <- unchanged
        +-- Detect(P3, P4, P5), nc=20
```

### Inside the block

```
                       x  (256 ch, 80x80)
                       |
      +----------------+----------------+------------------------+
      |                |                                         |
  proj 1x1        SHARED expert                         GATE  Linear(256->3)
  256->64         (always on)                     global-avg-pool -> sigmoid
      |                |                                         |
      |                |          +----------------+-------------+-------------+
      |                |     p_clear > 0.5     p_fog > 0.5           p_night > 0.5
      |                |          |                |                     |
      |                |    [ IDENTITY ]    [ TRANSMISSION ]    [ ILLUMINATION- ]
      |                |    [  prior    ]   [    prior     ]    [   INVARIANT   ]
      |                |    [ conv 3x3  ]   [   conv 5x5   ]    [   conv 3x3    ]
      |                |          |  x p_clear     |  x p_fog       |  x p_night
      +----------------+----------+----------------+----------------+
                                  |  (sum)
                                  v
                            out (64 ch, 80x80)
```

**Multi-label, not top-1.** The gate is a sigmoid per branch with a 0.5
threshold, so zero, one, or several experts can fire on the same image. A
softmax would force fog and night to compete for one probability budget, and an
image can genuinely be both.

**The priors are fixed, not learned.** That is what stops the branches
collapsing into copies of each other — the first MoE differed only by kernel
size and converged to inter-expert CKA 0.996.

| prior | operation | targets |
|---|---|---|
| `IdentityPrior` | none | clear |
| `TransmissionPrior` | subtract a wide (15x15) low-frequency veil, concat a dark-channel cue | haze — ASM's smooth additive veil |
| `IlluminationInvariant` | `log` then subtract a local (7x7) mean | low light — multiplicative gain becomes additive in log space and cancels |

Each prior is followed by a **learnable** conv, so the design is static and
dynamic together: the prior fixes *what the branch looks at*, the convolution
learns *what to do about it*.

### Parameter count

| | params | GFLOPs @640 |
|---|---:|---:|
| stock YOLO11n (dense control) | 2,593,740 | 6.52 |
| **`cond3b_gated` (final)** | **3,014,031** | **8.26** |
| overhead | **+420,291 (+16.2%)** | +1.74 |

Inside the block (452,387 params total):

| component | params | note |
|---|---:|---|
| shared expert (always on) | 75,840 | identity prior, 3x3 |
| clear expert | 75,840 | identity prior, 3x3 |
| **fog expert** | **207,712** | transmission prior, **5x5** — wider kernel for airlight |
| night expert | 75,840 | illumination-invariant prior, 3x3 |
| **gate** | **771** | `Linear(256 -> 3)`, 0.026% of the model |

The router is essentially free. The cost is in the experts, and only some of
them run per image.

---

## 2. Conditions

Three, all built on **the same DIOR images**, which is what prevents the router
separating them on platform statistics instead of on the degradation.

| condition | source | strength |
|---|---|---|
| `clear` | DIOR originals, 23,463 images | — |
| `fog2` | our ASM synthesis, calibrated to real haze | dark-channel **+110** (real: +100 moderate, +143 thick) |
| `night` | inverse-ISP + Poisson-Gaussian sensor model | brightness ratio 0.315 |

Both degradations sample **per-image random parameters** seeded by DIOR id, so
no two images share a simulator setting and the corpus still rebuilds
identically.

**Training data actually consumed** (verified from the subset lists):

| split | clear | fog2 | night | total |
|---|---:|---:|---:|---:|
| train | 1,954 | 1,954 | 1,954 | **5,862** |
| val | 666 | 666 | 666 | 1,998 |

Balanced three ways, and 5,862 per epoch matches every other arm in the project
so the comparisons stay valid.

---

## 3. Losses — and how the router is trained

**One optimiser. One backward pass. The router is NOT trained separately.**

```
L_total = L_box(CIoU) + L_cls(BCE) + L_dfl  +  lambda_gate * L_gate(BCE)
          \____________ Ultralytics v8DetectionLoss ____/     lambda_gate = 1.0
```

The gate term is added into the detection loss tensor before `.backward()`
(`src/models/moe_trainer.py::_add_gate_supervision`), so detector and router
train **jointly, end to end, from step 0**. There is no router pre-training
stage, no frozen-backbone phase, and no alternating schedule.

| loss | form | on |
|---|---|---|
| box | CIoU | detection |
| classification | BCE | detection |
| DFL | distribution focal | detection |
| **gate** | **BCE vs multi-hot condition label**, lambda = 1.0 | **the router** |

**Where the gate labels come from:** the corpus filenames. Every union image is
named `<condition>_<id>`, so `fog2_00042` yields the target `[0, 1, 0]`. The
supervision is therefore free — no extra annotation. Multi-hot rather than
one-hot, so a future `fog_night_*` compound image already produces `[0, 1, 1]`
with no code change.

Two guards, both added after they bit:

- Supervision is **training-only**. The validator calls `model.loss(batch, preds)`
  with predictions precomputed, so no forward re-runs and the stashed gate logits
  belong to a previous batch.
- Aliases (`fog2 -> fog`) are resolved, or the filename token matches no branch
  and every sample is silently dropped from the supervision while the loss still
  looks healthy.

Also present in the block but **not** the main objective:

- `noise_std = 0.5` Gaussian noise on gate logits, **training only**, for
  exploration. Eval routing is deterministic.
- The entropy/switch auxiliary balance loss exists in the older `MoEBlock` and is
  **inactive** here — supervision replaced it.
- The NWD box loss was a **separate experiment arm**, not part of this model.

---

## 4. Training recipe

Identical to every other arm in the project — that is the point, since it makes
the arms comparable.

| setting | value |
|---|---|
| init | COCO-pretrained `yolo11n.pt` (448/499 tensors transfer; head is 20-class) |
| epochs | 100 |
| batch | 16 |
| image size | 640 |
| optimiser | Ultralytics `auto` -> SGD, lr0 0.01, momentum 0.937, wd 5e-4 |
| schedule | cosine, 3 warmup epochs |
| seed | 0, `deterministic: true` |
| patience | 30 (never triggered) |
| hardware | RTX 5070 Ti, ~76 min |

---

## 5. How the configuration evolved

Every row is a completed 100-epoch run unless noted. This is the path, including
the parts that were wrong.

| # | run | data | model | what it tested | outcome |
|---|---|---|---|---|---|
| 1 | `fog_yolo11n` | fog, 651 scenes | stock | first end-to-end run, **from scratch** | 0.335 test mAP50 — plumbing only |
| 2 | `clear/fog_yolo11n_obb` | 651 scenes | stock OBB | box format | OBB/HBB not comparable; NIRNet protocol replicated |
| — | *data recovery* | — | — | pixel-hash id recovery | **651 -> 5,835 scenes** |
| 3 | `clear_full_yolo11n` | clear, 5,862 | stock | **Gate 1** | **65.2 VOC07** vs 57.1 reference, passed |
| 4 | `fog_full`, `night_full` | each condition | stock | specialists | interference matrix |
| 5 | `union_full`, `union3_full` | 2- / 3-condition | stock | dense controls | fog deficit +0.0005 -> "no interference" |
| 6 | `moe_backbone`, `moe_neck` | union | `MoEBlock` x2 | placement | neck > backbone; **both null** |
| 7 | `moe3_neck` | union3 | `MoEBlock` x3 | expert count | null again |
| — | *router measured* | — | — | NMI, CKA | **NMI 0.000, CKA 0.996 — never routed. Results withdrawn** |
| 8 | `union_nwd`, `moe_neck_nwd` | union | + NWD loss | tiny-object loss | +0.008 aggregate, **-0.006 on small classes** |
| 9 | `cond3_gated` | union3 | **`CondMoEBlock`** | supervision + static priors | **NMI 0.473**, CKA 0.79 |
| 10 | `cond3_nogate` | union3 | same block | supervision ablation | NMI **0.000** — supervision is the cause |
| 11 | `plain3_gated` | union3 | identical priors | prior ablation | CKA **0.94** — priors are the cause |
| — | *fog recalibrated* | — | — | vs RRSHID real haze | old fog weaker than real **moderate** |
| 12 | `fog2_full` | fog2 | stock | calibrated specialist | 0.6883 |
| 13 | `union3b_full` | union3b | stock | dense control | 0.6627 |
| 14 | **`cond3b_gated`** | **union3b** | **`CondMoEBlock`** | **final** | **NMI 0.542**, -0.0044 vs dense |
| 15 | `dv_dark`, `dv_lit` | DroneVehicle | stock (nc=2) | real low light | 93.3% retention @mAP50, 78.9% @mAP50-95 |

### The three architectural generations

| | gen 1 `MoEBlock` | gen 2 `CondMoEBlock` |
|---|---|---|
| experts differ by | kernel size only (3x3 / 5x5) | **fixed physical priors** |
| gate | unsupervised, entropy balance loss | **supervised BCE on condition labels** |
| selection | top-1 hard (softmax argmax) | **multi-label sigmoid, threshold 0.5** |
| always-on branch | optional, off | **on** |
| **NMI(route ; condition)** | **0.000** | **0.542** |
| **inter-expert CKA** | **0.968 – 0.996** | **0.831 – 0.941** |

---

## 6. Where it stands

**Works:** the router routes (NMI 0.542), the experts compute different
functions (CKA 0.83), interference exists to be recovered (~2 points on each
condition), and the gate costs 771 parameters.

**Does not work yet:** the block does not convert any of that into accuracy —
0.6583 against the dense control's 0.6627 on the same data.

**Most likely cause — SUPERSEDED, see `results-routing-cost.md`.** The original
diagnosis here was that the gate predicted uniformly low. That was measured and
found wrong: `sum(p)` is already 1.012, so the total probability mass is
correct. The gate **spreads** it — roughly (0.50, 0.29, 0.22) rather than
(1.0, 0, 0) — so the maximum barely clears the 0.5 threshold.

A routing cost was implemented and run (`cond3c_cost_yolo11n`). It improved
every routing metric (NMI 0.542 -> 0.596, fog purity 0.596 -> 0.682) and made
accuracy slightly **worse** (0.6583 -> 0.6558). That is now the third
intervention to improve routing without improving detection, so the open problem
has moved again: **routing quality and accuracy have decoupled**, and the next
work belongs on the experts, not the router.
