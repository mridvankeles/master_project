# Inference cost, and why the experts are clones

> **Date:** 2026-08-30 · `scripts/benchmark_speed.py`, `scripts/probe_expert_priors.py`
> All timings **without test-time augmentation** (`augment=False`, single scale,
> no flips), fp32, imgsz 640, RTX 5070 Ti, CUDA-synchronised, 30 warmup + 150
> timed iterations, on real `union3b` test images.

---

## 1. Inference cost

| run | params | GFLOPs static | **GFLOPs activated** | ms @b1 | FPS @b1 | FPS @b16 |
|---|---:|---:|---:|---:|---:|---:|
| `union3b_full` dense | 2.594 M | 6.52 | 6.52 | 8.92 | 112.1 | **1353.9** |
| `cond3b_gated` MoE | 3.014 M | 9.23 | **5.13** | 9.76 | 102.5 | 972.8 |
| `union3d_nomosaic` dense | 2.594 M | 6.52 | 6.52 | 8.81 | **113.6** | 1275.9 |
| `cond3d_nomosaic` MoE | 3.014 M | 9.23 | **6.10** | 10.42 | 96.0 | 772.9 |
| `cond3e_nonwd` MoE | 3.014 M | 9.23 | **6.05** | 9.42 | 106.2 | 782.9 |

Three numbers are all called "cost" and they disagree, which is the point:

- **static GFLOPs** (9.23) is what `model.info()` prints. It traces every branch,
  so for a conditional model it describes a network that never runs.
- **activated GFLOPs** (6.10) is what the design promises: 6.10 against the dense
  model's 6.52, so conditional routing genuinely does **6% less arithmetic**.
- **wall clock** goes the other way. The MoE is **18% slower at batch 1** and
  **39% slower at batch 16**.

**The conditional block saves FLOPs and loses time.** Skipping two of three
experts costs a boolean mask, a `nonzero`, a gather, a variable-sized batch per
expert, and an `index_add` — several kernel launches on tensors too small to
amortise them. The convolution it skips is cheaper than the machinery that skips
it. The penalty grows with batch size because a dense conv keeps getting more
efficient with more work while the scatter/gather does not.

### Where the compute goes

| | `cond3b` | `cond3d` | `cond3e_nonwd` |
|---|---:|---:|---:|
| experts active per image | 0.433 | 0.993 | 0.967 |
| **shortcut-only (no expert at all)** | **0.567** | 0.020 | 0.043 |
| clear fires / costs | 0.183 / 0.97 GF | 0.330 / 0.97 GF | 0.373 / 0.97 GF |
| fog fires / costs | 0.050 / **2.66 GF** | 0.300 / **2.66 GF** | 0.287 / **2.66 GF** |
| night fires / costs | 0.200 / 0.97 GF | 0.363 / 0.97 GF | 0.307 / 0.97 GF |
| all experts | 4.60 GF | 4.60 GF | 4.60 GF |
| → activated | 0.51 GF | 1.47 GF | 1.42 GF |
| **gate** | **0.000005 GF** | same | same |

Two things worth naming.

**`cond3b` really was taking the shortcut on 57% of test images.** That was the
original claim, and it was right — it just had the wrong cause. Mosaic was
corrupting the gate label; with mosaic off (`cond3d`) it is 2%.

**The experts are far too expensive for what they are.** One expert is 0.97
GFLOPs — **15% of the entire dense model** — and the fog branch, with its 5×5
kernel and extra channel, is 2.66 GFLOPs, **41% of the whole model**. These were
described as "corrections" on top of a working representation. They are priced
like a second backbone stage. The gate, by contrast, is 5 kFLOPs: free.

---

## 2. Why the experts are clones — measured, not guessed

`cond3d` has the best router the project has produced (NMI 0.8748) and
inter-expert CKA of 0.956–0.975. I hypothesised that collapse was a
**late-training** effect, since the 40-epoch arms were less collapsed. **That
hypothesis is wrong.** Measuring the fixed priors directly, on the real P3
features, with no learned weights involved:

| pair | **prior vs prior** (no learning) | learned expert outputs |
|---|---:|---:|
| clear vs fog | **0.9461** | 0.8282 |
| clear vs night | **0.9319** | 0.8210 |
| **fog vs night** | **0.9828** | **0.9767** |

The experts did not collapse. **They were never apart.** The learned CKA is
inherited almost exactly from the priors, including the ordering — fog vs night
is the most similar pair before training (0.983) and after (0.977), in every run
measured.

### The reason

Both "different physics" priors are the same operator family:

```
TransmissionPrior      x - avgpool_15(x)                      subtractive high-pass
IlluminationInvariant  log(x') - avgpool_7(log(x'))           subtractive high-pass, in log
```

On positive features `log` is locally near-affine, so the log-domain version is
approximately a scaled linear high-pass. Measured: the night prior scores
**0.9814** against a plain `x - avgpool_7(x)`, and the fog prior **0.9991**
against `x - avgpool_15(x)`. They differ in *kernel width*, which is exactly the
difference the first MoE had — and which
`finding-router-never-specialised.md` already established is not enough.

The docstring in `experts.py` claims "two experts cannot converge to the same
function no matter how the gate behaves". The measurement says the fixed part
imposes almost no distinguishing bias at all.

### What a genuinely different operator looks like — RETRACTED

**This section originally claimed that a divisive prior `x / mu` scores CKA 0.30
against all three, and that "the redesign is three lines". That was wrong, and
the error was mine: a numerical artifact, not a finding.**

70% of the local means of a post-SiLU feature map are negative and 2% are within
1e-3 of zero, so `x / (mu + 1e-3)` divides by near-zero and produces values up
to **2.4e7**. A handful of exploded locations then dominate the Gram matrix and
drag CKA down. Written stably the operator is not different at all:

| variant | CKA vs input | max abs output |
|---|---:|---:|
| `x / (mu + eps)` — what was published | **0.3192** | **24,235,042** |
| `x / (abs(mu) + eps)` | 0.9539 | 3,762 |
| `x / (abs(mu) + 0.1)` | 0.9796 | 61 |
| `log(abs(x)) - log(abs(mu))` | 0.9753 | 8 |

A CKA that falls only because a few locations explode is not diversity.

### The corrected result, from an actual search

`scripts/search_expert_priors.py` scores **36 candidate priors** — subtractive,
divisive, contrast-normalising, log-domain, local-range, dark-channel, Sobel,
phase-only and Fourier high-pass, across kernel widths 3–21 and dilations 1–2 —
on the real P3 features, with no training. Objective: minimise the *worst*
pairwise CKA in the triple, with `clear` pinned to identity.

| | worst pair |
|---|---:|
| current design (identity / sub_mu k15 / log_sub k7) | 0.9883 |
| **best of 36 candidates** (identity / range k15 / div_mu k7 d2) | **0.9504** |

**The entire searchable space buys 0.04 of CKA.** And the binding constraint is
always `identity vs X`: every candidate scores ≥0.95 against the raw input.

That is a much stronger negative than the one it replaces, and it points
somewhere else entirely. No local spatial filter of an already-normalised
feature map can be very different from it — these are all near-linear maps of a
common input, so they necessarily carry the same information. **The priors are
not badly chosen; they are in the wrong place.** By P3 the backbone's
BatchNorms have removed the first- and second-order statistics that *define*
fog and night, and `x.amin(dim=1)` over 256 SiLU channels is not a dark channel.
Light- and haze-invariance are properties of RGB, and that is where they have
to be computed.

### A caveat on the metric itself

Linear CKA is generous to linear maps — any full-rank linear operator largely
preserves sample-space similarity structure. Cosine similarity between the same
learned expert outputs is **+0.020 / +0.059 / +0.026**: near-orthogonal. The two
metrics disagree completely, and the project has been steering on CKA alone.

The honest reading: the experts produce **different outputs carrying the same
information**. Both facts matter, and reporting one without the other has been
misleading. Future routing reports should carry both.

---

## 3. The decisive test: force each expert and see if anything changes

Every diversity metric argued about above is a **correlation**. The question
that matters is a **causal** one: if you send every image through the wrong
expert, does the detector get worse? `scripts/expert_intervention.py` replaces
the routing policy with a fixed one and re-runs validation.

**mAP50, `cond3d_nomosaic`, per condition:**

| data | gate | **none** | all | force:clear | force:fog | force:night | **spread** |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 0.7512 | **0.7510** | 0.7499 | 0.7515 | 0.7511 | 0.7506 | **0.0016** |
| fog2 | 0.7457 | **0.7456** | 0.7445 | 0.7449 | 0.7451 | 0.7450 | **0.0011** |
| night | 0.7281 | **0.7283** | 0.7267 | 0.7270 | 0.7266 | 0.7279 | **0.0017** |

**mAP50-95** behaves identically: spread 0.0020–0.0024.

Read the `none` column. That is **every expert switched off** — the shared
always-on branch alone — and it scores 0.7456 on fog against the trained gate's
0.7457, and 0.7283 on night against 0.7281. Switching off the entire mixture
costs nothing. Forcing the *wrong* expert costs nothing. Forcing *all three*
costs nothing.

**The experts are functionally inert.** The block is a dense model with an
expensive unused branch. That single fact explains everything the project has
been unable to explain:

- why four interventions improved routing and none improved accuracy — nothing
  downstream depended on the route;
- why the MoE tracks its dense control to within 0.005 — it *is* the dense
  control;
- why `cond3b` (57% shortcut-only) scored the same as `cond3d` (2%) — the
  shortcut is as good as the experts;
- why the CKA argument was unresolvable — it was measuring the geometry of a
  signal that never reaches the loss.

### Why they are inert

| | proj | shared | clear | fog | night |
|---|---:|---:|---:|---:|---:|
| `cond3d` output RMS | 0.361 | 0.322 | 0.101 | 0.108 | 0.088 |
| as % of (proj+shared) | — | — | **19.9%** | **21.1%** | **17.2%** |
| `cond3b` as % | — | — | 10.8% | 10.0% | 9.9% |
| `cond3e_nonwd` as % | — | — | 12.3% | 12.4% | 18.5% |

An expert contributes 10–21% of the always-on path's magnitude — and is then
multiplied by its gate probability (~0.65–0.77), so ~7–16% reaches the sum. A
perturbation that size on one neck feature map, with a full detection head
downstream, moves mAP by less than 0.002.

Two design decisions produce this, and both were deliberate:

1. **`zero_init_output()` starts every expert at exactly zero** while the shared
   branch starts working immediately. The shared branch therefore takes the job
   on step 1, the detection loss is satisfied, and no gradient pressure ever
   builds to make the experts grow. This is the classic residual-MoE failure:
   an always-competent bypass makes the specialists optional, and optional
   branches stay small.
2. **The expert output is scaled by the gate probability** `p_i`, so selection
   and magnitude are the same number. A branch can only reach full strength if
   the gate is fully confident, and confidence is capped by the BCE.

### What this makes testable

- **Decouple selection from magnitude.** Straight-through hard mask: the expert
  runs at weight 1.0 when selected, with the gradient still flowing to the gate
  through `p`. Removes cause (2) with no architecture change.
- **Weaken or remove the shared branch.** If it is always competent the experts
  are never needed. Halving its width, or zero-initialising *it* instead, forces
  the routed branches to carry real function. Removes cause (1).
- **Then, and only then**, prior design and kernel width become worth searching.
  Tuning the shape of a branch that contributes 7% of one feature map is
  measuring noise.

## 3. What this implies

- **The efficiency claim needs rewriting.** Conditional routing here reduces
  arithmetic by 6% and reduces throughput by 39%. On this hardware, at this
  block size, sparsity is a loss. It becomes a win only if the experts are
  large enough for the saved convolution to dominate the gather/scatter, or if
  the whole batch routes the same way (condition-sorted batching), or with a
  fused kernel.
- **The prior redesign is NOT the highest-leverage change.** A 36-candidate
  search buys 0.04 of CKA, and the intervention above shows the branches those
  priors feed contribute nothing measurable anyway. Expert *contribution* has to
  be fixed before expert *design* means anything.
- **The experts should get smaller, not larger.** At 15–41% of the model each,
  they are not corrections; and a smaller expert makes the sparsity arithmetic
  worse, not better, which is another reason the efficiency story has to be
  argued on activated FLOPs rather than wall clock.
