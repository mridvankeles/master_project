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

### What a genuinely different operator looks like

| candidate | vs clear | vs fog | vs night |
|---|---:|---:|---:|
| `x - mu` (subtractive) | 0.9411 | **0.9991** | 0.9814 |
| `(x - mu)/sd` (contrast norm) | 0.9275 | 0.9720 | **0.9981** |
| **`x / mu` (divisive)** | **0.3003** | **0.3087** | **0.3287** |

Everything that removes the local mean lands in the same subspace. The divisive
family does not. That is a factor-of-three difference in the project's own
diversity metric, available from a one-line change.

### A caveat on the metric itself

Linear CKA is generous to linear maps — any full-rank linear operator largely
preserves sample-space similarity structure. Cosine similarity between the same
learned expert outputs is **+0.020 / +0.059 / +0.026**: near-orthogonal. The two
metrics disagree completely, and the project has been steering on CKA alone.

The honest reading: the experts produce **different outputs carrying the same
information**. Both facts matter, and reporting one without the other has been
misleading. Future routing reports should carry both.

---

## 3. What this implies

- **The efficiency claim needs rewriting.** Conditional routing here reduces
  arithmetic by 6% and reduces throughput by 39%. On this hardware, at this
  block size, sparsity is a loss. It becomes a win only if the experts are
  large enough for the saved convolution to dominate the gather/scatter, or if
  the whole batch routes the same way (condition-sorted batching), or with a
  fused kernel.
- **The prior redesign is the highest-leverage change available**, and it is
  cheap: three lines in `src/models/experts.py`.
- **The experts should get smaller, not larger.** At 15–41% of the model each,
  they are not corrections; and a smaller expert makes the sparsity arithmetic
  worse, not better, which is another reason the efficiency story has to be
  argued on activated FLOPs rather than wall clock.
