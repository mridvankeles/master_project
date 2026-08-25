# Routing cost: making the gate commit to an expert

> **Date:** 2026-08-26 · answers "can we charge the gate for using the wrong
> expert, or none at all — and is there literature on it?"
> **Short answer:** yes to both, the idea has a name, and our version is a
> supervised variant the standard one cannot express.

---

## 0. First, a correction worth making explicitly

**Firing more experts makes inference SLOWER, not faster.** Two goals are being
conflated and they pull in opposite directions:

| goal | wants |
|---|---|
| **specialisation** (accuracy, robustness) | the *correct* expert to fire |
| **efficiency** (inference time) | *few* experts to fire |

Right now `cond3b_gated` fires **0.466 experts per image**. That is cheap, but it
is the wrong kind of cheap: it is not efficient routing, it is **most images
getting no specialist at all** and passing through the shared branch alone. 53%
of images run only the 75,840-parameter shared expert.

So the target is not "more firing". It is **~1.0 experts per image, and the
right one** — which is *more* compute than now (0.466 → 1.0) but far less than a
dense 3-expert block (3.0).

**And a separate problem the routing cost will not fix:** our MoE is already
slower than the dense baseline (8.96 vs 6.52 GFLOPs) because the block we
inserted is more expensive than the `C3k2` it replaced. Perfect routing still
would not make it faster until the experts are sized *below* the block they
replace. That is a sizing change, independent of everything below.

---

## 1. The idea has a name — several, in fact

| mechanism | what it charges for | source |
|---|---|---|
| **Per-sample sparsity / commitment loss** | activating many experts for one sample; "encourages individual samples to commit to a single expert" | sparse-MoE literature; LD-MoLE (2025) uses an explicit sparsity target |
| **Sparsity L1** | L1 norm of normalised router probabilities, pushing them toward the axes | same |
| **Load-balancing (CV / mean-importance)** | uneven expert usage **across a batch** | Switch Transformers, GLaM |
| **z-loss** | router logit magnitude, for stability | ST-MoE |
| **Capacity factor + token dropping** | more tokens than an expert's quota | GShard, Switch |
| **Expert-choice routing** | inverts the direction — experts pick a fixed number of samples | Zhou et al. 2022 |
| **Adaptive-k** | more experts for harder examples | survey ref [81] |
| **Auxiliary-loss-free balancing** | per-expert **bias terms** updated from recent routing statistics, no extra gradients | Wang et al., survey ref [85] |
| **Dual-ascent with sparsemax** | target usage ratios as a constraint, task loss untouched | Thaman, survey ref [86] |

The MoE survey in `other-papers/` covers most of these (17 mentions of load
balancing, plus z-loss, expert-choice, adaptive-k and the auxiliary-loss-free
line), so the ground is well trodden.

### Why the standard tools do not fit our failure

**Load balancing is the wrong instrument.** Switch-style balancing constrains
usage *across a batch*: it ensures each expert gets a fair share overall. Our
failure is *per image* — the batch-level distribution can look perfectly
reasonable while every individual image activates nothing. A batch statistic
cannot see that, which is the same blind spot that made the entropy auxiliary
report a collapsed router as healthy earlier in this project.

**Capacity factors solve the opposite problem.** They cap over-subscription.
Ours is under-subscription.

**Auxiliary-loss-free bias adaptation (ref [85]) is genuinely relevant** and
worth trying second: per-expert biases nudged from recent routing statistics
would raise activation without touching the objective at all.

---

## 2. What we implemented

One term, doing both jobs asked for:

```
L_count = | sum_i p_i  -  n_true |
```

with `n_true` = the number of conditions actually present, read from the same
free filename labels the gate is already supervised with.

- **too few experts fire** (sum below `n_true`) → a **coverage** cost. This is
  "charging for choosing no expert".
- **too many fire** (sum above `n_true`) → a **sparsity** cost. This is
  "charging for going through experts the image does not need".

Plus `pos_weight = 2.0` on the gate BCE. With three outputs and typically one
positive, two thirds of every target is zero, so BCE is minimised by predicting
low — which is exactly the measured pathology (probabilities peak on the correct
branch but reach only 0.336). `n_experts − 1 = 2` restores the balance between
the positive and negative halves of the objective.

**Why the supervised target matters.** The standard commitment loss uses a fixed
`k`. Ours reads `n_true` per image, so a compound `fog_night` image is asked for
**two** experts while a single-condition image is asked for one. A fixed `k=1`
would actively fight the multi-label design — and compound conditions are the
case the whole architecture exists to handle.

Verified behaviour on synthetic gates:

| gate state | cost |
|---|---:|
| confident single expert (0.9, 0.05, 0.05) | **0.20** |
| under-firing — our actual bug (0.1, 0.1, 0.1) | **0.90** |
| over-firing (0.9, 0.9, 0.9) | **1.50** |

Both failure directions are charged, and the healthy state is cheapest.

Two new per-epoch diagnostics, both **threshold-free**, so calibration is
visible without re-running anything: `expected_experts` (`sum(p)`) and
`gate_max_prob`.

---

## 3. Is the next move certain?

**No — and it is worth being precise about which parts are certain.**

| claim | confidence | why |
|---|---|---|
| `pos_weight` + count cost will **raise activation** | **near-certain** | mechanical: both terms directly penalise low `sum(p)`; the gradient points that way by construction |
| it will make the gate **better calibrated** | likely | it is the textbook fix for the exact class imbalance measured |
| it will **improve mAP** | **genuinely uncertain** | more of the right expert firing *should* help, but every MoE configuration so far has landed 0.003–0.006 below dense, and no experiment has yet shown routing converting into accuracy in this network |
| it will **improve inference time** | **no** | more firing costs more compute; efficiency needs the separate sizing fix |

So this is one experiment, not a certainty — which is why it is being run rather
than assumed, and why the acceptance criteria are checked before the accuracy
number is looked at.

**A cheaper check ran first.** The activation threshold is an *inference-time*
attribute, so the accuracy-vs-firing curve can be swept on the existing
checkpoint with no retraining. If accuracy is flat or falling as more experts
fire, then firing is not the bottleneck and the training run would have been
wasted.

### If the routing cost does not help

Ordered by what the evidence would then point at:

1. **Expert capacity.** Inter-expert CKA is still 0.83–0.94 — distinct, but not
   very. Bottleneck width or prior strength may be limiting.
2. **Placement.** The block sits at one neck level (P3); the degradation acts on
   the *input*, so backbone placement may be where interference actually lives.
3. **Auxiliary-loss-free bias adaptation** (survey ref [85]) as an alternative
   calibration route that leaves the objective alone.
4. **The honest possibility:** ~2 points of interference may simply not be
   recoverable by a 450k-parameter block on a 2.6M-parameter network.

---

## 4. Configuration

```yaml
gate:
  lambda: 1.0        # weight on the whole gate objective
  pos_weight: 2.0    # n_experts - 1; rebalances BCE
  count_lambda: 0.5  # weight on |sum(p) - n_true|
```

Run: `configs/train/cond3c_cost_yolo11n.yaml` — identical to `cond3b_gated` in
every other respect, so the routing cost is the only variable.

## Sources

Local: `other-papers/JGEO-...pdf` (MoE in Remote Sensing survey — load
balancing, z-loss, expert-choice, adaptive-k, auxiliary-loss-free balancing,
capacity factors).

Web: LD-MoLE (learnable dynamic routing with an explicit sparsity target,
arXiv 2509.25684) · Switch Transformers / GLaM load balancing · ST-MoE z-loss ·
expert-choice routing · adaptive-expert-weight load balance schemes (2025).
