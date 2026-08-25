# Routing cost: the term worked, my diagnosis did not

> **Date:** 2026-08-26 · single seed · run `cond3c_cost_yolo11n`
> **Corrects** the "next experiment" section of `MODEL-AND-TRAINING-CONFIG.md`
> and §2 of `research-routing-cost.md`.

---

## 1. What was predicted, and what happened

The prediction was: the gate under-fires because BCE with one positive in three
is minimised by predicting low, so `sum(p)` is too small; charging
`|sum(p) − n_true|` plus `pos_weight=2` will raise activation and, plausibly,
accuracy.

| | `cond3b` (no cost) | **`cond3c` (routing cost)** |
|---|---:|---:|
| NMI(route ; condition) | 0.5421 | **0.5960** |
| fog purity | 0.596 | **0.682** |
| night purity | 0.882 | **0.905** |
| clear purity | 0.941 | 0.941 |
| experts active / image | 0.466 | **0.557** |
| **mAP50 on union3b** | **0.6583** | **0.6558** |
| vs dense (0.6627) | −0.0044 | **−0.0069** |

**Routing got better on every measure. Accuracy got worse.**

---

## 2. The diagnosis was wrong, and the measurement says so

I claimed the gate was predicting *low* — that `sum(p)` fell short of the number
of true conditions. Measured on both checkpoints:

| | `cond3b` (no cost) | `cond3c` (cost) |
|---|---:|---:|
| **sum(p)** | **1.012** | 1.030 |
| max p | 0.500 | 0.532 |
| 2nd p | 0.290 | 0.284 |
| 3rd p | 0.222 | 0.214 |
| **concentration** (max / sum) | **0.492** | 0.514 |
| fraction with max > 0.5 | 0.457 | 0.560 |

**`sum(p)` was already 1.012 before the cost was ever added.** The gate was not
predicting low. It was **spreading** its mass — roughly (0.50, 0.29, 0.22)
instead of (1.0, 0.0, 0.0).

So `|sum(p) − n_true|` was aimed at a quantity that was already correct, and it
admits precisely the degenerate solution the gate had already found: a uniform
(0.33, 0.33, 0.33) satisfies `sum = 1` exactly as well as a confident
(1.0, 0.0, 0.0). The term could not distinguish the failure from the fix.

It still helped slightly — concentration 0.492 → 0.514, firing 0.457 → 0.560 —
because `pos_weight` pushed all probabilities up. But that is the BCE doing the
work, not the count term.

**The real quantity is concentration, not count.** The correct term penalises a
*spread* distribution: a per-sample entropy penalty on the gate, or a margin
between the top-1 and top-2 probabilities. The BCE nominally wants (1, 0, 0)
already, so the question is why it loses — most likely because at λ=1.0 against
a detection loss of ~2–3, the gate objective is simply outvoted once it is
"roughly right".

---

## 3. The finding that matters more

This is now the **third independent** piece of evidence that routing quality is
not what is holding the MoE back:

| improvement | routing | accuracy vs dense |
|---|---|---:|
| supervision added (`cond3_gated`) | NMI 0.000 → 0.473 | −0.0044 |
| fog calibrated (`cond3b_gated`) | NMI 0.473 → 0.542 | −0.0044 |
| routing cost (`cond3c_cost`) | NMI 0.542 → **0.596** | **−0.0069** |

NMI has improved monotonically from 0.000 to 0.596 across three interventions.
Accuracy has not improved once, and on the last step moved the wrong way. Better
routing is not translating into better detection — and the correlation, if
anything, is now mildly negative.

That reframes the open problem. It is not "make the router better". Candidate
explanations, in the order the evidence supports them:

1. **The experts are too weak to be worth routing to.** Inter-expert CKA is
   0.82–0.96 and *rose* in `cond3c` (0.884 / 0.824 / 0.957). Routing more
   confidently to branches that compute nearly the same thing cannot help, and
   the gate spending capacity on the routing objective may be costing the
   detector directly.
2. **The gate weight scales the expert down.** Each expert's contribution is
   multiplied by its probability, which averages ~0.5. Experts run at half
   strength even when selected. Straight-through (multiply by 1, gradient
   through p) would decouple selection from magnitude.
3. **λ = 1.0 on the gate is too high.** The gate objective may be competing with
   detection for the same capacity rather than complementing it — consistent
   with accuracy falling as routing improved.

---

## 4. What I would do next, and what I would not

**Not** another routing-quality intervention. Three have now improved NMI
without improving accuracy; a fourth would most likely produce a fourth data
point on the same flat line.

**Instead**, in order:

1. **Decouple selection from magnitude** (straight-through weighting). Cheap,
   and directly tests explanation 2 — currently the best expert available is
   run at half strength.
2. **Sweep λ_gate** (1.0 → 0.3 → 0.1). If accuracy recovers as the gate
   objective weakens, explanation 3 is confirmed and the trade-off is explicit
   rather than hidden.
3. **Strengthen the experts** — wider bottleneck, or stronger priors — and
   re-measure CKA. Routing is worth nothing if the destinations are the same.

**The threshold sweep is still worth completing** since it costs no training:
if accuracy is flat or falling as more experts fire, that is direct evidence for
explanation 1 or 2 and rules out "the threshold was simply wrong".

---

## 5. Honest status

- The routing cost **works as specified** and improves every routing metric.
- **My stated reason for needing it was wrong** — `sum(p)` was already ≈ 1.0, so
  the term addressed a non-problem and admits the exact degenerate solution the
  gate had already found.
- Accuracy is now **−0.0069** against dense, the worst of any gated
  configuration, though still within the single-seed noise floor.
- The useful output is not the term. It is the demonstration that **routing
  quality and detection accuracy have decoupled in this architecture**, which
  points the next work at the experts rather than the router.
