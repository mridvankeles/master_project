# The residual MoE line is closed — five designs, one verdict

> **Date:** 2026-09-15. Final entry for the condition-gated *residual* MoE at the
> P3 neck. Every number measured, every arm matched on data, schedule, seed and
> augmentation to its own dense control.

---

## The result

`scripts/expert_intervention.py` replaces the routing policy with a fixed one and
re-validates. The **`none`** column is *every expert switched off*.

| design | what it added | spread (max − min over 6 policies) |
|---|---|---:|
| 2 `cond3d` | supervised gate, heterogeneous static priors | 0.0011 – 0.0017 |
| 3 `cond3h` | per-degradation architectures, hard mask, rich proj, non-zero init, block-level restoration | 0.0010 – 0.0020 |
| 3i `cond3i` | **per-expert reconstruction losses, bypass penalty** | 0.0013 – 0.0023 |
| 3j `cond3j` | **the above, with the shared branch deleted entirely** | 0.0017 – 0.0039 |

**Criterion set in advance: > 0.01. Never reached.**

The most direct statement is in `cond3j`. With no shared branch, `shortcut_only`
is **0.333** — for a third of all images the block output is *literally just
`proj(x)`*, one convolution block, no expert at all. Test mAP is unaffected.

---

## What makes this conclusive rather than merely negative

The experts are not failing to learn. They learn exactly what they are asked to,
and it changes nothing.

| | `cond3h` | `cond3i` | `cond3j` |
|---|---:|---:|---:|
| restoration gain, **fog** | −0.0368 | +0.0453 | **+0.0980** (88% of images improved) |
| restoration gain, **night** | −0.0095 | +0.0284 | **+0.0805** (93% improved) |
| `expert_share` | 0.047 → — | 0.320 | 0.227 |
| **intervention spread** | 0.0010–0.0020 | 0.0013–0.0023 | 0.0017–0.0039 |

Across three designs the restoration gain went from **negative to +0.098** — the
experts move degraded P3 features nearly 10% closer to their clear twin, on
88–93% of images — and the intervention did not move.

**Feature-space restoration and detection are decoupled.** That is the finding.
It is stronger than "the MoE didn't help", because it isolates *which link in the
chain fails*: not the router, not the expert objective, not expert magnitude —
the step from a restored neck feature to a changed prediction.

A specialisation ordering did finally emerge, and it is correct:

| data | clear | fog | night |
|---|---:|---:|---:|
| fog2 | 0.1259 | **0.0980** | 0.0886 |
| night | 0.1466 | 0.0371 | **0.0805** |

Fog beats night on fog data; night beats fog on night data. `clear` beats both,
because "move towards the clear twin" is most directly served by a branch that
behaves like a gentle identity — which is what `ClearExpert` is.

---

## Accuracy and cost, final

| run | model | test mAP50 | test mAP50-95 | vs its dense control |
|---|---|---:|---:|---:|
| **`union3b_full`** | **dense, mosaic on** | **0.6627** | **0.4632** | — |
| `cond3b_gated` | MoE d2 | 0.6583 | 0.4597 | −0.0044 |
| `union3h_dense` | dense | 0.6296 | 0.4390 | — |
| `cond3h_hetero` | MoE d3 | 0.6199 | 0.4316 | −0.0097 |
| `cond3i_perexpert` | MoE d3 + per-expert losses | 0.6220 | 0.4327 | −0.0076 |
| `cond3j_noshared` | MoE d3, no shared branch | 0.6158 | 0.4266 | −0.0138 |

No MoE arm ever beat its dense control. The best model in the project is plain
YOLO11n.

| run | params | GF activated | FPS @b16 |
|---|---:|---:|---:|
| `union3h_dense` | 2.594 M | 6.52 | **1381.3** |
| `cond3d` | 3.014 M | **6.10** | 772.9 |
| `cond3j` | 3.202 M | 6.70 | 724.8 |
| `cond3h` / `cond3i` | 3.278 M | 7.62 / 7.65 | 696.2 / 477.5 |

Only design 2 ever did *less* arithmetic than dense, and it still ran 44% slower.
Masking, `nonzero`, gather and `index_add` cost more than the convolution they
skip, and the penalty grows with batch size.

---

## What DOES work, and belongs in the thesis

- **The router.** NMI(route ; condition) **0.83–0.87** across designs. The fog
  branch fires on **100%** of fog images, night on **100%** of night, purity
  0.85–0.99. It costs **771 parameters and ~5 kFLOPs**. Condition routing from a
  globally-pooled neck feature is solved.
- **Supervision is what creates routing.** Unsupervised, NMI 0.0000 — the gate
  splits on nothing. Supervised from free filename labels, 0.87.
- **Gate 1.** 65.2 VOC07 mAP on clear DIOR against a 57.1 published reference.
- **The measurement chain**, which is the real contribution: an intervention test
  that asks a causal question instead of a correlational one, and which overturned
  four conclusions that CKA, cosine similarity and training-loss curves all got
  wrong.

---

## Why it fails, stated precisely

The block is a **residual**, not a **mixture**:

```
out = proj(x) [+ shared(x)] + Σ_i p_i · expert_i(x)
      └──── always competent ────┘   └─ optional addition ─┘
```

The always-on path alone minimises the detection loss. Everything added to it is
optional *by construction*. Five designs, four auxiliary losses, two
architectures, and deleting the shared branch outright did not change that,
because `proj` cannot be deleted — it is what maps 256 channels to 64.

The one structural alternative is **AW-MoE** (Lin et al., in the papers folder):
each expert owns a **detection head** and pays its **own** loss,

```
L_CW = Σ_{w ∈ S} P_w · L_w(WSE_w)
```

so no expert can be ignored, because nothing else computes its output. That is a
different architecture, not a different loss — and it is the only remaining path
to a positive result.

---

## Recommendation

**Write this up as the thesis result.** It is a complete, well-instrumented
negative with a positive core:

> Condition routing in a small remote-sensing detector is easy, cheap and
> reliable — 771 parameters, NMI 0.87, 100% correct firing. Making the routed
> experts *matter* is the hard part, and a residual formulation at the neck
> cannot do it: experts that provably restore degraded features by 10% change no
> predictions, and conditional execution costs more wall-clock than it saves
> below ~6.5 GFLOPs.

Two things would strengthen it and both are cheap:

1. **The seed-noise floor.** 3 seeds, one config, ~1.5 h. Every deficit quoted
   above is 0.004–0.014 and we have never measured what a seed alone does. This
   is the cheapest credibility available and it should be done before submission.
2. **A P2/stride-4 head.** Orthogonal to the MoE and the largest effect in the
   project: 100% of objects under 8 px are missed, 63.7% under 16 px, and 98% of
   misses produce no box at any confidence threshold.

## What to stop

Tuning the router (solved), adding auxiliary losses to the residual block (five
designs of evidence), and judging expert diversity by CKA — it disagrees with
cosine on the same tensors and neither predicts the intervention.
