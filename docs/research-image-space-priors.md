# Classical priors belong on the image, not on the feature map

> **Date:** 2026-09-14 · `scripts/search_image_priors.py`, training-free,
> 120 real val images (40 per condition), operators at 640 px then pooled to the
> 80×80 grid an expert actually receives.

---

## The question

Design 2's experts already *were* classical methods — `TransmissionPrior` is
dark-channel + veil removal, `IlluminationInvariant` is homomorphic filtering /
single-scale Retinex. A 36-operator search over P3 features then found the whole
space buys **0.04** of CKA (best worst-pair 0.9504 against the current 0.9883).

The conclusion recorded at the time was that the priors are not badly chosen,
they are in the wrong **place**: by P3 the backbone's BatchNorms have removed
the first- and second-order statistics that define fog and night, and a
channel-min over 256 SiLU activations is not a dark channel. This tests that.

## Answer: yes, by a factor of five

| | best achievable worst-pair CKA |
|---|---:|
| current design (feature space) | 0.9883 |
| best of 36 operators, **feature space** | 0.9504 |
| best of 12 operators, **image space** | **0.2011** |

Computed where they mean what they are supposed to mean, classical operators are
genuinely different from each other. `sobel` vs `identity` is **0.020**;
`harris` vs `identity` **0.012**; `lcn` vs `identity` **0.083**. In feature space
nothing scored below 0.95 against the input.

So the place *was* the problem.

---

## The twist: diversity and condition-sensitivity are anti-correlated

The second table is the one that changes the design. Effect size = between-
condition spread of the operator's mean response, in within-condition SDs:

| operator | suits | **condition effect** | CKA vs identity |
|---|---|---:|---:|
| `dark_channel` | fog | **2.99** | 0.926 |
| `identity` | — | 2.69 | 1.000 |
| `value` | night | 2.61 | **0.998** |
| `transmission` | fog | 2.28 | 0.823 |
| `clahe` | night | 2.01 | 0.944 |
| `dcp_dehazed` | fog | 1.43 | 0.640 |
| `saturation` | fog | 1.38 | 0.607 |
| `sobel` | edges | 1.23 | **0.020** |
| `retinex` | night | 0.86 | 0.181 |
| `lcn` | night | **0.46** | **0.083** |
| `harris` | corners | 0.33 | 0.012 |
| `laplacian` | edges | **0.13** | 0.028 |

**The operators that respond most strongly to the condition are near-duplicates
of each other**, because they are all measuring roughly the same thing — how
bright or how hazy the scene is. `value` vs `identity` is 0.998; `dark_channel`
vs `transmission` 0.954.

**The operators that are mutually distinct are nearly invariant to the
condition** — `laplacian` 0.13, `harris` 0.33, `lcn` 0.46.

That is not a defect. It is a job assignment:

- **Condition-sensitive operators belong to the GATE.** They separate clear from
  fog from night, which is what a router needs. (Ours already achieves NMI 0.84
  reading P3, so this is confirmation rather than a change.)
- **Condition-invariant operators belong to the EXPERTS.** An operator whose
  response barely moves between clear, fog and night is precisely a
  *light-invariant* or *haze-invariant* feature — the thing this project has
  been trying to build since the FeatEnHancer discussion. `lcn` with an effect
  size of 0.46 is that property, measured rather than asserted.

---

## The design this points to

| expert | operator | why | CKA vs the others |
|---|---|---|---|
| clear | `identity` | control branch | — |
| **fog** | **`dcp_dehazed`** | actually inverts the ASM: `J = (I−A)/max(t,t₀) + A`. A real cleaning filter, not an analogy | 0.640 vs identity, 0.172 vs lcn |
| **night** | **`lcn`** | local contrast normalisation — "increase contrast", and the most condition-invariant operator with real signal | 0.083 vs identity |

Worst pair **0.640**, against the current design's **0.9883**. Every operator is
the textbook classical method for its condition, and `sobel`/`harris` are held
in reserve for a tiny-object expert, where their near-zero CKA against
everything else is exactly what a fourth branch would want.

---

## Two caveats, and they decide the ordering

**1. This is downstream of the contribution problem.** `expert_intervention.py`
has shown three times that switching every expert off costs 0.001–0.002 mAP.
Diverse-but-inert experts are still inert. Prior diversity only becomes
measurable once an expert's output changes the prediction at all, which is what
`cond3i` (per-expert reconstruction + bypass penalty) and `cond3j` (no shared
branch) are testing now. **Do those first.**

**2. It needs plumbing that does not exist.** The block sits at neck index 16 and
receives only the P3 feature map; the raw image is not in the graph there.
Getting it means caching the input with a forward pre-hook on `model.model[0]`
and reading it from the block — the same pattern the probe scripts already use,
but it has to survive `save_model`, EMA deepcopy, and export. Budget half a day,
not an afternoon.

## What this retires

The idea that *any* re-selection of fixed priors at P3 could have worked. It
could not: 36 candidates, none below 0.95 against the raw input. The three
design-2 priors were not a poor choice within that space — the space itself was
empty.
