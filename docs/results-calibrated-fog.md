# Calibrated fog reopens the interference claim

> **Date:** 2026-08-14 · single seed · continues `results-conditional-gating.md`
> and `research-expert-design.md`

Fog was re-synthesised at realistic strength (ASM calibrated to RRSHID,
dark-channel +110 against the release's +37/+62/+93). Everything else — model,
schedule, budget, seed — is unchanged, so haze strength is the only variable.

---

## 1. The headline: interference appears

| condition | specialist | union (dense) | **deficit** |
|---|---:|---:|---:|
| clear | 0.6984 | 0.6760 | +0.0225 |
| night | 0.6612 | 0.6405 | +0.0208 |
| fog — **weak** (Hazy-DIOR release) | 0.6594 | 0.6578 | **+0.0016** |
| fog — **calibrated** (fog2) | **0.6883** | 0.6648 | **+0.0235** |

**The fog deficit grows 15x, from +0.0016 to +0.0235**, purely by making the
haze as strong as real haze. `results-full-scale-and-moe.md` concluded there was
"essentially no interference for routing to recover" on fog. That conclusion was
an artefact of a corpus whose thickest setting was weaker than real *moderate*
fog. **Claim C2 is reopened.**

All three conditions now sit at a consistent ~2-point deficit, which is a far
more coherent picture than one condition behaving completely differently from
the other two.

### The confound, and why it does not explain the result

The specialist sees more unique scenes than the union model does, and that ratio
is not identical across arms:

| arm | specialist scenes | union share | ratio | deficit |
|---|---:|---:|---:|---:|
| clear | 5,862 | 1,954 | **3.00x** | +0.0225 |
| night | 5,862 | 1,954 | **3.00x** | +0.0208 |
| fog2 | 5,862 | 1,954 | **3.00x** | +0.0235 |
| fog (weak) | 4,138 | 1,734 | 2.39x | +0.0016 |

Weak fog carries three severities per scene, so 5,862 images cover only 4,138
distinct scenes; fog2 has one render per scene and covers 5,862.

At **identical** 3.00x ratios, clear, night and calibrated fog all land within
0.003 of each other. Weak fog, at a only slightly lower 2.39x, should have shown
roughly +0.018 if scene diversity were driving the effect. It shows +0.0016.
**Diversity cannot account for a 15x difference**, and the remaining explanation
is the one the synthesis measurement already predicted: a degradation that
barely degrades cannot interfere with anything.

Honest residue: some part of +0.0235 is diversity rather than interference. The
clean version of this experiment gives every arm one render per scene, and is
one training run away.

---

## 2. Routing also improves

Same architecture, only haze strength differs:

| | weak fog (`cond3_gated`) | **calibrated (`cond3b_gated`)** |
|---|---:|---:|
| **NMI(route ; condition)** | 0.4731 | **0.5421** |
| fog purity | 0.522 | **0.596** |
| clear / night purity | 0.952 / 0.872 | 0.941 / 0.882 |
| experts active per image | 0.287 | **0.466** |
| clear branch fires on clear | 0.324 | **0.655** |
| fog branch fires on fog | 0.016 | **0.137** |
| inter-expert CKA | 0.789–0.927 | 0.831–0.941 |

Routing improves and the gate becomes markedly more confident — activation
roughly doubles. Fog remains the hardest condition at 0.596 purity, so stronger
haze narrows the clear/fog confusion without closing it; some of it is
intrinsic, and thin haze on a high-contrast scene genuinely does resemble clear.

---

## 3. But the MoE still does not exploit the interference

| tested on | dense | cond3b MoE | Δ |
|---|---:|---:|---:|
| union3b | 0.6627 | 0.6583 | −0.0044 |
| fog2 | 0.6648 | 0.6615 | −0.0033 |

This is the same −0.003 to −0.006 seen in every previous configuration. So the
situation has changed in an important way:

- **Before:** no interference existed, so the MoE had nothing to recover, and
  the null result said nothing about routing.
- **Now:** interference demonstrably exists (~2 points on every condition), the
  router demonstrably routes (NMI 0.54), the experts are demonstrably distinct
  (CKA 0.83) — **and the MoE still does not recover it.**

That is a much more informative negative, and it points at the block rather than
at the premise. The most likely candidates, in order:

1. **The gate is miscalibrated.** Only 0.466 experts activate per image at the
   0.5 threshold; the correct branch fires 13.7% of the time on fog. Most images
   are effectively passing through the shared branch alone, so most of the time
   there is no routing to benefit from. **This is the first thing to fix** and it
   is a `pos_weight` change, not a redesign.
2. **The experts are still too similar** (CKA 0.83–0.94). Distinct enough not to
   be clones, not obviously distinct enough to specialise.
3. **The block sits at one neck level.** Interference may live in the backbone,
   where the degradation acts, rather than at P3.

---

## 4. Real low-light, for reference

DroneVehicle, real night imagery, no simulator involved:

| | mAP50 | mAP50-95 |
|---|---:|---:|
| dark (brightness 32.5) | 0.8567 | 0.5324 |
| lit (brightness 118.9) | 0.9184 | 0.6752 |
| **retention** | **93.3%** | **78.9%** |

Detection survives darkness; precise localisation does not. Our synthetic night
— uniformly dimmed, 3.5x too little contrast — would probably not reproduce that
split, which is a concrete target for improving it.

---

## 5. Status of the claims

- **C2 (interference): reopened and supported.** ~2 points on all three
  conditions once each is genuinely degrading.
- **C1 (efficiency): still unsupported.** Every MoE arm costs more per image.
- **The MoE's failure is now a block-level problem, not a premise-level one.**
  Fixing gate calibration is the single highest-value next step, and it is
  cheap.

### Limitations

Single seed throughout. Part of the fog2 deficit is scene diversity rather than
interference (§1). The calibrated fog matches real haze on dark channel but
loses more contrast than real haze (−23.2 vs −13.4). Night remains uncalibrated
against real data.
