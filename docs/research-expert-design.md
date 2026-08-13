# Research: designing expert skills, connecting experts, and what data we need

> **Date:** 2026-08-13 · sources: `main-papers/`, `other-papers/`, literature
> search, plus four measurements run against our own checkpoints and corpora.
> Answers the four questions raised after `results-conditional-gating.md`.

---

## 0. Two measurements that reframe everything below

### 0.1 The gate already works on REAL fog — better than on ours

`cond3_gated`'s gate, run on RRSHID (real paired clear/hazy remote sensing, no
labels needed for this):

| source | clear | fog | night | argmax = fog |
|---|---:|---:|---:|---:|
| REAL moderate — hazy | 0.376 | **0.453** | 0.190 | **98.0 %** |
| REAL thick — hazy | 0.355 | **0.446** | 0.198 | **100.0 %** |
| REAL moderate — clear | 0.334 | 0.361 | 0.288 | 53.9 % |
| **SYNTH fog (ours)** | 0.364 | 0.336 | 0.293 | **31.2 %** |
| SYNTH clear (ours) | **0.472** | 0.288 | 0.253 | 0.6 % |

Paired within RRSHID, controlling for domain: fog-branch probability rises
**+0.092** (moderate) and **+0.079** (thick) from clear to hazy.

The gate fires on real haze at 98–100 % while firing on *our own* synthetic fog
only 31 % of the time. It has not overfitted the simulator — if anything the
simulator is the weaker signal. (Caveat: RRSHID *clear* also routes to fog
54–60 %, so there is a domain bias on top; the paired delta is the clean number.)

### 0.2 Our synthetic fog is far weaker than real fog

Dark-channel increase relative to each source's own clear images:

| | dark channel Δ | contrast Δ |
|---|---:|---:|
| REAL moderate | **+100.5** | −13.4 |
| REAL thick | **+142.8** | −14.4 |
| SYNTH thin | +36.8 | −9.5 |
| SYNTH moderate | +62.3 | −9.0 |
| SYNTH thick | +93.0 | −21.2 |

**Our thickest fog is weaker than real *moderate* fog.** Real thick fog is 1.5×
our thickest. Two thirds of our fog corpus (thin + moderate) is barely degraded
at all — which fully explains the 38.5 % fog routing accuracy in
`results-conditional-gating.md`. The gate is not failing; **most of our "fog"
images are nearly clear**, and the label is wrong more often than the gate is.

This single fact reframes questions 1 and 3 below.

---

## 1. How to give the fog expert a real skill

The night branch got illumination-invariance (log + local-mean removal). The
literature gives three concrete equivalents for haze, in increasing cost.

### 1.1 DFENet — the closest precedent, and it is in your folder

`other-papers/An_Oriented_Object_Detector_for_Hazy_Remote_Sensing_Images.pdf`
(Liu et al., IEEE TGRS 2024) is almost exactly the design being asked about:

> "DFENet ... is a two-branch structure ... We use the **haze-predict module
> (HPM)** to extract haze information and predict masks, while the **cross-fuse
> module (CFM)** combines the features of two branches with the haze mask to
> obtain the fused features."

Three things to steal:

1. **A haze *mask*, not a haze *scalar*.** Their HPM predicts a spatial haze map.
   Our gate emits one probability per image — but haze is spatially
   non-uniform, and a per-pixel mask is strictly more expressive. This is the
   single most promising upgrade to the fog branch.
2. **The mask *guides fusion*, it does not select.** CFM uses the haze map to
   weight the two branches per location. That is soft, spatial, and continuous —
   which is what a physically smooth degradation deserves.
3. **Their stated motivation is our exact failure mode:** it "can dynamically
   balance the weight of the input image and the dehazing image", so the dehazing
   path does not damage clean images. That is the always-on shared branch
   argument, arrived at independently.

They also ship **HRSI**, real hazy remote sensing imagery (airport, large
vehicle, ship) with oriented boxes — a real-fog *detection* set, which RRSHID is
not.

### 1.2 CM-YOLO — a cheap prior we do not have

`other-papers/CM-YOLO...pdf` proposes **component-decoupling background
suppression (CDBS)**, which "combines the **optical properties** of environmental
elements and a **background subtraction** strategy, adaptively suppressing
environmental interference and highlighting target's contrast".

Our current `TransmissionPrior` does a plain wide average-pool subtraction. CDBS
is the same idea done properly: estimate the haze *component* from its optical
signature and subtract that, rather than subtracting a generic low-pass.

### 1.3 Concrete proposal for the fog branch

In order of effort:

| change | what it adds | cost |
|---|---|---|
| Predict a spatial haze mask in the fog branch and multiply the branch output by it | spatial haze awareness; fixes the "thin fog on part of the image" case | small |
| Replace the average-pool veil estimate with a dark-channel-based transmission estimate | a physically correct prior instead of a generic low-pass | small |
| Add a per-location fusion weight (CFM-style) instead of one image-level gate | soft, spatial routing for a spatially smooth degradation | medium |

Note the third makes the *fog* branch spatial while the *night* gate stays
image-level, which is correct: illumination is global, haze is not.

---

## 2. More experts, and connecting them

### 2.1 Connecting them: yes, and the literature is unanimous on why

- **AW-MoE** (`other-papers/AW-MoE...pdf`) runs a **dedicated lightweight weather
  classifier on the image** to route, reaching **99 % routing accuracy**, where
  routing from internal features reached only 71.3 % — and **51.9 % on fog
  specifically**. Our gate reads pooled internal features and gets 38.5 % on fog.
  **The pattern matches exactly, and their fix is a separate classifier head fed
  from the input rather than a linear probe on neck features.**
- AW-MoE's **K ablation** directly supports the multi-label design already built:
  K=2 beats K=1 *specifically under ambiguous weather*, because "routing to
  multiple experts mitigates the impact of classification errors". Our fog/clear
  confusion is precisely that ambiguity.
- **CWE-Net** (Cross-Weather Expert Network, 2026) reports that "cross-weather
  fusion improves expert collaboration and **alleviates task interference**".
- **WM-MoE** decouples *content* embedding from *weather* embedding before
  routing — worth noting, because our gate currently mixes both.

### 2.2 More experts: probably not yet

Our own evidence says expert *count* is not the bottleneck: going 2 → 3 experts
changed nothing (−0.004 everywhere), while going unsupervised → supervised moved
NMI from 0.000 to 0.473. Depth of specialisation, not number of branches, is
where the return is. Add the fourth expert only when a fourth condition is
genuinely present in the data.

### 2.3 One warning from the tiny-object literature

`other-papers/Bridging the Scale.pdf` (ScaleBridge-Det) motivates its routing
module by "the tendency of **standard MoE models to favor dominant scales**".
That is a documented MoE failure mode with the same shape as ours: the gate
drifts to whatever dominates. Worth measuring per-severity before adding
capacity.

---

## 3. Do we need more fog / night imagery?

**Not more — harder, and some of it real.** §0.2 shows our synthetic fog is
weaker than real moderate fog even at its thickest setting.

### 3.1 Fog

- **Regenerate at realistic strength.** Target a dark-channel increase of
  +100 to +145 to match RRSHID, versus our current +37/+62/+93. This is a
  parameter change in the synthesis, not new data, and it should fix both the
  routing accuracy and the "fog shows no interference" result — a condition that
  barely degrades the image cannot interfere with anything.
- **You already hold real fog**: RRSHID (1,220 + 611 paired train images, real,
  with clear counterparts) — usable now for router evaluation and for calibrating
  the synthesiser, though it has no detection labels.
- **HRSI** (from the DFENet paper) is real hazy RS *with* boxes, and is the
  obvious acquisition target.

### 3.2 Night

Synthetic is a genuine limitation, and the literature is blunt: domain gap
"persists even with highly realistic simulation". But **you already have real
night imagery** — `dronevehicle/DroneVehiclesDatasetYOLO` (1.1 GB, 2 vehicle
classes, YOLO format, with a night split). Taxonomy does not match DIOR, so it
cannot join training, but it is a valid **held-out router test**: does the night
branch fire on real night images? That is the same free experiment §0.1 just
ran for fog, and it needs no labels.

### 3.3 The cheap experiment this all points to

Re-synthesise fog at realistic strength → re-train `cond3_gated` → re-measure
routing. If fog routing rises from 38.5 % toward clear/night's 88–94 %, the
weak-fog explanation is confirmed and every downstream fog result improves for
free.

---

## 4. A tiny-object expert

### 4.1 The scale numbers say DIOR qualifies

`other-papers/AI-TOD...pdf` defines the tiny regime: **mean object size 12.8 px**.
Our measured DIOR medians: `vehicle` **12.0 px**, `storagetank` 22.0, `ship`
25.3. So `vehicle` is exactly AI-TOD-scale, and it is one of the two weakest
classes (AP50 0.486). The regime is real in our data.

### 4.2 But route it by scale, not by image

This is the granularity problem `01-scope-and-claim.md` flagged from the start,
and the literature resolves it the same way: ScaleBridge-Det routes **per
scale**, via a Routing-Enhanced Mixture Attention over scale-specific experts,
plus a density-guided query allocation so tiny objects are not "overwhelmed by
the representational dominance of large targets".

So a "tiny" folder alongside clear/fog/night would be a **category error**:
tininess is a property of objects, not images, and nearly every DIOR image
contains both tiny and large objects. Two coherent alternatives:

1. **Per-FPN-level experts** — a tiny expert at P3, a general expert at P5. The
   route is the pyramid level, which is known, so it needs no gate at all. This
   is the honest form of "static expert".
2. **Keep it in the loss/assigner** — where our NWD experiment already found
   +0.0076 overall, and where the per-class analysis showed the real fix is the
   *assigner* (NWD-RKA), not the regression loss.

### 4.3 What a "tiny" folder IS good for

Not a fourth condition, but a **stratified evaluation slice**: images whose
objects are predominantly < 16 px, scored separately. That turns "does this help
tiny objects?" into a number, which is currently only visible by reading 20
per-class rows.

---

## 5. Recommended order

1. **Re-synthesise fog at realistic strength** (§3.1). Cheapest, and it plausibly
   explains three separate anomalies at once.
2. **Route from a dedicated classifier head on the input**, not a linear probe on
   neck features (§2.1, AW-MoE 99 % vs 71 %).
3. **Give the fog branch a spatial haze mask** (§1.3, DFENet HPM/CFM).
4. **Calibrate the gate** (`pos_weight`) so the deployed threshold matches the
   analysed one.
5. Run the **real-night router test** on DroneVehicle (§3.2) — free, no labels.
6. Only then consider more experts, and per-scale rather than per-image.

## Sources

Local: `An_Oriented_Object_Detector_for_Hazy_Remote_Sensing_Images.pdf` (DFENet,
HRSI) · `CM-YOLO...pdf` (CDBS) · `AW-MoE...pdf` (IWR routing, K ablation) ·
`Bridging the Scale.pdf` (ScaleBridge-Det) · `AI-TOD_ICPR_camera_ready.pdf` ·
`FeatEnHancer` (ICCV 2023) · `JGEO-...pdf` (MoE in RS survey).

Web: CWE-Net (cross-weather expert collaboration, 2026) · WM-MoE (weather-aware
multi-scale MoE) · RRSHID real-world RS dehazing benchmark · LAD-Enhancer ·
Deweather-MoE with uncertainty-aware FiLM.

Measured here: RRSHID router response, RRSHID vs Hazy-DIOR haze statistics,
DIOR per-class object scale, HazyDet EDA.

---

## 6. CONFIRMED (2026-08-13, no retraining required)

The weak-fog hypothesis in §0.2 and §3.1 predicted that the *existing* gate —
trained only on the weak Hazy-DIOR fog — would route a stronger fog corpus
better, because the gate had learned haze and the corpus simply lacked it.

Tested by running `cond3_gated`'s unchanged gate over the newly calibrated
`fog2` corpus:

| corpus | clear | fog | night | argmax = fog |
|---|---:|---:|---:|---:|
| clear | 0.471 | 0.289 | 0.253 | 0.7 % |
| fog **OLD** (Hazy-DIOR release) | 0.361 | 0.342 | 0.290 | **36.5 %** |
| fog2 **NEW** (calibrated ASM) | 0.295 | **0.386** | 0.312 | **66.4 %** |
| night | 0.252 | 0.251 | 0.521 | 1.0 % |

Fog-branch probability above clear: **+0.053 → +0.097**. Routing accuracy nearly
doubles, with no false positives introduced on clear (0.7 %) or night (1.0 %).

**The router was not the problem; the data was.** A weak degradation cannot be
routed because there is nothing to detect, and the 38.5 % fog accuracy reported
in `results-conditional-gating.md` was measuring the corpus rather than the gate.

Residual gap: real haze (RRSHID) still routes at 98–100 % against `fog2`'s
66.4 %, so the calibrated synthesis is closer to real but not equivalent —
consistent with its contrast falling further than real haze (−23.2 vs −13.4).
Matching the dark channel is not the same as matching haze.
