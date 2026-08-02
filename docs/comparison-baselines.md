# Comparison targets — what the four main papers actually report

Extracted from `../main-papers/`. Every number here is quoted from the paper;
nothing is inferred unless marked. Page numbers are PDF pages.

**The headline: only one of the four touches our dataset, and its numbers cannot
be placed in the same column as ours.** Read the blockers before building any
table.

---

## Summary

| | NIRNet | WRRT-DETR | UDDet | FeatEnHancer |
|---|---|---|---|---|
| Venue | IEEE TGRS 2025 | Drones (MDPI) 2025 | Electronics Letters 2024 | ICCV 2023 |
| Domain | Optical RS | Drone maritime | RS / UAV | **Ground-level** |
| Degradation | Cloud/haze (synth) | Fog + low-light + flare | Low-light + modality loss | Low-light (real) |
| **Boxes** | **OBB** | HBB | HBB *(inferred)* | HBB |
| Base detector | Oriented R-CNN R50-FPN | RT-DETR | YOLO-World | RetinaNet / FQ R-CNN |
| Uses Hazy-DIOR | **yes — 40.86 mAP** | no | no | no |
| Comparable to ours | **no** (see below) | no | no | no |

---

## 1. NIRNet — the only one on our data

Zhang, Cheng, Lang, Xie, Han. *NIRNet: Noise Incentive Robust Network in Remote
Sensing Object Detection Under Cloud Corruption.* IEEE TGRS 63 (2025), art.
5629713. Code: `github.com/zhangpeng2001/nirnet`.

### Headline table (Table I, p.6) — Hazy-DIOR, all 20 DIOR classes

| Method | mAP | rPC |
|---|---|---|
| Rotated RetinaNet | 33.97 | 62.20 |
| Rotated FCOS | 35.51 | 59.43 |
| R³Det | 36.21 | 63.90 |
| S²ANet | 38.36 | 63.50 |
| Rotated Faster R-CNN | 36.82 | 61.31 |
| RoI Transformer | 37.27 | 62.37 |
| Oriented R-CNN | 38.70 | 61.76 |
| **NIRNet** | **40.86** | **64.32** |

`rPC = mAP_hazy / mAP_clean`. Clean DIOR-R for the same models (Tables V–VI, p.9):

| Model | clean | hazy | absolute drop | rPC |
|---|---|---|---|---|
| Oriented R-CNN (R50) | 62.66 | 38.70 | −23.96 | 61.76 % |
| **NIRNet** (R50) | 63.53 | 40.86 | −22.67 | 64.32 % |
| NIRNet (Swin-T) | 65.10 | 48.70 | −16.40 | 74.81 % |

Prose (p.7): *"an approximately 40% performance decline across all detectors."*

### Protocol

- Split **5,862 / 5,863 / 11,738** — identical to DIOR's, and to ours.
- Images 800×800, all 20 classes.
- Synthesis: `I = δ(R)·t + P·(1−t)`, 360 scraped cloud patches, brightness ×0.8,
  transparency `t ∈ [0.5, 0.8]` random **per image** — not discrete severity tiers.
- MMRotate, Oriented R-CNN, ResNet-50+FPN, **SGD lr 0.005, momentum 0.9,
  wd 1e-4, batch 2, 12 epochs, 1× RTX 3090**.

### Why its 40.86 cannot sit next to our mAP50 — four blockers

1. **OBB, and it says so.** p.7: *"oriented bounding boxes are employed to
   evaluate the localization precision."* DIOR-**R**, Oriented R-CNN. HBB AP is
   systematically **higher** than OBB AP for elongated/rotated classes — harbor,
   bridge, ship, vehicle, groundtrackfield. Different quantity, not a harder or
   easier version of the same one.
2. **Metric convention unstated.** The paper says only "mAP". MMRotate's default
   is VOC07 11-point AP@0.5; Ultralytics reports 101-point COCO-style. The
   11-point convention typically reads 1–3 points lower. We do not know which
   they used.
3. **Opposite training protocol — the big one.** NIRNet trains on **clean images
   only** and evaluates zero-shot on hazy (p.1, p.10). It measures *robustness
   transfer*. Our `fog_yolo11n` config trains **on** fog, which is a strictly
   easier problem. Beating 40.86 would mean nothing.
4. **Provenance mismatch with the release on disk.** p.5 says Hazy-DIOR was
   synthesized *"from the DIOR-R testing set"* (11,738 images). The Hugging Face
   release we hold does not match that: our 2,607 DIOR-ID-keyed images split
   **651 / 651 / 1,305** across DIOR train/val/test, i.e. half of them come from
   DIOR **trainval**, not test. Either the HF packaging differs from what the
   paper describes, or the paper's description is loose. **Unresolved — do not
   claim to be evaluating on "the NIRNet benchmark" until this is settled.**

### How to use it honestly

Cite NIRNet as related work, annotated *"OBB, VOC-style mAP, clean-trained"*, and
anchor any comparison on the **relative** degradation rather than absolute mAP:

> Oriented R-CNN loses 61.76 % → i.e. retains 61.76 % of its clean mAP under
> Hazy-DIOR cloud corruption; NIRNet raises that to 64.32 %.

A relative drop is a defensible thing to compare against our own relative drop,
because it cancels most of the box-format and metric-convention difference. An
absolute mAP comparison does not.

---

## 2. WRRT-DETR — metric-compatible, dataset-incompatible

Liu, Jin, Zhang, Sun. *WRRT-DETR: Weather-Robust RT-DETR for Drone-View Object
Detection in Adverse Weather.* Drones 2025, 9, 369.

Drone-view **maritime** (AWOD: 20,000 images, 6 classes), fog + low-light +
flare. **No DIOR.** COCO mAP50 / mAP50:95, HBB, 640×640, 100 epochs, AdamW
lr 1e-4, batch 16, 1× RTX 4090.

Useful because it is the only one of the four that is **metric- and
box-compatible with us**, and it publishes stock **YOLOv11m** on a degraded
benchmark (Table 6, p.15):

| Difficulty | Easy | Normal | Difficult | Particularly difficult | All |
|---|---|---|---|---|---|
| YOLOv11m mAP50 | 73.1 | 68.4 | 66.6 | 51.8 | 66.9 |
| YOLOv11m mAP50-95 | 50.6 | 41.4 | 37.5 | 33.8 | 36.1 |
| WRRT-DETR mAP50 | 86.3 | 80.2 | 78.8 | 76.4 | 82.3 |

So a stock YOLO11 loses **~21 mAP50 points** from mild to severe degradation on a
comparable HBB/COCO benchmark. Use that as a *sanity band* for our own
degradation curve — never as a row in a DIOR table.

Two internal inconsistencies worth knowing: prose p.16 says P = 87.7 while
Table 7 says 86.7; Table 5's FSAE-only row lists mAP50-95 = 64.3, out of family
with every neighbouring value and almost certainly a typo. **AWOD's train/val/test
split is never stated anywhere in the paper.**

---

## 3. UDDet — not a comparison target

Sun, Yu, Cheng. *Unified diffusion-based object detection in multi-modal and
low-light remote sensing images.* Electronics Letters 60(22), Nov 2024.

VEDAI / DroneVehicle / VisDrone / UAVDT, RGB-IR multi-modal + low light,
YOLO-World base, COCO AP protocol. Headline AP (=mAP@0.5:0.95): **50.5 / 55.3 /
25.1 / 20.7** respectively.

Excluded because: no DIOR, different degradation, and the input is **RGB-IR
multi-modal**, which a stock RGB-only YOLO11 cannot consume. Also **epochs,
batch size, optimizer, learning rate and input resolution are all unreported**,
so the setup is not reproducible for a controlled comparison. Box format is not
stated either — inferred HBB from the COCO protocol, though VEDAI and
DroneVehicle both ship oriented annotations natively.

---

## 4. FeatEnHancer — methodological only, not remote sensing

Hashmi, Kallempudi, Stricker, Afzal. ICCV 2023, pp. 6702–6712.

ExDark / DARK FACE / ACDC / DarkVision — **all ground-level natural imagery.**
Not remote sensing at all, so it cannot share a table with anything here. This is
the same scope objection `03-datasets.md` raises against exDark.

Its value is one specific finding, which is a direct argument for the
end-to-end approach over a restoration cascade (baseline B4 in
`05-experiment-plan.md`) — Table 5, p.7, DarkVision mAP@0.5:

| | Illum 3.2 | Illum 0.2 |
|---|---|---|
| Baseline (SELSA, no enhancement) | **32.8** | **10.4** |
| + RAUS / EnGAN / MBLLEN / KIND / Zero-DCE | 7.4 – 7.8 | 5.0 – 5.4 |
| + FeatEnHancer (feature-level) | **34.6** | **11.2** |

Every *image-level* low-light enhancement method collapses the detector from
32.8 to ~7.5 mAP. Feature-level enhancement helps. NIRNet cites this paper
(ref [39]) as the inspiration for its calibration mechanism.

---

## Consequence for GATE 1

`05-experiment-plan.md` Gate 1 requires: *"Stock YOLO11 (HBB) reproduced at a
published mAP on clear DIOR. Pick a specific paper and a specific DIOR mAP
number, and write it down before training."*

**None of these four papers supplies that number.** The only clean-DIOR figure
available here is NIRNet's Oriented R-CNN **62.66**, which is **OBB on DIOR-R** —
the wrong box format for our gate. A published **HBB DIOR** mAP must be sourced
from elsewhere before Gate 1 can be assessed, and it has to be written down
before the full clear-DIOR run, not after.

Note also that our current clear config trains on 651 images, not DIOR's 5,862,
because of the pairing loss documented in the pairing report. Gate 1 is therefore
blocked twice over: no reference number, and not yet the right training set.
