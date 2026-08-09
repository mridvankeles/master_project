# 02 — Related Work

> **Status:** needs work
> **Sources:** papers-datasets__2_.xlsx (rows 2–15), chat 2026-07-28, advisor web search 2026-07-28
> **Open questions:** 4  |  **Conflicts:** 2

Every entry from the source spreadsheet is preserved below. Missing fields are marked
`—` rather than filled in by guessing. Citation counts are as recorded in the sheet;
the date they were recorded is unknown (see `## Conflicts`).

## Group A — Haze / cloud / mist RS detection (from your sheet)

| # | Paper | Datasets used | Cites | Year | Venue | Repo |
|---|---|---|---|---|---|---|
| A1 | An Oriented Object Detector for Hazy Remote Sensing Images | HRSI | 11 | 2024 | Published — IEEE TGRS | — |
| A2 | DHC-Net: A Remote Sensing Object Detection Under Haze and Class Imbalance | DOTA-v2.0, DOTA-v2.0Haze, RTTS, HazeNet | 1 | 2025 | Published — IEEE TGRS | github.com/Linghuaqian1/DHC_Net |
| A3 | CM-YOLO: Typical Object Detection Method in Remote Sensing Cloud and Mist Scene Images | Mar20, HRSC216 *(recorded thus in source; almost certainly HRSC2016 — verify)*, CORS-ADD | 39 | 2025 | Published — Remote Sensing (MDPI) | github.com/JimmyRSlab/CM-YOLO-Typical-Object-Detection-Method-in-Remote-Sensing-Cloud-and-Mist-Scene-Images |
| A4 | NIRNet: Noise Incentive Robust Network in Remote Sensing Object Detection Under Cloud Corruption | Hazy-DIOR, DOTA-Cloud | 3 | 2025 | Published — IEEE TGRS | github.com/zhangpeng2001/nirnet |
| A5 | HazyDet: Open-Source Benchmark for Drone-View Object Detection with Depth-Cues in Hazy Scenes | HazyDet | 55 | 2025 | recorded as "Published — ECCV" — **disputed, see Conflicts** | github.com/grokcv/hazydet |
| A6 | HAE-Net: Haze-Aware Remote Sensing Object Detection With Local–Global Defogging | — | — | — | — | — |

`[from: papers-datasets__2_.xlsx rows 2, 3, 4, 5, 8, 15]`

## Group B — Other degradations / adverse weather (from your sheet)

| # | Paper | Datasets used | Cites | Year | Venue | Repo |
|---|---|---|---|---|---|---|
| B1 | Unified diffusion-based object detection in multi-modal and low-light remote sensing images | Multimodal VEDAI, DroneVehicle, unimodal VisDrone and UAVDT | 1 | 2024 | Published — IEEE IGARSS | note in sheet: "RGB-IR data fusion to detect from low light images" (recorded in the repository column) |
| B2 | WRRT-DETR: Weather-Robust RT-DETR for Drone-View Object Detection in Adverse Weather | SOTA methods on the AWOD dataset | 15 | 2025 | Published — Drones (MDPI) | github.com/bei-liu/AWOD-datasets |

`[from: papers-datasets__2_.xlsx rows 6, 7]`

**B1 is your only low-illumination paper.** For a thesis with a low-illumination expert, one
IGARSS paper is not a literature review. See `## Additions` below.

## Group C — Scale / tiny objects (from your sheet)

| # | Paper | Datasets used | Year | Venue |
|---|---|---|---|---|
| C1 | Bridging the Scale Gap: Balanced Tiny and General *(title truncated in source)* | AI-TOD-V2 and DTOD, VisDrone | 1 Dec 2025 | Preprint — arXiv |

`[from: papers-datasets__2_.xlsx row 10]`

**One paper for the tiny-object expert.** Same problem as B1.

## Group D — Mixture of Experts (from your sheet)

| # | Paper | Datasets used | Year | Venue | Repo |
|---|---|---|---|---|---|
| D1 | AW-MoE: All-Weather Mixture of Experts for Robust Multi-Modal 3D Object Detection | K-Radar | 2026-03 | preprint — arXiv | github.com/windlinsherlock/AW-MoE |
| D2 | Mixture-of-Experts in Remote Sensing: A Survey | recorded as "Object Detection in Remote Sensing Imagery" — this is a topic, not a dataset | 23 March 2026 | Preprint — arXiv | — |
| D3 | Heterogeneous Mixture of Experts for Remote *(title truncated in source)* | UCMerced and AID dataset | 3 April 2025 | Preprint — arXiv | — |

`[from: papers-datasets__2_.xlsx rows 9, 11, 12]`

Notes on relevance:
- **D1** is 3D detection on radar/LiDAR (K-Radar). Its routing mechanism — Image-guided Weather-aware
  Routing, which classifies weather from camera images and routes to the matching expert — is
  conceptually the closest thing in your sheet to your proposal. Different modality and task.
- **D3** uses UCMerced and AID, which are **scene classification** benchmarks, not detection.
- **D2** is a survey. Note that once a survey of MoE in remote sensing exists, "we introduce
  MoE to remote sensing" is background, not contribution.

## Group E — Resource links (from your sheet)

- "all adverse condition codes and papers" —
  github.com/ChunmingHe/awesome-diffusion-models-in-low-level-vision/blob/main/README.md
  `[from: papers-datasets__2_.xlsx row 14]`

## The gap, stated precisely

Across all 12 entries in the source sheet:

- Papers on MoE: 3 (D1, D2, D3)
- Papers on 2D remote sensing object detection under degradation: 8 (A1–A6, B1, B2)
- **Papers doing both: 0**

That intersection is the target. `[advisor's addition]` The gap appears real, but the sheet
does not currently contain the evidence needed to claim it — because the nearest published
competitor is absent (see PMFN below).

## [advisor's addition] Additions — must be read before the proposal

Marked as advisor additions because they were not in the source sheet. Verify each yourself
before citing; author lists and exact venues are not recorded here and must be filled in
from the source.

### Priority 1 — the nearest competitor

- **PMFN — "Multimodal Remote Sensing Object Detection Based on Prior-Enhanced
  Mixture-of-Experts Fusion Network"**, IEEE (2025), ieeexplore.ieee.org/document/11071289.
  Proposes a prior-information-enhanced MoE fusion network for multimodal RS object detection,
  with a dynamic gating network that combines prior information and image features to give the
  system environmental perception, used to dynamically allocate weights to sub-fusion experts
  optimized for different environmental conditions.

  **This is environment-conditioned expert selection for RS object detection, already published.**
  Read it before writing the proposal. Your delta must be stated against it explicitly.
  Candidate deltas (to be verified against the actual paper): PMFN is multimodal-fusion-centric
  (its experts are fusion strategies, not degradation specialists); it does not appear to
  address tiny objects; it does not analyse router failure. `[advisor's addition]`

### Priority 2 — non-MoE competitors on your exact benchmarks

- **RShDet — "An Adaptive Spectral-Aware Network for Remote Sensing Object Detection Under
  Haze Corruption"**, Remote Sensing (MDPI) 18(7):1020, 2026. Constructs **Hazy-DOTA** by
  synthesizing haze at varying density over DOTA-v2.0 (11,268 images, ~1,793,658 annotated
  objects, 18 categories). Reports +7.1% mAP50 over baseline on Hazy-DOTA, +3.1% mAP on
  HazyDet, +6.63% mAP on RTTS; and over prior SOTA, +2.4% mAP50 / +1.9% mAP / +2.33% mAP
  respectively. Trained on a single RTX 4090, batch size 4, 100 epochs.
  → **Direct competitor on DOTA-v2.0 + haze. Its numbers are the bar.** The 4090 detail also
  proves a single-GPU haze-robust detector on DOTA is feasible on your hardware.
- **UniDet-D — "A Unified Dynamic Spectral Attention Model for Object Detection under Adverse
  Weathers"**, arXiv 2506.12324. Claims superior generalization to *unseen* adverse conditions
  including sandstorms and rain-fog mixtures. → This is the generalization claim your MoE must
  beat or explicitly not contest.
- **LEGNet — "A Lightweight Edge-Gaussian Network for Low-Quality Remote Sensing Image Object
  Detection"**, arXiv 2503.14012. Evaluated on DOTA-v1.0, DOTA-v1.5, DIOR-R, FAIR1M-v1.0,
  VisDrone2019; targets low-quality objects arising from limited sensor resolution, atmospheric
  interference, motion blur, variable illumination and occlusion. → A single-model competitor
  covering *all three* of your tasks at once. If LEGNet already handles fog + illumination +
  small scale in one dense network, "why do you need three experts?" is the first question you
  will be asked.

### Priority 3 — routing failure (the seam for claim C3)

None of this is in your sheet, and it is the literature your analysis chapter depends on.

- The **MoE in Remote Sensing survey** (D2, already in your sheet) contains the argument
  directly: under distribution shift — new geography, seasonal change, atypical atmospheric
  conditions — out-of-distribution samples can be routed to experts that were not trained for
  the relevant regime, producing confident but incorrect predictions; the risk increases with
  cloud-obscured optical imagery and missing modalities, where routing may rely on spurious
  signals. It also notes expert imbalance and representational collapse intensify as expert
  count grows and the corpus becomes more heterogeneous, and that systematic validation in
  RS-specific pipelines remains limited. → **Read the failure-modes section of the survey you
  already have.**
- **"Robust Experts: the Effect of Adversarial Training on CNNs with Sparse Mixture-of-Experts
  Layers"**, arXiv 2509.05086. Finds the switch auxiliary loss fails to prevent dying experts
  under adversarial training, while an entropy loss keeps multiple experts active; also finds
  MoE placement deeper in the network (second residual block) gives more meaningful routing and
  more stable expert specialization. → **Directly actionable: use an entropy loss, and place
  the MoE deep.** With only 3 experts, collapse is less likely than at scale, but it is the
  single cheapest failure to guard against.
- **"Toward Calibrated Mixture-of-Experts Under Distribution Shift"**, arXiv 2606.20544 (2026).
  Argues a soft-routed MoE can become miscalibrated when routing patterns differ between train
  and test, *even if every expert is perfectly calibrated on its own routing-induced view of
  the data.* → Relevant if you use soft routing rather than top-1.

### Priority 4 — gaps in your own task coverage

- Low illumination in remote sensing: you have one paper (B1). Search terms to run:
  low-light aerial detection, nighttime UAV detection, illumination-invariant remote sensing
  detection, RGB-IR fusion nighttime vehicle detection.
- Tiny objects: you have one paper (C1). The tiny-object detection literature is large and has
  its own standard machinery (label-assignment strategies designed for tiny objects,
  normalized-Wasserstein-style localization metrics, high-resolution feature paths). You need
  this literature regardless of the MoE decision, because your tiny-object expert will be
  compared against it. `[CITE NEEDED]` — specific method names to be confirmed from the
  AI-TOD-v2 literature rather than from memory.

## Conflicts

- **HazyDet venue.** Your sheet records "Published — ECCV", 2025 `[from: papers-datasets__2_.xlsx row 8]`.
  The authors' own BibTeX on the dataset's Hugging Face page lists it as an arXiv preprint,
  arXiv:2409.19833, year 2025; ResearchGate shows a publication date of Jan 1, 2026
  `[from: advisor web search 2026-07-28]`. Additionally, ECCV is held in even-numbered years,
  so "ECCV 2025" is unlikely to exist (advisor confidence: fairly sure, not certain).
  **Unresolved — verify from the publisher before citing.**
- **DOTA-v2.0 object count.** Your sheet records "18 sınıf, 280k+ örnek" (18 classes, 280k+
  instances) `[from: papers-datasets__2_.xlsx row 23]`. The RShDet paper describes DOTA-v2.0 as
  11,268 images with 1,793,658 annotated objects over 18 categories
  `[from: advisor web search 2026-07-28]`. The class count agrees; the instance count differs by
  roughly 6×. Possible explanations: different DOTA version, train-split-only vs. full count, or
  an error in one source. **Unresolved — do not cite either number until checked against the
  official DOTA page.**

## Open questions

1. Does your delta survive PMFN? (Blocking — nothing else matters until this is answered.)
2. What is HAE-Net's venue and year, and does it overlap A1/A4?
3. Are the citation counts in the sheet from one snapshot date, and which?
4. What is the full title of C1 and D3? Both are truncated in the source.
