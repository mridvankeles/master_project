# 03 — Datasets

> **Status:** needs work
> **Sources:** papers-datasets__2_.xlsx (rows 19–40), chat 2026-07-28, advisor web search 2026-07-28
> **Open questions:** 5  |  **Conflicts:** 2

## Dataset decision (REVISED 2026-08-01)

One training corpus, three conditions, exact image-level pairing, **horizontal boxes**.

### Training corpus: DIOR family

20 classes, native HBB, uniform 800×800, no tiling required.

| Condition | Source | Status |
|---|---|---|
| Clear | DIOR originals | to download |
| Fog | Hazy-DIOR (synthesized from DIOR, so pairing is exact) | have, with HBB annotations |
| Low illumination | synthetic darkening of the same DIOR images | to generate |

DIOR classes: airplane, airport, baseball field, basketball court, bridge, chimney, dam,
expressway service area, expressway toll station, golf course, ground track field, harbor,
overpass, ship, stadium, storage tank, tennis court, train station, vehicle, windmill.

**Why HBB, not OBB** — the objection will come, so the answer goes in the thesis: DIOR's native
annotation is horizontal, so this uses the benchmark as published. The contribution is
degradation-conditioned routing, which is orthogonal to box geometry — the same MoE would work with
an oriented head. Adding OBB would introduce a confound without testing the claim.

**What this removes:** the `le135` angle-convention conversion, corner-ordering bugs, rotated NMS,
and — because DIOR images are uniform 800×800 with no tiling — the tile-overlap leakage path that
was the most likely silent failure in the whole project.

Low-light synthesis must be a citable physical model, not a gamma hack: inverse-ISP darkening plus
Poisson-Gaussian sensor noise, with **randomized parameters per sample** (see
`04-method-open-questions.md` § Router domain robustness).

**Verify before building anything:** that Hazy-DIOR filenames map 1:1 onto clear DIOR filenames and
that the splits match. If they do, the condition grid is exact and leakage is trivially checkable.

### Test-only sets — never trained on

| Set | Tests | Note |
|---|---|---|
| Fog + dark compound on DIOR | compound-condition router failure | one synthesis pass; **the money experiment** |
| DOTA-Cloud, or synthetic cloud on DIOR | **unseen** degradation | cloud ≠ fog: opaque occlusion vs. volumetric scattering |
| HazyDet / RDDTS | real fog | evaluate on classes shared with DIOR |
| DroneVehicle night split | real low illumination | RGB channel only — **never IR** |
| DOTA-v2.0 + DOTA-v2.0Haze | cross-platform, cross-taxonomy | secondary; shared classes only |

### Cut or parked

xView, AI-TOD-v2, HRSID, exDark, AWOD, DIOR-YOLO, HRSC2016, CORS-ADD, Mar20/HazeNet, VisDrone,
UAVDT, VEDAI. All preserved in `99-unsorted.md`. DOTA demoted from main arm to secondary test.

### Why not pool several datasets for training

Pooling is not pairing. If clear comes from DOTA and fog from Hazy-DIOR, "which condition" and
"which dataset" stay perfectly correlated after any amount of preprocessing, and the router learns
dataset identity instead of degradation. Pooling only becomes safe once every corpus appears in
every condition — a second research problem (multi-dataset detection across mismatched taxonomies)
and **out of scope for a one-year thesis**. `[advisor's addition]`

## [advisor's addition] Three data-design risks, in order of severity

### Risk 1 (fatal if unaddressed) — synthetic-degradation monoculture

Your fog data — Hazy-DIOR, DOTA-v2.0Haze, DOTA-Cloud, Foggy-DOTA — is entirely **synthetic
haze overlaid on clear benchmarks**. If experts are trained on synthetic haze and the router
is trained to recognize synthetic haze, the system may have learned to detect the synthesis
parameters rather than atmospheric degradation. An examiner will raise this, and no rhetorical
answer survives it.

**The fix is already in your inventory.** HazyDet (row 35, 7 GB) contains 383,000 real-world
instances from *both* naturally hazy environments and synthetically hazed clear scenes, and the
authors constructed **RDDTS — an independent Real-hazy Drone Detection Testing Set** inside
HazyDet specifically to evaluate detectors under real conditions `[from: advisor web search 2026-07-28]`.
The synthetic haze was generated with the Atmospheric Scattering Model (ASM).

→ **Protocol: train on synthetic, report headline numbers on real (RDDTS).** Add HazyDet to
the main dataset list. Treat AWOD as unavailable and stop waiting on the email.

### Risk 2 (serious) — platform confound in the router

If your fog data is DIOR-derived (satellite) and your low-light data is DroneVehicle (drone),
then the router can separate the two conditions on **platform statistics alone** — ground
sampling distance, object density, viewing geometry — without ever learning anything about fog
or illumination. Router accuracy would look excellent and mean nothing.

**Fix:** generate synthetic low illumination on the *same* imagery used for the fog condition
(gamma / ISP-style darkening plus sensor noise on DIOR or DOTA), so the identical source images
appear under both conditions. Real night imagery (dronevehicle-night) then serves as held-out
test only. This controls the confound and is cheap.

### Risk 3 (serious) — mixed box representation

DOTA-v2.0 and DIOR-R use **oriented bounding boxes (OBB)**; AI-TOD-v2 and xView use
**horizontal boxes (HBB)**; HRSID is OBB-annotated. You cannot train one model on the union
without deciding on a single box representation and converting, which loses information in one
direction and fabricates it in the other.

**Decision required before any training:** either commit to OBB throughout (and convert or drop
HBB sets), or run the tiny-object leg as a separate HBB experiment with its own baseline.
Do not discover this mid-experiment.

### [advisor's addition] Scope error to resolve

**exDark** (row 21) is ground-level photography — street scenes, vehicles and pedestrians shot
from human eye level. It is not remote sensing. If it stays in a thesis on remote sensing
detection, an examiner will ask why. Either cut it, or justify it explicitly as a
cross-domain transfer experiment with that framing stated up front.

## [advisor's addition] Compute budget against your hardware

Hardware: RTX 5070 Ti (16 GB) personal; 2× A6000 (48 GB) at work `[from: chat 2026-07-28]`.

Useful calibration point: RShDet trained a haze-robust detector on DOTA-v2.0-derived data on a
**single RTX 4090, batch size 4, 100 epochs** `[from: advisor web search 2026-07-28]`. So the
fog leg on DOTA-scale data is feasible on hardware comparable to yours. Confidence: fairly sure
this generalizes to your setup; their exact tiling and image size are unrecorded.

**Recommendation: cut xView.** Reasons: ~1M+ objects across 60 classes with a different label
taxonomy and very large source rasters; it is clear-weather, so it duplicates the role DOTA-v2.0
already fills; and 60 classes forces either a separate head or a taxonomy mapping that adds a
confound. The compute it consumes is better spent on seeds and ablations, which is where the
defense is actually won. Keep it in `99-unsorted.md` as a possible generalization test if the
core experiments finish early.

Reduced main set: **Hazy-DIOR (fog) + synthetic-dark DIOR/DOTA (low light) + AI-TOD-v2 (tiny) +
DOTA-v2.0 (clear control) + HazyDet/RDDTS (real-degradation test)**.

The 16 GB card will not comfortably hold three experts plus backbone at 1024×1024 OBB training.
Use it for router prototyping, data pipeline work, and single-expert debugging; reserve the
A6000s for full runs and all seeded final numbers.

---

## Full dataset inventory (preserved from source)

All 22 rows preserved. Descriptions translated to English; original Turkish annotations kept in
parentheses where they carry information. `[from: papers-datasets__2_.xlsx rows 19–40]`

| Row | Dataset | Description | Key properties | Link / note |
|---|---|---|---|---|
| 19 | **AI-TOD-v2** | Optimized for detecting "very small" objects in aerial imagery | 8 classes, 700k+ objects, micro-scale target detection | github.com/jwwangchn/AI-TOD |
| 20 | **xView** | One of the largest and most complex satellite-imagery object detection sets | 1M+ objects, 60 classes, high-resolution satellite data | xviewdataset.org |
| 21 | **exDark** | Built for object detection in low-light and night conditions | Night shots, low-light enhancement + detection, vehicle/pedestrian focused | github.com/cs-chan/Exclusively-Dark-Image-Dataset |
| 22 | **HRSID** | Ship detection in SAR (Synthetic Aperture Radar) imagery | 5,604 SAR images, ship/harbour classes, OBB labelling | github.com/chaozhong2010/HRSID |
| 23 | **DOTA-v2.0** | Standard benchmark for object detection in aerial imagery | 18 classes, 280k+ instances, oriented bounding boxes (OBB) | captain-whu.github.io/DOTA/dataset.html |
| 24 | **DOTA-v2.0Haze** | DOTA-v2.0 with synthetic fog added | Foggy-weather simulation, multi-scale objects, OBB | github.com/GrokCV/HazyDet — **VPN needed** ("VPN lazım"); noted as oriented bbox |
| 25 | **Hazy-DIOR** | The popular DIOR dataset with fog and cloud noise added | 20 classes, variable fog density, remote sensing | huggingface.co/datasets/SmileShaun/Hazy-DIOR/tree/main |
| 26 | **DOTA-Cloud** | Built by adding cloud corruptions over DOTA imagery | Cloudy scenes, 15+ categories, difficult weather conditions | github.com/zhangpeng2001/nirnet |
| 27 | **HazeNet / Mar20** | Base set for the 2020-era dehazing and detection work | Dehazing + object detection integration | github.com/GrokCV/HazyDet |
| 28 | **HRSC2016** | High-resolution optical dataset for detecting ships at sea | Complex harbour/sea backgrounds, detailed ship classes | github.com/wmchen/HRSC2016-MS |
| 29 | **CORS-ADD** | Focused on aircraft detection against complex optical backgrounds | 7,337 images, 32k aircraft, civil/military distinction | *(no link recorded)* |
| 30 | **Multimodal VEDAI** | Provides both RGB and Infrared for aerial vehicle detection | RGB-IR paired data, small vehicles, multiple orientations | ieee-dataport.org/documents/multimodal-object-detection-dataset — **IEEE subscription required** ("IEEE ABONELIK ISTENİYOR") |
| 31 | **DroneVehicle** | Large-scale day–night RGB-IR dataset captured from UAVs | 28,439 image pairs, 5 vehicle classes, dense traffic | github.com/VisDrone/DroneVehicle |
| 32 | **VisDrone** | Main set for object detection and tracking from a drone perspective | Very small objects, video sequences, crowded scenes | github.com/VisDrone/VisDrone-Dataset |
| 33 | **UAVDT** | Focused on UAV-based vehicle detection and tracking | 80,000 frames, vehicle-focused, varying altitudes and angles | datasetninja.com/uavdt |
| 34 | **AWOD** | UAV detection under adverse weather (fog, low light) | 20,000 images, maritime vehicles, 3 types of weather degradation | aimh.isti.cnr.it/dataset/mobdrone/ ; seadronessee.cs.uni-tuebingen.de/ ; datasetninja.com/vis-drone-2019-det — **email sent, no reply** ("mail atıldı dönüş yok") |
| 35 | **HazyDet** | Designed for detection in foggy weather and complex traffic flow | Mixed traffic classes, hybrid labelling, fog robustness | github.com/GrokCV/HazyDet — **7 GB** |
| 36 | **Foggy-DOTA** | *(no description recorded)* | *(none recorded)* | github.com/PhucNDA/Foggy-DOTA |
| 37 | **dronevehicle-night** *(recorded as "dronevvehicle-night")* | *(no description recorded)* | *(none recorded)* | scidb.cn/en/detail?dataSetId=19bef4848c2c4a7d8bdb63896670f96c — **7.7 GB** |
| 38 | **VEDAI** | *(no description recorded)* | *(none recorded)* | downloads.greyc.fr/vedai/ |
| 39 | **DIOR-YOLO** | *(no description recorded)* | *(none recorded)* | pan.baidu.com/s/1V5zHcFK3lMPcVBuHgmPPgg?pwd=6vug (path: sharelink4261821403-227222958648208/MFAE-YOLO) — **VPN needed** ("VPN lazım") |
| 40 | **Hazy-DIOR** *(recorded as "hazzy-dıor")* | Used by the NIRNet paper; many classes, ship-heavy ("çok class, gemi ağırlıklı") | train 5k and val 5k, test 11k images | huggingface.co/datasets/SmileShaun/Hazy-DIOR/tree/main — **27 GB** |

## Conflicts

- **DOTA-v2.0 instance count.** Row 23 records 18 classes, 280k+ instances. RShDet describes
  DOTA-v2.0 as 11,268 images / 1,793,658 objects / 18 categories `[from: advisor web search 2026-07-28]`.
  Classes agree; instance counts differ ~6×. **Unresolved** — check the official DOTA page.
- **Rows 25 and 40 are the same dataset** (Hazy-DIOR, same Hugging Face URL) recorded twice with
  different, non-overlapping details: row 25 gives 20 classes and variable fog density; row 40
  gives the split sizes, the 27 GB figure, the NIRNet association, and the "ship-heavy" note.
  Both preserved above rather than merged. Similarly, **rows 30 and 38** may or may not be the
  same VEDAI data (one behind an IEEE subscription, one free from GREYC) — **unresolved.**

## Shared-provenance warning

Rows 24 (DOTA-v2.0Haze), 27 (HazeNet/Mar20) and 35 (HazyDet) all point to the same repository,
`github.com/GrokCV/HazyDet`. These may be three distributions from one project rather than three
independent datasets. Verify before claiming dataset breadth in the thesis. Advisor confidence:
fairly sure they share a repository; not certain they are the same underlying data.

## Access status summary

| Status | Datasets |
|---|---|
| Open, direct download | AI-TOD-v2, xView, exDark, HRSID, DOTA-v2.0, Hazy-DIOR, DOTA-Cloud, HRSC2016, DroneVehicle, VisDrone, UAVDT, HazyDet, Foggy-DOTA, dronevehicle-night, VEDAI (GREYC) |
| VPN required | DOTA-v2.0Haze, DIOR-YOLO |
| Subscription required | Multimodal VEDAI (IEEE) |
| Unresolved / unanswered request | AWOD |
| No link recorded | CORS-ADD |

## Open questions

1. Which dataset supplies the low-illumination expert's training data?
2. Synthetic-dark generation: gamma-based, ISP-simulation-based, or something else? Which
   parameters, and are they held fixed across train and test?
3. OBB or HBB as the single box representation?
4. Is xView cut, or kept?
5. Are rows 24/27/35 independent datasets or one project's distributions?
