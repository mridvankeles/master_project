# OBB arm, and the NIRNet comparison

Three runs, all stock YOLO11n, all from scratch, all identical hyperparameters
(100 epochs, 640px, batch 16, `optimizer: auto`, `cos_lr`, seed 0, RTX 5070 Ti).
The only variables are box format and training condition.

| Run | Boxes | Trained on | Config |
|---|---|---|---|
| `fog_yolo11n` | HBB | fog | `configs/train/fog_yolo11n.yaml` |
| `fog_yolo11n_obb` | OBB | fog | `configs/train/fog_yolo11n_obb.yaml` |
| `clear_yolo11n_obb` | OBB | clear | `configs/train/clear_yolo11n_obb.yaml` |

---

## 1. HBB vs OBB, both trained on fog, evaluated on fog test

| Arm | COCO mAP50 | COCO mAP50-95 | VOC07 mAP@0.5 | P | R |
|---|---|---|---|---|---|
| HBB | 0.3345 | 0.1936 | **0.3161** | 0.5666 | 0.3230 |
| OBB | **0.3493** | **0.2385** | 0.3061 | 0.5598 | **0.3472** |

**The two metric conventions disagree on which arm wins.** OBB leads on COCO
mAP50 (+0.015) and clearly on mAP50-95 (+0.045, ~23 % relative), but trails on
VOC07 (−0.010). Any table reporting one number without naming the convention is
therefore reporting an arbitrary winner.

### These two columns are not actually comparable

HBB matches detections with axis-aligned IoU; OBB with exact rotated-polygon IoU.
For a rotated elongated object two axis-aligned boxes overlap far more readily
than two oriented ones, so **OBB@0.5 is a strictly harder criterion**. This is
the same objection `comparison-baselines.md` raises against setting NIRNet's
40.86 beside an HBB number — and it applies to our own two arms just as much.

Read the table as "each arm's score under its own geometry", not as a ranking.
The quantity that survives the comparison is the *ratio* in §2.

### Where orientation helps, per class (COCO AP50, OBB − HBB)

| Helped | Δ | | Hurt | Δ |
|---|---|---|---|---|
| windmill | +0.151 | | harbor | −0.099 |
| golffield | +0.129 | | dam | −0.062 |
| **ship** (14,280 inst) | **+0.075** | | overpass | −0.058 |
| expressway-toll-station | +0.049 | | airport | −0.041 |
| stadium | +0.042 | | expressway-service-area | −0.023 |

Ship is the substantive one: the most frequent class in DIOR, +0.075, and the
class where the oriented annotation carries the most extra information. Harbor
losing 0.099 is genuinely surprising for an elongated class and is worth a look.

---

## 2. The NIRNet comparison — train clean, evaluate hazy

NIRNet trains on **clean** images and evaluates zero-shot on hazy, reporting
`rPC = mAP_hazy / mAP_clean`. Our fog-trained runs train *on* fog, a strictly
easier problem, so they cannot be set against it at all. `clear_yolo11n_obb`
reproduces NIRNet's protocol.

All numbers below are **OBB** with **VOC07 11-point AP at IoU 0.50** — the same
convention NIRNet's own code uses (`use_07_metric=True` in
`nirnet-main/mmrotate/datasets/dior.py`).

| Model | Backbone | clean | hazy | **rPC** |
|---|---|---|---|---|
| Rotated RetinaNet | R50 | — | 33.97 | 62.20 % |
| Rotated Faster R-CNN | R50 | — | 36.82 | 61.31 % |
| RoI Transformer | R50 | 59.76 | 37.27 | 62.37 % |
| Oriented R-CNN | R50 | 62.66 | 38.70 | 61.76 % |
| **NIRNet** | R50 | 63.53 | 40.86 | **64.32 %** |
| NIRNet | Swin-T | 65.10 | 48.70 | 74.81 % |
| **ours — yolo11n OBB, from scratch** | — | **22.02** | **11.90** | **54.06 %** |

Under the COCO convention the same checkpoint gives clean 25.80, hazy 10.98,
**rPC 42.55 %** — an 11-point swing in rPC from the metric choice alone. Report
the convention or the ratio means nothing.

### What this says

**Our model is markedly less haze-robust than any published baseline**: it
retains 54 % of its clean performance where Oriented R-CNN retains 62 % and
NIRNet 64 %. That is the expected direction and worth stating plainly rather
than explaining away — a 2.6 M-parameter yolo11n trained from scratch on 651
images has no reason to out-generalise a pretrained ResNet-50 trained on 5,862.

The absolute numbers (22.02 clean vs their 62.66) are dominated by that same
gap and should not be presented as a comparison at all.

**The useful reading is that rPC is measurable in our pipeline and lands in a
plausible place.** That makes it a usable axis for the MoE work: the thesis
claim is about degradation robustness, and rPC is the quantity to move.

### What still does not match — state all of it in the thesis

| | NIRNet | ours |
|---|---|---|
| Eval set | 11,738 DIOR-R test images | 3,915 (1,305 unique ids × 3 severities) |
| Train set | 5,862 | 651 |
| Backbone | ImageNet-pretrained ResNet-50 | yolo11n, **from scratch** |
| Detector | Oriented R-CNN (two-stage) | YOLO11n (one-stage) |
| Degradation | cloud synthesis, per-image random transparency 0.5–0.8 | release's three fixed haze severities |
| Out-of-bounds boxes | kept raw | clipped; 294 of 301 lost orientation (see `voc_obb.py`) |
| Epochs / batch | 12 / 2 | 100 / 16 |

The eval-set difference is not a detail: only 2,607 of 23,463 DIOR ids survive
the Hazy-DIOR release with their filename intact, so we cannot evaluate on the
set NIRNet used. See the pairing report.

There is also an unresolved provenance conflict: NIRNet states Hazy-DIOR was
synthesized from the DIOR-R **test** set, but our aligned ids split 651/651/1,305
across DIOR train/val/test. Until that is settled we should not describe our
evaluation as "the Hazy-DIOR benchmark".

---

## 3. Reproduce

```bash
make train CONFIG=configs/train/fog_yolo11n_obb.yaml
make train CONFIG=configs/train/clear_yolo11n_obb.yaml

python scripts/eval.py --config configs/train/fog_yolo11n_obb.yaml --split test
python scripts/eval.py --config configs/train/clear_yolo11n_obb.yaml --split test \
       --name clearOBB_on_clear
python scripts/eval.py --checkpoint outputs/runs/clear_yolo11n_obb/weights/best.pt \
       --config configs/train/fog_yolo11n_obb.yaml --split test --name clearOBB_on_fog
```

Each writes `metrics.json`, `voc07.json`, `results.md` and pred-vs-GT panels.

## 4. Obvious next steps

1. **A clear-trained HBB run** completes the 2×2 grid and gives an HBB rPC, so
   "does box format change robustness?" becomes answerable. One command.
2. **`pretrained: yolo11n.pt`** — every run here is from scratch. This is the
   single largest available gain and closes much of the gap to a pretrained
   ResNet-50.
3. **3 seeds**, per `05-experiment-plan.md`. Everything here is one seed, so no
   difference under ~2 points should be treated as real.
