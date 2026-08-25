# Result — `fog_yolo11n`, first real training run

Stock YOLO11n, horizontal boxes, trained **on** Hazy-DIOR fog. First end-to-end
run of the pipeline; a plumbing result, not a thesis result. Read the caveats.

- Commit: `46604ef` · seed 0 · RTX 5070 Ti · torch 2.9.1+cu128 · Ultralytics 8.4.115
- Config: `configs/train/fog_yolo11n.yaml` — yolo11n @640, 100 epochs, batch 16,
  **from scratch**, `optimizer: auto`, `cos_lr: true`

> **Correction.** This run was first recorded as COCO-pretrained. It was not.
> The config said `pretrained: true`, which is a **silent no-op** with a `.yaml`
> model — Ultralytics only loads weights when `pretrained` is a string path
> (`engine/trainer.py:817-820`); given a bool it leaves `weights = None` and
> builds from the yaml. The config now says `pretrained: false` explicitly. To
> genuinely transfer from COCO, set `pretrained: yolo11n.pt`, which is the
> obvious next improvement and should raise these numbers substantially.
- Data: fog condition, 1,953 train / 1,953 val / 3,915 test images
  (651 / 651 / 1,305 unique DIOR ids × 3 haze severities)
- Metric convention: Ultralytics/COCO, 101-point interpolation

## Headline

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| val | 0.4738 | 0.2985 | 0.6224 | 0.4524 |
| **test** | **0.3345** | **0.1936** | 0.5666 | 0.3230 |

Speed on the 5070 Ti: 0.89 ms inference, 0.70 ms preprocess, 0.76 ms postprocess
per image at 640.

## Three things to be honest about

**1. The val→test gap is 14 mAP50 points.** `best.pt` was selected on val, so val
is optimistically biased — but 0.474 → 0.335 is far more than selection noise.
val and test are disjoint DIOR id ranges (val 05863–11725, test 11726–23463), and
the model saw only 651 unique scenes in training. Treat **test** as the honest
number and val as a tuning signal only.


**2. It had not converged.** Best epoch = **100 of 100**; mAP50 was still rising
at the last epoch and `patience: 30` never triggered. The run is
under-trained, not over-trained — a longer schedule should improve it. Training
from scratch (see the correction above) makes that expected: 100 epochs on 1,953
images is not much for a randomly initialised backbone.

**3. Losses were still falling.** train box 3.65 → 1.18, cls 5.47 → 1.05,
val box 2.81 → 1.49, val cls → 1.44 across 100 epochs, all monotone with no
divergence between train and val. No overfitting signature; just too few epochs.

## Per-class AP50 on test — the imbalance shows through

| Strong (AP50 > 0.4) | | Weak (AP50 < 0.2) | |
|---|---|---|---|
| ship | 0.650 | expressway-toll-station | 0.194 |
| baseballfield | 0.621 | storagetank | 0.186 |
| tenniscourt | 0.612 | windmill | 0.171 |
| expressway-service-area | 0.515 | airport | 0.129 |
| airplane | 0.509 | bridge | 0.120 |
| chimney | 0.437 | dam | 0.107 |
| basketballcourt | 0.429 | trainstation | 0.068 |

Two patterns, both expected:

- **Large, high-contrast, well-represented classes win** — ship (14,280 test
  instances), baseballfield, tenniscourt.
- **`storagetank` is the diagnostic case: P = 0.905 but R = 0.093.** When it
  fires it is almost always right, but it finds 9 % of them. That is a
  small-dense-object recall failure, not a classification failure, and it is
  exactly the regime `01-scope-and-claim.md` argues belongs in the neck/FPN
  rather than in an expert. `vehicle` shows the same shape (P 0.657, R 0.174).

The qualitative panels show the same thing: `predvsgt_22094_thin.jpg` has 17
ground-truth airplanes packed on an apron and **0 detections** at conf 0.25.

## What this result is NOT

- **Not comparable to NIRNet's 40.86.** That is OBB, an unstated mAP convention,
  and — decisively — trained on *clean* images and evaluated zero-shot on hazy.
  We trained on fog, a strictly easier problem. See `comparison-baselines.md`.
- **Not a DIOR baseline.** 651 unique training scenes against DIOR's 5,862. The
  full-scale run is blocked on downloading clear DIOR and recovering the
  train/val id mapping.
- **Not a tuned result.** One seed, default hyperparameters, no sweep.
  `05-experiment-plan.md` requires 3 seeds and mean ± std for any headline number.

## Reproduce

```bash
make train CONFIG=configs/train/fog_yolo11n.yaml
make eval  CONFIG=configs/train/fog_yolo11n.yaml SPLIT=test
```

MLflow run `fog_yolo11n-f5f4804-dirty` in experiment `dior-hbb`
(`outputs/mlruns`) carries 119 hyperparameters, 13 per-epoch metrics and the
provenance tags.
