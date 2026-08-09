# Gate 1 — the reference number, written down before training

`05-experiment-plan.md` requires: *"Pick a specific paper and a specific DIOR mAP
number, and write it down **before** training."* This file is that commitment.
It was written before any full-scale run started; git history is the evidence.

## Source

Li, K., Wan, G., Cheng, G., Meng, L., Han, J. (2020). *Object detection in
optical remote sensing images: A survey and a new benchmark.* ISPRS Journal of
Photogrammetry and Remote Sensing 159, 296–307. arXiv:1909.00133v2.

This is the paper that introduced DIOR, so its baselines are the canonical HBB
numbers on this dataset — and unlike NIRNet, DHC-Net, CM-YOLO or MPRNet, they
are **horizontal boxes on clear DIOR**, which is what our detector produces.
`comparison-baselines.md` records that none of the four main papers supplied
such a number; this is the missing reference.

## Protocol the paper used

Quoted from §5.1:

> "we randomly selected 11725 remote sensing images (i.e., 50% of the dataset)
> as trainval set, and the remaining 11738 images are used as test set."

That is exactly the split shipped in `ImageSets/Main` and exactly the one this
project uses — 5,862 train + 5,863 val = 11,725 trainval, 11,738 test.

## Reported mAP on the DIOR test set (HBB, 20 classes)

| Detector | Backbone | mAP |
|---|---|---|
| YOLOv3 | Darknet-53 | **57.1** |
| SSD | VGG16 | 58.6 |
| Faster R-CNN | ResNet-50 | 63.1 |
| RetinaNet | ResNet-50 | 63.8 |
| CornerNet | Hourglass-104 | 64.9 |
| RetinaNet | ResNet-101 | **66.1** |
| PANet | ResNet-101 | **66.1** |

## The number this gate is assessed against

> **YOLOv3 / Darknet-53 = 57.1 mAP.**

Chosen because it is the only one-stage YOLO-family entry in the table, so it is
the closest architectural analogue to YOLO11n. The 66.1 ceiling is recorded as
the best published result on the benchmark.

## Three differences that must be stated with any comparison

1. **Training data.** The paper trains on **trainval (11,725)**. We train on
   **train only (5,862)** and hold `val` back for checkpoint selection, because
   selecting on the test split would be tuning on test. So we train on half the
   data the published number used. Our result is expected to land below 57.1 for
   that reason alone, and that is a protocol difference, not a failed gate.
2. **Model capacity.** YOLO11n is 2.6 M parameters. Darknet-53 is ~62 M, and the
   66.1 entries are ResNet-101. We are comparing a nano model against backbones
   20–40x its size.
3. **Metric convention.** The paper reports VOC-style mAP@0.5. Ultralytics
   reports 101-point COCO mAP50. Both are computed here — `metrics.json` for
   COCO, `voc07.json` for VOC07 11-point — and the VOC07 figure is the one that
   belongs beside 57.1.

## Gate criterion

Gate 1 passes if stock YOLO11n on clear DIOR lands in a **defensible relationship**
to 57.1 once those three differences are accounted for: same ballpark, correct
ordering against its own capacity, and no sign of a broken pipeline (per-class AP
distributed sensibly rather than a handful of classes carrying everything).

It fails if the number is far enough below that a data or conversion bug is the
better explanation than capacity — which is exactly what the gate exists to catch
before months of downstream work rest on it.
