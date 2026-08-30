# Auditing the pipeline: what the model actually sees

> **Date:** 2026-08-30 · prompted by "we did EDA on the files, but never on what
> the model consumes on the fly, and we never looked at our worst predictions".
> That distinction turned out to matter more than anything else measured so far.
>
> Reproduce: `scripts/audit_training_batches.py` (tensors reaching the loss),
> `scripts/audit_corpus_labels.py` (labels on disk, per class),
> `scripts/worst_samples.py` (error analysis with the best model).

---

## Summary

| question asked | answer |
|---|---|
| are the label folders mislabelled? | **no** — corpus is clean, verified three ways |
| are the augmentations proper? | **no** — one of them silently destroys the gate label |
| where do the errors come from? | **tiny objects**, not classes, not conditions, not the threshold |

The corpus was the one thing that turned out to be fine. The augmentation
pipeline and the object-size ceiling are the two real problems, and the second
one is larger than anything the MoE work has been chasing.

---

## 1. The corpus is clean

`scripts/audit_corpus_labels.py`, train split, 5,862 images × 3 conditions:

| condition | images | labels | objects | classes |
|---|---:|---:|---:|---:|
| clear | 5,862 | 5,862 | 32,633 | 20/20 |
| fog2 | 5,862 | 5,862 | 32,633 | 20/20 |
| night | 5,862 | 5,862 | 32,633 | 20/20 |

**Cross-condition identity.** `fog2` and `night` are renderings of the *same*
DIOR scenes, so their labels must match `clear` exactly. Compared box by box:

```
fog2  vs clear :  5862 compared, 0 differ
night vs clear :  5862 compared, 0 differ
```

**Per class**, all 20 classes have byte-identical counts across all three
conditions — airplane 1,238 / ship 13,027 / vehicle 6,709 / bridge 506 … no
class is missing from any condition, and no count drifts.

**Malformed labels:** one duplicated box, in the same source file, propagated to
all three conditions. Nothing else: no coordinate out of range, no class id
outside `nc`, no image without a label, no label without an image.

**Visual check.** `scripts/audit_corpus_labels.py` also crops 16 ground-truth
instances of every class into a contact sheet
(`outputs/analysis/class_crops_clear_train/`). Inspected: bridges are bridges,
windmills are windmills, vehicles are vehicles. Two observations from looking:

- **bridge / overpass / dam are visually near-identical** — long thin road or
  water crossings. This looked like an obvious source of error. **It is not**
  (measured in §4).
- **windmill** is a thin bright blade plus a long shadow on low-contrast desert,
  and **vehicle** is frequently a ~10 px blur. These are the two classes fog and
  darkness should hurt most, and they are among the worst three.

So: the class folders are not mislabelled. That hypothesis is closed.

---

## 2. The augmentation problem: mosaic destroys the condition label

Ultralytics composes each training image from **four randomly drawn images**,
then records only the **first** one in `im_file` — documented in
`ultralytics/data/augment.py:759` as *"File path of the first image in the
mosaic."*

Our gate supervision reads its condition label from exactly that field.

Measured on the real training dataloader (300 sampled mosaics, `cond3b_gated`'s
own corpus and settings):

| | measured |
|---|---:|
| mosaic probability | **1.0** (every training image, for 90 of 100 epochs) |
| images per mosaic | 4 |
| mosaics whose tiles are **not all one condition** | **97.3%** |
| mean fraction of canvas matching the reported label | **0.506** |

**So for 90 of 100 epochs the gate was told "this image is fog" about a canvas
that is on average half something else.**

### This explains the behaviour we could not explain

The measured gate output on `cond3b_gated`:

| | value |
|---|---:|
| max probability | **0.500** |
| sum of probabilities | 1.012 |
| concentration (max / sum) | 0.492 |

**The gate's confidence in the labelled condition (0.500) matches the fraction
of the canvas that actually *is* that condition (0.506) to within 0.006.**

The gate was never miscalibrated. It was **correctly calibrated to a corrupted
label** — it learned the true mixture statistics of its training distribution.
Asked to put one-hot mass on a half-and-half canvas, spreading the mass is the
right answer, and it is what any well-fitted model would do.

That retires three earlier conclusions:

- "the gate is miscalibrated, fix it with `pos_weight`" — **wrong**, it was
  fitting its data correctly;
- "`sum(p)` is too low" — **wrong**, it was 1.012 all along;
- the routing cost `|sum(p) − n_true|` addressed a symptom of a label bug, which
  is why it moved routing metrics without moving accuracy.

It also gives a cleaner reading of the whole MoE arc. Every arm was trained on
condition-mixed canvases, so **the experts could not specialise either** — the
"fog expert" was trained on images that were half clear and half night. That is
a better explanation of three flat results than any property of routing.

### The second augmentation problem

| augmentation | value | what it does to our conditions |
|---|---:|---|
| `hsv_v` | **0.40** | jitters **brightness ±40%** — night is *defined* by brightness |
| `hsv_s` | **0.70** | jitters **saturation ±70%** — haze *desaturates*, that is its signature |
| `erasing` | 0.40 | random occlusion |
| `scale` / `translate` | 0.50 / 0.10 | geometric, harmless here |
| `fliplr` | 0.50 | harmless |

Sensible defaults for ordinary detection, and here they randomise the very
evidence the gate is asked to classify. Measured on 120 val images per condition
(HSV channel means, and the range each spans once the jitter is applied):

| | clear | fog2 | night |
|---|---:|---:|---:|
| **brightness V** | 112.9 | 172.8 | 35.8 |
| after `hsv_v=0.4` | **67.8 – 158.1** | **103.7 – 242.0** | 21.5 – 50.1 |
| **saturation S** | 48.5 | 8.6 | 66.1 |
| after `hsv_s=0.7` | **14.5 – 82.4** | 2.6 – **14.6** | **19.8 – 112.4** |

Undisturbed, all three conditions are cleanly separated on both channels. After
the jitter:

- **clear and fog2 overlap on brightness** across a wide band (103.7–158.1),
- **clear and night overlap on saturation** almost completely,
- clear/night stay apart on brightness (50.1 vs 67.8) and clear/fog2 barely stay
  apart on saturation (14.6 vs 14.5).

So each condition keeps exactly one usable cue and loses the other. Not fatal on
its own — but it halves the evidence available to a 771-parameter gate, and it
compounds with the mosaic problem above. Both values were inherited silently:
never chosen, never questioned, and not visible in any config file we wrote.

---

## 3. Error analysis: where the detector actually fails

`cond3b_gated` on 450 union3b val images, greedy IoU-0.5 matching per image,
conf 0.25:

| | value |
|---|---:|
| ground-truth objects | 2,427 |
| **missed** | **471 (19.4%)** |
| false positives | 441 |

### Miss rate by object size — the cliff

| size | missed | of | **miss rate** | share of all misses |
|---|---:|---:|---:|---:|
| **< 8 px** | 30 | 30 | **100.0%** | 6.4% |
| **8–16 px** | 165 | 259 | **63.7%** | 35.0% |
| 16–32 px | 121 | 719 | 16.8% | 25.7% |
| 32–64 px | 73 | 565 | 12.9% | 15.5% |
| ≥ 64 px | 82 | 854 | 9.6% | 17.4% |

**Every object under 8 px is missed. Every one.** Under 16 px, roughly two in
three. Above 16 px the model is a normal detector at ~10–17%.

This is the single largest effect in the whole project, and it is a factor of
six between the bands — far outside the ~2-point noise floor that every MoE
result has been living inside.

### Worst classes

| class | objects | missed |
|---|---:|---:|
| **bridge** | 44 | **61.4%** |
| **windmill** | 141 | **40.4%** |
| **vehicle** | 477 | **36.7%** |
| overpass | 62 | 35.5% |
| expressway-service-area | 35 | 28.6% |
| airplane | 72 | 26.4% |
| basketballcourt | 56 | 25.0% |
| … | | |
| ship | 948 | 6.4% |
| tenniscourt | 129 | 6.2% |
| baseballfield | 76 | 1.3% |

Thin and elongated (bridge, overpass), or tiny (windmill, vehicle). Large
compact classes are nearly solved.

### Per-image F1 by condition

| clear | fog2 | night |
|---:|---:|---:|
| 0.782 | 0.775 | 0.781 |

Indistinguishable. The model is **not** failing on degradation as such. It is
failing on small and thin objects, in every condition equally.

---

## 4. Two plausible explanations, both killed by measurement

Both of these were reasonable, and both are wrong. Recording them because
neither would have been settled by argument.

### "It is class confusion — bridge, overpass and dam look identical"

For every missed object, we asked whether **any** prediction, of any class, was
sitting on it at IoU ≥ 0.5:

| | count | share of misses |
|---|---:|---:|
| **not detected at all** | **462** | **98.1%** |
| detected, wrong class | 9 | 1.9% |

Nine. Across 471 misses. The top confusion pair is bridge → overpass, and it
happens **twice**. The visual similarity is real and it costs essentially
nothing. The model is not confusing these objects; it is not seeing them.

### "It is the confidence threshold — the objects are found but scored low"

Sweeping conf from 0.25 down to 0.01:

| conf | missed | false positives | 8–16px miss rate | bridge |
|---:|---:|---:|---:|---:|
| 0.25 | 471 (19.4%) | 441 | 63.7% | 61.4% |
| 0.10 | 415 (17.1%) | 1,092 | 58.3% | 56.8% |
| 0.05 | 377 (15.5%) | 1,908 | 55.2% | 52.3% |
| **0.01** | **324 (13.3%)** | **5,847** | 49.0% | **50.0%** |

A **13× increase in false positives** buys back 147 of 471 misses. Bridge stays
at 50% missed. `airplane` (26.4%), `basketballcourt` (25.0%) and
`expressway-service-area` (28.6%) do not move **at any threshold** — those
objects produce no box at all, at any score.

The missing detections are genuinely absent from the model's output. This is a
representation limit, not a calibration one.

---

## 5. Testing the obvious fix: more pixels

If 8–16 px objects are the problem, give them more pixels. DIOR images are
800×800 and we train and test at 640, so every object is already shrunk to 0.8×
before the network sees it. Evaluating the *same* checkpoint at higher input:

| imgsz | total missed | < 8 px | 8–16 px | 16–32 px | **≥ 64 px** |
|---:|---:|---:|---:|---:|---:|
| 640 (trained) | 471 | 100.0% | 63.7% | 16.8% | **9.6%** |
| 800 | 445 | 93.3% | 58.3% | 15.0% | **10.4%** |
| 1024 | 438 | 96.7% | 56.4% | 14.0% | **11.1%** |

Small objects improve (63.7% → 56.4%) and **large objects get worse**
(9.6% → 11.1%) — the classic train/test resolution mismatch. Net gain is 33
objects out of 471.

So test-time upscaling is not the fix. But the *direction* is confirmed: the
tiny-object failure is a resolution-and-stride problem, and the levers that
address it are structural —

1. **train at 800 or 1024** (removes the mismatch that spoils the test-time
   version; ~2.5× the training cost at 1024),
2. **add a P2 head at stride 4** — the finest head today is stride 8, so an
   8 px object occupies exactly one cell and a 4 px object cannot be centred in
   one at all,
3. **tile the input** at inference.

None of these is a routing change. That is the point of this section.

---

## 6. What is now running

`cond3d_nomosaic_yolo11n` — identical to `cond3b_gated` except:

```yaml
mosaic: 0.0     # the label-corrupting augmentation
hsv_v: 0.15     # was 0.40 — brightness is the night signal
hsv_s: 0.30     # was 0.70 — saturation is the haze signal
```

plus `union3d_nomosaic_yolo11n`, the dense control under the *same* augmentation
settings, so the comparison stays like-for-like.

**Prediction, recorded before the result:** the gate's max probability should
rise well above 0.5, because the label will finally describe the image. Whether
accuracy follows is a separate question — but for the first time the experts
will be trained on single-condition images, which is the precondition for
specialisation that has been missing all along.

**Early result, epoch 29 of 100:** `gate_max_prob` is **0.676** — already well
above `cond3b_gated`'s *final* 0.500, and still climbing. `gate_loss` has fallen
0.62 → 0.334. The prediction is holding: once the label describes the image, the
gate concentrates. Whether detection follows is still open.

---

## 7. The pipeline, end to end

What happens between disk and loss:

```
corpus  data/dior_hbb_full/<condition>/{images,labels}/<split>/
   |
   |  scripts/train.py -> stratified_subset()  (class-balanced, seeded)
   v
subset list  data/dior_hbb_full/subsets/<run>_train.txt  (5,862 paths)
   |
   |  Ultralytics YOLODataset + Compose
   v
  Mosaic(p=1.0, n=4)   <- 4 images, ONE im_file, condition label corrupted
  CopyPaste, RandomPerspective(scale 0.5, translate 0.1)
  MixUp(0), CutMix(0), Albumentations
  RandomHSV(h .015, s .70, v .40)   <- attacks the condition signal
  RandomFlip(lr 0.5), Format
   |
   v
batch { img, cls, bboxes, im_file }   <- gate label derived from im_file HERE
   |
   +--> v8DetectionLoss  -> box(CIoU) + cls(BCE) + dfl
   +--> gate BCE vs condition_from_paths(im_file)
   |
   v
one backward pass, one optimiser (joint; no separate router stage)
```

Validation and test use `rect`/letterbox with **no augmentation**, so eval sees
clean single-condition images while training saw mixed ones — a train/test
distribution mismatch on the gate's input that nothing was measuring.

---

## 8. What this changes

The MoE work has been optimising a router whose supervision signal was
half-noise, on a detector whose dominant error has nothing to do with routing.
Two consequences:

- **Every routing result in the project needs re-reading** under the mosaic
  finding. `cond3d` is the first arm trained on labels that match their images.
- **The accuracy ceiling is a small-object ceiling.** A 100% miss rate under
  8 px and 63.7% under 16 px is not something a neck-level expert block can
  recover. If the thesis wants a number to move, the next experiments belong at
  P2/resolution, not at the gate.

Neither of these was visible in mAP, in the training loss, or in static EDA.

## 9. Method note

Every bug of consequence in this project has been found the same way: by looking
at a **disaggregated** view of something an aggregate called fine. Per-class AP
caught NWD. Per-image routing caught the collapsed router. Per-batch inspection
caught mosaic. Per-size recall caught the small-object cliff. And the same
method killed two of my own hypotheses in §4 before they became conclusions.
