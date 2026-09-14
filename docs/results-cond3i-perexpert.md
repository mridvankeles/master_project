# cond3i — per-expert reconstruction + bypass penalty

> **Date:** 2026-09-15 · 100 epochs, seed 0, `cond3i_perexpert_yolo11n.yaml`.
> Control: `union3h_dense_yolo11n` (same data, schedule, seed, augmentation).

---

## Verdict

**The two new losses did exactly what they were designed to do, and the
intervention is still flat.** That combination is the finding.

| criterion (set before the run) | target | result |
|---|---|---|
| intervention spread | **> 0.01** | **0.0013 – 0.0023** ✗ |
| fog branch wins on fog data | yes | partial |
| `expert_share` rises and stays up | yes | 0.047 → **0.320** ✓ |
| restoration gain positive | — | **first time positive** ✓ |

---

## 1. The losses worked

| | before (`cond3h`) | after (`cond3i`) |
|---|---:|---:|
| `expert_share` (fraction of block output the experts carry) | 0.047 at init | **0.320** |
| `restore_fog` | — | 0.543 → **0.064** |
| `restore_night` | — | 0.650 → **0.228** |
| `bypass_loss` | — | 0.219 → **0.051** |
| **restoration gain, fog** | **−0.0368** | **+0.0453** (81% of images improved) |
| **restoration gain, night** | **−0.0095** | **+0.0284** (83% improved) |

The per-expert reconstruction term reversed the sign of the thing it targets.
The experts now measurably move degraded P3 features **towards their clear
twin** — the operational definition of "the fog branch removes fog before the
prediction head". Under `cond3h`, where the same objective was applied to the
block *sum*, the experts moved features the wrong way.

## 2. And the detector does not care

```
data      gate     none      all  force:clear  force:fog  force:night   spread
clear   0.7629   0.7630   0.7610       0.7633     0.7623       0.7631   0.0023
fog2    0.7476   0.7473   0.7466       0.7479     0.7477       0.7473   0.0013
night   0.7346   0.7346   0.7330       0.7333     0.7336       0.7346   0.0017
```

Switching **every expert off** still ties the trained gate on every condition.

**This is the cleanest statement of the problem the project has produced:
feature restoration and detection are decoupled.** The experts provably move the
P3 features 4.5% closer to their clear twin, and that movement provably does not
change a single prediction. Either 4.5% is too small to matter, or the detection
head is insensitive to the block output in the direction the experts move it.

## 3. The bypass penalty behaved exactly as warned

Its docstring, written before the run:

> It forces expert MAGNITUDE, not expert USEFULNESS. A model can satisfy it by
> inflating the experts and shrinking the bypass with no functional change at
> all — the ratio moves, the prediction does not.

`expert_share` 0.047 → 0.320, intervention unchanged. The prediction was correct
and is worth keeping as a record: a healthy-looking auxiliary metric is not
evidence of a healthy mechanism.

## 4. One genuine first: a specialisation ordering appeared

Restoration gain with each branch forced alone:

| data | clear | **fog** | **night** | winner |
|---|---:|---:|---:|---|
| fog2 | **0.0633** | **0.0453** | 0.0050 | clear |
| night | **0.0459** | 0.0205 | **0.0284** | clear |

`clear` still wins outright — but look at the other two columns. **On fog data
the fog branch beats the night branch 0.0453 to 0.0050; on night data the night
branch beats the fog branch 0.0284 to 0.0205.** That ordering is correct, and it
is the first time in the project that any condition-appropriate specialisation
has been measured at all.

Why `clear` still wins is explicable: "move towards the clear twin's features" is
most directly achieved by a branch that behaves like a gentle identity, which is
precisely what `ClearExpert` is. The degraded branches start from priors that
transform features *away* from the raw input, so they have further to travel.

## 5. Accuracy and cost

| run | test mAP50 | test mAP50-95 | vs dense control |
|---|---:|---:|---:|
| `union3h_dense` | **0.6296** | **0.4390** | — |
| `cond3h_hetero` | 0.6199 | 0.4316 | −0.0097 |
| **`cond3i_perexpert`** | **0.6220** | **0.4327** | **−0.0076** |

Slightly better than `cond3h`, still below its dense control.

| run | params | GF activated | FPS @b16 |
|---|---:|---:|---:|
| `union3h_dense` | 2.594 M | 6.52 | **1026.9** |
| `cond3i_perexpert` | 3.278 M | 7.65 | 477.5 |

(Absolute FPS is lower than the August benchmark for both arms — the machine was
busier. The ratio is what transfers: 0.46 here against 0.51 then.)

Routing is unchanged and healthy: NMI **0.8338**, fog branch fires on 100% of
fog images, night on 100% of night.

---

## What this leaves

Four designs have now produced a flat intervention:

| design | what it added | spread |
|---|---|---:|
| 2 `cond3d` | supervised gate, static priors | 0.0011–0.0017 |
| 3 `cond3h` | heterogeneous architectures, hard mask, rich proj, non-zero init, block-level restoration | 0.0010–0.0020 |
| 3i `cond3i` | **per-expert reconstruction, bypass penalty** | 0.0013–0.0023 |

`cond3j` (no shared branch at all) is running and is the last cheap test. If the
intervention is flat there too, the remaining always-on path is `proj`, and the
conclusion is that a **residual** MoE at the P3 neck of a 2.6M-parameter detector
cannot be made to matter by any training-side intervention — which is a result,
and a well-evidenced one.

The two structural alternatives left, in order of cost:

1. **Per-expert detection heads + confidence-weighted loss** (AW-MoE Eq. 8). An
   expert cannot be ignored when nothing else computes its output. This is the
   only remaining change that makes the experts non-optional *by construction*
   rather than by penalty.
2. **Image-space classical priors** (`research-image-space-priors.md`). Worth
   doing, but downstream: it improves expert *diversity*, and diversity has never
   been the binding constraint.
