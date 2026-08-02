# Thesis scaffold — stock YOLO11 (HBB) on the DIOR / Hazy-DIOR condition grid

Infrastructure for *Robust Remote Sensing Object Detection in Adverse Conditions
via Task-Specific Mixture of Experts*. **Infrastructure only.** No MoE, no custom
modules, no architecture changes — the detector is stock Ultralytics YOLO11 with
horizontal boxes. Modelling comes later.

Design decisions live in the `.md` files one directory up (`01-scope-and-claim.md`,
`03-datasets.md`, `04-method-open-questions.md`, `05-experiment-plan.md`). This
repo implements them; it does not decide anything.

---

## Read this first: the data is not what the spec assumed

The scaffold spec asked for a check that "Hazy-DIOR filenames map 1:1 onto clear
DIOR filenames and that the splits match." **They do not.** The release is a
*dehazing* dataset, and the difference changes what can be built.

| Directory | Files | Keyed by |
|---|---|---|
| `test/{gt,haze}/{thin,moderate,thick}/` | 2,607 ids | **DIOR ID** (`.jpg`) |
| `train/{gt,haze}/` | 56,310 (18,770 unique) | **renumbered 1..N** (`.png`) |
| `val/{gt,haze}/` | 6,258 (2,086 unique) | **renumbered 1..N** (`.png`) |

- Only **2,607 of 23,463** images keep their DIOR filename — exactly the stride-9
  subsample, every id ≡ 8 (mod 9).
- `train/` and `val/` repeat each image three times (one per severity) under
  sequential renumbering, destroying the DIOR id. So `train/haze/00100.png` is
  **not** the hazy version of `Annotations/…/00100.xml`. Filename-based pairing
  silently mispairs 20,856 images.
- `18,770 + 2,086 + 2,607 = 23,463` — the partition closes exactly.
- The release ships **two incompatible splits**. `ImageSets/Main/*.txt` is DIOR's
  *detection* split (5,862 / 5,863 / 11,738) — the one NIRNet's config consumes.
  The directory names are a *restoration* split. **This project keys splits off
  `ImageSets/Main` only.** Using the directory split would put **10,433 DIOR-test
  images into training**, the leak `05-experiment-plan.md` lists first.
- Clear DIOR `JPEGImages/` is not on disk. `test/gt/` *is* clear DIOR for the
  2,607 aligned ids, and is byte-identical across the three severity folders.

### What this repo therefore builds on

| Condition | Images | train / val / test |
|---|---|---|
| Clear | 2,607 | 651 / 651 / 1,305 |
| Fog (all three severities) | 7,821 | 1,953 / 1,953 / 3,915 |
| Dark | — | stub only (`src/data/degradation.py`) |

651 training images is small. It is enough to prove the loop, the converter and
the metrics plumbing — it proves nothing about detection accuracy. The full-scale
run on clear DIOR is blocked until clear DIOR is downloaded **and** the
`train/`+`val/` id mapping is recovered by content-matching the `gt/` renders
against the DIOR originals.

`make data` regenerates all of the above into
`outputs/verification/pairing_report.md`.

---

## Setup from scratch

Works on both target machines: **RTX 5070 Ti (Blackwell, sm_120, Windows)** and
**2× A6000 (Linux)**.

```bash
git clone <this repo> MasterRepoistory
cd MasterRepoistory

conda env create -f environment.yml     # python 3.12, make, git
conda activate rsmoe

make setup                              # pinned deps + editable Ultralytics + DVC
```

`make setup` clones Ultralytics as a **sibling** of this repo (`../ultralytics`),
checks out the pinned commit, and installs it editable so custom modules can be
added later without vendoring a fork.

| Component | Pin |
|---|---|
| Python | 3.12 |
| torch / torchvision | `2.9.1+cu128` / `0.24.1+cu128` |
| Ultralytics | `8fbfccb8e6139f4e995f55d1272dc96e66699346` (tag `v8.4.115`) |
| MLflow | 3.6.0 |
| DVC | 3.65.0 |

**The cu128 wheels are not optional on the 5070 Ti.** Blackwell is sm_120, and
the default PyPI wheels (cu126 and older) carry no kernels for it — they install
cleanly and then fail at the first kernel launch with *"no kernel image is
available for execution on the device."* cu128 is also fine on the A6000s
(sm_86), so both machines run the same pins.

### If `conda activate rsmoe` does nothing (Windows)

`conda init powershell` writes its hook into `Documents\WindowsPowerShell\profile.ps1`
— the *CurrentUserAllHosts* profile, not the `Microsoft.PowerShell_profile.ps1`
that `$PROFILE` prints. On a stock Windows install the execution policy defaults
to `Restricted`, so that profile never loads, the hook never runs, and `conda`
stays a bare `conda.exe`. Activation then silently fails: `conda activate` has to
mutate the *current* shell's environment, and an `.exe` can only change its own
child process.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
conda init powershell      # no-op if already initialised
```

Check it took: `(Get-Command conda).CommandType` should say `Alias`, not
`Application`. On Linux the equivalent is `conda init bash` and there is no
policy to change.

### Point the repo at your data

`configs/paths.yaml` is the only machine-specific file. Set one key:

```yaml
hazy_dior_root: C:/Users/Ridvan/Desktop/tez/NIR-Net_hazyDior/Hazy-DIOR
```

It must be the directory directly containing `Annotations/`, `ImageSets/` and
`Hazy-DIOR/`. The release nests a second `Hazy-DIOR/` inside the first — point at
the **outer** one. No other file in the repo contains an absolute path.

### DVC

`make setup` initialises DVC and configures a **local remote** at `../dvcstore`
(i.e. `tez/dvcstore/`). Track the materialised corpus with:

```bash
dvc add data/dior_hbb
dvc push
```

Caveat: the source imagery lives outside the repo, so DVC records the derived
corpus but cannot hash the release itself. `scripts/check_pairing.py` is what
guards against the release changing underneath you.

---

## Running it

```bash
make test      # converter round-trip + release invariants
make data      # verify the release, convert annotations, materialise the corpus
make verify    # draw boxes onto images for inspection
```

Then look at `outputs/verification/`:

| File | What it is |
|---|---|
| `pairing_report.md` | every structural check, with evidence |
| `boxes_NN_<cond>_<split>_<id>.jpg` | 20 samples, boxes drawn from the label files |
| `pair_<id>.jpg` | one id as clear \| thin \| moderate \| thick, side by side |

**Boxes must sit ON the objects.** The rendered images read boxes back from the
materialised `.txt` files, so they exercise the whole round trip
`xml → to_yolo → label.txt → from_yolo → pixels`. Pairing arithmetic can be right
while the converter is wrong; a transposed axis or an off-by-one survives every
automated check and is obvious here.

`make eda`, `make train` and `make eval` are Tasks 3–5 and currently exit with a
message saying so.

---

## Layout

```
configs/
  paths.yaml            the only machine-specific file
  data/dior_*.yaml      generated by prepare_dataset.py
  train/                Task 4
data/                   DVC-tracked, gitignored
src/
  data/
    dior_classes.py     the 20 classes — order IS the label encoding
    voc_hbb.py          VOC HBB xml -> normalised `class cx cy w h`
    pairing.py          every release invariant, with its evidence
    build_dataset.py    materialises the Ultralytics layout (hardlinks)
    degradation.py      low-light synthesis — STUB, deliberately unimplemented
  eval/                 Task 5
  utils/                paths, seeding + git provenance, logging
scripts/
  check_pairing.py      -> outputs/verification/pairing_report.{md,json}
  prepare_dataset.py    -> data/dior_hbb/ + configs/data/*.yaml
  render_verification.py-> outputs/verification/*.jpg
tests/
outputs/                gitignored (runs, checkpoints, figures, mlruns)
```

### Two implementation notes worth knowing

**Images are hardlinked, not copied.** Ultralytics locates a label by
string-replacing the last `/images/` in the image path with `/labels/`. Pointed
at the release tree directly it finds no `/images/` segment and would write
labels *into* the read-only source directories. Hence a materialised copy — but
hardlinked, so 10,428 images cost effectively nothing. Copying is the automatic
fallback when `data/` and the release sit on different volumes, which will be the
case on the A6000 machine.

**Bad geometry is reported, never silently dropped.** Out-of-bounds and
zero-area boxes are counted and printed by `prepare_dataset.py`. Clamping to
[0,1] happens only at label-write time, after the raw geometry has been recorded,
so the report shows what the source data actually contained.

---

## Experiment tracking

MLflow is driven **entirely** through Ultralytics' built-in callback. There is no
`import mlflow` anywhere in `src/` or `scripts/`, and there will not be.

```bash
yolo settings mlflow=True                      # done by `make setup`
export MLFLOW_TRACKING_URI=file:outputs/mlruns # $env:MLFLOW_TRACKING_URI on Windows
mlflow ui --backend-store-uri outputs/mlruns
```

---

## Standing rules this repo follows

- Stock model only. No custom architecture, no MoE, no extra losses.
- Horizontal boxes (`detect`), never `obb`. The oriented annotation set ships in
  the release and is never read.
- No degradation synthesis yet — `src/data/degradation.py` raises
  `NotImplementedError` and documents what goes there.
- Nothing in the test-only list in `03-datasets.md` ever enters a training split.
- Splits come from `ImageSets/Main`. Never re-split, never reshuffle.
