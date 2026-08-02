# Entrypoints. Every target assumes the `rsmoe` conda env is ACTIVE:
#
#     conda env create -f environment.yml     # first time only
#     conda activate rsmoe
#
# `make` itself comes from that env (conda-forge make), so it is available on
# the Windows box as well as on the A6000 machine.
#
# Recipes use only `python`, `git`, `pip`, `dvc` and `yolo` — no shell builtins,
# no `&&`, no globbing — so they behave identically under cmd.exe and sh.

PYTHON        ?= python
# NOT `../ultralytics`: a sibling directory with the package's own name shadows
# the installed package for any python started from the parent directory.
ULTRALYTICS_DIR ?= ../ultralytics-src
# The COMMIT for tag v8.4.115. `git ls-remote --tags` hands back the annotated
# tag object instead (8fbfccb...), which checkout peels but `git log` rejects.
ULTRALYTICS_SHA ?= 98a9cfde3119079568620bd43c26bb541c61ac8d
DVC_REMOTE    ?= ../dvcstore

.PHONY: setup data verify eda train eval test clean help

help:
	@echo "setup   - install pinned deps + editable Ultralytics, init DVC"
	@echo "data    - check the release, convert annotations, materialise the corpus"
	@echo "verify  - render boxes onto images for visual inspection"
	@echo "train   - train from CONFIG (default configs/train/smoke.yaml)"
	@echo "eval    - evaluate CONFIG's checkpoint on SPLIT (default val)"
	@echo "mlflow  - launch the MLflow UI on outputs/mlruns"
	@echo "eda     - execute the EDA notebook            (Task 3, not yet written)"
	@echo "test    - run the test suite"

# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
# Ultralytics is cloned as a SIBLING of this repo and installed editable, so
# custom modules can be added later without vendoring a fork. The `-` prefix on
# the clone makes a re-run a no-op instead of an error; the checkout that
# follows is what actually pins the version, and it fails loudly on drift.
setup:
	-git clone https://github.com/ultralytics/ultralytics.git $(ULTRALYTICS_DIR)
	git -C $(ULTRALYTICS_DIR) fetch --tags --quiet
	git -C $(ULTRALYTICS_DIR) checkout --quiet $(ULTRALYTICS_SHA)
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e $(ULTRALYTICS_DIR) --no-deps
	-dvc init
	dvc remote add -d -f local $(DVC_REMOTE)
	yolo settings mlflow=True
	@echo "setup complete. Ultralytics pinned at $(ULTRALYTICS_SHA)"

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
# check_pairing runs FIRST and exits non-zero on a structural surprise, so a
# release that differs from the one this project was built against stops the
# pipeline instead of quietly producing a mispaired corpus.
data:
	$(PYTHON) scripts/check_pairing.py
	$(PYTHON) scripts/prepare_dataset.py

verify:
	$(PYTHON) scripts/render_verification.py

# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------
# Override the config to pick a run:
#     make train CONFIG=configs/train/fog_yolo11n.yaml
CONFIG ?= configs/train/smoke.yaml
SPLIT  ?= val

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

eval:
	$(PYTHON) scripts/eval.py --config $(CONFIG) --split $(SPLIT)

mlflow:
	mlflow ui --backend-store-uri outputs/mlruns

eda:
	@echo "Task 3 (notebooks/01-eda.ipynb) is not written yet."
	@exit 1

# --------------------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests -v

clean:
	$(PYTHON) -c "import shutil,pathlib; shutil.rmtree(pathlib.Path('outputs/verification'), ignore_errors=True)"
