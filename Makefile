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
ULTRALYTICS_DIR ?= ../ultralytics
# Pinned in requirements.txt; repeated here because make does the checkout.
ULTRALYTICS_SHA ?= 8fbfccb8e6139f4e995f55d1272dc96e66699346
DVC_REMOTE    ?= ../dvcstore

.PHONY: setup data verify eda train eval test clean help

help:
	@echo "setup   - install pinned deps + editable Ultralytics, init DVC"
	@echo "data    - check the release, convert annotations, materialise the corpus"
	@echo "verify  - render boxes onto images for visual inspection"
	@echo "eda     - execute the EDA notebook            (Task 3, not yet written)"
	@echo "train   - train from a config                 (Task 4, not yet written)"
	@echo "eval    - evaluate a checkpoint               (Task 5, not yet written)"
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
	$(PYTHON) -m pip install -e $(ULTRALYTICS_DIR)
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
# not yet implemented — fail loudly rather than confusingly
# --------------------------------------------------------------------------
eda:
	@echo "Task 3 (notebooks/01-eda.ipynb) is not written yet."
	@exit 1

train:
	@echo "Task 4 (scripts/train.py + configs/train/) is not written yet."
	@exit 1

eval:
	@echo "Task 5 (scripts/eval.py) is not written yet."
	@exit 1

# --------------------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests -v

clean:
	$(PYTHON) -c "import shutil,pathlib; shutil.rmtree(pathlib.Path('outputs/verification'), ignore_errors=True)"
