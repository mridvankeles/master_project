"""Seeding and run provenance.

Every run records its seed and the git commit that produced it, so a number in
the thesis can be traced back to a command (`05-experiment-plan.md`,
Reproducibility).
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

DEFAULT_SEED = 0


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed python, numpy and (if installed) torch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def git_commit(repo: Path | None = None, short: bool = False) -> str:
    """Current commit hash, or 'unknown' outside a git work tree.

    Appends '-dirty' when the work tree has uncommitted changes, so a run logged
    against a commit cannot be quietly misattributed.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    args = ["git", "-C", str(repo), "rev-parse"]
    args += ["--short", "HEAD"] if short else ["HEAD"]
    try:
        sha = subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    try:
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""

    return f"{sha}-dirty" if dirty else sha
