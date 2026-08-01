"""Path resolution. Nothing in this project hardcodes a dataset location.

`configs/paths.yaml` is the only machine-specific file in the repo. It is
version-controlled so the two machines (RTX 5070 Ti / Windows, 2x A6000) can
each keep their own copy of one small file rather than sprinkling absolute paths
through the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
VERIFICATION_DIR = OUTPUT_DIR / "verification"

DEFAULT_PATHS_YAML = CONFIG_DIR / "paths.yaml"


@dataclass(frozen=True)
class SourcePaths:
    """Resolved locations inside the read-only Hazy-DIOR release.

    Nothing in this project writes into these directories.
    """

    hazy_dior_root: Path

    @property
    def annotations_hbb(self) -> Path:
        return self.hazy_dior_root / "Annotations" / "Horizontal Bounding Boxes"

    @property
    def annotations_obb(self) -> Path:
        """Present in the release; never used — see 03-datasets.md (HBB decision)."""
        return self.hazy_dior_root / "Annotations" / "Oriented Bounding Boxes"

    @property
    def imagesets_main(self) -> Path:
        """DIOR's *detection* splits. The only split definition this project uses."""
        return self.hazy_dior_root / "ImageSets" / "Main"

    @property
    def images_root(self) -> Path:
        """The nested `Hazy-DIOR/Hazy-DIOR/` image tree."""
        return self.hazy_dior_root / "Hazy-DIOR"

    @property
    def aligned_clear_root(self) -> Path:
        """`test/gt/` — clear DIOR originals, keyed by DIOR ID.

        Byte-identical across the three severity sub-folders (verified), so any
        one of them is the clear image.
        """
        return self.images_root / "test" / "gt"

    @property
    def aligned_fog_root(self) -> Path:
        """`test/haze/` — hazy renders, keyed by DIOR ID, one per severity."""
        return self.images_root / "test" / "haze"

    def unaligned_root(self, split: str) -> Path:
        """`train/` or `val/` — RENUMBERED, DIOR id destroyed. Do not use.

        Retained so `check_pairing` can inspect and report on them. See the
        pairing report: these directories are a restoration split whose
        filenames are sequential indices, not DIOR ids.
        """
        if split not in {"train", "val"}:
            raise ValueError(f"unaligned splits are train/val, got {split!r}")
        return self.images_root / split

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means all good."""
        problems: list[str] = []
        for label, path in [
            ("hazy_dior_root", self.hazy_dior_root),
            ("annotations_hbb", self.annotations_hbb),
            ("imagesets_main", self.imagesets_main),
            ("aligned_clear_root", self.aligned_clear_root),
            ("aligned_fog_root", self.aligned_fog_root),
        ]:
            if not path.exists():
                problems.append(f"{label}: does not exist -> {path}")
        return problems


def load_paths(config: str | Path | None = None) -> SourcePaths:
    """Load `configs/paths.yaml` and resolve it against the repo root."""
    config = Path(config) if config else DEFAULT_PATHS_YAML
    if not config.exists():
        raise FileNotFoundError(
            f"{config} not found. Copy configs/paths.example.yaml to "
            f"configs/paths.yaml and point it at your Hazy-DIOR release."
        )
    with config.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if "hazy_dior_root" not in raw:
        raise KeyError(f"{config}: missing required key 'hazy_dior_root'")

    root = Path(raw["hazy_dior_root"]).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()

    return SourcePaths(hazy_dior_root=root)


def dataset_root() -> Path:
    """Where `prepare_dataset` materialises the Ultralytics-format corpus."""
    return DATA_DIR / "dior_hbb"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
