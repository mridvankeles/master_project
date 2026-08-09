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
    """Resolved locations inside the read-only source releases.

    Nothing in this project writes into these directories.

    Two releases, with different roles:

    * `hazy_dior_root` — the Hazy-DIOR release. Supplies the fog imagery, and
      also carries DIOR's annotations and split lists (verified byte-identical
      to the official ones).
    * `dior_root` — the official DIOR release. Supplies CLEAR imagery for all
      23,463 ids. Optional, because the aligned-scope build predates it and
      still works without it; required for `scope="full"`.
    """

    hazy_dior_root: Path
    dior_root: Path | None = None

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

    # --- official DIOR release ------------------------------------------
    # The Hazy-DIOR release carries clear imagery for only the 2,607 aligned
    # ids (`aligned_clear_root`). These give the other 20,856.

    @property
    def dior_images_trainval(self) -> Path:
        """DIOR ids 00001-11725. Split membership comes from ImageSets, not here."""
        if self.dior_root is None:
            raise ValueError("dior_root is not configured; add it to configs/paths.yaml")
        return self.dior_root / "JPEGImages-trainval"

    @property
    def dior_images_test(self) -> Path:
        """DIOR ids 11726-23463."""
        if self.dior_root is None:
            raise ValueError("dior_root is not configured; add it to configs/paths.yaml")
        return self.dior_root / "JPEGImages-test"

    def dior_image(self, image_id: str) -> Path | None:
        """Clear image for a DIOR id, or None if absent.

        The release splits its imagery across two directories by id range, but
        that division is a packaging detail and carries no split meaning — the
        detection split is `ImageSets/Main`. So this looks in both rather than
        computing which one to use from the id.
        """
        if self.dior_root is None:
            return None
        for directory in (self.dior_images_trainval, self.dior_images_test):
            candidate = directory / f"{image_id}.jpg"
            if candidate.exists():
                return candidate
        return None

    # --- renumbered Hazy-DIOR subtrees ----------------------------------

    def renumbered_root(self, split: str, condition: str) -> Path:
        """`train/`|`val/` x `gt`|`haze` — sequentially numbered, DIOR id absent.

        These are the subtrees `pairing.py` documents as unusable: filenames are
        indices, not DIOR ids. `src/data/recovery.py` recovers the mapping by
        pixel hash, which is what makes them usable after all.
        """
        if split not in {"train", "val"}:
            raise ValueError(f"renumbered subtrees are train/val, got {split!r}")
        if condition not in {"gt", "haze"}:
            raise ValueError(f"condition must be gt|haze, got {condition!r}")
        return self.images_root / split / condition

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
        if self.dior_root is not None:
            for label, path in [
                ("dior_root", self.dior_root),
                ("dior_images_trainval", self.dior_images_trainval),
                ("dior_images_test", self.dior_images_test),
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

    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    dior_root = raw.get("dior_root")
    return SourcePaths(
        hazy_dior_root=_resolve(raw["hazy_dior_root"]),
        # Optional: the aligned-scope build predates the DIOR download and does
        # not need it. `scope="full"` does, and says so.
        dior_root=_resolve(dior_root) if dior_root else None,
    )


def dataset_root(task: str = "detect", scope: str = "aligned") -> Path:
    """Where `prepare_dataset` materialises the Ultralytics-format corpus.

    HBB and OBB get separate roots. Sharing one would let an oriented label be
    handed to a detect model (or the reverse), which trains without complaint
    and reports nonsense.

    Scope gets a separate root for the same reason: the aligned corpus (2,607
    ids) and the full one (~23,385) are different datasets, and a run that
    silently switched between them would be uncomparable to its own history.
    """
    base = "dior_hbb" if task == "detect" else "dior_obb"
    return DATA_DIR / (base if scope == "aligned" else f"{base}_{scope}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
