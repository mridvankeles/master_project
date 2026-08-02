"""Class-stratified subsetting, for the smoke config.

WHY STRATIFIED
--------------
DIOR is severely imbalanced — ship has ~62,500 instances, trainstation ~1,000.
A naive first-N or uniform-random slice of a few hundred images misses the rare
classes entirely, and the resulting run reports AP=0 rows for them. Those zeros
say nothing about the model; they say the class was absent. That is exactly the
kind of confusing output a smoke test should not produce.

The greedy cover below picks images that contribute the rarest still-uncovered
class first, so every class present in the split appears in the subset if any
image contains it, and only then tops up to the requested size at random.

This proves the loop runs. It proves nothing about accuracy.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from .dior_classes import DIOR_CLASSES

__all__ = ["stratified_subset", "write_image_list"]


def _labels_for(image_path: Path) -> Path:
    """Ultralytics' own convention: swap the last /images/ segment for /labels/."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _classes_in(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    out: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(int(line.split()[0]))
    return out


def stratified_subset(
    images_dir: Path,
    n: int,
    seed: int = 0,
    min_per_class: int = 3,
) -> tuple[list[Path], dict[str, int]]:
    """Pick `n` images covering as many classes as possible.

    Returns (selected paths, class -> image count in the selection).

    Args:
        images_dir: a `.../images/<split>` directory.
        n: target number of images.
        seed: sampling seed.
        min_per_class: how many images to try to guarantee per class before
            topping up randomly.
    """
    rng = random.Random(seed)

    images = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"no images in {images_dir}")

    by_image: dict[Path, set[int]] = {}
    by_class: dict[int, list[Path]] = defaultdict(list)
    for img in images:
        classes = _classes_in(_labels_for(img))
        by_image[img] = classes
        for c in classes:
            by_class[c].append(img)

    # Rarest class first, so a common class cannot consume the budget before a
    # rare one has been covered.
    order = sorted(by_class, key=lambda c: len(by_class[c]))

    selected: list[Path] = []
    chosen: set[Path] = set()
    for cls in order:
        have = sum(1 for p in selected if cls in by_image[p])
        pool = [p for p in by_class[cls] if p not in chosen]
        rng.shuffle(pool)
        for p in pool[: max(0, min_per_class - have)]:
            if len(selected) >= n:
                break
            selected.append(p)
            chosen.add(p)

    # Top up at random so the subset is not composed purely of rare-class images,
    # which would give a wildly unrepresentative class distribution.
    remaining = [p for p in images if p not in chosen]
    rng.shuffle(remaining)
    for p in remaining:
        if len(selected) >= n:
            break
        selected.append(p)
        chosen.add(p)

    selected.sort()

    counts = Counter()
    for p in selected:
        for c in by_image[p]:
            counts[DIOR_CLASSES[c]] += 1

    return selected, dict(counts)


def write_image_list(paths: list[Path], dst: Path) -> Path:
    """Write an Ultralytics image-list txt (one absolute path per line)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        "\n".join(str(p.resolve()) for p in paths) + "\n", encoding="utf-8"
    )
    return dst
