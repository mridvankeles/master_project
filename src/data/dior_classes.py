"""The 20 DIOR classes, in canonical order.

This is the single source of truth for class identity in the project. The order
here IS the integer class id used in every Ultralytics label file, every
`data.yaml`, and every per-class AP table. Changing it silently invalidates
every label file and every checkpoint already produced.

The order is taken verbatim from the DIOR release and matches
`nirnet-main/mmrotate/datasets/dior.py`, so per-class numbers line up with
NIRNet's published tables without a remapping step.

Names are lower-case and unspaced, exactly as they appear in the `<name>` tag of
the annotation XMLs (`expressway-service-area`, not `expressway service area`).
"""

from __future__ import annotations

DIOR_CLASSES: tuple[str, ...] = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "expressway-service-area",
    "expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)

NUM_CLASSES: int = len(DIOR_CLASSES)

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(DIOR_CLASSES)}
ID_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(DIOR_CLASSES)}

assert NUM_CLASSES == 20, f"DIOR has 20 classes, got {NUM_CLASSES}"


def class_id(name: str) -> int:
    """Map an XML `<name>` value to its integer class id.

    Raises KeyError on an unknown name rather than returning a sentinel — an
    unrecognised class means the annotation set is not the one we think it is,
    and that must surface immediately, not as a silently dropped object.
    """
    key = name.lower().strip()
    if key not in CLASS_TO_ID:
        raise KeyError(
            f"{name!r} is not a DIOR class. Known: {', '.join(DIOR_CLASSES)}"
        )
    return CLASS_TO_ID[key]
