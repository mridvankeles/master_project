"""PASCAL-VOC horizontal-bounding-box XML -> Ultralytics `detect` labels.

DIOR ships two annotation sets. This module reads only
`Annotations/Horizontal Bounding Boxes/`, per the HBB decision in
`03-datasets.md` ("Why HBB, not OBB"). The oriented set is never touched.

An HBB XML looks like this (00001.xml, verbatim):

    <annotation>
        <filename>00001.jpg</filename>
        <source><database>DIOR</database></source>
        <size><width>800</width><height>800</height><depth>3</depth></size>
        <segmented>0</segmented>
        <object>
            <name>golffield</name>
            <pose>Unspecified</pose>
            <bndbox><xmin>133</xmin><ymin>237</ymin>
                    <xmax>684</xmax><ymax>672</ymax></bndbox>
        </object>
    </annotation>

Note there is no `<difficult>` and no `<truncated>` tag in the HBB set, so
there is nothing to filter on and no "difficult" objects are excluded. (The OBB
set does carry those tags — a difference worth remembering if the project ever
switches representation.)

Coordinate convention
---------------------
Corners are treated as continuous pixel coordinates, not inclusive integer
indices. That makes `to_yolo` and `from_yolo` exact inverses in float
arithmetic, which is what `tests/test_voc_hbb.py` asserts. It also means a box
recorded as xmin=xmax has width 0 and is reported as degenerate rather than
being silently widened to one pixel.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .dior_classes import class_id

__all__ = [
    "Box",
    "ImageAnn",
    "BoxIssue",
    "to_yolo",
    "from_yolo",
    "parse_hbb_xml",
    "format_label_file",
]


@dataclass(frozen=True)
class Box:
    """One annotated object, in absolute pixel corner coordinates."""

    cls: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class BoxIssue:
    """A box that is geometrically suspect.

    Issues are *recorded and reported*, never silently dropped. A converter that
    quietly discards malformed boxes turns a data problem into a mysterious
    accuracy problem three weeks later.
    """

    image_id: str
    index: int
    kind: str  # "out_of_bounds" | "non_positive_area" | "size_mismatch"
    detail: str


@dataclass
class ImageAnn:
    """Every object annotated on one image, plus the declared image size."""

    image_id: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    issues: list[BoxIssue] = field(default_factory=list)


def to_yolo(
    box: Box, img_w: int, img_h: int
) -> tuple[int, float, float, float, float]:
    """Absolute corners -> normalised `class cx cy w h`."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
    cx = (box.xmin + box.xmax) / 2.0 / img_w
    cy = (box.ymin + box.ymax) / 2.0 / img_h
    w = (box.xmax - box.xmin) / img_w
    h = (box.ymax - box.ymin) / img_h
    return box.cls, cx, cy, w, h


def from_yolo(
    cls: int, cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> Box:
    """Normalised `class cx cy w h` -> absolute corners. Inverse of `to_yolo`."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
    half_w = w * img_w / 2.0
    half_h = h * img_h / 2.0
    acx = cx * img_w
    acy = cy * img_h
    return Box(
        cls=cls,
        xmin=acx - half_w,
        ymin=acy - half_h,
        xmax=acx + half_w,
        ymax=acy + half_h,
    )


def _require_int(node: ET.Element | None, tag: str, where: str) -> int:
    if node is None:
        raise ValueError(f"{where}: missing <{tag}>")
    text = node.findtext(tag)
    if text is None or not text.strip():
        raise ValueError(f"{where}: empty <{tag}>")
    return int(float(text.strip()))


def parse_hbb_xml(
    path: str | Path, expected_size: tuple[int, int] | None = None
) -> ImageAnn:
    """Parse one HBB annotation file.

    `expected_size` is the *actual* (width, height) read from the image on disk.
    When given, a disagreement with the XML's `<size>` is recorded as a
    `size_mismatch` issue and the XML value is kept, because the XML is what the
    box coordinates were authored against.
    """
    path = Path(path)
    image_id = path.stem
    root = ET.parse(path).getroot()

    size = root.find("size")
    width = _require_int(size, "width", f"{path.name}")
    height = _require_int(size, "height", f"{path.name}")

    ann = ImageAnn(image_id=image_id, width=width, height=height)

    if expected_size is not None and (width, height) != expected_size:
        ann.issues.append(
            BoxIssue(
                image_id=image_id,
                index=-1,
                kind="size_mismatch",
                detail=f"xml says {width}x{height}, image is "
                f"{expected_size[0]}x{expected_size[1]}",
            )
        )

    for i, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip()
        cls = class_id(name)

        bnd = obj.find("bndbox")
        if bnd is None:
            raise ValueError(f"{path.name}: object {i} ({name}) has no <bndbox>")
        xmin = float(bnd.findtext("xmin"))  # type: ignore[arg-type]
        ymin = float(bnd.findtext("ymin"))  # type: ignore[arg-type]
        xmax = float(bnd.findtext("xmax"))  # type: ignore[arg-type]
        ymax = float(bnd.findtext("ymax"))  # type: ignore[arg-type]

        box = Box(cls=cls, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        if box.width <= 0 or box.height <= 0:
            ann.issues.append(
                BoxIssue(
                    image_id=image_id,
                    index=i,
                    kind="non_positive_area",
                    detail=f"{name} w={box.width:g} h={box.height:g}",
                )
            )
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            ann.issues.append(
                BoxIssue(
                    image_id=image_id,
                    index=i,
                    kind="out_of_bounds",
                    detail=f"{name} ({xmin:g},{ymin:g},{xmax:g},{ymax:g}) "
                    f"vs {width}x{height}",
                )
            )

        ann.boxes.append(box)

    return ann


def format_label_file(ann: ImageAnn, precision: int = 6) -> str:
    """Render an `ImageAnn` as the text of an Ultralytics label `.txt`.

    Boxes are clamped to [0, 1] on write — Ultralytics rejects out-of-range
    coordinates outright. Clamping happens here and only here, after the
    unclamped geometry has already been recorded as an `out_of_bounds` issue, so
    the report still shows what the source data actually contained.

    Boxes with non-positive area are omitted: a zero-area box is not a
    trainable target. They too are already recorded as issues.
    """
    lines: list[str] = []
    for box in ann.boxes:
        if box.width <= 0 or box.height <= 0:
            continue
        cls, cx, cy, w, h = to_yolo(box, ann.width, ann.height)
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append(
            f"{cls} {cx:.{precision}f} {cy:.{precision}f} "
            f"{w:.{precision}f} {h:.{precision}f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")
