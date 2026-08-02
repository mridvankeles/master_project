"""PASCAL-VOC oriented-bounding-box XML -> Ultralytics `obb` labels.

Reads `Annotations/Oriented Bounding Boxes/`. Companion to `voc_hbb.py`; the two
arms share nothing but the class list, deliberately, so a change to one cannot
silently alter the other.

An OBB annotation object looks like this (00001.xml, verbatim):

    <object>
        <type>robndbox</type>
        <name>golffield</name>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <robndbox>
            <x_left_top>133</x_left_top>     <y_left_top>237</y_left_top>
            <x_right_top>684</x_right_top>   <y_right_top>237</y_right_top>
            <x_right_bottom>684</x_right_bottom><y_right_bottom>672</y_right_bottom>
            <x_left_bottom>133</x_left_bottom><y_left_bottom>672</y_left_bottom>
        </robndbox>
        <angle>0</angle>
    </object>

CORNER ORDER, AND WHY IT IS COPIED EXACTLY
------------------------------------------
left_top -> right_top -> right_bottom -> left_bottom. That is the order NIRNet's
own loader uses (`nirnet-main/mmrotate/datasets/dior.py`, lines 111-120), and
corner ordering is the classic silent OBB bug: a transposed pair produces a
bow-tie polygon whose area is wrong but which still trains, still converges, and
quietly reports a lower mAP. The order here matches theirs so our numbers are
built on the same geometry. `scripts/render_verification.py --task obb` draws the
polygons so this is checked by eye and not just asserted.

DIFFICULT / TRUNCATED ARE NOT FILTERED
--------------------------------------
The tags exist in the OBB set (they do not in the HBB set), but NIRNet's loader
never reads them — every object is kept. We match that, because filtering them
would silently change the evaluation population relative to the numbers we want
to compare against.

WHY POLYGONS AND NOT (cx, cy, w, h, angle)
------------------------------------------
Ultralytics' `obb` task takes 8 normalised polygon coordinates. That sidesteps
the angle-convention problem entirely — no `le135`/`oc`/`le90` conversion, no
period ambiguity, no corner reordering to satisfy a convention. `03-datasets.md`
listed that conversion as one of the things HBB was chosen to avoid; using
polygons avoids it while still being oriented.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .dior_classes import class_id
from .voc_hbb import BoxIssue

__all__ = ["Polygon", "ImageAnnOBB", "to_yolo_obb", "from_yolo_obb", "parse_obb_xml", "format_obb_label_file"]

CORNER_TAGS = (
    ("x_left_top", "y_left_top"),
    ("x_right_top", "y_right_top"),
    ("x_right_bottom", "y_right_bottom"),
    ("x_left_bottom", "y_left_bottom"),
)


@dataclass(frozen=True)
class Polygon:
    """One oriented object: 4 corners in absolute pixel coordinates."""

    cls: int
    coords: tuple[float, ...]  # x1,y1,x2,y2,x3,y3,x4,y4

    def __post_init__(self) -> None:
        if len(self.coords) != 8:
            raise ValueError(f"a polygon needs 8 coordinates, got {len(self.coords)}")

    @property
    def xs(self) -> tuple[float, ...]:
        return self.coords[0::2]

    @property
    def ys(self) -> tuple[float, ...]:
        return self.coords[1::2]

    @property
    def area(self) -> float:
        """Shoelace area. Sign-independent, so corner winding does not matter."""
        xs, ys = self.xs, self.ys
        total = 0.0
        for i in range(4):
            j = (i + 1) % 4
            total += xs[i] * ys[j] - xs[j] * ys[i]
        return abs(total) / 2.0


@dataclass
class ImageAnnOBB:
    image_id: str
    width: int
    height: int
    polygons: list[Polygon] = field(default_factory=list)
    issues: list[BoxIssue] = field(default_factory=list)


def to_yolo_obb(poly: Polygon, img_w: int, img_h: int) -> tuple[int, ...]:
    """Absolute corners -> (class, x1,y1,...,x4,y4) normalised to [0,1]."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
    out: list[float] = []
    for i in range(4):
        out.append(poly.coords[2 * i] / img_w)
        out.append(poly.coords[2 * i + 1] / img_h)
    return (poly.cls, *out)


def from_yolo_obb(cls: int, coords: tuple[float, ...], img_w: int, img_h: int) -> Polygon:
    """Normalised polygon -> absolute corners. Exact inverse of `to_yolo_obb`."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
    if len(coords) != 8:
        raise ValueError(f"expected 8 coordinates, got {len(coords)}")
    out: list[float] = []
    for i in range(4):
        out.append(coords[2 * i] * img_w)
        out.append(coords[2 * i + 1] * img_h)
    return Polygon(cls=cls, coords=tuple(out))


def _require_int(node: ET.Element | None, tag: str, where: str) -> int:
    if node is None:
        raise ValueError(f"{where}: missing <{tag}>")
    text = node.findtext(tag)
    if text is None or not text.strip():
        raise ValueError(f"{where}: empty <{tag}>")
    return int(float(text.strip()))


def parse_obb_xml(
    path: str | Path, expected_size: tuple[int, int] | None = None
) -> ImageAnnOBB:
    """Parse one oriented annotation file."""
    path = Path(path)
    image_id = path.stem
    root = ET.parse(path).getroot()

    size = root.find("size")
    width = _require_int(size, "width", path.name)
    height = _require_int(size, "height", path.name)

    ann = ImageAnnOBB(image_id=image_id, width=width, height=height)

    if expected_size is not None and (width, height) != expected_size:
        ann.issues.append(
            BoxIssue(
                image_id=image_id, index=-1, kind="size_mismatch",
                detail=f"xml says {width}x{height}, image is "
                       f"{expected_size[0]}x{expected_size[1]}",
            )
        )

    for i, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip()
        cls = class_id(name)

        rb = obj.find("robndbox")
        if rb is None:
            # A handful of DIOR-R objects carry only an axis-aligned <bndbox>.
            # Promote it to a degenerate rectangle rather than dropping the
            # object, and record that we did.
            bnd = obj.find("bndbox")
            if bnd is None:
                raise ValueError(f"{path.name}: object {i} ({name}) has neither box")
            x1 = float(bnd.findtext("xmin"))  # type: ignore[arg-type]
            y1 = float(bnd.findtext("ymin"))  # type: ignore[arg-type]
            x2 = float(bnd.findtext("xmax"))  # type: ignore[arg-type]
            y2 = float(bnd.findtext("ymax"))  # type: ignore[arg-type]
            coords = (x1, y1, x2, y1, x2, y2, x1, y2)
            ann.issues.append(
                BoxIssue(
                    image_id=image_id, index=i, kind="hbb_fallback",
                    detail=f"{name} had no <robndbox>; used axis-aligned <bndbox>",
                )
            )
        else:
            coords = tuple(
                float(rb.findtext(tag))  # type: ignore[arg-type]
                for pair in CORNER_TAGS
                for tag in pair
            )

        poly = Polygon(cls=cls, coords=coords)

        if poly.area <= 0:
            ann.issues.append(
                BoxIssue(
                    image_id=image_id, index=i, kind="non_positive_area",
                    detail=f"{name} area={poly.area:g}",
                )
            )
        if min(poly.xs) < 0 or min(poly.ys) < 0 or max(poly.xs) > width or max(poly.ys) > height:
            ann.issues.append(
                BoxIssue(
                    image_id=image_id, index=i, kind="out_of_bounds",
                    detail=f"{name} x[{min(poly.xs):g},{max(poly.xs):g}] "
                           f"y[{min(poly.ys):g},{max(poly.ys):g}] vs {width}x{height}",
                )
            )

        ann.polygons.append(poly)

    return ann


def _clip_to_rect(
    pts: list[tuple[float, float]], w: float, h: float
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a convex polygon against [0,w] x [0,h]."""
    def clip_edge(poly, inside, intersect):
        out: list[tuple[float, float]] = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i - 1]
            cur_in, prev_in = inside(cur), inside(prev)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
        return out

    def lerp(p, q, t):
        return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)

    poly = list(pts)
    edges = (
        (lambda p: p[0] >= 0.0, lambda p, q: lerp(p, q, (0.0 - p[0]) / (q[0] - p[0]))),
        (lambda p: p[0] <= w, lambda p, q: lerp(p, q, (w - p[0]) / (q[0] - p[0]))),
        (lambda p: p[1] >= 0.0, lambda p, q: lerp(p, q, (0.0 - p[1]) / (q[1] - p[1]))),
        (lambda p: p[1] <= h, lambda p, q: lerp(p, q, (h - p[1]) / (q[1] - p[1]))),
    )
    for inside, intersect in edges:
        if not poly:
            return []
        poly = clip_edge(poly, inside, intersect)
    return poly


def clip_polygon(poly: Polygon, width: int, height: int) -> tuple[Polygon | None, str]:
    """Bring an out-of-bounds oriented box back inside the image, keeping its angle.

    WHY NOT JUST CLAMP THE CORNERS
    ------------------------------
    Ultralytics rejects labels with coordinates outside [0, 1] outright
    (`data/utils.py`: `assert points.max() <= 1.01`), and a rejected label drops
    the whole image from the dataset. So something must be done — 301 objects in
    the aligned subset are out of bounds, mostly large partially-visible things
    (stadium, airport, groundtrackfield) and vehicles clipped by the frame edge.

    Clamping each corner independently is the obvious fix and it is wrong: for a
    rotated rectangle it shears the shape into a non-rectangle, changing both the
    angle and the area. Instead we clip the polygon against the image rectangle
    properly (Sutherland-Hodgman) and re-fit a minimum-area rotated rectangle to
    what survives. That preserves orientation and yields the tightest oriented
    box around the visible part of the object.

    PROTOCOL DIFFERENCE, RECORD IT
    ------------------------------
    NIRNet's loader keeps the raw out-of-bounds coordinates and lets MMRotate
    handle them. We cannot, so our labels differ from theirs for these 301
    objects (~1.4% of the aligned subset). Noted in docs/comparison-baselines.md.

    THE SUBTLETY THAT BIT US
    -----------------------
    `minAreaRect` returns the minimum *enclosing* rectangle, and its corners can
    lie outside the convex hull of its input — so a rotated refit of an
    in-bounds clipped polygon can still poke outside the image. (A 45-degree
    diamond clipped at y=0 refits to exactly the original diamond.) When that
    happens we fall back to the axis-aligned bounding box of the clipped
    polygon, which is in-bounds by construction. Orientation is lost for those
    objects, which is worse than keeping it and better than a sheared box or a
    dropped image.

    Returns (polygon, method) where method is "unchanged", "rotated",
    "axis_aligned", or "dropped" (polygon None) when nothing survives.
    """
    import cv2
    import numpy as np

    pts = [(poly.coords[2 * i], poly.coords[2 * i + 1]) for i in range(4)]
    clipped = _clip_to_rect(pts, float(width), float(height))
    if len(clipped) < 3:
        return None, "dropped"

    arr = np.array(clipped, dtype=np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(arr)).reshape(-1)
    fitted = Polygon(cls=poly.cls, coords=tuple(float(v) for v in box))

    tol = 1e-3
    if (
        min(fitted.xs) >= -tol and min(fitted.ys) >= -tol
        and max(fitted.xs) <= width + tol and max(fitted.ys) <= height + tol
    ):
        return fitted, "rotated"

    x0, y0 = float(arr[:, 0].min()), float(arr[:, 1].min())
    x1, y1 = float(arr[:, 0].max()), float(arr[:, 1].max())
    return Polygon(cls=poly.cls, coords=(x0, y0, x1, y0, x1, y1, x0, y1)), "axis_aligned"


def format_obb_label_file(ann: ImageAnnOBB, precision: int = 6) -> str:
    """Render an `ImageAnnOBB` as an Ultralytics `obb` label `.txt`.

    Out-of-bounds polygons are clipped by `clip_polygon` (which preserves the
    angle) rather than corner-clamped. Zero-area polygons are omitted: they are
    not trainable targets. Both cases have already been recorded as issues by
    `parse_obb_xml`, so the report still reflects the source data.
    """
    lines: list[str] = []
    for poly in ann.polygons:
        if poly.area <= 0:
            continue

        if (
            min(poly.xs) < 0 or min(poly.ys) < 0
            or max(poly.xs) > ann.width or max(poly.ys) > ann.height
        ):
            fitted, method = clip_polygon(poly, ann.width, ann.height)
            ann.issues.append(
                BoxIssue(
                    image_id=ann.image_id, index=-1, kind=f"clipped_{method}",
                    detail=f"class {poly.cls} refitted via {method}",
                )
            )
            if fitted is None or fitted.area <= 0:
                continue
            poly = fitted

        cls, *coords = to_yolo_obb(poly, ann.width, ann.height)
        # Belt and braces: the refit is in-bounds by construction, but floating
        # point can land a hair outside and Ultralytics' assert is unforgiving.
        clamped = [min(max(c, 0.0), 1.0) for c in coords]
        body = " ".join(f"{c:.{precision}f}" for c in clamped)
        lines.append(f"{cls} {body}")
    return "\n".join(lines) + ("\n" if lines else "")
