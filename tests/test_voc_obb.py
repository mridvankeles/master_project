"""Oriented-box converter tests.

The round trip is necessary but far from sufficient here: an OBB converter can
round-trip perfectly while having the corner order wrong, because the same eight
numbers come back regardless of what they mean. So these tests also pin the
corner ORDER against NIRNet's loader, and check that the polygon is convex and
non-self-intersecting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dior_classes import class_id  # noqa: E402
from src.data.voc_obb import (  # noqa: E402
    Polygon,
    clip_polygon,
    format_obb_label_file,
    from_yolo_obb,
    parse_obb_xml,
    to_yolo_obb,
)

W, H = 800, 800

# The real 00001.xml golffield object.
SAMPLE_XML = """<annotation>
    <folder>x</folder>
    <filename>00001.jpg</filename>
    <source><database>Unknown</database></source>
    <size><width>800</width><height>800</height><depth>3</depth></size>
    <segmented>0</segmented>
    <object>
        <type>robndbox</type>
        <name>golffield</name>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <robndbox>
            <x_left_top>133</x_left_top><y_left_top>237</y_left_top>
            <x_right_top>684</x_right_top><y_right_top>237</y_right_top>
            <x_right_bottom>684</x_right_bottom><y_right_bottom>672</y_right_bottom>
            <x_left_bottom>133</x_left_bottom><y_left_bottom>672</y_left_bottom>
        </robndbox>
        <angle>0</angle>
    </object>
</annotation>
"""

ROTATED_XML = SAMPLE_XML.replace(
    """<x_left_top>133</x_left_top><y_left_top>237</y_left_top>
            <x_right_top>684</x_right_top><y_right_top>237</y_right_top>
            <x_right_bottom>684</x_right_bottom><y_right_bottom>672</y_right_bottom>
            <x_left_bottom>133</x_left_bottom><y_left_bottom>672</y_left_bottom>""",
    """<x_left_top>400</x_left_top><y_left_top>100</y_left_top>
            <x_right_top>700</x_right_top><y_right_top>400</y_right_top>
            <x_right_bottom>400</x_right_bottom><y_right_bottom>700</y_right_bottom>
            <x_left_bottom>100</x_left_bottom><y_left_bottom>400</y_left_bottom>""",
)


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_known_polygon_round_trips_unchanged():
    original = Polygon(cls=class_id("golffield"), coords=(133, 237, 684, 237, 684, 672, 133, 672))
    cls, *norm = to_yolo_obb(original, W, H)
    back = from_yolo_obb(cls, tuple(norm), W, H)
    assert back.cls == original.cls
    for a, b in zip(back.coords, original.coords):
        assert a == pytest.approx(b)


def test_rotated_polygon_round_trips_unchanged():
    """A genuinely rotated diamond — the axis-aligned case cannot catch a swap."""
    original = Polygon(cls=3, coords=(400, 100, 700, 400, 400, 700, 100, 400))
    cls, *norm = to_yolo_obb(original, W, H)
    back = from_yolo_obb(cls, tuple(norm), W, H)
    for a, b in zip(back.coords, original.coords):
        assert a == pytest.approx(b)


def test_round_trip_on_non_square_image():
    """DIOR is all-square, so only a non-square case can catch an x/y swap."""
    original = Polygon(cls=1, coords=(10, 20, 110, 20, 110, 220, 10, 220))
    cls, *norm = to_yolo_obb(original, 640, 480)
    back = from_yolo_obb(cls, tuple(norm), 640, 480)
    for a, b in zip(back.coords, original.coords):
        assert a == pytest.approx(b)


def test_normalised_values_are_exact():
    poly = Polygon(cls=0, coords=(100, 200, 300, 200, 300, 600, 100, 600))
    cls, *norm = to_yolo_obb(poly, 1000, 2000)
    assert cls == 0
    assert norm == pytest.approx([0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3])


def test_zero_image_size_rejected():
    poly = Polygon(cls=0, coords=(0, 0, 1, 0, 1, 1, 0, 1))
    with pytest.raises(ValueError):
        to_yolo_obb(poly, 0, 800)
    with pytest.raises(ValueError):
        from_yolo_obb(0, (0, 0, 1, 0, 1, 1, 0, 1), 800, 0)


def test_polygon_rejects_wrong_length():
    with pytest.raises(ValueError):
        Polygon(cls=0, coords=(1, 2, 3, 4))


# --------------------------------------------------------------------------
# corner order — the bug the round trip cannot see
# --------------------------------------------------------------------------


def test_corner_order_matches_nirnet(tmp_path: Path):
    """left_top, right_top, right_bottom, left_bottom.

    Pinned against nirnet-main/mmrotate/datasets/dior.py lines 111-120. A
    transposed pair yields a bow-tie polygon: the eight numbers still round-trip,
    the model still trains, and mAP is quietly wrong.
    """
    xml = tmp_path / "00001.xml"
    xml.write_text(SAMPLE_XML, encoding="utf-8")
    ann = parse_obb_xml(xml)
    assert len(ann.polygons) == 1
    assert ann.polygons[0].coords == (133.0, 237.0, 684.0, 237.0, 684.0, 672.0, 133.0, 672.0)


def test_shoelace_area_is_correct_and_not_bowtie(tmp_path: Path):
    """The rectangle's area must be exactly w*h. A bow-tie gives less."""
    xml = tmp_path / "00001.xml"
    xml.write_text(SAMPLE_XML, encoding="utf-8")
    poly = parse_obb_xml(xml).polygons[0]
    assert poly.area == pytest.approx((684 - 133) * (672 - 237))


def test_rotated_area_is_correct(tmp_path: Path):
    """The diamond has diagonals of 600 and 600, so area = d1*d2/2 = 180000."""
    xml = tmp_path / "r.xml"
    xml.write_text(ROTATED_XML, encoding="utf-8")
    poly = parse_obb_xml(xml).polygons[0]
    assert poly.area == pytest.approx(180000.0)


def test_area_is_winding_independent():
    clockwise = Polygon(cls=0, coords=(0, 0, 10, 0, 10, 5, 0, 5))
    counter = Polygon(cls=0, coords=(0, 0, 0, 5, 10, 5, 10, 0))
    assert clockwise.area == pytest.approx(counter.area) == pytest.approx(50.0)


# --------------------------------------------------------------------------
# parsing behaviour
# --------------------------------------------------------------------------


def test_difficult_objects_are_kept(tmp_path: Path):
    """NIRNet's loader never reads <difficult>; filtering would change the
    evaluation population relative to the numbers we compare against."""
    xml = tmp_path / "d.xml"
    xml.write_text(
        SAMPLE_XML.replace("<difficult>0</difficult>", "<difficult>1</difficult>"),
        encoding="utf-8",
    )
    assert len(parse_obb_xml(xml).polygons) == 1


def test_missing_robndbox_falls_back_to_bndbox(tmp_path: Path):
    xml = tmp_path / "f.xml"
    xml.write_text(
        """<annotation><size><width>800</width><height>800</height></size>
        <object><name>ship</name>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>220</ymax></bndbox>
        </object></annotation>""",
        encoding="utf-8",
    )
    ann = parse_obb_xml(xml)
    assert ann.polygons[0].coords == (10.0, 20.0, 110.0, 20.0, 110.0, 220.0, 10.0, 220.0)
    assert "hbb_fallback" in {i.kind for i in ann.issues}


def test_out_of_bounds_recorded_not_dropped(tmp_path: Path):
    xml = tmp_path / "o.xml"
    xml.write_text(SAMPLE_XML.replace("<x_right_top>684</x_right_top>", "<x_right_top>900</x_right_top>"), encoding="utf-8")
    ann = parse_obb_xml(xml)
    assert "out_of_bounds" in {i.kind for i in ann.issues}
    assert len(ann.polygons) == 1


def test_size_mismatch_recorded(tmp_path: Path):
    xml = tmp_path / "s.xml"
    xml.write_text(SAMPLE_XML, encoding="utf-8")
    ann = parse_obb_xml(xml, expected_size=(800, 787))
    assert "size_mismatch" in {i.kind for i in ann.issues}


# --------------------------------------------------------------------------
# label file
# --------------------------------------------------------------------------


def test_label_file_has_nine_fields(tmp_path: Path):
    xml = tmp_path / "00001.xml"
    xml.write_text(SAMPLE_XML, encoding="utf-8")
    line = format_obb_label_file(parse_obb_xml(xml)).strip().split()
    assert len(line) == 9  # class + 8 coordinates
    assert int(line[0]) == class_id("golffield")
    assert all(0.0 <= float(v) <= 1.0 for v in line[1:])


def test_label_file_survives_reparse(tmp_path: Path):
    xml = tmp_path / "r.xml"
    xml.write_text(ROTATED_XML, encoding="utf-8")
    ann = parse_obb_xml(xml)
    line = format_obb_label_file(ann).strip().split()
    back = from_yolo_obb(int(line[0]), tuple(float(v) for v in line[1:9]), W, H)
    for a, b in zip(back.coords, ann.polygons[0].coords):
        assert a == pytest.approx(b, abs=1e-2)


def test_zero_area_polygon_omitted():
    from src.data.voc_obb import ImageAnnOBB

    ann = ImageAnnOBB(image_id="x", width=W, height=H)
    ann.polygons.append(Polygon(cls=0, coords=(10, 10, 10, 10, 10, 10, 10, 10)))
    ann.polygons.append(Polygon(cls=1, coords=(10, 10, 110, 10, 110, 110, 10, 110)))
    assert len(format_obb_label_file(ann).splitlines()) == 1


# --------------------------------------------------------------------------
# out-of-bounds clipping
# --------------------------------------------------------------------------


def test_inside_polygon_survives_clipping_unchanged():
    p = Polygon(cls=0, coords=(100, 100, 200, 100, 200, 200, 100, 200))
    out, _ = clip_polygon(p, W, H)
    assert out.area == pytest.approx(p.area, rel=1e-3)


def test_axis_aligned_overflow_is_cropped_not_sheared():
    """100x100 box hanging 50px off the right edge of a 150-wide image."""
    p = Polygon(cls=0, coords=(100, 10, 200, 10, 200, 110, 100, 110))
    out, _ = clip_polygon(p, 150, H)
    assert max(out.xs) == pytest.approx(150, abs=1e-3)
    assert min(out.xs) == pytest.approx(100, abs=1e-3)
    assert out.area == pytest.approx(50 * 100, rel=1e-3)


def test_rotated_overflow_stays_in_bounds_and_stays_a_rectangle():
    """A 45-degree diamond hanging off the top edge.

    This is the case that caught the minAreaRect bug: the minimum *enclosing*
    rectangle of the clipped shape is the original diamond, which is still out
    of bounds, so the axis-aligned fallback has to take over.
    """
    import math

    p = Polygon(cls=0, coords=(400, -100, 600, 100, 400, 300, 200, 100))
    out, method = clip_polygon(p, W, H)
    assert out is not None
    assert min(out.ys) >= -1e-3
    assert method == "axis_aligned"

    def side(i, j):
        return math.hypot(out.xs[i] - out.xs[j], out.ys[i] - out.ys[j])

    assert side(0, 1) == pytest.approx(side(2, 3), rel=1e-3)
    assert side(1, 2) == pytest.approx(side(3, 0), rel=1e-3)


def test_fully_outside_polygon_is_dropped():
    p = Polygon(cls=0, coords=(900, 900, 1000, 900, 1000, 1000, 900, 1000))
    poly, method = clip_polygon(p, W, H)
    assert poly is None and method == "dropped"


def test_label_file_never_emits_out_of_range_coordinates():
    """Ultralytics asserts points.max() <= 1.01 and min >= -0.01; a violation
    drops the whole image from the dataset."""
    from src.data.voc_obb import ImageAnnOBB

    ann = ImageAnnOBB(image_id="x", width=W, height=H)
    ann.polygons.append(Polygon(cls=0, coords=(-55, -96, 723, -96, 723, 740, -55, 740)))
    ann.polygons.append(Polygon(cls=1, coords=(400, -100, 600, 100, 400, 300, 200, 100)))
    lines = format_obb_label_file(ann).splitlines()
    assert len(lines) == 2
    for line in lines:
        vals = [float(v) for v in line.split()[1:]]
        assert all(-0.01 <= v <= 1.01 for v in vals), vals


def test_fully_outside_polygon_omitted_from_label_file():
    from src.data.voc_obb import ImageAnnOBB

    ann = ImageAnnOBB(image_id="x", width=W, height=H)
    ann.polygons.append(Polygon(cls=0, coords=(900, 900, 1000, 900, 1000, 1000, 900, 1000)))
    assert format_obb_label_file(ann) == ""
