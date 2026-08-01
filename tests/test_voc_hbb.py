"""Converter tests. The round-trip is the one that matters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dior_classes import CLASS_TO_ID, DIOR_CLASSES, class_id  # noqa: E402
from src.data.voc_hbb import (  # noqa: E402
    Box,
    ImageAnn,
    format_label_file,
    from_yolo,
    parse_hbb_xml,
    to_yolo,
)

W, H = 800, 800  # every DIOR image is 800x800


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------


def test_known_box_round_trips_unchanged():
    """A known box converted to YOLO and back must return unchanged.

    Uses the golffield box from the real 00001.xml.
    """
    original = Box(cls=class_id("golffield"), xmin=133, ymin=237, xmax=684, ymax=672)

    cls, cx, cy, w, h = to_yolo(original, W, H)
    recovered = from_yolo(cls, cx, cy, w, h, W, H)

    assert recovered.cls == original.cls
    assert recovered.xmin == pytest.approx(original.xmin)
    assert recovered.ymin == pytest.approx(original.ymin)
    assert recovered.xmax == pytest.approx(original.xmax)
    assert recovered.ymax == pytest.approx(original.ymax)


def test_known_box_normalised_values_are_exact():
    """Guards the normalisation itself, not just its invertibility.

    A converter that divided x by height and y by width would still round-trip
    perfectly on a square image. DIOR is all-square, so the round trip alone
    cannot catch a transposed axis — these literals can.
    """
    box = Box(cls=0, xmin=100, ymin=200, xmax=300, ymax=600)
    cls, cx, cy, w, h = to_yolo(box, 1000, 2000)
    assert (cls, cx, cy, w, h) == (0, 0.2, 0.2, 0.2, 0.2)


def test_round_trip_on_non_square_image_catches_axis_swap():
    box = Box(cls=3, xmin=10, ymin=20, xmax=110, ymax=220)
    cls, cx, cy, w, h = to_yolo(box, 640, 480)
    back = from_yolo(cls, cx, cy, w, h, 640, 480)
    assert back.xmin == pytest.approx(10)
    assert back.ymin == pytest.approx(20)
    assert back.xmax == pytest.approx(110)
    assert back.ymax == pytest.approx(220)


@pytest.mark.parametrize(
    "box",
    [
        Box(0, 0, 0, W, H),  # full image
        Box(1, 0, 0, 1, 1),  # single pixel at the origin
        Box(2, W - 1, H - 1, W, H),  # single pixel at the far corner
        Box(3, 399.5, 399.5, 400.5, 400.5),  # sub-pixel, centred
        Box(4, 12.25, 7.75, 33.125, 91.0625),  # awkward fractions
    ],
)
def test_round_trip_edge_cases(box):
    cls, cx, cy, w, h = to_yolo(box, W, H)
    back = from_yolo(cls, cx, cy, w, h, W, H)
    assert back.xmin == pytest.approx(box.xmin)
    assert back.ymin == pytest.approx(box.ymin)
    assert back.xmax == pytest.approx(box.xmax)
    assert back.ymax == pytest.approx(box.ymax)


def test_round_trip_every_class_id():
    for name in DIOR_CLASSES:
        box = Box(cls=class_id(name), xmin=10, ymin=10, xmax=110, ymax=210)
        cls, cx, cy, w, h = to_yolo(box, W, H)
        assert from_yolo(cls, cx, cy, w, h, W, H).cls == CLASS_TO_ID[name]


def test_zero_image_size_rejected():
    with pytest.raises(ValueError):
        to_yolo(Box(0, 0, 0, 10, 10), 0, 800)
    with pytest.raises(ValueError):
        from_yolo(0, 0.5, 0.5, 0.1, 0.1, 800, 0)


# --------------------------------------------------------------------------
# label file rendering
# --------------------------------------------------------------------------


def test_label_file_clamps_out_of_bounds_but_keeps_the_box():
    ann = ImageAnn(image_id="x", width=W, height=H)
    ann.boxes.append(Box(cls=0, xmin=-50, ymin=-50, xmax=100, ymax=100))
    text = format_label_file(ann)
    values = [float(v) for v in text.split()[1:]]
    assert len(text.splitlines()) == 1
    assert all(0.0 <= v <= 1.0 for v in values)


def test_label_file_omits_zero_area_boxes():
    ann = ImageAnn(image_id="x", width=W, height=H)
    ann.boxes.append(Box(cls=0, xmin=10, ymin=10, xmax=10, ymax=100))  # zero width
    ann.boxes.append(Box(cls=1, xmin=10, ymin=10, xmax=100, ymax=100))  # fine
    assert len(format_label_file(ann).splitlines()) == 1


def test_empty_annotation_yields_empty_file():
    assert format_label_file(ImageAnn(image_id="x", width=W, height=H)) == ""


def test_label_file_survives_a_reparse():
    """label -> text -> parse -> pixels must match the source geometry."""
    ann = ImageAnn(image_id="x", width=W, height=H)
    ann.boxes.append(Box(cls=13, xmin=133, ymin=237, xmax=684, ymax=672))
    line = format_label_file(ann).strip().split()
    back = from_yolo(int(line[0]), *(float(v) for v in line[1:5]), W, H)
    assert back.cls == 13
    assert back.xmin == pytest.approx(133, abs=1e-2)
    assert back.ymax == pytest.approx(672, abs=1e-2)


# --------------------------------------------------------------------------
# xml parsing
# --------------------------------------------------------------------------

SAMPLE_XML = """<annotation>
    <filename>00001.jpg</filename>
    <source><database>DIOR</database></source>
    <size><width>800</width><height>800</height><depth>3</depth></size>
    <segmented>0</segmented>
    <object>
        <name>golffield</name>
        <pose>Unspecified</pose>
        <bndbox><xmin>133</xmin><ymin>237</ymin><xmax>684</xmax><ymax>672</ymax></bndbox>
    </object>
</annotation>
"""


def test_parse_hbb_xml(tmp_path: Path):
    xml = tmp_path / "00001.xml"
    xml.write_text(SAMPLE_XML, encoding="utf-8")

    ann = parse_hbb_xml(xml)
    assert ann.image_id == "00001"
    assert (ann.width, ann.height) == (800, 800)
    assert len(ann.boxes) == 1
    assert ann.boxes[0].cls == class_id("golffield")
    assert (ann.boxes[0].xmin, ann.boxes[0].ymax) == (133.0, 672.0)
    assert ann.issues == []


def test_parse_records_size_mismatch(tmp_path: Path):
    xml = tmp_path / "00001.xml"
    xml.write_text(SAMPLE_XML, encoding="utf-8")
    ann = parse_hbb_xml(xml, expected_size=(1024, 1024))
    assert [i.kind for i in ann.issues] == ["size_mismatch"]
    assert (ann.width, ann.height) == (800, 800)  # xml wins; boxes match it


def test_parse_records_out_of_bounds(tmp_path: Path):
    xml = tmp_path / "x.xml"
    xml.write_text(
        SAMPLE_XML.replace("<xmax>684</xmax>", "<xmax>900</xmax>"), encoding="utf-8"
    )
    ann = parse_hbb_xml(xml)
    assert "out_of_bounds" in {i.kind for i in ann.issues}
    assert len(ann.boxes) == 1  # recorded, not dropped


def test_parse_records_non_positive_area(tmp_path: Path):
    xml = tmp_path / "x.xml"
    xml.write_text(
        SAMPLE_XML.replace("<xmin>133</xmin>", "<xmin>684</xmin>"), encoding="utf-8"
    )
    ann = parse_hbb_xml(xml)
    assert "non_positive_area" in {i.kind for i in ann.issues}


def test_unknown_class_raises(tmp_path: Path):
    xml = tmp_path / "x.xml"
    xml.write_text(SAMPLE_XML.replace("golffield", "helipad"), encoding="utf-8")
    with pytest.raises(KeyError):
        parse_hbb_xml(xml)


# --------------------------------------------------------------------------
# class list
# --------------------------------------------------------------------------


def test_class_list_is_stable():
    """The order IS the label encoding. Changing it invalidates every label."""
    assert len(DIOR_CLASSES) == 20
    assert DIOR_CLASSES[0] == "airplane"
    assert DIOR_CLASSES[-1] == "windmill"
    assert DIOR_CLASSES[13] == "ship"
    assert len(set(DIOR_CLASSES)) == 20


def test_class_id_is_case_insensitive():
    assert class_id("Ship") == class_id("ship") == CLASS_TO_ID["ship"]
