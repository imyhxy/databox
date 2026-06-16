import math
import xml.etree.ElementTree as ET

import pytest
import yaml
from databox.segmentation.mmseg_dataset_card import (
    analyze_dataset,
    compute_ocnet_weights,
    parse_labelmap,
    write_dataset_card,
)
from PIL import Image


def _write_labelmap(path):
    path.write_text(
        """# label:color_rgb:parts:actions
background:0,0,0::
object:255,0,0::
empty:0,255,0::
"""
    )


def _save_mask(path, values):
    image = Image.new("P", (len(values[0]), len(values)))
    image.putdata([value for row in values for value in row])
    palette = [0] * 768
    palette[:9] = [0, 0, 0, 255, 0, 0, 0, 255, 0]
    palette[255 * 3 : 255 * 3 + 3] = [128, 128, 128]
    image.putpalette(palette)
    image.save(path)


def _make_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    _write_labelmap(root / "labelmap.txt")
    Image.new("RGB", (2, 2)).save(root / "images" / "one.jpg")
    Image.new("RGB", (2, 2)).save(root / "images" / "two.jpg")
    _save_mask(root / "annotations" / "one.png", [[0, 1], [1, 255]])
    _save_mask(root / "annotations" / "two.png", [[0, 0], [1, 7]])
    (root / "train.txt").write_text("one\n")
    (root / "val.txt").write_text("two\nmissing\n")
    return root


def _make_voc_dataset(tmp_path):
    root = tmp_path / "voc_dataset"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "SegmentationClass").mkdir()
    (root / "ImageSets" / "Segmentation").mkdir(parents=True)
    _write_labelmap(root / "labelmap.txt")
    Image.new("RGB", (2, 2)).save(root / "JPEGImages" / "one.jpg")
    Image.new("RGB", (2, 2)).save(root / "JPEGImages" / "two.png")
    _save_mask(root / "SegmentationClass" / "one.png", [[0, 1], [1, 255]])
    _save_mask(root / "SegmentationClass" / "two.png", [[0, 0], [1, 7]])
    (root / "ImageSets" / "Segmentation" / "train.txt").write_text("one\n")
    (root / "ImageSets" / "Segmentation" / "val.txt").write_text("two\nmissing\n")
    (root / "ImageSets" / "Segmentation" / "trainval.txt").write_text("one\ntwo\n")
    return root


def test_parse_labelmap_reads_ordered_classes(tmp_path):
    labelmap = tmp_path / "labelmap.txt"
    _write_labelmap(labelmap)

    classes = parse_labelmap(labelmap)

    assert [item.name for item in classes] == ["background", "object", "empty"]
    assert classes[1].id == 1
    assert classes[1].color_rgb == (255, 0, 0)


def test_analyze_dataset_counts_pixels_images_ignore_and_warnings(tmp_path):
    root = _make_dataset(tmp_path)

    data = analyze_dataset(root)

    rows = {row["name"]: row for row in data["classes"]}
    assert rows["background"]["pixel_count"] == 3
    assert rows["background"]["image_count"] == 2
    assert rows["object"]["pixel_count"] == 3
    assert rows["object"]["image_count"] == 2
    assert rows["empty"]["pixel_count"] == 0
    assert rows["empty"]["class_weight_ocnet"] is None
    assert data["class_weights_ocnet"] == [
        (
            round(row["class_weight_ocnet"], 4)
            if row["class_weight_ocnet"] is not None
            else 0.0
        )
        for row in data["classes"]
    ]
    assert data["ignore_index"]["pixel_count"] == 1
    assert data["ignore_index"]["image_count"] == 1
    assert data["total_pixels"] == 8
    assert data["splits"] == {"train": 1, "val": 2}
    assert data["split_stats"]["train"]["stem_count"] == 1
    assert data["split_stats"]["train"]["mask_count"] == 1
    assert data["split_stats"]["train"]["total_pixels"] == 4
    train_rows = {row["name"]: row for row in data["split_stats"]["train"]["classes"]}
    assert train_rows["background"]["pixel_count"] == 1
    assert train_rows["object"]["pixel_count"] == 2
    assert data["split_stats"]["train"]["class_weights_ocnet"] == [
        (
            round(row["class_weight_ocnet"], 4)
            if row["class_weight_ocnet"] is not None
            else 0.0
        )
        for row in data["split_stats"]["train"]["classes"]
    ]
    assert data["split_stats"]["val"]["stem_count"] == 2
    assert data["split_stats"]["val"]["mask_count"] == 1
    val_rows = {row["name"]: row for row in data["split_stats"]["val"]["classes"]}
    assert val_rows["background"]["pixel_count"] == 2
    assert val_rows["object"]["pixel_count"] == 1
    assert any("Unknown mask values" in item for item in data["validation_warnings"])
    assert any(
        "val.txt contains 1 stems" in item for item in data["validation_warnings"]
    )
    assert any(
        "Classes with zero pixels: empty" == item
        for item in data["validation_warnings"]
    )


def test_analyze_dataset_auto_detects_voc_layout(tmp_path):
    root = _make_voc_dataset(tmp_path)

    data = analyze_dataset(root)

    rows = {row["name"]: row for row in data["classes"]}
    assert data["layout"] == "voc"
    assert data["image_count"] == 2
    assert data["mask_count"] == 2
    assert data["splits"] == {"train": 1, "val": 2, "trainval": 2}
    assert data["split_stats"]["trainval"]["stem_count"] == 2
    assert data["split_stats"]["trainval"]["mask_count"] == 2
    assert rows["background"]["pixel_count"] == 3
    assert rows["object"]["pixel_count"] == 3
    assert data["ignore_index"]["pixel_count"] == 1
    assert any("Unknown mask values" in item for item in data["validation_warnings"])
    assert any(
        "val.txt contains 1 stems" in item for item in data["validation_warnings"]
    )


def test_analyze_dataset_reports_ambiguous_layout(tmp_path):
    root = _make_dataset(tmp_path)
    (root / "JPEGImages").mkdir()
    (root / "SegmentationClass").mkdir()
    Image.new("RGB", (2, 2)).save(root / "JPEGImages" / "one.jpg")
    _save_mask(root / "SegmentationClass" / "one.png", [[0, 1], [1, 255]])

    with pytest.raises(ValueError, match="multiple layouts are present"):
        analyze_dataset(root)


def test_analyze_dataset_ignores_empty_stale_layout_directories(tmp_path):
    root = _make_dataset(tmp_path)
    (root / "JPEGImages").mkdir()
    (root / "SegmentationClass").mkdir()

    data = analyze_dataset(root)

    assert data["layout"] == "mmseg"
    assert data["image_count"] == 2
    assert data["mask_count"] == 2


def test_compute_ocnet_weights_match_formula_and_skip_zero_counts():
    counts = {0: 100, 1: 10, 2: 0}

    weights = compute_ocnet_weights(counts)

    raw0 = 1 / math.log1p(100)
    raw1 = 1 / math.log1p(10)
    assert weights[0] == pytest.approx(3 * raw0 / (raw0 + raw1))
    assert weights[1] == pytest.approx(3 * raw1 / (raw0 + raw1))
    assert weights[2] is None


def test_write_dataset_card_creates_yaml_and_svg(tmp_path):
    root = _make_dataset(tmp_path)

    card = write_dataset_card(root)

    assert card.yaml_path.exists()
    assert card.svg_path.exists()
    assert not (root / "dataset_card.png").exists()
    assert card.svg_path.stat().st_size > 0
    svg_text = card.svg_path.read_text()
    assert "<svg" in svg_text
    assert "Train Pixel Count by Class" in svg_text
    assert "Val Pixel Count by Class" in svg_text
    ET.parse(card.svg_path)
    data = yaml.safe_load(card.yaml_path.read_text())
    assert data["image_count"] == 2
    assert data["mask_count"] == 2
    yaml_text = card.yaml_path.read_text()
    assert "class_weights_ocnet: [1.5000, 1.5000, 0.0000]" in yaml_text
    assert "split_stats:" in yaml_text
    assert "train:" in yaml_text
    assert "val:" in yaml_text
