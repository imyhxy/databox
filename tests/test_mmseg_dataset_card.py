import math
import xml.etree.ElementTree as ET

import pytest
import yaml
from databox.segmentation.mmseg_dataset_card import (
    analyze_dataset,
    compute_ocnet_weights,
    compute_sigmoid_pos_weights,
    parse_labelmap,
    write_dataset_card,
)
from PIL import Image


def _write_labelmaps(root):
    # polyline class ids are local (1, 2) but global ids in labelmap.txt are (3, 4).
    (root / "labelmap.txt").write_text(
        """# label:color_rgb:parts:actions
background:0,0,0::
poly_a:255,0,0::
poly_b:0,255,0::
line_a:0,0,255::
line_b:255,255,0::
"""
    )
    (root / "labelmap_polygon.txt").write_text(
        """background:0,0,0::
poly_a:255,0,0::
poly_b:0,255,0::
"""
    )
    (root / "labelmap_polyline.txt").write_text(
        """background:0,0,0::
line_a:0,0,255::
line_b:255,255,0::
"""
    )


def _save_mask(path, values):
    image = Image.new("P", (len(values[0]), len(values)))
    image.putdata([value for row in values for value in row])
    # Distinct palette entries keep PNG from remapping the class indices.
    image.putpalette([channel for i in range(256) for channel in (i, 255 - i, 0)])
    image.save(path)


def _write_sample(masks_dir, stem, main, polygon, polyline, polygon_txt, polyline_txt):
    _save_mask(masks_dir / f"{stem}.png", main)
    _save_mask(masks_dir / f"{stem}_polygon.png", polygon)
    _save_mask(masks_dir / f"{stem}_polyline.png", polyline)
    (masks_dir / f"{stem}_polygon.txt").write_text(polygon_txt)
    (masks_dir / f"{stem}_polyline.txt").write_text(polyline_txt)


def _make_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    _write_labelmaps(root)
    Image.new("RGB", (2, 2)).save(root / "images" / "one.jpg")
    Image.new("RGB", (2, 2)).save(root / "images" / "two.jpg")
    masks = root / "annotations"
    _write_sample(
        masks,
        "one",
        main=[[0, 1], [3, 255]],
        polygon=[[0, 1], [0, 255]],
        polyline=[[0, 0], [1, 255]],
        polygon_txt="1 0 0 1 0 1 1\n255 0 0 1 1 0 1\n",
        polyline_txt="3 0 1 1 1\n",
    )
    _write_sample(
        masks,
        "two",
        main=[[0, 2], [4, 4]],
        polygon=[[0, 2], [0, 0]],
        polyline=[[0, 0], [2, 2]],
        polygon_txt="2 0 0 1 0 1 1\n",
        polyline_txt="4 0 1 0 1\n4 1 1 1 1\n",
    )
    (root / "train.txt").write_text("one\n")
    (root / "val.txt").write_text("two\nmissing\n")
    return root


def _make_voc_dataset(tmp_path):
    root = tmp_path / "voc_dataset"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "SegmentationClass").mkdir()
    (root / "ImageSets" / "Segmentation").mkdir(parents=True)
    _write_labelmaps(root)
    Image.new("RGB", (2, 2)).save(root / "JPEGImages" / "one.jpg")
    Image.new("RGB", (2, 2)).save(root / "JPEGImages" / "two.png")
    _save_mask(root / "SegmentationClass" / "one.png", [[0, 1], [1, 255]])
    _save_mask(root / "SegmentationClass" / "two.png", [[0, 0], [1, 2]])
    splits = root / "ImageSets" / "Segmentation"
    (splits / "train.txt").write_text("one\n")
    (splits / "val.txt").write_text("two\n")
    (splits / "trainval.txt").write_text("one\ntwo\n")
    return root


def _rows(data, branch, split):
    return {row["name"]: row for row in data["branches"][branch][split]["classes"]}


def test_parse_labelmap_reads_ordered_classes(tmp_path):
    root = tmp_path
    _write_labelmaps(root)

    classes = parse_labelmap(root / "labelmap_polyline.txt")

    assert [item.name for item in classes] == ["background", "line_a", "line_b"]
    assert classes[1].id == 1
    assert classes[2].color_rgb == (255, 255, 0)


def test_analyze_dataset_counts_images_instances_pixels_per_branch_and_split(tmp_path):
    data = analyze_dataset(_make_dataset(tmp_path))

    assert data["splits"] == {"train": 1, "val": 2}
    assert data["mask_count"] == 2
    assert set(data["branches"]) == {"main", "polygon", "polyline"}
    assert set(data["branches"]["main"]) == {"train", "val"}

    train = data["branches"]["main"]["train"]
    assert train["mask_count"] == 1
    assert train["total_pixels"] == 4
    assert train["instance_count"] == 2
    rows = _rows(data, "main", "train")
    assert list(rows) == ["background", "poly_a", "poly_b", "line_a", "line_b", "ignore"]
    assert rows["background"]["pixel_count"] == 1
    assert rows["background"]["image_count"] == 1
    assert rows["background"]["instance_count"] is None
    assert rows["poly_a"]["pixel_count"] == 1
    assert rows["poly_a"]["instance_count"] == 1
    assert rows["poly_b"]["image_count"] == 0
    assert rows["poly_b"]["instance_count"] == 0
    assert rows["line_a"]["instance_count"] == 1
    assert rows["ignore"]["id"] == 255
    assert rows["ignore"]["pixel_count"] == 1
    assert rows["ignore"]["image_count"] == 1
    assert rows["ignore"]["pixel_ratio"] == 0.25
    assert rows["ignore"]["instance_count"] == 1  # "255 ..." txt line
    assert not any("instance class ids" in w for w in data["validation_warnings"])

    val_rows = _rows(data, "main", "val")
    assert val_rows["line_b"]["pixel_count"] == 2
    assert val_rows["line_b"]["instance_count"] == 2
    assert val_rows["line_b"]["image_ratio"] == 1.0
    assert val_rows["ignore"]["pixel_count"] == 0

    polygon_val = _rows(data, "polygon", "val")
    assert list(polygon_val) == ["background", "poly_a", "poly_b", "ignore"]
    assert polygon_val["poly_b"]["pixel_count"] == 1
    assert polygon_val["poly_b"]["instance_count"] == 1
    assert data["branches"]["polygon"]["val"]["instance_count"] == 1


def test_polyline_uses_local_mask_ids_and_maps_global_txt_ids_by_name(tmp_path):
    data = analyze_dataset(_make_dataset(tmp_path))

    train = _rows(data, "polyline", "train")
    assert train["line_a"]["id"] == 1
    assert train["line_a"]["pixel_count"] == 1
    assert train["line_a"]["instance_count"] == 1  # txt id 3 -> local id 1
    assert train["line_b"]["instance_count"] == 0
    val = _rows(data, "polyline", "val")
    assert val["line_b"]["id"] == 2
    assert val["line_b"]["pixel_count"] == 2
    assert val["line_b"]["instance_count"] == 2  # txt id 4 -> local id 2
    assert not any("instance class ids" in item for item in data["validation_warnings"])


def test_ocnet_weights_are_computed_per_branch_and_split(tmp_path):
    data = analyze_dataset(_make_dataset(tmp_path))

    for branch in ("main", "polygon", "polyline"):
        for split in ("train", "val"):
            stats = data["branches"][branch][split]
            class_rows = stats["classes"][:-1]  # without ignore
            expected = compute_ocnet_weights(
                {row["id"]: row["pixel_count"] for row in class_rows}
            )
            assert stats["class_weights_ocnet"] == [
                round(expected[row["id"]] or 0.0, 4) for row in class_rows
            ]
    assert data["branches"]["polygon"]["train"]["class_weights_ocnet"] == [1.1606, 1.8394, 0.0]
    assert "pos_weights_sigmoid" not in data["branches"]["polygon"]["train"]


def test_polyline_sigmoid_pos_weights_per_split(tmp_path):
    data = analyze_dataset(_make_dataset(tmp_path))

    train = data["branches"]["polyline"]["train"]
    # train: 4 px, 1 ignored, line_a pos=1 neg=2 -> sqrt(2); line_b has no pixels -> cap.
    assert train["pos_weights_sigmoid"] == [round(math.sqrt(2), 4), 50.0]
    rows = _rows(data, "polyline", "train")
    assert rows["line_a"]["pos_weight_sigmoid"] == pytest.approx(math.sqrt(2))
    assert rows["background"]["pos_weight_sigmoid"] is None
    assert rows["ignore"]["pos_weight_sigmoid"] is None
    # val: 4 px, line_b pos=2 neg=2 -> 1; line_a has no pixels -> cap.
    assert data["branches"]["polyline"]["val"]["pos_weights_sigmoid"] == [50.0, 1.0]


def test_compute_sigmoid_pos_weights_formula_cap_and_zero_pixels():
    weights = compute_sigmoid_pos_weights({1: 100, 2: 1, 3: 0, 4: 1}, valid_pixels=1_000)

    assert weights[1] == pytest.approx(3.0)  # sqrt(900 / 100)
    assert weights[2] == pytest.approx(math.sqrt(999))  # ~31.6, under the cap
    assert weights[3] == 50.0  # no positives
    capped = compute_sigmoid_pos_weights({1: 1}, valid_pixels=100_001)
    assert capped[1] == 50.0  # sqrt(100000) ~ 316 -> capped
    assert compute_sigmoid_pos_weights({1: 1}, 100_001, cap=10.0)[1] == 10.0


def test_analyze_dataset_reports_warnings(tmp_path):
    root = _make_dataset(tmp_path)
    _save_mask(root / "annotations" / "two.png", [[0, 2], [4, 7]])

    warnings = analyze_dataset(root)["validation_warnings"]

    assert any("val.txt contains 1 stems without images" in item for item in warnings)
    assert any("main/val: unknown mask values found: 7 (1 px)" in item for item in warnings)
    assert "polygon/train: classes with zero pixels: poly_b" in warnings


def test_analyze_dataset_only_reports_train_and_val_splits(tmp_path):
    data = analyze_dataset(_make_voc_dataset(tmp_path))

    assert data["layout"] == "voc"
    assert data["image_count"] == 2
    assert data["mask_count"] == 2
    assert data["splits"] == {"train": 1, "val": 1}
    assert set(data["branches"]["main"]) == {"train", "val"}
    assert not any("multiple split" in item for item in data["validation_warnings"])
    assert _rows(data, "main", "train")["poly_a"]["pixel_count"] == 2
    # polygon/polyline masks are absent in this fixture.
    assert data["branches"]["polygon"]["train"]["mask_count"] == 0
    assert any("polygon/train: 1 masks are missing" == w for w in data["validation_warnings"])


def test_analyze_dataset_reports_ambiguous_layout(tmp_path):
    root = _make_dataset(tmp_path)
    (root / "JPEGImages").mkdir()
    (root / "SegmentationClass").mkdir()
    (root / "ImageSets" / "Segmentation").mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(root / "JPEGImages" / "one.jpg")
    _save_mask(root / "SegmentationClass" / "one.png", [[0, 1], [1, 255]])
    (root / "ImageSets" / "Segmentation" / "train.txt").write_text("one\n")

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
    svg_text = card.svg_path.read_text()
    assert "<svg" in svg_text
    for branch in ("Polygon", "Polyline"):
        for metric in ("Images", "Instances", "Pixels", "Pixels (All)"):
            assert f"{branch}: {metric} by Class" in svg_text
    ET.parse(card.svg_path)
    yaml_text = card.yaml_path.read_text()
    data = yaml.safe_load(yaml_text)
    assert data["image_count"] == 2
    assert data["branches"]["polyline"]["train"]["instance_count"] == 1
    assert "class_weights_ocnet: [1.1606, 1.8394, 0.0000]" in yaml_text
    assert "pos_weights_sigmoid: [1.4142, 50.0000]" in yaml_text
    assert data["sigmoid_pos_weight"]["cap"] == 50.0
