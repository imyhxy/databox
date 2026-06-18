import pytest
from databox.segmentation.make_slave_voc import build_slave_voc_dataset, scene_key


def _make_master(root):
    master = root / "master"
    (master / "SegmentationClass").mkdir(parents=True)
    (master / "ImageSets" / "Segmentation").mkdir(parents=True)
    (master / "labelmap.txt").write_text("background:0,0,0::\n")
    (master / "SegmentationClass" / "scene-a_0G_080.png").write_text("mask-a")
    (master / "SegmentationClass" / "scene-a_0G_080_polygon.png").write_text(
        "mask-a-polygon"
    )
    (master / "SegmentationClass" / "scene-a_0G_080_polyline.png").write_text(
        "mask-a-polyline"
    )
    (master / "SegmentationClass" / "scene-b_0G_080.png").write_text("mask-b")
    (master / "SegmentationClass" / "scene-b_0G_080_polygon.png").write_text(
        "mask-b-polygon"
    )
    (master / "SegmentationClass" / "scene-b_0G_080_polyline.png").write_text(
        "mask-b-polyline"
    )
    (master / "ImageSets" / "Segmentation" / "train.txt").write_text("scene-a_0G_080\n")
    (master / "ImageSets" / "Segmentation" / "val.txt").write_text("scene-b_0G_080\n")
    return master


def _make_slave(root):
    slave = root / "slave"
    (slave / "0G_100").mkdir(parents=True)
    (slave / "0G_120").mkdir()
    (slave / "0G_060").mkdir()
    (slave / "0G_100" / "scene-a_0G_100.jpg").write_text("image-a-100")
    (slave / "0G_120" / "scene-a_0G_120.jpg").write_text("image-a-120")
    (slave / "0G_100" / "scene-b_0G_100.jpg").write_text("image-b-100")
    (slave / "0G_060" / "unknown_0G_060.jpg").write_text("unknown")
    return slave


def test_scene_key_strips_brightness_suffix():
    assert scene_key("260522-115018_00DA8940606_0G_080") == (
        "260522-115018_00DA8940606"
    )
    assert scene_key("no_brightness") == "no_brightness"


def test_build_slave_voc_dataset_reuses_masks_and_master_splits(tmp_path):
    master = _make_master(tmp_path)
    slave = _make_slave(tmp_path)
    output = tmp_path / "out"

    count = build_slave_voc_dataset(master, slave, output)

    assert count == 3
    assert sorted(path.name for path in (output / "JPEGImages").iterdir()) == [
        "scene-a_0G_100.jpg",
        "scene-a_0G_120.jpg",
        "scene-b_0G_100.jpg",
    ]
    assert (output / "SegmentationClass" / "scene-a_0G_100.png").read_text() == (
        "mask-a"
    )
    assert (
        output / "SegmentationClass" / "scene-a_0G_100_polygon.png"
    ).read_text() == "mask-a-polygon"
    assert (
        output / "SegmentationClass" / "scene-a_0G_100_polyline.png"
    ).read_text() == "mask-a-polyline"
    assert (output / "SegmentationClass" / "scene-a_0G_120.png").read_text() == (
        "mask-a"
    )
    assert (
        output / "SegmentationClass" / "scene-a_0G_120_polygon.png"
    ).read_text() == "mask-a-polygon"
    assert (
        output / "SegmentationClass" / "scene-a_0G_120_polyline.png"
    ).read_text() == "mask-a-polyline"
    assert (output / "SegmentationClass" / "scene-b_0G_100.png").read_text() == (
        "mask-b"
    )
    assert (
        output / "SegmentationClass" / "scene-b_0G_100_polygon.png"
    ).read_text() == "mask-b-polygon"
    assert (
        output / "SegmentationClass" / "scene-b_0G_100_polyline.png"
    ).read_text() == "mask-b-polyline"
    assert (
        output / "ImageSets" / "Segmentation" / "train.txt"
    ).read_text().splitlines() == [
        "scene-a_0G_100",
        "scene-a_0G_120",
    ]
    assert (
        output / "ImageSets" / "Segmentation" / "val.txt"
    ).read_text().splitlines() == ["scene-b_0G_100"]
    assert (output / "labelmap.txt").read_text() == "background:0,0,0::\n"


def test_build_slave_voc_dataset_rejects_duplicate_master_scene_keys(tmp_path):
    master = _make_master(tmp_path)
    (master / "SegmentationClass" / "scene-a_0G_100.png").write_text("mask-a-100")
    (master / "SegmentationClass" / "scene-a_0G_100_polygon.png").write_text(
        "mask-a-100-polygon"
    )
    (master / "SegmentationClass" / "scene-a_0G_100_polyline.png").write_text(
        "mask-a-100-polyline"
    )
    (master / "ImageSets" / "Segmentation" / "val.txt").write_text("scene-a_0G_100\n")
    slave = _make_slave(tmp_path)

    with pytest.raises(ValueError, match="Duplicate master scene key"):
        build_slave_voc_dataset(master, slave, tmp_path / "out")


def test_build_slave_voc_dataset_requires_branch_masks(tmp_path):
    master = _make_master(tmp_path)
    (master / "SegmentationClass" / "scene-a_0G_080_polygon.png").unlink()

    with pytest.raises(FileNotFoundError, match="scene-a_0G_080_polygon.png"):
        build_slave_voc_dataset(master, _make_slave(tmp_path), tmp_path / "out")


def test_build_slave_voc_dataset_rejects_branch_mask_name_collisions(tmp_path):
    master = _make_master(tmp_path)
    (master / "SegmentationClass" / "scene-a_0G_100_polygon_0G_080.png").write_text(
        "mask-a-colliding-stem"
    )
    (
        master / "SegmentationClass" / "scene-a_0G_100_polygon_0G_080_polygon.png"
    ).write_text("mask-a-colliding-stem-polygon")
    (
        master / "SegmentationClass" / "scene-a_0G_100_polygon_0G_080_polyline.png"
    ).write_text("mask-a-colliding-stem-polyline")
    (master / "ImageSets" / "Segmentation" / "val.txt").write_text(
        "scene-b_0G_080\nscene-a_0G_100_polygon_0G_080\n"
    )
    slave = _make_slave(tmp_path)
    (slave / "0G_100" / "scene-a_0G_100_polygon.jpg").write_text("image-collision")

    with pytest.raises(ValueError, match="overwrite masks"):
        build_slave_voc_dataset(master, slave, tmp_path / "out")
