import pytest
from databox.segmentation.dataset_manifest import read_manifest, write_manifest
from databox.segmentation.make_slave_voc import build_slave_voc_dataset, scene_key
from PIL import Image


def _write_image(path, size=(8, 6)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def _make_master(root):
    master = root / "master"
    (master / "JPEGImages").mkdir(parents=True)
    (master / "SegmentationClass").mkdir(parents=True)
    (master / "ImageSets" / "Segmentation").mkdir(parents=True)
    for filename in (
        "labelmap.txt",
        "labelmap_polygon.txt",
        "labelmap_polyline.txt",
    ):
        (master / filename).write_text(f"{filename}\n")
    (master / "SegmentationClass" / "scene-a_0G_080.png").write_text("mask-a")
    (master / "SegmentationClass" / "scene-a_0G_080_polygon.png").write_text(
        "mask-a-polygon"
    )
    (master / "SegmentationClass" / "scene-a_0G_080_polyline.png").write_text(
        "mask-a-polyline"
    )
    (master / "SegmentationClass" / "scene-a_0G_080_polygon.txt").write_text(
        "1 1.25 1.5 6.75 1.125 6.5 6.25\n"
    )
    (master / "SegmentationClass" / "scene-a_0G_080_polyline.txt").write_text(
        "2 1.25 1.5 6.75 6.125\n"
    )
    (master / "SegmentationClass" / "scene-b_0G_080.png").write_text("mask-b")
    (master / "SegmentationClass" / "scene-b_0G_080_polygon.png").write_text(
        "mask-b-polygon"
    )
    (master / "SegmentationClass" / "scene-b_0G_080_polyline.png").write_text(
        "mask-b-polyline"
    )
    (master / "SegmentationClass" / "scene-b_0G_080_polygon.txt").write_text("")
    (master / "SegmentationClass" / "scene-b_0G_080_polyline.txt").write_text(
        "2 2.25 2.5 7.75 7.125\n"
    )
    (master / "ImageSets" / "Segmentation" / "train.txt").write_text("scene-a_0G_080\n")
    (master / "ImageSets" / "Segmentation" / "val.txt").write_text("scene-b_0G_080\n")
    _write_image(master / "JPEGImages" / "scene-a_0G_080.jpg", (10, 8))
    _write_image(master / "JPEGImages" / "scene-b_0G_080.jpg", (12, 9))
    write_manifest(
        master,
        [
            {
                "sample_id": f"cvat_13_105_{frame_id}",
                "task_name": "master-task",
                "split": split,
                "image_path": f"JPEGImages/{stem}.jpg",
                "mask_paths": {
                    "polygon": f"SegmentationClass/{stem}_polygon.png",
                    "polyline": f"SegmentationClass/{stem}_polyline.png",
                    "main": f"SegmentationClass/{stem}.png",
                },
                "width": 10 if stem.startswith("scene-a") else 12,
                "height": 8 if stem.startswith("scene-a") else 9,
                "task_id": 13,
                "job_id": 105,
                "frame_id": frame_id,
            }
            for frame_id, stem, split in (
                (1, "scene-a_0G_080", "train"),
                (2, "scene-b_0G_080", "val"),
            )
        ],
    )
    return master


def _make_slave(root):
    slave = root / "slave"
    (slave / "0G_100").mkdir(parents=True)
    (slave / "0G_120").mkdir()
    (slave / "0G_060").mkdir()
    _write_image(slave / "0G_100" / "scene-a_0G_100.jpg", (20, 10))
    _write_image(slave / "0G_120" / "scene-a_0G_120.jpg", (22, 11))
    _write_image(slave / "0G_100" / "scene-b_0G_100.jpg", (24, 12))
    _write_image(slave / "0G_060" / "unknown_0G_060.jpg")
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
    assert (
        output / "SegmentationClass" / "scene-a_0G_100_polygon.txt"
    ).read_text() == "1 1.25 1.5 6.75 1.125 6.5 6.25\n"
    assert (
        output / "SegmentationClass" / "scene-a_0G_100_polyline.txt"
    ).read_text() == "2 1.25 1.5 6.75 6.125\n"
    assert (output / "SegmentationClass" / "scene-a_0G_120.png").read_text() == (
        "mask-a"
    )
    assert (
        output / "SegmentationClass" / "scene-a_0G_120_polygon.png"
    ).read_text() == "mask-a-polygon"
    assert (
        output / "SegmentationClass" / "scene-a_0G_120_polyline.png"
    ).read_text() == "mask-a-polyline"
    assert (
        output / "SegmentationClass" / "scene-a_0G_120_polygon.txt"
    ).read_text() == "1 1.25 1.5 6.75 1.125 6.5 6.25\n"
    assert (
        output / "SegmentationClass" / "scene-a_0G_120_polyline.txt"
    ).read_text() == "2 1.25 1.5 6.75 6.125\n"
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
        output / "SegmentationClass" / "scene-b_0G_100_polygon.txt"
    ).read_text() == ""
    assert (
        output / "SegmentationClass" / "scene-b_0G_100_polyline.txt"
    ).read_text() == "2 2.25 2.5 7.75 7.125\n"
    assert (
        output / "ImageSets" / "Segmentation" / "train.txt"
    ).read_text().splitlines() == [
        "scene-a_0G_100",
        "scene-a_0G_120",
    ]
    assert (
        output / "ImageSets" / "Segmentation" / "val.txt"
    ).read_text().splitlines() == ["scene-b_0G_100"]
    for filename in (
        "labelmap.txt",
        "labelmap_polygon.txt",
        "labelmap_polyline.txt",
    ):
        assert (output / filename).read_text() == f"{filename}\n"
    manifest = read_manifest(output)
    assert [record["sample_id"] for record in manifest] == [
        "cvat_13_105_1_derived_scene-a_0G_100",
        "cvat_13_105_1_derived_scene-a_0G_120",
        "cvat_13_105_2_derived_scene-b_0G_100",
    ]
    assert manifest[0]["mask_paths"] == {
        "polygon": "SegmentationClass/scene-a_0G_100_polygon.png",
        "polyline": "SegmentationClass/scene-a_0G_100_polyline.png",
        "main": "SegmentationClass/scene-a_0G_100.png",
    }
    assert manifest[0]["width"] == 20
    assert manifest[0]["height"] == 10


def test_build_slave_voc_dataset_rejects_duplicate_master_scene_keys(tmp_path):
    master = _make_master(tmp_path)
    (master / "SegmentationClass" / "scene-a_0G_100.png").write_text("mask-a-100")
    (master / "SegmentationClass" / "scene-a_0G_100_polygon.png").write_text(
        "mask-a-100-polygon"
    )
    (master / "SegmentationClass" / "scene-a_0G_100_polyline.png").write_text(
        "mask-a-100-polyline"
    )
    (master / "SegmentationClass" / "scene-a_0G_100_polygon.txt").write_text(
        "1 1 1 2 1 2 2\n"
    )
    (master / "SegmentationClass" / "scene-a_0G_100_polyline.txt").write_text(
        "2 1 1 2 2\n"
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
    (
        master / "SegmentationClass" / "scene-a_0G_100_polygon_0G_080_polygon.txt"
    ).write_text("1 1 1 2 1 2 2\n")
    (
        master / "SegmentationClass" / "scene-a_0G_100_polygon_0G_080_polyline.txt"
    ).write_text("2 1 1 2 2\n")
    (master / "ImageSets" / "Segmentation" / "val.txt").write_text(
        "scene-b_0G_080\nscene-a_0G_100_polygon_0G_080\n"
    )
    collision_stem = "scene-a_0G_100_polygon_0G_080"
    (master / "JPEGImages" / f"{collision_stem}.jpg").write_text(
        "image-colliding-master"
    )
    manifest = read_manifest(master)
    manifest.append(
        {
            "sample_id": "cvat_13_105_3",
            "task_name": "master-task",
            "split": "val",
            "image_path": f"JPEGImages/{collision_stem}.jpg",
            "mask_paths": {
                "polygon": f"SegmentationClass/{collision_stem}_polygon.png",
                "polyline": f"SegmentationClass/{collision_stem}_polyline.png",
                "main": f"SegmentationClass/{collision_stem}.png",
            },
            "width": 8,
            "height": 6,
            "task_id": 13,
            "job_id": 105,
            "frame_id": 3,
        }
    )
    write_manifest(master, manifest)
    slave = _make_slave(tmp_path)
    _write_image(slave / "0G_100" / "scene-a_0G_100_polygon.jpg")

    with pytest.raises(ValueError, match="overwrite masks"):
        build_slave_voc_dataset(master, slave, tmp_path / "out")
