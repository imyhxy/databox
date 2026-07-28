import pytest
from databox.segmentation.dataset_manifest import read_manifest, write_manifest
from databox.segmentation.make_slave_voc import build_slave_voc_dataset, scene_key


def _make_master(root):
    master = root / "master"
    (master / "JPEGImages").mkdir(parents=True)
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
    (master / "SegmentationClass" / "scene-b_0G_080_polyline.txt").write_text(
        "2 2.25 2.5 7.75 7.125\n"
    )
    (master / "ImageSets" / "Segmentation" / "train.txt").write_text("scene-a_0G_080\n")
    (master / "ImageSets" / "Segmentation" / "val.txt").write_text("scene-b_0G_080\n")
    (master / "JPEGImages" / "scene-a_0G_080.jpg").write_text("image-a")
    (master / "JPEGImages" / "scene-b_0G_080.jpg").write_text("image-b")
    write_manifest(
        master,
        [
            {
                "sample_id": f"cvat:13:105:{frame_id}",
                "task_name": "master-task",
                "split": split,
                "image_name": f"{stem}.jpg",
                "image_path": f"JPEGImages/{stem}.jpg",
                "gt_mask_path": f"SegmentationClass/{stem}.png",
                "polygon_mask_path": f"SegmentationClass/{stem}_polygon.png",
                "polyline_mask_path": f"SegmentationClass/{stem}_polyline.png",
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
    assert (output / "labelmap.txt").read_text() == "background:0,0,0::\n"
    manifest = read_manifest(output)
    assert [record["sample_id"] for record in manifest] == [
        "cvat:13:105:1:derived:scene-a_0G_100",
        "cvat:13:105:1:derived:scene-a_0G_120",
        "cvat:13:105:2:derived:scene-b_0G_100",
    ]
    assert manifest[0]["polygon_mask_path"] == (
        "SegmentationClass/scene-a_0G_100_polygon.png"
    )


def test_build_slave_voc_dataset_rejects_duplicate_master_scene_keys(tmp_path):
    master = _make_master(tmp_path)
    (master / "SegmentationClass" / "scene-a_0G_100.png").write_text("mask-a-100")
    (master / "SegmentationClass" / "scene-a_0G_100_polygon.png").write_text(
        "mask-a-100-polygon"
    )
    (master / "SegmentationClass" / "scene-a_0G_100_polyline.png").write_text(
        "mask-a-100-polyline"
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
            "sample_id": "cvat:13:105:3",
            "task_name": "master-task",
            "split": "val",
            "image_name": f"{collision_stem}.jpg",
            "image_path": f"JPEGImages/{collision_stem}.jpg",
            "gt_mask_path": f"SegmentationClass/{collision_stem}.png",
            "polygon_mask_path": (
                f"SegmentationClass/{collision_stem}_polygon.png"
            ),
            "polyline_mask_path": (
                f"SegmentationClass/{collision_stem}_polyline.png"
            ),
            "task_id": 13,
            "job_id": 105,
            "frame_id": 3,
        }
    )
    write_manifest(master, manifest)
    slave = _make_slave(tmp_path)
    (slave / "0G_100" / "scene-a_0G_100_polygon.jpg").write_text("image-collision")

    with pytest.raises(ValueError, match="overwrite masks"):
        build_slave_voc_dataset(master, slave, tmp_path / "out")
