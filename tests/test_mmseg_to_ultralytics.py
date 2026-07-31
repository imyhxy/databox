import os

import pytest
from databox.segmentation.dataset_manifest import read_manifest, write_manifest
from databox.segmentation.mmseg_to_ultralytics import build_ultralytics_dataset
from PIL import Image


def _make_dataset(root):
    dataset = root / "dataset"
    image_dir = dataset / "JPEGImages"
    mask_dir = dataset / "SegmentationClass"
    split_dir = dataset / "ImageSets" / "Segmentation"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    split_dir.mkdir(parents=True)

    (image_dir / "foo.jpg").write_text("image-foo")
    (image_dir / "bar.jpeg").write_text("image-bar")
    _write_palette_mask(mask_dir / "foo.png", [0, 1, 2, 255])
    _write_palette_mask(mask_dir / "bar.png", [3, 4, 5, 6])
    _write_palette_mask(mask_dir / "foo_polygon.png", [0, 1, 0, 255])
    _write_palette_mask(mask_dir / "bar_polygon.png", [0, 2, 0, 255])
    _write_palette_mask(mask_dir / "foo_polyline.png", [0, 3, 0, 255])
    _write_palette_mask(mask_dir / "bar_polyline.png", [0, 4, 0, 255])
    (split_dir / "train.txt").write_text("foo\n")
    (split_dir / "val.txt").write_text("bar\n")
    write_manifest(
        dataset,
        [
            {
                "sample_id": f"cvat:12:21:{frame_id}",
                "task_name": "task",
                "split": split,
                "image_path": f"JPEGImages/{image_name}",
                "mask_paths": {
                    "polygon": f"SegmentationClass/{stem}_polygon.png",
                    "polyline": f"SegmentationClass/{stem}_polyline.png",
                    "main": f"SegmentationClass/{stem}.png",
                },
                "width": 2,
                "height": 2,
                "task_id": 12,
                "job_id": 21,
                "frame_id": frame_id,
            }
            for frame_id, stem, image_name, split in (
                (1, "foo", "foo.jpg", "train"),
                (2, "bar", "bar.jpeg", "val"),
            )
        ],
    )
    return dataset


def _write_palette_mask(path, pixels):
    image = Image.new("P", (2, 2))
    image.putpalette([0, 0, 0] * 256)
    image.putdata(pixels)
    image.save(path)


def test_build_ultralytics_dataset_writes_root_splits_and_image_symlinks(tmp_path):
    dataset = _make_dataset(tmp_path)
    output = tmp_path / "dataset-ultralytics"

    count = build_ultralytics_dataset(dataset, output)

    assert count == 2
    assert (output / "train.txt").read_text() == "./images/foo.jpg\n"
    assert (output / "val.txt").read_text() == "./images/bar.jpeg\n"

    output_image = output / "images" / "foo.jpg"
    assert output_image.is_symlink()
    assert os.readlink(output_image) == "../../dataset/JPEGImages/foo.jpg"


def test_build_ultralytics_dataset_converts_palette_masks_to_l_mode(tmp_path):
    dataset = _make_dataset(tmp_path)
    output = tmp_path / "dataset-ultralytics"

    build_ultralytics_dataset(dataset, output)

    with Image.open(output / "labels" / "foo.png") as image:
        assert image.mode == "L"
        assert list(image.getdata()) == [0, 1, 2, 255]
    with Image.open(output / "polygon_masks" / "foo.png") as image:
        assert image.mode == "L"
        assert list(image.getdata()) == [0, 1, 0, 255]
    with Image.open(output / "polyline_masks" / "foo.png") as image:
        assert image.mode == "L"
        assert list(image.getdata()) == [0, 3, 0, 255]

    manifest = read_manifest(output)
    assert manifest[0]["image_path"] == "images/foo.jpg"
    assert manifest[0]["mask_paths"] == {
        "polygon": "polygon_masks/foo.png",
        "polyline": "polyline_masks/foo.png",
        "main": "labels/foo.png",
    }
    assert manifest[0]["width"] == 2
    assert manifest[0]["height"] == 2


def test_build_ultralytics_dataset_rejects_missing_image(tmp_path):
    dataset = _make_dataset(tmp_path)
    (dataset / "JPEGImages" / "foo.jpg").unlink()

    with pytest.raises(ValueError, match="missing images or masks"):
        build_ultralytics_dataset(dataset, tmp_path / "dataset-ultralytics")


def test_build_ultralytics_dataset_rejects_missing_mask(tmp_path):
    dataset = _make_dataset(tmp_path)
    (dataset / "SegmentationClass" / "foo.png").unlink()

    with pytest.raises(ValueError, match="missing images or masks"):
        build_ultralytics_dataset(dataset, tmp_path / "dataset-ultralytics")
