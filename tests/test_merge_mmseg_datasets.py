import pytest
from databox.segmentation.dataset_manifest import read_manifest, write_manifest
from databox.segmentation.merge_mmseg_datasets import merge_datasets


def _make_dataset(root, name, stems=("one",)):
    dataset = root / name
    (dataset / "JPEGImages").mkdir(parents=True)
    (dataset / "SegmentationClass").mkdir()
    (dataset / "ImageSets" / "Segmentation").mkdir(parents=True)
    for filename in (
        "labelmap.txt",
        "labelmap_polygon.txt",
        "labelmap_polyline.txt",
        "labelmap_vehicle.txt",
    ):
        (dataset / filename).write_text(f"{filename}\n")
    for stem in stems:
        (dataset / "JPEGImages" / f"{stem}.jpg").write_text(f"image-{stem}")
        (dataset / "SegmentationClass" / f"{stem}.png").write_text(f"mask-{stem}")
        (dataset / "SegmentationClass" / f"{stem}_polygon.png").write_text(
            f"polygon-{stem}"
        )
        (dataset / "SegmentationClass" / f"{stem}_polyline.png").write_text(
            f"polyline-{stem}"
        )
        (dataset / "SegmentationClass" / f"{stem}_vehicle.png").write_text(
            f"vehicle-{stem}"
        )
        (dataset / "SegmentationClass" / f"{stem}_polygon.txt").write_text(
            f"1 1.25 1.5 6.75 1.125 6.5 6.25 {stem}\n"
        )
        (dataset / "SegmentationClass" / f"{stem}_polyline.txt").write_text(
            f"2 1.25 1.5 6.75 6.125 {stem}\n"
        )
        (dataset / "SegmentationClass" / f"{stem}_vehicle.txt").write_text(
            "1 1.25 1.5 6.75 1.125 6.5 6.25\n"
        )
    (dataset / "ImageSets" / "Segmentation" / "train.txt").write_text(
        "".join(f"{stem}\n" for stem in stems)
    )
    task_name = f"{dataset.parent.name}-{name}"
    task_id = sum(ord(character) for character in task_name)
    write_manifest(
        dataset,
        [
            {
                "sample_id": f"cvat_{task_id}_2_{frame_id}",
                "task_name": task_name,
                "split": "train",
                "image_path": f"JPEGImages/{stem}.jpg",
                "mask_paths": {
                    "polygon": f"SegmentationClass/{stem}_polygon.png",
                    "polyline": f"SegmentationClass/{stem}_polyline.png",
                    "vehicle": f"SegmentationClass/{stem}_vehicle.png",
                    "main": f"SegmentationClass/{stem}.png",
                },
                "width": 640,
                "height": 480,
                "task_id": task_id,
                "job_id": 2,
                "frame_id": frame_id,
            }
            for frame_id, stem in enumerate(stems)
        ],
    )
    return dataset


def test_merge_datasets_copies_base_polygon_and_polyline_masks(tmp_path):
    first = _make_dataset(tmp_path, "first", stems=("one",))
    second = _make_dataset(tmp_path, "second", stems=("two",))
    output = tmp_path / "merged"

    merge_datasets([first, second], output)

    assert (output / "SegmentationClass" / "first__one.png").read_text() == "mask-one"
    assert (
        output / "SegmentationClass" / "first__one_polygon.png"
    ).read_text() == "polygon-one"
    assert (
        output / "SegmentationClass" / "first__one_polyline.png"
    ).read_text() == "polyline-one"
    assert (
        output / "SegmentationClass" / "first__one_vehicle.png"
    ).read_text() == "vehicle-one"
    assert (
        output / "SegmentationClass" / "first__one_polygon.txt"
    ).read_text() == "1 1.25 1.5 6.75 1.125 6.5 6.25 one\n"
    assert (
        output / "SegmentationClass" / "first__one_polyline.txt"
    ).read_text() == "2 1.25 1.5 6.75 6.125 one\n"
    assert (
        output / "SegmentationClass" / "first__one_vehicle.txt"
    ).read_text() == "1 1.25 1.5 6.75 1.125 6.5 6.25\n"
    assert (output / "SegmentationClass" / "second__two.png").read_text() == (
        "mask-two"
    )
    assert (
        output / "SegmentationClass" / "second__two_polygon.png"
    ).read_text() == "polygon-two"
    assert (
        output / "SegmentationClass" / "second__two_polyline.png"
    ).read_text() == "polyline-two"
    assert (
        output / "SegmentationClass" / "second__two_polygon.txt"
    ).read_text() == "1 1.25 1.5 6.75 1.125 6.5 6.25 two\n"
    assert (
        output / "SegmentationClass" / "second__two_polyline.txt"
    ).read_text() == "2 1.25 1.5 6.75 6.125 two\n"
    assert (
        output / "ImageSets" / "Segmentation" / "train.txt"
    ).read_text().splitlines() == ["first__one", "second__two"]
    records = read_manifest(output)
    assert [record["image_path"] for record in records] == [
        "JPEGImages/first__one.jpg",
        "JPEGImages/second__two.jpg",
    ]
    assert records[0]["mask_paths"] == {
        "polygon": "SegmentationClass/first__one_polygon.png",
        "polyline": "SegmentationClass/first__one_polyline.png",
        "vehicle": "SegmentationClass/first__one_vehicle.png",
        "main": "SegmentationClass/first__one.png",
    }
    assert records[0]["width"] == 640
    assert records[0]["height"] == 480
    for filename in (
        "labelmap.txt",
        "labelmap_polygon.txt",
        "labelmap_polyline.txt",
        "labelmap_vehicle.txt",
    ):
        assert (output / filename).read_text() == f"{filename}\n"


def test_merge_datasets_requires_branch_masks(tmp_path):
    dataset = _make_dataset(tmp_path, "dataset", stems=("one",))
    (dataset / "SegmentationClass" / "one_polygon.png").unlink()

    with pytest.raises(ValueError, match="Missing branch masks"):
        merge_datasets([dataset], tmp_path / "merged")


def test_merge_datasets_requires_shape_annotations(tmp_path):
    dataset = _make_dataset(tmp_path, "dataset", stems=("one",))
    (dataset / "SegmentationClass" / "one_polygon.txt").unlink()

    with pytest.raises(ValueError, match="polygon_annotations"):
        merge_datasets([dataset], tmp_path / "merged")


def test_merge_datasets_rejects_branch_mask_name_collisions(tmp_path):
    dataset = _make_dataset(tmp_path, "dataset", stems=("one", "one_polygon"))

    with pytest.raises(ValueError, match="reused for multiple stems"):
        merge_datasets([dataset], tmp_path / "merged")


def test_merge_datasets_uses_explicit_prefixes(tmp_path):
    first = _make_dataset(tmp_path / "first_parent", "shadow", stems=("one",))
    second = _make_dataset(tmp_path / "second_parent", "shadow", stems=("two",))
    output = tmp_path / "merged"

    merge_datasets([first, second], output, ["street_map", "proprietary"])

    split = output / "ImageSets" / "Segmentation" / "train.txt"
    assert split.read_text().splitlines() == ["street_map__one", "proprietary__two"]


def test_merge_datasets_requires_one_unique_prefix_per_input(tmp_path):
    first = _make_dataset(tmp_path / "first_parent", "shadow", stems=("one",))
    second = _make_dataset(tmp_path / "second_parent", "shadow", stems=("two",))

    with pytest.raises(ValueError, match="exactly one value per input"):
        merge_datasets([first, second], tmp_path / "merged", ["street_map"])

    with pytest.raises(ValueError, match="prefixes must be unique"):
        merge_datasets([first, second], tmp_path / "merged", ["shadow", "shadow"])
