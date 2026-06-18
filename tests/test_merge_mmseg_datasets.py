import pytest
from databox.segmentation.merge_mmseg_datasets import merge_datasets


def _make_dataset(root, name, stems=("one",)):
    dataset = root / name
    (dataset / "JPEGImages").mkdir(parents=True)
    (dataset / "SegmentationClass").mkdir()
    (dataset / "ImageSets" / "Segmentation").mkdir(parents=True)
    (dataset / "labelmap.txt").write_text("background:0,0,0::\n")
    for stem in stems:
        (dataset / "JPEGImages" / f"{stem}.jpg").write_text(f"image-{stem}")
        (dataset / "SegmentationClass" / f"{stem}.png").write_text(f"mask-{stem}")
        (dataset / "SegmentationClass" / f"{stem}_polygon.png").write_text(
            f"polygon-{stem}"
        )
        (dataset / "SegmentationClass" / f"{stem}_polyline.png").write_text(
            f"polyline-{stem}"
        )
    (dataset / "ImageSets" / "Segmentation" / "train.txt").write_text(
        "".join(f"{stem}\n" for stem in stems)
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
        output / "ImageSets" / "Segmentation" / "train.txt"
    ).read_text().splitlines() == ["first__one", "second__two"]


def test_merge_datasets_requires_branch_masks(tmp_path):
    dataset = _make_dataset(tmp_path, "dataset", stems=("one",))
    (dataset / "SegmentationClass" / "one_polygon.png").unlink()

    with pytest.raises(ValueError, match="Missing branch masks"):
        merge_datasets([dataset], tmp_path / "merged")


def test_merge_datasets_rejects_branch_mask_name_collisions(tmp_path):
    dataset = _make_dataset(tmp_path, "dataset", stems=("one", "one_polygon"))

    with pytest.raises(ValueError, match="reused for multiple stems"):
        merge_datasets([dataset], tmp_path / "merged")
