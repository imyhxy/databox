import pytest
from databox.segmentation.dataset_manifest import read_manifest, write_manifest


def _record(stem, *, split, frame_id):
    return {
        "sample_id": f"cvat:104:208:{frame_id}",
        "task_name": "batch_260618",
        "split": split,
        "image_name": f"{stem}.jpg",
        "image_path": f"images/{stem}.jpg",
        "gt_mask_path": f"labels/{stem}.png",
        "polygon_mask_path": f"polygon_masks/{stem}.png",
        "polyline_mask_path": f"polyline_masks/{stem}.png",
        "task_id": 104,
        "job_id": 208,
        "frame_id": frame_id,
    }


def _write_files(root, stem):
    for dirname, suffix in (
        ("images", ".jpg"),
        ("labels", ".png"),
        ("polygon_masks", ".png"),
        ("polyline_masks", ".png"),
    ):
        path = root / dirname / f"{stem}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stem)


def test_manifest_round_trip_is_deterministically_sorted(tmp_path):
    _write_files(tmp_path, "train_sample")
    _write_files(tmp_path, "val_sample")
    records = [
        _record("val_sample", split="val", frame_id=2),
        _record("train_sample", split="train", frame_id=1),
    ]

    write_manifest(tmp_path, records)

    assert [record["image_name"] for record in read_manifest(tmp_path)] == [
        "train_sample.jpg",
        "val_sample.jpg",
    ]


def test_manifest_rejects_duplicate_sample_ids(tmp_path):
    _write_files(tmp_path, "one")
    _write_files(tmp_path, "two")
    first = _record("one", split="train", frame_id=1)
    second = _record("two", split="val", frame_id=2)
    second["sample_id"] = first["sample_id"]

    with pytest.raises(ValueError, match="Duplicate manifest sample_id"):
        write_manifest(tmp_path, [first, second])


def test_manifest_rejects_missing_branch_mask(tmp_path):
    _write_files(tmp_path, "one")
    (tmp_path / "polygon_masks" / "one.png").unlink()

    with pytest.raises(FileNotFoundError, match="polygon_mask_path"):
        write_manifest(tmp_path, [_record("one", split="train", frame_id=1)])
