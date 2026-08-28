from databox.pose.split_ultralytics_pose import (
    main,
    split_ultralytics_pose_dataset,
)


def _make_pose_dataset(root):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    entries = []
    for index, suffix in enumerate((".jpg", ".png", ".jpeg", ".webp", ".bmp")):
        image_name = f"frame-{index}{suffix}"
        image_path = root / "images" / image_name
        image_path.write_bytes(b"image")
        (root / "labels" / f"frame-{index}.txt").write_text("")
        entries.append(f"./images/{image_name}")
    (root / "data.txt").write_text("\n".join(entries) + "\n")
    return entries


def test_split_uses_data_manifest(tmp_path):
    root = tmp_path / "pose"
    entries = _make_pose_dataset(root)
    data_yaml = root / "data.yaml"
    data_yaml.write_text("train: data.txt\nval: data.txt\n")

    stats = split_ultralytics_pose_dataset(root, train_ratio=0.8, seed=7)

    assert stats == {"train_count": 4, "val_count": 1}
    train = (root / "train.txt").read_text().splitlines()
    val = (root / "val.txt").read_text().splitlines()
    assert set(train) | set(val) == set(entries)
    assert set(train).isdisjoint(val)
    assert data_yaml.read_text() == "train: data.txt\nval: data.txt\n"


def test_split_is_deterministic_and_ratio_can_change_without_conversion(tmp_path):
    root = tmp_path / "pose"
    _make_pose_dataset(root)

    split_ultralytics_pose_dataset(root, train_ratio=0.8, seed=3)
    first_train = (root / "train.txt").read_text()
    first_val = (root / "val.txt").read_text()

    split_ultralytics_pose_dataset(root, train_ratio=0.4, seed=3)

    assert (root / "train.txt").read_text() != first_train
    assert (root / "val.txt").read_text() != first_val
    assert len((root / "train.txt").read_text().splitlines()) == 2
    assert len((root / "val.txt").read_text().splitlines()) == 3


def test_split_cli_reads_wheel_contact_params(tmp_path):
    root = tmp_path / "pose"
    _make_pose_dataset(root)
    params = tmp_path / "params.yaml"
    params.write_text("wheel_contact_split:\n  train: 0.4\n  seed: 3\n")

    assert main(["--dataset-root", str(root), "--params", str(params)]) == 0
    assert len((root / "train.txt").read_text().splitlines()) == 2
    assert len((root / "val.txt").read_text().splitlines()) == 3


def test_split_rejects_manifest_paths_without_labels(tmp_path):
    root = tmp_path / "pose"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "images" / "frame.jpg").write_bytes(b"image")
    (root / "data.txt").write_text("./images/frame.jpg\n")

    try:
        split_ultralytics_pose_dataset(root)
    except FileNotFoundError as error:
        assert "no label file" in str(error)
    else:
        raise AssertionError("missing pose labels should be rejected")
