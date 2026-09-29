import pytest
from databox.segmentation.make_slave_ultralytics import build_slave_dataset


def test_build_slave_dataset_reuses_labels_and_master_split(tmp_path):
    master = tmp_path / "master"
    (master / "labels").mkdir(parents=True)
    (master / "labels" / "scene-a_0G_080.txt").write_text("0 0.1 0.1 0.2 0.2 0.1 0.2\n")
    (master / "labels" / "scene-b_0G_080.txt").write_text("0 0.5 0.5 0.6 0.6 0.5 0.6\n")
    (master / "train.txt").write_text("./images/scene-a_0G_080.jpg")
    (master / "val.txt").write_text("./images/scene-b_0G_080.jpg")
    (master / "data.yaml").write_text("names:\n- vehicle\n")

    slave = tmp_path / "slave"
    for rel in (
        "0G_040/scene-a_0G_040.jpg",
        "0G_120/scene-a_0G_120.jpg",
        "0G_100/scene-b_0G_100.jpg",
        "0G_100/unknown_0G_100.jpg",
    ):
        (slave / rel).parent.mkdir(parents=True, exist_ok=True)
        (slave / rel).write_bytes(rel.encode())

    output = tmp_path / "out"
    assert build_slave_dataset(master, slave, output) == 3

    image = output / "images" / "scene-a_0G_040.jpg"
    assert image.is_symlink() and image.read_bytes() == b"0G_040/scene-a_0G_040.jpg"
    assert not (output / "images" / "unknown_0G_100.jpg").exists()
    assert (output / "labels" / "scene-a_0G_120.txt").read_text() == (
        master / "labels" / "scene-a_0G_080.txt"
    ).read_text()
    assert (output / "train.txt").read_text().split() == [
        "./images/scene-a_0G_040.jpg",
        "./images/scene-a_0G_120.jpg",
    ]
    assert (output / "val.txt").read_text().split() == ["./images/scene-b_0G_100.jpg"]
    assert (output / "data.yaml").read_text() == "names:\n- vehicle\n"

    # rerun cleans previous output instead of failing on existing symlinks
    assert build_slave_dataset(master, slave, output) == 3

    with pytest.raises(ValueError, match="No slave images"):
        build_slave_dataset(master, tmp_path / "out" / "labels", tmp_path / "x")
