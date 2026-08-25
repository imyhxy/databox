import os

import cv2
import numpy as np
import yaml
from databox.segmentation.cityscapes_to_ultralytics_instance import (
    PARAM_NAME,
    build_ultralytics_instance_dataset,
    config_from_args,
    parse_args,
)


def _write_instance_ids(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.asarray(values, dtype=np.uint16))


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def test_build_ultralytics_instance_dataset_uses_visible_instance_masks(tmp_path):
    raw = tmp_path / "raw"
    train_city = "aachen"
    val_city = "bochum"

    _write_image(raw / "leftImg8bit" / "train" / train_city / "sample_leftImg8bit.png")
    instance_ids = np.zeros((10, 20), dtype=np.uint16)
    instance_ids[1:5, 2:6] = 26000  # car
    instance_ids[2:4, 8:10] = 27000  # truck
    instance_ids[0, 0] = 7  # stuff, not an instance
    _write_instance_ids(
        raw / "gtFine" / "train" / train_city / "sample_gtFine_instanceIds.png",
        instance_ids,
    )

    # A polygon file must not affect the conversion anymore.
    polygon_path = raw / "gtFine" / "train" / train_city / "sample_gtFine_polygons.json"
    polygon_path.write_text(
        '{"imgWidth": 20, "imgHeight": 10, "objects": []}', encoding="utf-8"
    )

    _write_image(raw / "leftImg8bit" / "val" / val_city / "empty_leftImg8bit.png")
    _write_instance_ids(
        raw / "gtFine" / "val" / val_city / "empty_gtFine_instanceIds.png",
        np.zeros((10, 20), dtype=np.uint16),
    )

    output = tmp_path / "ultralytics-instance"
    stats = build_ultralytics_instance_dataset(
        raw,
        output,
        categories=["car", "truck"],
        category_map={"car": "vehicle", "truck": "vehicle"},
        splits=["train", "val"],
    )

    assert stats.image_count == 2
    assert stats.instance_count == 2
    assert stats.skipped_instances == 0
    assert stats.splits == ("train", "val")
    assert stats.names == ("vehicle",)

    output_image = output / "images" / "train" / train_city / "sample_leftImg8bit.png"
    assert output_image.is_symlink()
    assert not os.readlink(output_image).startswith("/")
    assert (
        output_image.resolve()
        == (
            raw / "leftImg8bit" / "train" / train_city / "sample_leftImg8bit.png"
        ).resolve()
    )

    label_path = output / "labels" / "train" / train_city / "sample_leftImg8bit.txt"
    assert label_path.read_text(encoding="utf-8").splitlines() == [
        "0 0.100000 0.100000 0.100000 0.400000 0.250000 0.400000 0.250000 0.100000",
        "0 0.400000 0.200000 0.400000 0.300000 0.450000 0.300000 0.450000 0.200000",
    ]

    empty_label = output / "labels" / "val" / val_city / "empty_leftImg8bit.txt"
    assert empty_label.read_text(encoding="utf-8") == ""

    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data == {
        "path": ".",
        "nc": 1,
        "names": ["vehicle"],
        "train": "images/train",
        "val": "images/val",
    }


def test_disconnected_parts_of_one_instance_are_written_as_one_segment(tmp_path):
    raw = tmp_path / "raw"
    _write_image(raw / "leftImg8bit" / "train" / "aachen" / "one_leftImg8bit.png")
    instance_ids = np.zeros((12, 24), dtype=np.uint16)
    instance_ids[1:4, 1:4] = 26000
    instance_ids[7:10, 18:22] = 26000
    _write_instance_ids(
        raw / "gtFine" / "train" / "aachen" / "one_gtFine_instanceIds.png",
        instance_ids,
    )

    stats = build_ultralytics_instance_dataset(
        raw, tmp_path / "output", categories=["car"], splits=["train"]
    )

    assert stats.instance_count == 1
    line = (
        tmp_path / "output" / "labels" / "train" / "aachen" / "one_leftImg8bit.txt"
    ).read_text()
    assert line.startswith("0 ")
    assert len(line.split()) > 1 + 4 * 2


def test_config_from_args_requires_cli_paths_and_reads_other_values(tmp_path):
    raw = tmp_path / "raw"
    _write_image(raw / "leftImg8bit" / "train" / "aachen" / "one_leftImg8bit.png")
    _write_instance_ids(
        raw / "gtFine" / "train" / "aachen" / "one_gtFine_instanceIds.png",
        np.zeros((4, 6), dtype=np.uint16),
    )

    params = tmp_path / "params.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                PARAM_NAME: {
                    "input_root": str(tmp_path / "wrong-input"),
                    "output_root": str(tmp_path / "from-config"),
                    "splits": ["train"],
                    "categories": ["person", "car"],
                    "category_map": {"car": "vehicle"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(params),
            "--input-root",
            str(raw),
            "--output-root",
            str(tmp_path / "from-cli"),
            "--categories",
            "car",
            "truck",
            "--category-map",
            "car=vehicle",
            "truck=vehicle",
        ]
    )
    config = config_from_args(args)

    assert config.input_root == raw
    assert config.output_root == tmp_path / "from-cli"
    assert config.splits == ("train",)
    assert config.categories == ("car", "truck")
    assert config.category_map == {"car": "vehicle", "truck": "vehicle"}
