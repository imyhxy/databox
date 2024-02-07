# Author: imyhxy
# File: filter_yolov5.py
# Date: 2/6/24

import argparse
import os.path
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
import yaml
from fiftyone import ViewField as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=str, required=True, help="YOLOv5 dataset directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Treat subdirectories as YOLOv5 dataset"
    )

    return parser.parse_args()


def print_dataset(ds):
    print(ds.count())
    print(ds.count_values("ground_truth.detections.label"))


def filter_yolov5(input_dir: str, output_dir: str):
    assert not Path(output_dir).exists(), f"Directory {output_dir} already exists."
    with (Path(input_dir) / "dataset.yaml").open() as f:
        config = yaml.safe_load(f)

    for split in {"train", "test", "val"}:
        if split not in config:
            print(f"Skipping {split} for {input_dir}")
            continue

        ds = fo.Dataset.from_dir(
            dataset_dir=input_dir,
            dataset_type=fot.YOLOv5Dataset,
            split=split,
        )
        mapping = {
            "person": "person",
            "car": "vehicle",
            "truck": "vehicle",
            "bus": "vehicle",
        }

        view = ds.map_labels("ground_truth", mapping)

        view = view.filter_labels(
            "ground_truth.detections",
            F("label").is_in(set(mapping.values())),
            only_matches=False,
        )

        view.export(
            export_dir=output_dir,
            dataset_type=fot.YOLOv5Dataset,
            split=split,
            yaml_path="dataset.yaml",
            export_media="symlink",
        )

        shutil.rmtree(os.path.join(output_dir, "images"))

    src_images = os.path.join(input_dir, "images")
    dst_images = os.path.join(output_dir, "images")
    assert os.path.isdir(src_images), "Source directory does not exists."
    os.symlink(os.path.relpath(src_images, output_dir), dst_images)


def main():
    args = parse_args()

    assert Path(args.input_dir).exists()
    assert Path(args.input_dir) != Path(args.output_dir)

    if args.batch:
        for inp_dir in Path(args.input_dir).glob("*"):
            if inp_dir.is_dir():
                out_dir = Path(args.output_dir) / inp_dir.name
                filter_yolov5(str(inp_dir), str(out_dir))
    else:
        filter_yolov5(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
