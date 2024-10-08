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
            continue

        ds = fo.Dataset.from_dir(
            dataset_dir=input_dir,
            dataset_type=fot.YOLOv5Dataset,
            split=split,
            progress=True,
            include_all_data=True,
        )
        mapping = {
            "car": "vehicle",
            "truck": "vehicle",
            "bus": "vehicle",
        }

        classes = ["vehicle"]

        view = ds.map_labels("ground_truth", mapping)

        view = view.filter_labels(
            "ground_truth",
            F("label").is_in(classes),
            only_matches=False,
        )

        view.export(
            export_dir=output_dir,
            dataset_type=fot.YOLOv5Dataset,
            split=split,
            yaml_path="dataset.yaml",
            export_media="symlink",
            classes=classes,
        )

    src_images = os.path.join(input_dir, "images")
    dst_images = os.path.join(output_dir, "images")
    shutil.rmtree(dst_images)
    os.makedirs(dst_images, exist_ok=True)
    assert os.path.isdir(src_images), "Source directory does not exists."

    for root, _, files in os.walk(src_images, followlinks=True):
        dst_root = os.path.join(dst_images, os.path.relpath(root, src_images))
        os.makedirs(dst_root, exist_ok=True)

        for f in files:
            src_f = os.path.join(src_images, root, f)
            dst_f = os.path.join(dst_root, f)
            os.symlink(os.path.relpath(src_f, dst_root), dst_f)


def main():
    args = parse_args()

    assert Path(args.input_dir).exists()
    assert Path(args.input_dir) != Path(args.output_dir)

    if args.batch:
        for inp_dir in Path(args.input_dir).glob("*"):
            if inp_dir.is_dir():
                out_dir = Path(args.output_dir) / inp_dir.name
                try:
                    filter_yolov5(str(inp_dir), str(out_dir))
                except AssertionError as e:
                    print(e.args)
                    pass
    else:
        filter_yolov5(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
