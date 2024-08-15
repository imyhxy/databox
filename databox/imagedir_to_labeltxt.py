# Author: fkwong
# File: imagedir_to_labeltxt.py
# Date: 8/15/24
"""Convert a yolov5 classification style dataset to a mmlab style dataset.

YOLOv5-style dataset:
root/
    train/
        cls1/
        cls2/
    val/
        cls1/
        cls2/

mmlab-style dataset:
root/
    dir1/
    dir2/
    train.txt
    val.txt
"""
import argparse
import os
from pathlib import Path

from databox.utils.search_images import search_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root of the dataset")
    parser.add_argument(
        "--categories",
        "-c",
        type=str,
        required=True,
        nargs="+",
        help="List of ordered categories",
    )

    parser.add_argument(
        "--relative",
        action="store_true",
        help="Store relative path to root in label files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    categories = {c: n for n, c in enumerate(args.categories)}
    root_dir = Path(args.root)
    dataset = []
    for c, n in categories.items():
        cls_dir = root_dir / c
        images = search_images(cls_dir)
        for img in images:
            if args.relative:
                img = os.path.relpath(img, root_dir)
            dataset.append("./" + img.rstrip("./") + f" {n}\n")

    if len(dataset) != 0:
        with (Path(args.root) / "data.txt").open("w") as f:
            f.writelines(dataset)


if __name__ == "__main__":
    main()
