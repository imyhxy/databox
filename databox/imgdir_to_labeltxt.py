# Author: fkwong
# File: imgdir_to_labeltxt.py
# Date: 12/14/23
"""Convert directory dataset into a openmm style dataset.

YOLOv5-style dataset:

root/
    train/
        cls1/
        cls2/
    val/
        cls1/
        cls2/

openmm-style dataset:

root/
    dir1/
    dir2/
    train.txt
    val.txt
"""
import argparse
import os.path as osp
from glob import glob
from itertools import chain
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


def search_images(root: str):
    exts = {".jpg", ".png", ".jpeg"}
    return sorted(
        [  # noqa: C416
            x
            for x in chain(
                *[
                    glob(osp.join(root, "**", f"*{x}"), recursive=True)
                    for x in chain(*[(e.lower(), e.upper()) for e in exts])
                ]
            )
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--yaml", action="store_true", help="use params.yaml")

    # command line parameters
    exclusive_group.add_argument("--seed", type=int, help="random seed")
    parser.add_argument(
        "--categories",
        "-c",
        type=str,
        default=None,
        nargs="+",
        help="List of ordered categories",
    )
    parser.add_argument(
        "--train", type=float, default=0.8, help="train subset fraction"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.yaml:
        params = yaml.safe_load(open("params.yaml"))["imgdir_to_labeltxt"]
        train = params["train"]
        seed = params["seed"]
        categories = params["categories"]
    else:
        train = args.train
        seed = args.seed
        categories = args.categories

    inp_dir = Path(args.input)
    out_dir = Path(args.output)
    x = []
    y = []
    for idx, name in enumerate(categories):
        cls_dir = inp_dir / name
        images = search_images(cls_dir)
        x.extend(f"{i} {idx}" for i in images)
        y.extend([idx] * len(images))

    x = [osp.relpath(i, args.output) for i in x]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=train, random_state=seed, shuffle=True, stratify=y
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [x_train, x_val]
    preset_names = ["train", "val"]
    for idx, split in enumerate(splits):
        if idx < 3:
            name = preset_names[idx]
        else:
            name = f"split{idx}"

        with open(out_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(split))


if __name__ == "__main__":
    main()
