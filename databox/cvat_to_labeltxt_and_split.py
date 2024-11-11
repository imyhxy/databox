# Author: fkwong
# File: cvat_to_labeltxt_and_split.py
# Date: 11/11/24
"""Convert annotation.xml into labeltxt format and split into train and val."""
import argparse
import os.path as osp
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=str, help="Path to the annotation file")
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--yaml", action="store_true", help="use params.yaml")

    exclusive_group.add_argument("--seed", type=int, help="random seed")
    parser.add_argument(
        "--categories", type=str, nargs="+", help="List of ordered categories"
    )
    parser.add_argument("--train", type=float, default=0.8, help="train split ratio")

    parser.add_argument(
        "--strip", type=str, default=None, help="Prefix used to strip image path"
    )
    parser.add_argument(
        "--prefix", type=str, default=None, help="Prefix add to path after strip"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.yaml:
        params = yaml.safe_load(open("params.yaml"))["cvat_to_labeltxt_and_split"]
        train = params["train"]
        seed = params["seed"]
        categories = params["categories"]
    else:
        train = args.train
        seed = args.seed
        categories = args.categories

    assert 0 < train < 1, f"train ratio must be between 0 and 1, got {train}"

    tree = ET.parse(args.annotations)

    dc = [x.text for x in tree.findall(".//label/name")]
    assert all(c in categories for c in dc), "categories mismatch"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp_dir = Path(args.annotations).parent

    labels = []
    for img in tree.findall("./image"):
        path = img.attrib["name"]
        if args.strip and path.startswith(args.strip):
            removed_len = len(args.strip)
            path = path[removed_len:]
        if args.prefix is not None:
            path = osp.join(args.prefix, path)
        path = inp_dir / path

        path = osp.relpath(path, out_dir)

        tag = img.find("tag")
        if tag is None:
            continue

        name = tag.attrib["label"]
        idx = categories.index(name)

        label = f"{path} {idx}"
        labels.append(label)

    y = [int(r.rsplit(maxsplit=1)[1]) for r in labels]
    x_train, x_val, y_train, y_val = train_test_split(
        labels, y, train_size=train, random_state=seed, shuffle=True, stratify=y
    )

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
