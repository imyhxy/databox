# Author: fkwong
# File: cvat_to_labeltxt.py
# Date: 8/9/24
"""Convert annotation.xml into labeltxt format."""
import argparse
import os.path as osp
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=str, help="Path to the annotation file")
    parser.add_argument(
        "--strip", type=str, default=None, help="Prefix used to strip image path"
    )
    parser.add_argument(
        "--prefix", type=str, default=None, help="Prefix add to path after strip"
    )
    parser.add_argument(
        "--classes", type=str, required=True, nargs="+", help="classes order"
    )
    parser.add_argument(
        "--out-name", type=str, required=True, help="Name of output labels txt file"
    )

    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--yaml", action="store_true", help="use params.yaml")
    exclusive_group.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--train", type=float, default=0.8, help="train split ratio")

    return parser.parse_args()


def main():
    args = parse_args()
    tree = ET.parse(args.annotations)

    categories = [x.text for x in tree.findall(".//label/name")]
    if set(categories) != set(args.classes):
        exclude = set(categories) - set(args.classes)
        print(f"ignore classes: {exclude}")
    classes = args.classes

    labels = []
    for img in tree.findall("./image"):
        path = img.attrib["name"]
        if args.strip and path.startswith(args.strip):
            removed_len = len(args.strip)
            path = path[removed_len:]

        if args.prefix is not None:
            path = osp.join(args.prefix, path)

        tag = img.find("tag")
        if tag is None:
            continue

        name = tag.attrib["label"]
        idx = classes.index(name)

        label = f"{path} {idx}"
        labels.append(label)

    label_path = Path(args.annotations).with_name(args.out_name)
    with label_path.open("w") as f:
        f.write("\n".join(labels))

    if args.yaml:
        params = yaml.safe_load(open("params.yaml"))["split_labeltxt"]
        train = params["train"]
        seed = params["seed"]
    else:
        train = args.train
        seed = args.seed

    assert 0 < train < 1, f"train ratio must be between 0 and 1, got {train}"

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

        with label_path.with_stem(name).open("w") as f:
            f.write("\n".join(split))


if __name__ == "__main__":
    main()
