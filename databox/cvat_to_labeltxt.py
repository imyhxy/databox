# Author: fkwong
# File: cvat_to_labeltxt.py
# Date: 8/9/24
"""Convert annotation.xml into labeltxt format."""
import argparse
import os.path as osp
import xml.etree.ElementTree as ET
from pathlib import Path


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
        "--sort-classes", action="store_true", help="Sort classes by alphabetical order"
    )
    parser.add_argument(
        "--out-name", type=str, required=True, help="Name of output labels txt file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    tree = ET.parse(args.annotations)

    categories = [x.text for x in tree.findall(".//label/name")]
    if args.sort_classes:
        categories = sorted(categories)

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
        idx = categories.index(name)

        label = f"{path} {idx}"
        labels.append(label)

    with Path(args.annotations).with_name(args.out_name).open("w") as f:
        f.write("\n".join(labels))


if __name__ == "__main__":
    main()
