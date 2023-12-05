# Author: imyhxy
# File: anno2labeltxt.py
# Date: 11/30/23
import argparse
import os.path as osp
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("anno_file", type=str, help="Path to the annotation file")
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
        "--output", type=str, required=True, help="Path of output labels txt file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    tree = ET.parse(args.anno_file)

    categories = [x.text for x in tree.findall(".//label/name")]
    if args.sort_classes:
        categories = sorted(categories)

    labels = []
    for img in tree.findall("./image"):
        path = img.attrib["name"]
        if path.startswith(args.strip):
            removed_len = len(args.strip)
            path = path[removed_len:]

        if args.prefix is not None:
            path = osp.join(args.prefix, path)

        tag = img.find("tag")
        if tag is None:
            continue

        name = tag.attrib["label"]
        idx = categories.index(name)

        label = f"{path} {idx}\n"
        labels.append(label)

    with open(args.output, "w") as f:
        f.writelines(labels)


if __name__ == "__main__":
    main()
