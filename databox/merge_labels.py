# Author: imyhxy
# File: merge_labels.py
# Date: 11/25/23
"""
Recursively search for subdirectories "train.txt", "test.txt" and "val.txt"
files and merge them according to the name.
"""
import argparse
import os.path as osp
from glob import glob
from math import sqrt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of list of homogeneous datasets",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    for split in {"train", "test", "val"}:
        splits = glob(osp.join(args.root, "**", f"{split}.txt"))
        if len(splits) != 0:
            with open(osp.join(args.root, f"{split}.txt"), "w") as f:
                for file in splits:
                    dirname = Path(file).parent
                    with open(file) as d:
                        labels = [osp.join(dirname, x) for x in d.readlines()]
                    f.writelines(labels)

    cls_stat = dict()
    with open(osp.join(args.root, "train.txt")) as f:
        for p in f:
            key = p.split()[1]
            cls_stat[key] = cls_stat.get(key, 0) + 1

    total = sum(cls_stat.values())
    class_num = 3
    print(
        "Inverse weight     :",
        [round(total / cls_stat[str(x)], 2) for x in range(class_num)],
    )
    print(
        "Inverse sqrt weight:",
        [round(sqrt(total) / sqrt(cls_stat[str(x)]), 2) for x in range(class_num)],
    )


if __name__ == "__main__":
    main()
