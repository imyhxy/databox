# Author: imyhxy
# File: split_labeltxt.py
# Date: 12/5/23
"""
Split label text file created by anno2labeltxt.py into non-overlapping splits.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from utils.general import calculate_split_ratios


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-txt",
        type=str,
        required=True,
        help="Path to label text file, each line: '/path/to/image cid'",
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="8:1:1",
        help="A colon separated list of weights of splits",
    )

    return parser.parse_args()


def split_labeltxt(
    path: str,
    split_ratios: list[float],
):
    data = defaultdict(list)
    with open(path) as f:
        for item in f:
            cid = item.strip().split()[1]
            data[cid].append(item)

    sets = [[] for _ in split_ratios]
    for cid, labels in data.items():
        prev = 0
        labels = sorted(labels)
        total = len(labels)
        for idx, ratio in enumerate(split_ratios):
            curr = int(total * ratio)
            sets[idx].extend(p for p in labels[prev:curr])
            prev = curr

    return sets


def main():
    args = parse_args()

    split_ratios = calculate_split_ratios(args.split_ratios)
    splits = split_labeltxt(args.label_txt, split_ratios)

    preset_names = ["train", "val", "test"]
    for idx, split in enumerate(splits):
        if idx < 3:
            name = preset_names[idx]
        else:
            name = f"split{idx}"

        with Path(args.label_txt).with_stem(name).open("w") as f:
            f.writelines(split)


if __name__ == "__main__":
    main()
