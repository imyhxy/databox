# Author: fkwong
# File: merge_labeltxt.py
# Date: 8/15/24
"""Merge "train.txt", "test.txt" and "val.txt" files from inputs."""
import argparse
import os.path as osp
from math import sqrt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Root directory of list of homogeneous datasets",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Root directory of merged labels",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    for split in {"train", "test", "val"}:
        splits = []
        for inp in args.inputs:
            p = Path(inp) / f"{split}.txt"
            if p.is_file():
                splits.append(p)

        if len(splits) != 0:
            with open(Path(args.output) / f"{split}.txt", "w") as f:
                for file in splits:
                    with open(file) as q:
                        for line in q:
                            p, n = line.strip().split()
                            p = osp.join(osp.dirname(file), p)
                            p = osp.relpath(p, args.output)
                            f.write(
                                f"./{p.lstrip('./') if p.startswith('./') else p} {n}\n"
                            )

    cls_stat = {}
    with open(Path(args.output) / "train.txt") as f:
        for p in f:
            key = p.split()[1]
            cls_stat[key] = cls_stat.get(key, 0) + 1

    total = sum(cls_stat.values())
    class_num = len(cls_stat)
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
