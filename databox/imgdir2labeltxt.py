# Author: imyhxy
# File: imgdir2labeltxt.py
# Date: 12/14/23
import argparse
from pathlib import Path

from databox.utils.search_images import search_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", "-r", type=str, required=True, help="Root of the dataset"
    )
    parser.add_argument(
        "--categories",
        "-c",
        type=str,
        default=None,
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

    if args.categories is not None:
        categories = {c: n for n, c in enumerate(args.categories)}
    else:
        sub_dirs = set()
        for split in {"train", "test", "val"}:
            sub_dirs.update(
                i.stem for i in (Path(args.root) / split).glob("*") if i.is_dir()
            )
        categories = {c: n for n, c in enumerate(sorted(sub_dirs))}

    for split in {"train", "test", "val"}:
        sub_dir = Path(args.root) / split
        dataset = []
        for c, n in categories.items():
            cls_dir = sub_dir / c
            images = search_images(cls_dir)
            dataset.extend(i + f" {n}\n" for i in images)

        if args.relative:
            dataset = [
                x[len(args.root) :].lstrip("/") if x.startswith(args.root) else x
                for x in dataset
            ]

        if len(dataset) != 0:
            with (Path(args.root) / f"{split}.txt").open("w") as f:
                f.writelines(dataset)


if __name__ == "__main__":
    main()
