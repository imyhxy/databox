# Author: imyhxy
# File: normalize_filename.py
# Date: 11/25/23
import argparse
from pathlib import Path

from utils import search_images


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

    paths = search_images(args.root)

    for p in paths:
        path = Path(p)
        path.rename(path.with_stem(path.stem.replace(" ", "_")))


if __name__ == "__main__":
    main()
