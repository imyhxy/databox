# Author: imyhxy
# File: create_missing_empty.py
# Date: 2/6/24
import argparse
from pathlib import Path

from utils.search_images import search_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Path to the YOLO dataset images directory",
    )

    return parser.parse_args()


def create_missing_empty(image_dir):
    images = search_images(image_dir)

    for image in images:
        label = Path(image.replace("/images/", "/labels/")).with_suffix(".txt")
        if label.exists():
            continue

        with label.open("a") as _:
            pass


def main():
    opt = parse_args()
    create_missing_empty(opt.image_dir)


if __name__ == "__main__":
    main()
