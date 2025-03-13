# Author: imyhxy
# File: resave_images.py
# Date: 3/6/25
"""Open and save images to remove any metadata."""
from argparse import ArgumentParser
from pathlib import Path

import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--sources", type=str, required=True, help="path to files to be rename"
    )
    parser.add_argument("--suffix", default=None, type=str, help="save as suffix")

    args = parser.parse_args()

    sources = Path(args.sources)
    files = []
    if sources.is_dir():
        files.extend(sources.rglob("*"))

    for file in files:
        suffix = file.suffix.lower()
        if suffix in {".jpg", ".png", ".jpeg"}:
            img = cv2.imread(str(file))
            if img is None:
                print(file)
            else:
                if args.suffix is not None:
                    file = file.with_suffix(args.suffix)
                cv2.imwrite(str(file), img)


if __name__ == "__main__":
    main()
