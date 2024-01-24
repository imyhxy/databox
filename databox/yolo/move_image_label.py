# Author: imyhxy
# File: move_image_label.py
# Date: 1/24/24
import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, required=True, help="files to be moved")
    parser.add_argument("--output", type=str, required=True, help="output directory")

    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output

    if os.path.exists(output):
        assert os.path.isdir(output), "output is not a directory"
    else:
        os.makedirs(output)

    with open(args.filelist) as f:
        for image_path in f:
            image_path = image_path.strip()
            if not os.path.exists(image_path):
                continue

            label_path = image_path.replace("/images/", "/labels/").replace(
                ".jpg", ".txt"
            )
            shutil.move(image_path, output)
            shutil.move(label_path, output)


if __name__ == "__main__":
    main()
