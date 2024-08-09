# Author: imyhxy
# File: imagedir_to_cvat.py
# Date: 8/9/24
"""Convert dataset from Imagedir to CVAT format.

Origin format:
    root/
        cls_1/
            xxx.jpg
            xxx.jpg
        cls_2/
            xxx.jpg
            xxx.jpg

Target format:
    root/
        data/
            xxx.jpg
            xxx.jpg
            xxx.jpg
        annotations.xml
"""
import argparse

import fiftyone as fo
import fiftyone.types as fot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root of the dataset")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="List of ordered categories",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory of the CVAT dataset",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ds = fo.Dataset.from_dir(
        dataset_dir=args.root,
        dataset_type=fot.ImageClassificationDirectoryTree,
        recursive=True,
    )
    ds.default_classes = args.categories
    ds.export(
        export_dir=args.output,
        dataset_type=fot.CVATImageDataset,
        label_path="annotations.xml",
    )


if __name__ == "__main__":
    main()
