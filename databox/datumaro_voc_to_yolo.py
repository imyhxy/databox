# Author: imyhxy
# File: datumaro_voc_to_yolo.py
# Date: 8/15/24
"""Converts a VOC dataset to YOLO-Ultralytics by datumaro.

A VOC dataset must have the following files:
    - Annotations
    - JPEGImages
    - ImageSets
    - labelmap.txt
"""
import argparse

from datumaro import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to VOC directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # required `labelmap.txt` and `ImageSets`
    ds = Dataset.import_from(
        path=args.input_dir,
        format="voc_detection",
    )
    ds = ds.transform(
        "random_split", splits=[("train", 0.8), ("val", 0.2)]
    )  # CLI option is subset
    ds.export(
        args.output_dir,
        format="yolo_ultralytics",
        save_media=True,
    )


if __name__ == "__main__":
    main()
