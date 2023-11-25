# Author: imyhxy
# File: convert.py
# Date: 11/8/23
import argparse

import fiftyone as fo
import fiftyone.types as fot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-if",
        "--input-format",
        type=str,
        default="CVATImageDataset",
        help="Input dataset format",
    )
    parser.add_argument(
        "-of",
        "--output-format",
        type=str,
        default="ImageClassificationDirectoryTree",
        help="Output dataset format",
    )
    parser.add_argument(
        "-dp", "--data-path", type=str, required=True, help="Path to input directory"
    )
    parser.add_argument(
        "-lp", "--label-path", type=str, required=True, help="Path to input labels"
    )
    parser.add_argument(
        "-op", "--output-path", type=str, required=True, help="Output directory"
    )

    return parser.parse_args()


def main():
    opt = parse_args()

    dataset = fo.Dataset.from_dir(
        dataset_type=getattr(fot, opt.input_format),
        data_path=opt.data_path,
        labels_path=opt.label_path,
    )

    for sample in dataset.iter_samples(autosave=True):
        sample["light_status"] = fo.Classification(
            label=sample["classifications"].classifications[0]["led"]
        )

    dataset.export(
        export_dir=opt.output_path,
        label_field="light_status",
        dataset_type=getattr(fot, opt.output_format),
        export_media=True,
    )


if __name__ == "__main__":
    main()
