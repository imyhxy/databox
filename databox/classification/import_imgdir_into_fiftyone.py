# Author: imyhxy
# File: import_imgdir_to_fiftyone.py
# Date: 7/25/25
import argparse
import os

import fiftyone as fo
import fiftyone.types as fot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image directory",
    )
    parser.add_argument("--tag", type=str, help="Tag for the dataset")
    parser.add_argument("--output", type=str, help="Fiftyone dataset output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise ValueError(
            f"Input directory '{args.input}' does not exist. Please provide a valid path."
        )

    # Load the dataset from the input directory
    dataset = fo.Dataset.from_dir(
        dataset_dir=args.input,
        dataset_type=fot.ImageClassificationDirectoryTree,
        label_field="ground_truth",
        tags=args.tag,
    )

    # Save the dataset to FiftyOne
    dataset.persistent = True
    dataset.save()

    dataset.export(
        export_dir=args.output,
        dataset_type=fot.FiftyOneDataset,
    )


if __name__ == "__main__":
    main()
