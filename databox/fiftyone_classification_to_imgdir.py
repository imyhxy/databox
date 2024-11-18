# Author: fkwong
# File: fiftyone_classification_to_imgdir.py
# Date: 11/18/24
import argparse

import fiftyone as fo
import fiftyone.types as fot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to the fiftyone classification format dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to the output directory",
    )
    args = parser.parse_args()

    ds = fo.Dataset.from_dir(
        dataset_dir=args.input,
        dataset_type=fot.FiftyOneImageClassificationDataset,
    )

    ds.export(
        export_dir=args.output,
        dataset_type=fot.ImageClassificationDirectoryTree,
        label_field="ground_truth",
        progress=True,
    )


if __name__ == "__main__":
    main()
