# Author: fkwong
# File: import_fiftyone.py
# Date: 7/29/25
from argparse import ArgumentParser

import fiftyone as fo
import fiftyone.types as fot


def parse_args():
    parser = ArgumentParser(description="Import FiftyOne dataset")
    parser.add_argument(
        "--fiftyone-dataset",
        type=str,
        required=True,
        help="Path to the FiftyOne dataset directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the FiftyOne dataset to create or update",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the FiftyOne dataset from the specified directory
    dataset = fo.Dataset.from_dir(
        dataset_dir=args.fiftyone_dataset,
        dataset_type=fot.FiftyOneDataset,
        name=args.name,
    )

    # Set the dataset to persistent mode
    dataset.persistent = True

    # Save the dataset
    dataset.save()

    print(f"FiftyOne dataset '{args.name}' imported successfully.")


if __name__ == "__main__":
    main()
