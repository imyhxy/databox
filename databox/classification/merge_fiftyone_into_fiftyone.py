# Author: fkwong
# File: merge_fiftyone_into_fiftyone.py
# Date: 8/11/25
import argparse
from itertools import chain

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.types as fot

parser = argparse.ArgumentParser()


def parse_args():
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Paths to the fiftyone dataset directory",
    )
    parser.add_argument(
        "--add-dirs",
        nargs="+",
        type=str,
        required=True,
        help="Image directories",
    )
    parser.add_argument("--output", type=str, help="Fiftyone dataset output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    main_dataset = fo.Dataset.from_dir(
        dataset_dir=args.base_dir,
        dataset_type=fot.FiftyOneDataset,
    )

    for imgdir in args.add_dirs:
        dataset = fo.Dataset.from_dir(
            dataset_dir=imgdir,
            dataset_type=fot.FiftyOneDataset,
        )
        main_dataset.add_samples(dataset)

    duplicates_map = fob.compute_exact_duplicates(main_dataset)
    duplicates = list(chain(*duplicates_map.values()))
    if duplicates:
        print(f"Found {len(duplicates)} duplicate samples, removing them.")
        main_dataset.delete_samples(duplicates)

    if args.output:
        main_dataset.export(
            export_dir=args.output,
            dataset_type=fot.FiftyOneDataset,
        )


if __name__ == "__main__":
    main()
