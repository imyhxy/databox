# Author: imyhxy
# File: merge_imgdir_into_fiftyone.py
# Date: 7/25/25
import argparse
from itertools import chain

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.types as fot

parser = argparse.ArgumentParser()


def parse_args():
    parser.add_argument(
        "--fiftyone-dataset",
        type=str,
        required=True,
        help="Paths to the fiftyone dataset directory",
    )
    parser.add_argument(
        "--image-directories",
        nargs="+",
        type=str,
        required=True,
        help="Image directories",
    )
    parser.add_argument("--tags", nargs="+", type=str, help="Tags for the dataset")
    parser.add_argument("--output", type=str, help="Fiftyone dataset output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    assert len(args.image_directories) == len(args.tags)

    main_dataset = fo.Dataset.from_dir(
        dataset_dir=args.fiftyone_dataset,
        dataset_type=fot.FiftyOneDataset,
    )

    for imgdir, tag in zip(args.image_directories, args.tags):
        dataset = fo.Dataset.from_dir(
            dataset_dir=imgdir,
            dataset_type=fot.ImageClassificationDirectoryTree,
            label_field="ground_truth",
            tags=tag,
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
