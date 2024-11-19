# Author: fkwong
# File: cvat_to_folder.py
# Date: 11/14/24
"""Convert CVAT dataset to folder structure."""
import argparse

import fiftyone as fo
import fiftyone.types as fot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat-dir", type=str, required=True, help="CVAT directory")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ds = fo.Dataset.from_dir(
        dataset_dir=args.cvat_dir,
        dataset_type=fot.CVATImageDataset,
        data_path="images",
        labels_path="annotations.xml",
        progress=True,
    )

    for sample in ds.iter_samples(autosave=True):
        if sample["classifications"] is not None:
            sample["classification"] = sample["classifications"]["classifications"][0]

    ds.export(
        export_dir=args.output_dir,
        dataset_type=fot.ImageClassificationDirectoryTree,
        label_field="classification",
        progress=True,
    )


if __name__ == "__main__":
    main()
