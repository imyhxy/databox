# Author: fkwong
# File: merge_cvatvoc_into_fiftyone.py
# Date: 8/18/25
import os.path as osp
from argparse import ArgumentParser

import fiftyone as fo
import fiftyone.types as fot


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--fiftyone-dir", type=str, default=None, help="Path to origin directory"
    )
    parser.add_argument(
        "--cvat-dirs",
        type=str,
        default=(),
        nargs="+",
        help="Path to multiple CVAT datasets",
    )
    parser.add_argument(
        "--voc-dirs",
        type=str,
        default=(),
        nargs="+",
        help="Path to multiple VOC datasets",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if osp.exists(args.output):
        raise ValueError(f"Output directory '{args.output}' already exists")

    if args.origin_dir is not None:
        ds = fo.Dataset.from_dir(
            dataset_dir=args.origin_dir,
            dataset_type=fot.FiftyOneDataset,
        )
    else:
        ds = fo.Dataset()

    for cv in args.cvat_dirs:
        tag = cv.split("/raw/")[1].split("/")[0]
        ds.add_dir(
            dataset_dir=cv,
            dataset_type=fot.CVATImageDataset,
            data_path="images",
            labels_path="annotations.xml",
            tags=[tag],
            include_all_data=False,
        )

    for voc in args.voc_dirs:
        tag = voc.split("/JPEGImages/")[0].rsplit("/", 1)[1]
        ds.add_dir(
            dataset_dir=osp.join(voc, "JPEGImages"),
            dataset_type=fot.ImageDirectory,
            tags=[tag],
        )

    ds.export(
        dataset_type=fot.FiftyOneDataset,
        export_dir=args.output,
        export_media=True,
    )


if __name__ == "__main__":
    main()
