# Author: fkwong
# File: import_coco_to_fiftyone.py
# Date: 3/19/25
import argparse

import fiftyone as fo
import fiftyone.types as fot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to COCO dataset")
    parser.add_argument("--name", type=str, required=True, help="Name of the dataset")
    args = parser.parse_args()

    root_dir = args.input
    name = args.name
    ds = fo.Dataset(name=name, persistent=True)
    for split in {"train", "val"}:
        ds.add_dir(
            dataset_type=fot.COCODetectionDataset,
            dataset_dir=root_dir,
            data_path="manifest.json",
            labels_path=f"{split}.json",
            label_field="ground_truth",
            label_types=("detections",),
            progress=True,
        )

    with ds.save_context() as ctx:
        for sample in ds:
            suffix = sample.filepath.rsplit("/raw/", 1)[1]
            suffix = suffix.split("/", 1)[0]
            sample.tags.append(suffix)
            ctx.save(sample)
    ds.save()


if __name__ == "__main__":
    main()
