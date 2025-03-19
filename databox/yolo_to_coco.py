# Author: imyhxy
# File: yolo_to_coco.py
# Date: 3/19/25
import argparse
import json
import os.path as osp

import fiftyone as fo
import fiftyone.types as fot
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", type=str, required=True, help="Path to YOLO dataset"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()
    ds = fo.Dataset()

    for inp in args.inputs:
        for split in ["train", "val"]:
            ds.add_dir(
                dataset_dir=inp,
                dataset_type=fot.YOLOv5Dataset,
                data_path="images",
                label_path="dataset.yaml",
                tags=[split],
                include_all_data=True,
                split=split,
            )

    labels = ds.default_classes
    out_dir = osp.abspath(args.output)
    ds.compute_metadata()
    with ds.save_context() as ctx:
        for sam in ds.select_fields("filepath").iter_samples(progress=True):
            sam.filepath = osp.relpath(sam.filepath, out_dir)
            ctx.save(sam)

    train_view, val_view = ds.match_tags("train"), ds.match_tags("val")
    print("Exporting dataset")
    for name, view in [("train", train_view), ("val", val_view)]:
        view.export(
            export_dir=out_dir,
            dataset_type=fot.COCODetectionDataset,
            labels_path=f"{name}.json",
            export_media=False,
            abs_paths=True,
            label_field="ground_truth",
            iscrowd="iscrowd",
            classes=labels,
        )

    manifest = {
        sam.filepath: osp.abspath(osp.join(out_dir, sam.filepath))
        for sam in ds.select_fields("filepath").iter_samples()
    }

    with open(osp.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(osp.join(out_dir, "label_distribution.yaml"), "w") as f:
        yaml.safe_dump(ds.count_values("ground_truth.detections.label"), f)


if __name__ == "__main__":
    main()
