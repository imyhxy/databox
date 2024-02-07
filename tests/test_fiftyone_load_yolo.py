# Author: imyhxy
# File: test_fiftyone_load_yolo.py
# Date: 2/7/24

import fiftyone as fo
import fiftyone.types as fot


def test_fiftyone_load_yolo():
    fo.Dataset.from_dir(
        dataset_dir="/home/fkwong/datasets/r10_repo-det-mixup/10_fp-nansha",
        dataset_type=fot.YOLOv5Dataset,
        split="train",
    )
