# Author: fkwong
# File: cvat_attrdet_to_labeltxt.py
# Date: 8/21/24
"""Convert CVAT detection dataset into labeltxt format.

The CVAT object have different attributes, e.g.:

  <image id="2" name="00186ab5910f7e38b7f66645026d4445.jpg" width="800" height="450">
    <box label="manhole" source="manual" occluded="0" xtl="230.72" ytl="298.31" xbr="393.71" ybr="367.44" z_order="0">
      <attribute name="good">true</attribute>
      <attribute name="broken">false</attribute>
      <attribute name="lose">false</attribute>
      <attribute name="uncovered">false</attribute>
      <attribute name="circle">false</attribute>
    </box>
  </image>
"""
import argparse
import os.path as osp
from collections import OrderedDict
from pathlib import Path

import cv2
import fiftyone as fo
import fiftyone.types as fot
import fiftyone.utils.patches as fup
import yaml
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument("--yaml", action="store_true", help="use params.yaml")
    exclusive_group.add_argument("--seed", type=int, help="random seed")

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="path to the CVAT detection dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="path to the labeltxt dataset"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="only consider specified classes",
    )
    parser.add_argument(
        "--attr-index",
        type=str,
        nargs="+",
        help="attribute-index mappings, order matter",
    )
    parser.add_argument("--train", type=float, default=0.8, help="train split ratio")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="an optional expansion/contraction to apply to the patch before extracting it, \
                        in [-1, inf). If provided, the length and width of the box are expanded \
                        (or contracted, when alpha < 0) by (100 * alpha)%. For example, set alpha = 0.1 to expand the \
                        box by 10%, and set alpha = -0.1 to contract the box by 10%",
    )

    args = parser.parse_args()

    if args.yaml:
        params = yaml.safe_load(open("params.yaml"))["cvat_attrdet_to_labeltxt"]
    else:
        params = args

    seed = getattr(params, "seed")  # noqa: B009
    train = getattr(params, "train")  # noqa: B009
    alpha = getattr(params, "alpha")  # noqa: B009
    attr_index = getattr(params, "attr_index")  # noqa: B009
    classes = getattr(params, "classes")  # noqa: B009

    ais = OrderedDict()
    for ai in attr_index:
        attr, index = ai.split(":")
        ais[attr] = int(index)

    ds = fo.Dataset.from_dir(
        dataset_dir=args.input_dir,
        dataset_type=fot.CVATImageDataset,
        labels_path="annotations.xml",
        data_path="images",
    )

    out_dir = Path(args.output_dir)
    out_image_dir = out_dir / "images"

    out_image_dir.mkdir(parents=True, exist_ok=True)

    x = []
    y = []
    for sam in ds.iter_samples(progress=True):
        if sam.detections is None:
            continue

        im = cv2.imread(sam.filepath)
        for box in sam.detections.detections:
            if box.label not in classes:
                continue
            for k, v in ais.items():
                if box[k]:
                    patch = fup.extract_patch(im, detection=box, alpha=alpha)
                    save_path = str(out_image_dir / f"{box['id']}.jpg")
                    cv2.imwrite(save_path, patch)
                    x.append(osp.relpath(save_path, args.output_dir))
                    y.append(v)
                    break

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=train, random_state=seed, shuffle=True, stratify=y
    )

    splits = [(x_train, y_train), (x_val, y_val)]
    preset_names = ["train", "val"]
    for idx, split in enumerate(splits):
        if idx < 3:
            name = preset_names[idx]
        else:
            name = f"split{idx}"

        with open(out_dir / f"{name}.txt", "w") as f:
            f.write("\n".join([f"{m} {n}" for m, n in zip(*split)]))


if __name__ == "__main__":
    main()
