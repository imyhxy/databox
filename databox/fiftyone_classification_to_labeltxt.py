# Author: imyhxy
# File: fiftyone_classification_to_labeltxt.py
# Date: 8/20/24
"""Convert a fiftyone classification dataset to a labeltxt file."""
import argparse
import os.path as osp
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
import yaml
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=str,
        required=True,
        help="path to the fiftyone classification format dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="path to the output directory",
    )

    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--yaml", action="store_true", help="use params.yaml")
    exclusive_group.add_argument("--seed", type=int, help="random seed")

    parser.add_argument("--train", type=float, default=0.8, help="train split ratio")
    parser.add_argument(
        "--mappings",
        type=str,
        nargs="+",
        help="define old-new names mappings, oldname1:newname1, oldname2:newname2",
    )

    parser.add_argument("--categories", type=str, nargs="+", help="categories order")

    args = parser.parse_args()

    if args.yaml:
        params = yaml.safe_load(open("params.yaml"))[
            "fiftyone_classification_to_labeltxt"
        ]
        seed = params["seed"]
        train = params["train"]
        mappings = params["mappings"]
        categories = params["categories"]
    else:
        seed = args.seed
        train = args.train
        mappings = {k[0]: k[1] for k in [x.split(":") for x in args.mappings]}
        categories = args.categories

    assert set(categories) == set(mappings.values())

    ds = fo.Dataset()
    for inp in args.inputs:
        inp_dir = Path(inp)
        if (inp_dir / "labels.json").exists():
            ds.merge_dir(
                dataset_dir=inp,
                dataset_type=fot.FiftyOneImageClassificationDataset,
            )

    x = []
    y = []
    for sam in ds:
        fp = osp.relpath(sam.filepath, args.output_dir)
        x.append(fp)
        name = sam.ground_truth.label
        new_name = mappings[name]
        new_index = categories.index(new_name)
        y.append(new_index)

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

        with open(Path(args.input_dir) / f"{name}.txt", "w") as f:
            f.write("\n".join(f"{x} {y}" for x, y in zip(*split)))


if __name__ == "__main__":
    main()
