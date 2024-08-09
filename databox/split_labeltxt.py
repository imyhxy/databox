# Author: imyhxy
# File: split_labeltxt.py
# Date: 12/5/23
"""Split label text file into stratified shuffle split."""
import argparse
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--yaml", action="store_true", help="use params.yaml")
    exclusive_group.add_argument("--seed", type=int, help="random seed")

    parser.add_argument("--train", type=float, default=0.8, help="train split ratio")
    parser.add_argument("--val", type=float, default=0.2, help="validation split ratio")

    parser.add_argument(
        "--label-txt",
        type=str,
        required=True,
        help="Path to label text file, each line: '/path/to/image cid'",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.yaml:
        params = yaml.safe_load(open("params.yaml"))["split_labeltxt"]
        train = params["train"]
        seed = params["seed"]
    else:
        train = args.train
        seed = args.seed

    assert 0 < train < 1, f"train ratio must be between 0 and 1, got {train}"

    text = [line.strip() for line in open(args.label_txt).readlines() if line.strip()]
    y = [int(r.rsplit(maxsplit=1)[1]) for r in text]

    x_train, x_val, y_train, y_val = train_test_split(
        text, y, train_size=train, random_state=seed, shuffle=True, stratify=y
    )

    splits = [x_train, x_val]
    preset_names = ["train", "val"]
    for idx, split in enumerate(splits):
        if idx < 3:
            name = preset_names[idx]
        else:
            name = f"split{idx}"

        with Path(args.label_txt).with_stem(name).open("w") as f:
            f.write("\n".join(split))


if __name__ == "__main__":
    main()
