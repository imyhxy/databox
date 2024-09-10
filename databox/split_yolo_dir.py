# Author: fkwong
# File: split_yolo_dir.py
# Date: 8/14/24
"""Split YOLO dataset into train and val set."""
import argparse
import os.path as osp
import random
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--yaml", action="store_true", help="use params.yaml")
    exclusive_group.add_argument("--seed", type=int, help="random seed")

    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Path to multiply YOLO dataset root",
    )
    parser.add_argument(
        "--train", type=float, default=0.8, help="validation split ratio"
    )
    args = parser.parse_args()

    if args.yaml:
        params = yaml.safe_load(open("params.yaml"))["split_yolo_dir"]
        train = params["train"]
        seed = params["seed"]
    else:
        train = args.train
        seed = args.seed

    assert 0 < train < 1, f"train ratio must be between 0 and 1, got {train}"

    for inp in args.inputs:
        random.seed(seed)
        inp_dir = Path(inp)
        jpegs = sorted((inp_dir / "images").glob("*.jpg"))
        jpegs = [osp.relpath(str(j), str(inp_dir)) for j in jpegs]
        jpegs = [f"./{j}" if not j.startswith("./") else j for j in jpegs]
        random.shuffle(jpegs)
        train_num = int(train * len(jpegs))
        with open(inp_dir / "train.txt", "w") as f:
            f.write("\n".join(jpegs[:train_num]))
        with open(inp_dir / "val.txt", "w") as f:
            f.write("\n".join(jpegs[train_num:]))


if __name__ == "__main__":
    main()
