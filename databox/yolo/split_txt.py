# Author: imyhxy
# File: split_txt.py
# Date: 1/24/24
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--train", type=float, default=0.9)

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input) as f:
        data = f.read().split()
    total = len(data)

    train_num = int(total * args.train)
    train_set = data[:train_num]
    val_set = data[train_num:]

    train_file = Path(args.input).with_name("train.txt")
    val_file = Path(args.input).with_name("val.txt")

    with train_file.open("w") as f:
        f.write("\n".join(train_set))
    with val_file.open("w") as f:
        f.write("\n".join(val_set))


if __name__ == "__main__":
    main()
