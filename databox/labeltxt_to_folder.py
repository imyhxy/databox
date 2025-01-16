# Author: fkwong
# File: labeltxt_to_folder.py
# Date: 1/16/25
"""Convert labeltxt dataset to folder structure."""
import argparse
from pathlib import Path

import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir", type=str, required=True, help="Root of labeltxt dataset directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for split in ["train", "val", "test"]:
        split_file = Path(args.root_dir) / f"{split}.txt"
        if not split_file.is_file():
            continue
        with open(split_file) as f:
            for line in tqdm.tqdm(f):
                p, n = line.strip().split()
                p = Path(args.root_dir) / p
                out_dir = Path(args.output_dir) / n
                out_dir.mkdir(parents=True, exist_ok=True)
                p.rename(out_dir / p.name)


if __name__ == "__main__":
    main()
