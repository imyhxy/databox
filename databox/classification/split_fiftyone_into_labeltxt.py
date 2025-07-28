# Author: imyhxy
# File: split_fiftyone.py
# Date: 7/28/25
import argparse
import os
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fiftyone-dataset",
        type=str,
        required=True,
        help="Path to the FiftyOne dataset directory",
    )
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="List of ordered categories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for train and val datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the FiftyOne dataset
    dataset = fo.Dataset.from_dir(
        dataset_dir=args.fiftyone_dataset,
        dataset_type=fot.FiftyOneImageClassificationDataset,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    categories = args.categories

    x, y = [], []
    for sample in dataset.iter_samples():
        cat_idx = categories.index(sample["ground_truth"].label)
        relative_path = os.path.relpath(sample.filepath, out_dir)
        x.append(f"{relative_path} {cat_idx}")
        y.append(f"{sample.tags[0] if sample.tags else ''}-{cat_idx}")

    count_values = {k: y.count(k) for k in set(y)}
    unique_values = [k for k, v in count_values.items() if v == 1]
    dup_values = [k for k, v in count_values.items() if v > 1]
    for n, i in enumerate(y):
        if i in unique_values:
            y[n] = unique_values[0] if len(unique_values) > 1 else dup_values[0]

    train = args.train
    seed = args.seed

    x_train, x_val = train_test_split(
        x, train_size=train, random_state=seed, shuffle=True, stratify=y
    )

    splits = [x_train, x_val]
    preset_names = ["train", "val"]
    for idx, split in enumerate(splits):
        if idx < 3:
            name = preset_names[idx]
        else:
            name = f"split{idx}"

        with open(out_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(split))


if __name__ == "__main__":
    main()
