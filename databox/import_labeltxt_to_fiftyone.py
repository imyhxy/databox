# Author: fkwong
# File: import_coco_to_fiftyone.py
# Date: 3/19/25
import argparse
from pathlib import Path

import fiftyone as fo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", type=str, required=True, help="Path to label text"
    )
    parser.add_argument("--name", type=str, required=True, help="Name of the dataset")
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        required=True,
        help="List of ordered categories",
    )
    args = parser.parse_args()

    dataset = fo.Dataset(name=args.name, persistent=True)
    for input_path_str in args.inputs:
        input_path = Path(input_path_str).resolve()
        parent_dir = input_path.parent
        try:
            with input_path.open("r") as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parts = line.split(maxsplit=1)
                        if len(parts) != 2:
                            raise ValueError(
                                f"Line does not contain exactly two elements: '{line}'"
                            )
                        rel_filepath, label = parts
                        try:
                            label_idx = int(label)
                        except ValueError:
                            raise ValueError(f"Label '{label}' is not a valid integer.")
                        if not (0 <= label_idx < len(args.categories)):
                            raise ValueError(
                                f"Label index {label_idx} out of range for categories."
                            )
                        file_path_obj = Path(rel_filepath)
                        if not file_path_obj.is_absolute():
                            file_path_obj = (parent_dir / file_path_obj).resolve()
                        sample = fo.Sample(filepath=str(file_path_obj))
                        sample["ground_truth"] = fo.Classification(
                            label=args.categories[label_idx]
                        )
                        sample["tags"] = [input_path.stem]
                        dataset.add_sample(sample)
                    except ValueError as e:
                        print(
                            f"Value error parsing line {line_num} in {input_path}: {e}"
                        )
                    except Exception as e:
                        print(
                            f"Unexpected error parsing line {line_num} in {input_path}: {e}"
                        )
        except FileNotFoundError:
            print(f"Input file not found: {input_path}")
        except Exception as e:
            print(f"Error reading file {input_path}: {e}")

    dataset.save()


if __name__ == "__main__":
    main()
