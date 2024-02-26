# Author: imyhxy
# File: deduplicated.py
# Date: 2/26/24
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels", type=str, required=True, help="Path to labels file/directory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't actually write")

    return parser.parse_args()


def main():
    args = parse_args()
    paths = []
    p = Path(args.labels)
    if p.is_file():
        paths.append(p)
    elif p.is_dir():
        paths.extend(list(p.glob("**/*.txt")))
    else:
        raise AssertionError()

    for path in paths:
        with path.open() as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        lb = np.array(lb, dtype=np.float32)
        nl = len(lb)
        _, i = np.unique(lb, axis=0, return_index=True)
        if len(i) < nl:
            lb = lb[i]
            print(f"WARNING {path}: {nl - len(i)} duplicate labels removed")
            if not args.dry_run:
                np.savetxt(path, lb, fmt="%d" + " %.8f" * 4)


if __name__ == "__main__":
    main()
