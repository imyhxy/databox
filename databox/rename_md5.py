# Author: imyhxy
# File: rename_md5.py
# Date: 3/6/25
"""Rename files to their md5 hash."""
import subprocess
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--sources", type=str, required=True, help="path to files to be rename"
    )

    args = parser.parse_args()

    sources = Path(args.sources)
    files = []
    if sources.is_dir():
        files.extend(sources.rglob("*.*"))

    for file in files:
        ret = subprocess.check_output(["md5sum", str(file)])
        md5_hash = ret.decode().split()[0]
        suffix = file.suffix.lower()
        new_file = file.parent / f"{md5_hash}{suffix}"
        print(file, new_file)
        file.rename(new_file)


if __name__ == "__main__":
    main()
