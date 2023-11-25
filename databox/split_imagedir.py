# Author: imyhxy
# File: split_imagedir.py
# Date: 11/25/23
import argparse
import os.path as osp
from glob import glob
from itertools import chain


def search_images(root):
    exts = {".jpg", ".png", ".jpeg"}
    return sorted(
        [
            x
            for x in chain(
                *[
                    glob(osp.join(root, "**", f"*{x}"), recursive=True)
                    for x in chain(*[(e.lower(), e.upper()) for e in exts])
                ]
            )
            if "outputs" not in x
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root directory")
    parser.add_argument(
        "--categories",
        type=str,
        required=True,
        help="A comma-separated list of ordered categories used for filtering directory and creating labels",
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="8:1:1",
        help="A colon separated list of weights of train, val, test split",
    )

    return parser.parse_args()


def split_imagedir(
    root: str,
    categories: list[str],
    split_ratios: list[float],
):
    sets = [[] for _ in split_ratios]
    for cls_idx, cls_name in enumerate(categories):
        paths = search_images(osp.join(root, cls_name))
        total = len(paths)

        prev = 0
        for idx, ratio in enumerate(split_ratios):
            curr = int(total * ratio)
            sets[idx].extend(
                p.replace(root, "").lstrip("/") + f" {cls_idx}\n"
                for p in paths[prev:curr]
            )
            prev = curr

    return sets


def main():
    opt = parse_args()

    split_ratios = [float(x) for x in opt.split_ratios.split(":")]
    for i in range(1, len(split_ratios)):
        split_ratios[i] = split_ratios[i - 1] + split_ratios[i]
    split_ratios = [x / split_ratios[-1] for x in split_ratios]

    if opt.categories == "auto":
        categories = sorted(
            [osp.basename(x) for x in glob(osp.join(opt.root, "*")) if osp.isdir(x)]
        )
    else:
        categories = opt.categories.split(",")

    splits = split_imagedir(opt.root, categories, split_ratios)

    preset_names = ["train", "val", "test"]
    for idx, split in enumerate(splits):
        if idx < 3:
            name = preset_names[idx]
        else:
            name = f"split{idx}"

        with open(osp.join(opt.root, f"{name}.txt"), "w") as f:
            f.writelines(split)

    with open(osp.join(opt.root, "names.txt"), "w") as f:
        f.write("\n".join(categories))


if __name__ == "__main__":
    main()
