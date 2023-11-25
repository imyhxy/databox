# Author: imyhxy
# File: search_images.py
# Date: 11/25/23
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
