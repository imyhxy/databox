#!/usr/bin/env python3
# Author: imyhxy
# File: create_missed_voc_mask.py
# Date: 12/11/23
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    assert len(sys.argv) == 2, sys.argv

    os.chdir(sys.argv[1])

    imgs = list(Path("JPEGImages").glob("*"))
    for img in tqdm(imgs, total=len(imgs)):
        save_path = f"SegmentationClass/{img.stem}.png"
        if os.path.exists(save_path):
            continue

        im = cv2.imread(str(img))
        m = np.zeros(im.shape[:2], dtype=np.uint8)
        cv2.imwrite(f"SegmentationClass/{img.stem}.png", m)


if __name__ == "__main__":
    main()
