# Author: imyhxy
# File: yolo_patch.py
# Date: 6/6/24
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Path to YOLO images directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to patch output directory"
    )
    parser.add_argument(
        "--class-indexes",
        type=str,
        nargs="+",
        help="save multiply index into a directory, i.e.: cat:1,4,5 dog:0,2,3",
    )
    parser.add_argument(
        "--expand",
        type=float,
        default=0,
        help="Expand factor apply for the bounding box",
    )
    parser.add_argument(
        "--action",
        choices=["show", "crop"],
        default="show",
        help="Action to be performed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    total = 0
    jpgs = list(img_dir.glob("**/*.jpg"))

    idx2cat = {}
    for ai in args.class_indexes:
        cat, idxs = ai.split(":")
        idx2cat.update({int(i): cat for i in idxs.split(",")})

    for img in tqdm(jpgs, total=len(jpgs)):
        lab = Path(str(img.with_suffix(".txt")).replace("/images/", "/labels/"))
        if not lab.exists():
            continue
        boxes = np.loadtxt(lab, delimiter=" ", dtype=np.float64, ndmin=2)

        if len(boxes) > 0:
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
            boxes[:, 3:] *= 1 + args.expand
            boxes[:, 1:3] -= boxes[:, 3:5] / 2
            boxes[:, 3:5] += boxes[:, 1:3]
            np.clip(boxes[:, 1:5:2] * w, 0, w, boxes[:, 1:5:2])
            np.clip(boxes[:, 2:5:2] * h, 0, h, boxes[:, 2:5:2])

            if args.action == "show":
                for box in boxes:
                    im = cv2.rectangle(
                        im,
                        (int(box[1]), int(box[2])),
                        (int(box[3]), int(box[4])),
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow("", im)
                k = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                if k == ord("q"):
                    break
            else:
                for idx, box in enumerate(boxes):
                    cat = idx2cat.get(int(box[0]))
                    if cat is None:
                        continue  # not required categories
                    patch = im[int(box[2]) : int(box[4]), int(box[1]) : int(box[3])]
                    (out_dir / cat).mkdir(exist_ok=True)
                    cv2.imwrite(str(out_dir / cat / f"{lab.stem}_{idx:02d}.jpg"), patch)
                    total += 1

    print(f"Total patches: {total}")


if __name__ == "__main__":
    main()
