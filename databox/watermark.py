#!/bin/env python3
# author: fkwong
# email: huangfuqiang@transai.cn
# date: Fri 16 Apr 2021 07:02:48 PM CST

import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from functools import lru_cache
from glob import glob

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
COLOR = (250, 250, 250)
TEXT = "深研人工智能技术（深圳）有限公司"
LOC = (200, 492)  # with respect to 1920x1080 (w, h)
FONTSIZE = 96  # with respect to 1920x1080 (w, h)
OPACITY = int(0.55 * 255)
DST_DIR = "MJPEGImages"


@lru_cache
def get_font(fontsize):
    assert isinstance(fontsize, int), "font size must be integer"
    return ImageFont.truetype(
        "/usr/share/fonts/ms_fonts/simhei.ttf", fontsize, encoding="utf-8"
    )


def watermark(fpath):
    img = Image.open(fpath).convert("RGBA")
    width, height = img.size
    fw = width / DEFAULT_WIDTH
    fh = height / DEFAULT_HEIGHT
    fontsize = max(10, int(FONTSIZE * fw))
    loc = tuple(int(x * y) for x, y in zip(LOC, [fw, fh]))

    txt = Image.new("RGBA", img.size, (255, 255, 255, 0))
    fnt = get_font(fontsize)
    d = ImageDraw.Draw(txt)
    d.text(loc, TEXT, font=fnt, fill=COLOR + (OPACITY,))

    out = Image.alpha_composite(img, txt).convert("RGB")
    # out = np.asarray(out)
    # out = cv2.applyColorMap(out, cv2.COLORMAP_PINK)
    # out = cv2.blur(out, (5, 5))
    # out = Image.fromarray(out)
    out.save(osp.join(DST_DIR, osp.basename(fpath)))


def main(opacity=0.55, src_dir="JPEGImages", dst_dir="MJPEGImages"):
    assert osp.isdir(src_dir), src_dir

    os.makedirs(dst_dir, exist_ok=True)
    images = list(glob(osp.join(src_dir, "*.jpg")))
    with PoolExecutor() as executor:
        list(tqdm(executor.map(watermark, images), desc="Processing"))


if __name__ == "__main__":
    main()
