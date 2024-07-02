# Author: imyhxy
# File: overwrite_cvat_xml.py
# Date: 7/2/24
"""
将新的CVAT标签覆盖旧的标签.

此脚本只会覆盖分类标签。从数据集A中挑选一部分图片组成数据集B，对数据集进行标注并导出标签。
此脚本可以将数据集B的标签覆盖到数据集A中。
"""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="base xml file")
    parser.add_argument("--new", type=str, required=True, help="new xml file")
    parser.add_argument("--out", type=str, required=True, help="output xml file")
    args = parser.parse_args()

    new_tree = ET.parse(args.new)
    new_image_tags = {}
    for image in new_tree.iterfind("./image"):
        path = Path(image.attrib["name"])
        new_image_tags[path.name] = image.findall("./tag")

    base_tree = ET.parse(args.base)
    for image in base_tree.iterfind("./image"):
        path = Path(image.attrib["name"])
        if path.name in new_image_tags:
            for tag in image.findall("./tag"):
                image.remove(tag)
            for tag in new_image_tags[path.name]:
                image.insert(0, tag)

    out_str = ET.tostring(
        base_tree.getroot(),
        encoding="unicode",
        short_empty_elements=False,
        xml_declaration=True,
    )

    with open(args.out, "w") as f:
        f.write(out_str.replace("'", '"'))


if __name__ == "__main__":
    main()
