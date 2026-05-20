# Author: fkwong
# File: cvat_xml_to_mmseg.py
# Date: 5/19/26
"""Convert CVAT annotations.xml into an MMSegmentation-style dataset."""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import yaml


PARAM_NAME = "cvat_xml_to_mmseg"
SHAPE_TAGS = {"polygon", "polyline", "box", "ellipse", "mask", "points", "skeleton"}


@dataclass(frozen=True)
class Config:
    annotations: Path
    output: Path
    seed: int
    train: float
    categories: list[str]
    ignore_categories: list[str]
    ignore_index: int
    ignore_palette: tuple[int, int, int]
    polyline_width: int
    strict_categories: bool
    palette: list[tuple[int, int, int]]


def parse_args():
    parser = argparse.ArgumentParser()
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--yaml", action="store_true", help="use params.yaml")
    exclusive_group.add_argument("--seed", type=int, help="random seed")

    parser.add_argument(
        "--input", type=str, required=True, help="Path to annotations.xml"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--config", type=str, default="params.yaml", help="Path to params yaml"
    )
    parser.add_argument(
        "--param-name",
        type=str,
        default=PARAM_NAME,
        help="Name of the parameter to use from params.yaml",
    )
    parser.add_argument("--train", type=float, default=0.8, help="train split ratio")
    parser.add_argument(
        "--categories",
        "-c",
        type=str,
        default=None,
        nargs="+",
        help="List of ordered categories",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=None,
        nargs="+",
        help="Ordered RGB palette entries, one per category, as R,G,B values",
    )
    parser.add_argument(
        "--ignore-categories",
        type=str,
        default=[],
        nargs="*",
        help="Labels to write as ignore_index",
    )
    parser.add_argument(
        "--ignore-index", type=int, default=255, help="Pixel value for ignore labels"
    )
    parser.add_argument(
        "--ignore-palette",
        type=str,
        default=None,
        help="RGB palette entry for ignore_index, as R,G,B values",
    )
    parser.add_argument(
        "--polyline-width",
        type=int,
        default=5,
        help="Polyline drawing width in pixels, from 1 to 20",
    )
    parser.add_argument(
        "--strict-categories",
        action="store_true",
        help="Reject configured categories that do not exist in CVAT labels, except background",
    )
    return parser.parse_args()


def config_from_args(args) -> Config:
    if args.yaml:
        with open(args.config) as f:
            params = yaml.safe_load(f)[args.param_name]
        annotations = args.input
        output = args.output
        seed = params["seed"]
        train = params.get("train", 0.8)
        categories = params["categories"]
        ignore_categories = params.get("ignore_categories", [])
        ignore_index = params.get("ignore_index", 255)
        ignore_palette = params.get("ignore_palette")
        polyline_width = params.get("polyline_width", 5)
        strict_categories = params.get("strict_categories", False)
        palette = params.get("palette")
    else:
        missing = [
            name
            for name, value in (
                ("--input", args.input),
                ("--output", args.output),
                ("--categories", args.categories),
                ("--palette", args.palette),
                ("--ignore-palette", args.ignore_palette),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Missing required arguments: {', '.join(missing)}")
        annotations = args.input
        output = args.output
        seed = args.seed
        train = args.train
        categories = args.categories
        ignore_categories = args.ignore_categories
        ignore_index = args.ignore_index
        ignore_palette = args.ignore_palette
        polyline_width = args.polyline_width
        strict_categories = args.strict_categories
        palette = args.palette

    return Config(
        annotations=Path(annotations),
        output=Path(output),
        seed=int(seed),
        train=float(train),
        categories=list(categories),
        ignore_categories=list(ignore_categories),
        ignore_index=int(ignore_index),
        ignore_palette=parse_palette_color(ignore_palette, name="ignore_palette"),
        polyline_width=int(polyline_width),
        strict_categories=bool(strict_categories),
        palette=parse_palette(palette),
    )


def parse_palette_color(color, name: str = "palette color") -> tuple[int, int, int]:
    if color is None:
        raise ValueError(f"{name} is required")
    if isinstance(color, str):
        channels = color.split(",")
    else:
        channels = color
    if len(channels) != 3:
        raise ValueError(f"Invalid palette color: {color}")
    try:
        rgb = tuple(int(channel) for channel in channels)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid palette color: {color}") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"Palette color channels must be between 0 and 255: {color}")
    return rgb


def parse_palette(palette) -> list[tuple[int, int, int]]:
    if palette is None:
        raise ValueError("palette is required")
    if not palette:
        raise ValueError("palette must not be empty")

    parsed = []
    for color in palette:
        parsed.append(parse_palette_color(color))

    return parsed


def validate_config(config: Config) -> None:
    if not config.categories:
        raise ValueError("categories must not be empty")
    if not config.palette:
        raise ValueError("palette is required")
    if len(config.palette) < len(config.categories):
        raise ValueError(
            "palette must contain at least one RGB color for each category"
        )
    if len(config.palette) > 256:
        raise ValueError("palette must contain no more than 256 colors")
    if not 0 < config.train < 1:
        raise ValueError(f"train ratio must be between 0 and 1, got {config.train}")
    if not 1 <= config.polyline_width <= 20:
        raise ValueError(
            f"polyline_width must be between 1 and 20, got {config.polyline_width}"
        )
    if not 0 <= config.ignore_index <= 255:
        raise ValueError(
            f"ignore_index must be between 0 and 255, got {config.ignore_index}"
        )
    if len(set(config.categories)) != len(config.categories):
        raise ValueError("categories must not contain duplicates")
    if len(set(config.ignore_categories)) != len(config.ignore_categories):
        raise ValueError("ignore_categories must not contain duplicates")
    overlap = set(config.categories) & set(config.ignore_categories)
    if overlap:
        raise ValueError(
            f"categories and ignore_categories overlap: {sorted(overlap)}"
        )
    if len(config.categories) > 255:
        raise ValueError("categories must contain no more than 255 labels")


def parse_points(points: str) -> np.ndarray:
    parsed = []
    for point in points.split(";"):
        xy = point.split(",")
        if len(xy) != 2:
            raise ValueError(f"Invalid CVAT point: {point}")
        parsed.append([float(xy[0]), float(xy[1])])
    return np.rint(np.array(parsed, dtype=np.float32)).astype(np.int32)


def image_path(image_element: ET.Element, annotations_path: Path) -> Path:
    path = Path(image_element.attrib["name"])
    if path.is_absolute():
        return path
    direct = annotations_path.parent / path
    if direct.exists():
        return direct
    return annotations_path.parent / "images" / path


def _shape_priority(
    shape: ET.Element, categories: list[str], ignore_categories: list[str]
) -> int:
    label = shape.attrib["label"]
    if label in categories:
        return categories.index(label)
    if label in ignore_categories:
        return len(categories)
    raise ValueError(f"Unknown label: {label}")


def rasterize_image(
    image_element: ET.Element,
    categories: list[str],
    ignore_categories: list[str],
    ignore_index: int = 255,
    polyline_width: int = 5,
) -> np.ndarray:
    width = int(image_element.attrib["width"])
    height = int(image_element.attrib["height"])
    mask = np.zeros((height, width), dtype=np.uint8)

    shapes = []
    for order, child in enumerate(image_element):
        if child.tag in SHAPE_TAGS and child.tag not in {"polygon", "polyline"}:
            image_name = image_element.attrib["name"]
            raise ValueError(
                f"Unsupported CVAT shape '{child.tag}' in image {image_name}"
            )
        if child.tag not in {"polygon", "polyline"}:
            continue
        shapes.append(
            (_shape_priority(child, categories, ignore_categories), order, child)
        )

    shapes.sort(key=lambda item: (item[0], item[1]))
    for _, _, shape in shapes:
        label = shape.attrib["label"]
        value = ignore_index if label in ignore_categories else categories.index(label)
        points = parse_points(shape.attrib["points"])
        if shape.tag == "polygon":
            if len(points) < 3:
                raise ValueError(
                    f"Polygon for label '{label}' must have at least 3 points"
                )
            cv2.fillPoly(mask, [points], int(value))
        elif shape.tag == "polyline":
            if len(points) < 2:
                raise ValueError(
                    f"Polyline for label '{label}' must have at least 2 points"
                )
            cv2.polylines(
                mask,
                [points],
                isClosed=False,
                color=int(value),
                thickness=polyline_width,
            )

    return mask


def _check_unique_mask_stems(images: list[ET.Element]) -> None:
    seen = {}
    duplicates = []
    for image in images:
        stem = Path(image.attrib["name"]).stem
        if stem in seen:
            duplicates.append(stem)
        seen[stem] = image.attrib["name"]
    if duplicates:
        raise ValueError(
            f"Duplicate image stems would overwrite masks: {sorted(duplicates)}"
        )


def clean_output(output: Path) -> None:
    for dirname in ("images", "annotations"):
        path = output / dirname
        if path.exists():
            shutil.rmtree(path)
    for filename in ("train.txt", "val.txt"):
        path = output / filename
        if path.exists():
            path.unlink()


def make_split(
    images: list[ET.Element], train: float, seed: int
) -> tuple[list[ET.Element], list[ET.Element]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    train_num = int(train * len(shuffled))
    if train_num == 0 or train_num == len(shuffled):
        raise ValueError(
            f"train ratio {train} produces an empty split for {len(shuffled)} images"
        )
    return shuffled[:train_num], shuffled[train_num:]


def make_splits(images: list[ET.Element], config: Config) -> dict[str, list[ET.Element]]:
    train_images, val_images = make_split(images, config.train, config.seed)
    return {"train": train_images, "val": val_images}


def _write_split_files(output: Path, splits: dict[str, list[ET.Element]]) -> None:
    for split, split_images in splits.items():
        stems = [Path(image.attrib["name"]).stem for image in split_images]
        text = "\n".join(stems)
        if text:
            text += "\n"
        (output / f"{split}.txt").write_text(text)


def save_palette_mask(
    mask: np.ndarray,
    path: Path,
    palette: list[tuple[int, int, int]],
    ignore_index: int,
    ignore_palette: tuple[int, int, int],
) -> None:
    flat_palette = [channel for color in palette for channel in color]
    flat_palette.extend([0] * (768 - len(flat_palette)))
    start = ignore_index * 3
    flat_palette[start : start + 3] = list(ignore_palette)
    image = Image.fromarray(mask, mode="P")
    image.putpalette(flat_palette)
    image.save(path)


def convert_cvat_xml_to_mmseg(config: Config) -> None:
    validate_config(config)
    tree = ET.parse(config.annotations)
    root = tree.getroot()

    cvat_labels = [x.text for x in root.findall(".//label/name")]
    valid_labels = set(config.categories) | set(config.ignore_categories)
    missing = sorted(set(cvat_labels) - valid_labels)
    if missing:
        raise ValueError(f"CVAT labels missing from config: {missing}")
    if config.strict_categories:
        extra = sorted((valid_labels - set(cvat_labels)) - {"background"})
        if extra:
            raise ValueError(f"Config labels missing from CVAT: {extra}")

    images = root.findall("./image")
    if not images:
        raise ValueError(f"No images found in {config.annotations}")
    _check_unique_mask_stems(images)

    splits = make_splits(images, config)
    clean_output(config.output)
    config.output.mkdir(parents=True, exist_ok=True)
    _write_split_files(config.output, splits)

    img_dir = config.output / "images"
    ann_dir = config.output / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split_images in splits.values():
        for image in split_images:
            src = image_path(image, config.annotations)
            if not src.exists():
                raise FileNotFoundError(f"Image not found: {src}")

            dst_img = img_dir / src.name
            dst_mask = ann_dir / f"{src.stem}.png"
            shutil.copy2(src, dst_img)

            mask = rasterize_image(
                image,
                config.categories,
                config.ignore_categories,
                ignore_index=config.ignore_index,
                polyline_width=config.polyline_width,
            )
            save_palette_mask(
                mask,
                dst_mask,
                config.palette,
                config.ignore_index,
                config.ignore_palette,
            )


def main():
    args = parse_args()
    config = config_from_args(args)
    convert_cvat_xml_to_mmseg(config)


if __name__ == "__main__":
    main()
