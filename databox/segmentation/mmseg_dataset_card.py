"""Generate a compact dataset card for an MMSegmentation-style dataset."""

import argparse
import html
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_FILE_STEMS = ("train", "val", "trainval", "test")
BRANCH_MASK_SUFFIXES = {
    "polygon": "_polygon",
    "polyline": "_polyline",
}
OCNET_WEIGHT_SOURCE = (
    "https://github.com/openseg-group/OCNet.pytorch/issues/14#issuecomment-528144988"
)


@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str
    color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class DatasetCard:
    data: dict[str, Any]
    svg_path: Path
    yaml_path: Path


@dataclass(frozen=True)
class DatasetLayout:
    name: str
    images_dir: Path
    masks_dir: Path
    split_dir: Path
    labelmap_path: Path


class FourDecimalFloat(float):
    """Float marker for YAML values that should render with four decimals."""


class FlowList(list):
    """List marker for YAML values that should render in flow style."""


class DatasetCardYamlDumper(yaml.SafeDumper):
    pass


def _represent_four_decimal_float(
    dumper: yaml.Dumper,
    data: FourDecimalFloat,
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{data:.4f}")


def _represent_flow_list(
    dumper: yaml.Dumper,
    data: FlowList,
) -> yaml.nodes.SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


DatasetCardYamlDumper.add_representer(
    FourDecimalFloat,
    _represent_four_decimal_float,
)
DatasetCardYamlDumper.add_representer(FlowList, _represent_flow_list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SVG and YAML dataset card for an MMSeg dataset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help=(
            "Path containing either MMSeg images/ and annotations/ or Pascal VOC "
            "JPEGImages/ and SegmentationClass/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated card files. Defaults to dataset root.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
        help="Mask value to report as ignored and exclude from class weights.",
    )
    parser.add_argument(
        "--include-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the first labelmap class, normally background, in stats.",
    )
    parser.add_argument(
        "--card-name",
        default="dataset_card",
        help="Base filename for SVG and YAML outputs.",
    )
    return parser.parse_args()


def parse_labelmap(path: Path) -> list[ClassInfo]:
    if not path.exists():
        raise FileNotFoundError(f"labelmap not found: {path}")

    classes = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid labelmap line {line_number}: {raw_line}")
        name = parts[0].strip()
        color = _parse_color(parts[1], line_number)
        classes.append(ClassInfo(id=len(classes), name=name, color_rgb=color))

    if not classes:
        raise ValueError(f"No classes found in labelmap: {path}")
    if len(classes) > 256:
        raise ValueError("labelmap contains more than 256 classes")
    return classes


def _parse_color(text: str, line_number: int) -> tuple[int, int, int]:
    values = text.split(",")
    if len(values) != 3:
        raise ValueError(f"Invalid RGB value on labelmap line {line_number}: {text}")
    try:
        rgb = tuple(int(value) for value in values)
    except ValueError as exc:
        raise ValueError(
            f"Invalid RGB value on labelmap line {line_number}: {text}"
        ) from exc
    if any(value < 0 or value > 255 for value in rgb):
        raise ValueError(f"RGB channels must be 0..255 on line {line_number}: {text}")
    return rgb


def compute_ocnet_weights(pixel_counts: dict[int, int]) -> dict[int, float | None]:
    nonzero = {class_id: count for class_id, count in pixel_counts.items() if count > 0}
    if not nonzero:
        return dict.fromkeys(pixel_counts)

    raw_weights = {
        class_id: 1.0 / math.log1p(count) for class_id, count in nonzero.items()
    }
    normalizer = sum(raw_weights.values())
    class_count = len(pixel_counts)
    weights = {}
    for class_id, count in pixel_counts.items():
        if count == 0:
            weights[class_id] = None
        else:
            weights[class_id] = class_count * raw_weights[class_id] / normalizer
    return weights


def analyze_dataset(
    dataset_root: Path,
    ignore_index: int = 255,
    include_background: bool = True,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    layout = detect_dataset_layout(dataset_root)
    classes = parse_labelmap(layout.labelmap_path)
    if not include_background:
        classes = classes[1:]
    if not classes:
        raise ValueError("No classes remain after applying --no-include-background")

    class_ids = {item.id for item in classes}
    image_paths = _list_image_paths(layout.images_dir)
    mask_paths = _list_mask_paths(layout.masks_dir)
    polygon_mask_paths = _list_branch_mask_paths(layout.masks_dir, "polygon")
    polyline_mask_paths = _list_branch_mask_paths(layout.masks_dir, "polyline")
    image_stems = {path.stem for path in image_paths}
    mask_stems = {path.stem for path in mask_paths}
    split_stems = _read_split_stems(layout.split_dir)

    pixel_counts = {item.id: 0 for item in classes}
    image_counts = {item.id: 0 for item in classes}
    ignore_pixel_count = 0
    ignore_image_count = 0
    unknown_values: Counter[int] = Counter()
    image_sizes: Counter[str] = Counter()
    total_pixels = 0
    split_pixel_counts = {
        name: {item.id: 0 for item in classes} for name in split_stems
    }
    split_image_counts = {
        name: {item.id: 0 for item in classes} for name in split_stems
    }
    split_mask_counts: Counter[str] = Counter()
    split_total_pixels: Counter[str] = Counter()

    for mask_path in mask_paths:
        mask_counts, size = _read_mask_counts(mask_path)
        mask_total_pixels = sum(mask_counts.values())
        image_sizes[f"{size[0]}x{size[1]}"] += 1
        total_pixels += mask_total_pixels

        for value, count in mask_counts.items():
            if value in class_ids:
                pixel_counts[value] += count
            elif value == ignore_index:
                ignore_pixel_count += count
            else:
                unknown_values[value] += count

        for class_id in class_ids:
            if mask_counts.get(class_id, 0) > 0:
                image_counts[class_id] += 1
        if mask_counts.get(ignore_index, 0) > 0:
            ignore_image_count += 1

        for split_name, stems in split_stems.items():
            if mask_path.stem not in stems:
                continue
            split_mask_counts[split_name] += 1
            split_total_pixels[split_name] += mask_total_pixels
            for class_id in class_ids:
                count = mask_counts.get(class_id, 0)
                split_pixel_counts[split_name][class_id] += count
                if count > 0:
                    split_image_counts[split_name][class_id] += 1

    weights = compute_ocnet_weights(pixel_counts)
    polygon_stats = _analyze_branch_masks(
        polygon_mask_paths,
        classes,
        ignore_index,
        branch_name="polygon",
    )
    polyline_stats = _analyze_branch_masks(
        polyline_mask_paths,
        classes,
        ignore_index,
        branch_name="polyline",
    )
    warnings = _make_warnings(
        classes=classes,
        image_stems=image_stems,
        mask_stems=mask_stems,
        split_stems=split_stems,
        unknown_values=unknown_values,
        pixel_counts=pixel_counts,
    )

    class_rows = []
    for item in classes:
        pixel_count = pixel_counts[item.id]
        image_count = image_counts[item.id]
        class_rows.append(
            {
                "id": item.id,
                "name": item.name,
                "color_rgb": list(item.color_rgb),
                "pixel_count": pixel_count,
                "image_count": image_count,
                "pixel_ratio": _ratio(pixel_count, total_pixels),
                "image_ratio": _ratio(image_count, len(mask_paths)),
                "class_weight_ocnet": _round_float(weights[item.id]),
            }
        )
    class_weights = FlowList(
        _round_float_four_decimals(weights[item.id]) for item in classes
    )

    split_stats = _build_split_stats(
        classes=classes,
        split_stems=split_stems,
        split_pixel_counts=split_pixel_counts,
        split_image_counts=split_image_counts,
        split_mask_counts=split_mask_counts,
        split_total_pixels=split_total_pixels,
    )
    data = {
        "dataset_root": str(dataset_root),
        "layout": layout.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_count": len(image_paths),
        "mask_count": len(mask_paths),
        "total_pixels": total_pixels,
        "splits": {name: len(stems) for name, stems in split_stems.items()},
        "image_sizes": [
            {"size": size, "count": count}
            for size, count in sorted(image_sizes.items(), key=lambda item: item[0])
        ],
        "classes": class_rows,
        "class_weights_ocnet": class_weights,
        "polygon_branch": polygon_stats,
        "polyline_branch": polyline_stats,
        "class_weights_ocnet_polygon": polygon_stats["class_weights_ocnet"],
        "class_weights_ocnet_polyline": polyline_stats["class_weights_ocnet"],
        "split_stats": split_stats,
        "ignore_index": {
            "value": ignore_index,
            "pixel_count": ignore_pixel_count,
            "image_count": ignore_image_count,
            "pixel_ratio": _ratio(ignore_pixel_count, total_pixels),
            "image_ratio": _ratio(ignore_image_count, len(mask_paths)),
        },
        "ocnet_weight": {
            "formula": (
                "weight_i = n * (1 / log1p(pixel_count_i)) / "
                "sum_j(1 / log1p(pixel_count_j))"
            ),
            "source": OCNET_WEIGHT_SOURCE,
            "zero_pixel_classes": [
                item.name for item in classes if pixel_counts[item.id] == 0
            ],
        },
        "validation_warnings": warnings,
    }
    return data


def detect_dataset_layout(dataset_root: Path) -> DatasetLayout:
    dataset_root = dataset_root.resolve()
    labelmap_path = dataset_root / "labelmap.txt"
    candidates = [
        DatasetLayout(
            name="mmseg",
            images_dir=dataset_root / "images",
            masks_dir=dataset_root / "annotations",
            split_dir=dataset_root,
            labelmap_path=labelmap_path,
        ),
        DatasetLayout(
            name="voc",
            images_dir=dataset_root / "JPEGImages",
            masks_dir=dataset_root / "SegmentationClass",
            split_dir=dataset_root / "ImageSets" / "Segmentation",
            labelmap_path=labelmap_path,
        ),
    ]
    scored_matches = [
        (candidate, _layout_evidence(candidate))
        for candidate in candidates
        if _layout_exists(candidate)
    ]
    if not scored_matches:
        raise FileNotFoundError(
            "Could not detect dataset layout. Expected either MMSeg "
            "images/ and annotations/ or Pascal VOC JPEGImages/ and "
            "SegmentationClass/ under "
            f"{dataset_root}"
        )
    max_score = max(score for _, score in scored_matches)
    best_matches = [
        candidate for candidate, score in scored_matches if score == max_score
    ]
    if len(best_matches) == 1:
        return best_matches[0]
    names = ", ".join(candidate.name for candidate in best_matches)
    raise ValueError(
        "Could not auto detect dataset layout because multiple layouts are present: "
        f"{names}"
    )


def _layout_exists(layout: DatasetLayout) -> bool:
    return layout.images_dir.is_dir() and layout.masks_dir.is_dir()


def _layout_evidence(layout: DatasetLayout) -> int:
    score = 1
    if _list_image_paths(layout.images_dir):
        score += 1
    if _list_mask_paths(layout.masks_dir):
        score += 1
    if any((layout.split_dir / f"{split}.txt").exists() for split in SPLIT_FILE_STEMS):
        score += 1
    return score


def _read_mask_counts(mask_path: Path) -> tuple[Counter[int], tuple[int, int]]:
    with Image.open(mask_path) as image:
        if image.mode not in {"P", "L"}:
            raise ValueError(
                f"Mask must store integer class IDs in P or L mode: {mask_path}"
            )
        colors = image.getcolors(maxcolors=256)
        if colors is None:
            raise ValueError(f"Mask has more than 256 values: {mask_path}")
        return Counter({int(value): int(count) for count, value in colors}), image.size


def _read_mask_palette(mask_path: Path) -> list[tuple[int, int, int]]:
    with Image.open(mask_path) as image:
        palette = image.getpalette()
    if not palette:
        return []
    return [
        tuple(palette[index : index + 3])
        for index in range(0, min(len(palette), 768), 3)
    ]


def _list_image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _list_mask_paths(masks_dir: Path) -> list[Path]:
    if not masks_dir.exists():
        return []
    return sorted(
        path
        for path in masks_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".png"
            and not _is_branch_mask_path(path)
        )
    )


def _list_branch_mask_paths(masks_dir: Path, branch_name: str) -> list[Path]:
    suffix = BRANCH_MASK_SUFFIXES[branch_name]
    if not masks_dir.exists():
        return []
    return sorted(
        path
        for path in masks_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".png"
            and path.stem.endswith(suffix)
        )
    )


def _is_branch_mask_path(path: Path) -> bool:
    return any(path.stem.endswith(suffix) for suffix in BRANCH_MASK_SUFFIXES.values())


def _read_split_stems(split_dir: Path) -> dict[str, set[str]]:
    split_stems = {}
    if not split_dir.exists():
        return split_stems
    for split in SPLIT_FILE_STEMS:
        path = split_dir / f"{split}.txt"
        if not path.exists():
            continue
        split_stems[path.stem] = {
            line.strip() for line in path.read_text().splitlines() if line.strip()
        }
    return split_stems


def _make_warnings(
    classes: list[ClassInfo],
    image_stems: set[str],
    mask_stems: set[str],
    split_stems: dict[str, set[str]],
    unknown_values: Counter[int],
    pixel_counts: dict[int, int],
) -> list[str]:
    warnings = []
    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)
    if missing_masks:
        warnings.append(f"{len(missing_masks)} images have no matching mask")
    if missing_images:
        warnings.append(f"{len(missing_images)} masks have no matching image")
    if unknown_values:
        unknown_text = ", ".join(
            f"{value} ({count} px)" for value, count in sorted(unknown_values.items())
        )
        warnings.append(f"Unknown mask values found: {unknown_text}")

    split_union = set()
    for split, stems in split_stems.items():
        missing_split_images = sorted(stems - image_stems)
        missing_split_masks = sorted(stems - mask_stems)
        if missing_split_images:
            warnings.append(
                f"{split}.txt contains {len(missing_split_images)} stems without images"
            )
        if missing_split_masks:
            warnings.append(
                f"{split}.txt contains {len(missing_split_masks)} stems without masks"
            )
        overlap = split_union & stems
        if overlap:
            warnings.append(f"{len(overlap)} stems appear in multiple split files")
        split_union |= stems

    if split_stems:
        unsplit = (image_stems | mask_stems) - split_union
        if unsplit:
            warnings.append(f"{len(unsplit)} stems are not listed in any split file")

    empty_classes = [item.name for item in classes if pixel_counts[item.id] == 0]
    if empty_classes:
        warnings.append(f"Classes with zero pixels: {', '.join(empty_classes)}")
    return warnings


def _analyze_branch_masks(
    mask_paths: list[Path],
    labelmap_classes: list[ClassInfo],
    ignore_index: int,
    branch_name: str,
) -> dict[str, Any]:
    classes = _infer_branch_classes(mask_paths, labelmap_classes)
    class_ids = {item.id for item in classes}
    pixel_counts = {item.id: 0 for item in classes}
    image_counts = {item.id: 0 for item in classes}
    ignore_pixel_count = 0
    ignore_image_count = 0
    unknown_values: Counter[int] = Counter()
    total_pixels = 0

    for mask_path in mask_paths:
        mask_counts, _ = _read_mask_counts(mask_path)
        total_pixels += sum(mask_counts.values())
        for value, count in mask_counts.items():
            if value in class_ids:
                pixel_counts[value] += count
            elif value == ignore_index:
                ignore_pixel_count += count
            else:
                unknown_values[value] += count
        for class_id in class_ids:
            if mask_counts.get(class_id, 0) > 0:
                image_counts[class_id] += 1
        if mask_counts.get(ignore_index, 0) > 0:
            ignore_image_count += 1

    weights = compute_ocnet_weights(pixel_counts)
    class_rows = []
    for item in classes:
        pixel_count = pixel_counts[item.id]
        image_count = image_counts[item.id]
        class_rows.append(
            {
                "id": item.id,
                "name": item.name,
                "color_rgb": list(item.color_rgb),
                "pixel_count": pixel_count,
                "image_count": image_count,
                "pixel_ratio": _ratio(pixel_count, total_pixels),
                "image_ratio": _ratio(image_count, len(mask_paths)),
                "class_weight_ocnet": _round_float(weights[item.id]),
            }
        )

    return {
        "name": branch_name,
        "mask_count": len(mask_paths),
        "total_pixels": total_pixels,
        "classes": class_rows,
        "class_weights_ocnet": _class_weights_flow_list(classes, weights),
        "ignore_index": {
            "value": ignore_index,
            "pixel_count": ignore_pixel_count,
            "image_count": ignore_image_count,
            "pixel_ratio": _ratio(ignore_pixel_count, total_pixels),
            "image_ratio": _ratio(ignore_image_count, len(mask_paths)),
        },
        "unknown_values": [
            {"value": value, "pixel_count": count}
            for value, count in sorted(unknown_values.items())
        ],
    }


def _infer_branch_classes(
    mask_paths: list[Path],
    labelmap_classes: list[ClassInfo],
) -> list[ClassInfo]:
    if not labelmap_classes:
        return []
    if not mask_paths:
        return [labelmap_classes[0]]

    palette = _read_mask_palette(mask_paths[0])
    if not palette:
        max_value = 0
        for mask_path in mask_paths:
            mask_counts, _ = _read_mask_counts(mask_path)
            values = [value for value in mask_counts if value != 255]
            if values:
                max_value = max(max_value, max(values))
        return [
            ClassInfo(
                id=index,
                name=labelmap_classes[0].name if index == 0 else f"class_{index}",
                color_rgb=(0, 0, 0),
            )
            for index in range(max_value + 1)
        ]

    labels_by_color = {item.color_rgb: item.name for item in labelmap_classes}
    classes = [
        ClassInfo(
            id=0,
            name=labelmap_classes[0].name,
            color_rgb=labelmap_classes[0].color_rgb,
        )
    ]
    for index, color in enumerate(palette[1:], start=1):
        if color == labelmap_classes[0].color_rgb:
            break
        classes.append(
            ClassInfo(
                id=index,
                name=labels_by_color.get(color, f"class_{index}"),
                color_rgb=color,
            )
        )
    return classes


def _build_split_stats(
    classes: list[ClassInfo],
    split_stems: dict[str, set[str]],
    split_pixel_counts: dict[str, dict[int, int]],
    split_image_counts: dict[str, dict[int, int]],
    split_mask_counts: Counter[str],
    split_total_pixels: Counter[str],
) -> dict[str, dict[str, Any]]:
    split_stats = {}
    for split_name in split_stems:
        pixel_counts = split_pixel_counts[split_name]
        image_counts = split_image_counts[split_name]
        total_pixels = split_total_pixels[split_name]
        mask_count = split_mask_counts[split_name]
        weights = compute_ocnet_weights(pixel_counts)
        class_rows = []
        for item in classes:
            pixel_count = pixel_counts[item.id]
            image_count = image_counts[item.id]
            class_rows.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "color_rgb": list(item.color_rgb),
                    "pixel_count": pixel_count,
                    "image_count": image_count,
                    "pixel_ratio": _ratio(pixel_count, total_pixels),
                    "image_ratio": _ratio(image_count, mask_count),
                    "class_weight_ocnet": _round_float(weights[item.id]),
                }
            )

        split_stats[split_name] = {
            "stem_count": len(split_stems[split_name]),
            "mask_count": mask_count,
            "total_pixels": total_pixels,
            "classes": class_rows,
            "class_weights_ocnet": _class_weights_flow_list(classes, weights),
        }
    return split_stats


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 12)


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 12)


def _round_float_four_decimals(value: float | None) -> FourDecimalFloat:
    if value is None:
        return FourDecimalFloat(0.0)
    return FourDecimalFloat(round(value, 4))


def _class_weights_flow_list(
    classes: list[ClassInfo],
    weights: dict[int, float | None],
) -> FlowList:
    return FlowList(_round_float_four_decimals(weights[item.id]) for item in classes)


def _dump_dataset_card_yaml(data: dict[str, Any]) -> str:
    return yaml.dump(data, Dumper=DatasetCardYamlDumper, sort_keys=False)


def write_dataset_card(
    dataset_root: Path,
    output_dir: Path | None = None,
    ignore_index: int = 255,
    include_background: bool = True,
    card_name: str = "dataset_card",
) -> DatasetCard:
    output_dir = (output_dir or dataset_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = analyze_dataset(
        dataset_root=dataset_root,
        ignore_index=ignore_index,
        include_background=include_background,
    )
    yaml_path = output_dir / f"{card_name}.yaml"
    svg_path = output_dir / f"{card_name}.svg"

    yaml_path.write_text(_dump_dataset_card_yaml(data), encoding="utf-8")
    write_dataset_card_svg(data, svg_path)
    return DatasetCard(
        data=data,
        svg_path=svg_path,
        yaml_path=yaml_path,
    )


def draw_dataset_card(data: dict[str, Any], output_path: Path) -> None:
    classes = data["classes"]
    foreground_classes = _foreground_pixel_classes(classes)
    train_classes = _split_pixel_classes(data, "train")
    val_classes = _split_pixel_classes(data, "val")
    width = 1600
    legend_y = 1758
    legend_height = _class_key_height(classes)
    height = legend_y + legend_height + 46
    image = Image.new("RGB", (width, height), (246, 248, 251))
    draw = ImageDraw.Draw(image)
    font = _load_font(16)
    small_font = _load_font(13)
    tiny_font = _load_font(10)
    title_font = _load_font(28)
    heading_font = _load_font(20)

    draw.text((36, 28), "MMSeg Dataset Card", fill=(15, 23, 42), font=title_font)
    subtitle = (
        f"{data['image_count']} images / {data['mask_count']} masks / "
        f"{_format_int(data['total_pixels'])} labeled pixels"
    )
    draw.text((38, 66), subtitle, fill=(71, 85, 105), font=font)

    _draw_vertical_bar_chart(
        draw=draw,
        classes=classes,
        x=36,
        y=112,
        width=748,
        height=330,
        value_key="pixel_count",
        title="Pixel Count by Class",
        font=font,
        small_font=small_font,
        percent_font=tiny_font,
        percent_key="pixel_ratio",
    )
    _draw_vertical_bar_chart(
        draw=draw,
        classes=classes,
        x=816,
        y=112,
        width=748,
        height=330,
        value_key="image_count",
        title="Image Count by Class",
        font=font,
        small_font=small_font,
        percent_font=tiny_font,
        percent_key="image_ratio",
    )
    _draw_vertical_bar_chart(
        draw=draw,
        classes=classes,
        x=36,
        y=476,
        width=1528,
        height=330,
        value_key="pixel_count",
        title="Pixel Count by Class, Log10 Scale",
        font=font,
        small_font=small_font,
        percent_font=tiny_font,
        percent_key="pixel_ratio",
        value_transform=lambda value: math.log10(value + 1),
        axis_value_formatter=lambda value: f"{value:.1f}",
        subtitle="Bars show log10(pixel_count + 1)",
    )
    _draw_vertical_bar_chart(
        draw=draw,
        classes=train_classes,
        x=36,
        y=840,
        width=748,
        height=330,
        value_key="pixel_count",
        title="Train Pixel Count by Class",
        font=font,
        small_font=small_font,
        percent_font=tiny_font,
        percent_key="pixel_ratio",
    )
    _draw_vertical_bar_chart(
        draw=draw,
        classes=val_classes,
        x=816,
        y=840,
        width=748,
        height=330,
        value_key="pixel_count",
        title="Val Pixel Count by Class",
        font=font,
        small_font=small_font,
        percent_font=tiny_font,
        percent_key="pixel_ratio",
    )
    _draw_vertical_bar_chart(
        draw=draw,
        classes=foreground_classes,
        x=36,
        y=1204,
        width=1528,
        height=330,
        value_key="pixel_count",
        title="Pixel Count by Class, Excluding Background",
        font=font,
        small_font=small_font,
        percent_font=tiny_font,
        percent_key="foreground_pixel_ratio",
    )

    panel_y = 1568
    _draw_summary_panel(
        draw, data, 36, panel_y, 748, 160, heading_font, font, small_font
    )
    _draw_imbalance_panel(
        draw, classes, 816, panel_y, 748, 160, heading_font, font, small_font
    )
    _draw_class_key(draw, classes, 36, legend_y, 1528, legend_height, small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_vertical_bar_chart(
    draw: ImageDraw.ImageDraw,
    classes: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
    value_key: str,
    title: str,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    percent_font: ImageFont.ImageFont,
    percent_key: str,
    value_transform=None,
    axis_value_formatter=None,
    subtitle: str | None = None,
) -> None:
    _panel(draw, x, y, width, height, fill=(255, 255, 255))
    draw.text((x + 18, y + 14), title, fill=(15, 23, 42), font=font)
    if subtitle:
        draw.text((x + 18, y + 40), subtitle, fill=(100, 116, 139), font=small_font)

    transform = value_transform or (lambda value: value)
    axis_formatter = axis_value_formatter or _format_axis_value
    transformed = [transform(row[value_key]) for row in classes]
    max_value = max(transformed, default=0) or 1
    tick_count = 4
    plot_left = x + 64
    axis_top = y + 70
    bar_top_limit = axis_top + 22
    plot_right = x + width - 28
    plot_bottom = y + height - 56
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - bar_top_limit

    for tick in range(tick_count + 1):
        fraction = tick / tick_count
        grid_y = plot_bottom - int(fraction * plot_height)
        value = max_value * fraction
        draw.line((plot_left, grid_y, plot_right, grid_y), fill=(226, 232, 240))
        draw.text(
            (x + 14, grid_y - 8),
            axis_formatter(value),
            fill=(100, 116, 139),
            font=small_font,
        )

    draw.line((plot_left, axis_top, plot_left, plot_bottom), fill=(203, 213, 225))
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(203, 213, 225))

    count = max(len(classes), 1)
    slot_width = plot_width / count
    bar_width = max(6, min(34, int(slot_width * 0.58)))
    for index, row in enumerate(classes):
        value = transform(row[value_key])
        bar_height = int((value / max_value) * plot_height) if max_value else 0
        center_x = int(plot_left + index * slot_width + slot_width / 2)
        left = center_x - bar_width // 2
        right = center_x + bar_width // 2
        top = plot_bottom - bar_height
        color = tuple(row["color_rgb"])
        if bar_height > 0:
            draw.rounded_rectangle(
                (left, top, right, plot_bottom),
                radius=5,
                fill=color,
            )
        percent = _format_bar_percent(row[percent_key])
        percent_width = _text_width(draw, percent, percent_font)
        draw.text(
            (center_x - percent_width / 2, max(axis_top + 4, top - 15)),
            percent,
            fill=(30, 41, 59),
            font=percent_font,
        )
        draw.text(
            (center_x - 6, plot_bottom + 12),
            str(row["id"]),
            fill=(71, 85, 105),
            font=small_font,
        )


def _draw_class_key(
    draw: ImageDraw.ImageDraw,
    classes: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
    small_font: ImageFont.ImageFont,
) -> None:
    _panel(draw, x, y, width, height, fill=(255, 255, 255))
    columns = 4
    rows_per_column = math.ceil(len(classes) / columns)
    column_width = width // columns
    for index, row in enumerate(classes):
        col = index // rows_per_column
        row_index = index % rows_per_column
        item_x = x + 18 + col * column_width
        item_y = y + 14 + row_index * 21
        color = tuple(row["color_rgb"])
        draw.rounded_rectangle(
            (item_x, item_y + 3, item_x + 13, item_y + 16),
            radius=3,
            fill=color,
            outline=(203, 213, 225),
        )
        label = f"{row['id']}: {row['name']}"
        draw.text(
            (item_x + 20, item_y),
            _truncate(label, 32),
            fill=(51, 65, 85),
            font=small_font,
        )


def _foreground_pixel_classes(classes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    foreground = [dict(row) for row in classes[1:]]
    total = sum(row["pixel_count"] for row in foreground)
    for row in foreground:
        row["foreground_pixel_ratio"] = _ratio(row["pixel_count"], total)
    return foreground


def _split_pixel_classes(data: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    split_stats = data.get("split_stats", {}).get(split_name)
    if split_stats:
        return [dict(row) for row in split_stats["classes"]]

    zero_rows = []
    for row in data["classes"]:
        zero_row = dict(row)
        zero_row["pixel_count"] = 0
        zero_row["image_count"] = 0
        zero_row["pixel_ratio"] = 0.0
        zero_row["image_ratio"] = 0.0
        zero_row["class_weight_ocnet"] = None
        zero_rows.append(zero_row)
    return zero_rows


def _class_key_height(classes: list[dict[str, Any]]) -> int:
    columns = 4
    rows_per_column = max(1, math.ceil(len(classes) / columns))
    return 28 + rows_per_column * 21


def write_dataset_card_svg(data: dict[str, Any], output_path: Path) -> None:
    classes = data["classes"]
    foreground_classes = _foreground_pixel_classes(classes)
    train_classes = _split_pixel_classes(data, "train")
    val_classes = _split_pixel_classes(data, "val")
    width = 1600
    legend_y = 1758
    legend_height = _class_key_height(classes)
    height = legend_y + legend_height + 46
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>"
        "text{font-family:DejaVu Sans,Arial,sans-serif}"
        ".title{font-size:28px;font-weight:700;fill:#0f172a}"
        ".body{font-size:16px;fill:#475569}"
        ".small{font-size:13px;fill:#475569}"
        ".tiny{font-size:10px;fill:#1e293b}"
        ".heading{font-size:20px;font-weight:700;fill:#0f172a}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f6f8fb"/>',
        '<text x="36" y="50" class="title">MMSeg Dataset Card</text>',
    ]
    subtitle = (
        f"{data['image_count']} images / {data['mask_count']} masks / "
        f"{_format_int(data['total_pixels'])} labeled pixels"
    )
    parts.append(f'<text x="38" y="84" class="body">{_svg_text(subtitle)}</text>')
    parts.extend(
        _svg_vertical_bar_chart(
            classes=classes,
            x=36,
            y=112,
            width=748,
            height=330,
            value_key="pixel_count",
            percent_key="pixel_ratio",
            title="Pixel Count by Class",
        )
    )
    parts.extend(
        _svg_vertical_bar_chart(
            classes=classes,
            x=816,
            y=112,
            width=748,
            height=330,
            value_key="image_count",
            percent_key="image_ratio",
            title="Image Count by Class",
        )
    )
    parts.extend(
        _svg_vertical_bar_chart(
            classes=classes,
            x=36,
            y=476,
            width=1528,
            height=330,
            value_key="pixel_count",
            percent_key="pixel_ratio",
            title="Pixel Count by Class, Log10 Scale",
            value_transform=lambda value: math.log10(value + 1),
            axis_value_formatter=lambda value: f"{value:.1f}",
            subtitle="Bars show log10(pixel_count + 1)",
        )
    )
    parts.extend(
        _svg_vertical_bar_chart(
            classes=train_classes,
            x=36,
            y=840,
            width=748,
            height=330,
            value_key="pixel_count",
            percent_key="pixel_ratio",
            title="Train Pixel Count by Class",
        )
    )
    parts.extend(
        _svg_vertical_bar_chart(
            classes=val_classes,
            x=816,
            y=840,
            width=748,
            height=330,
            value_key="pixel_count",
            percent_key="pixel_ratio",
            title="Val Pixel Count by Class",
        )
    )
    parts.extend(
        _svg_vertical_bar_chart(
            classes=foreground_classes,
            x=36,
            y=1204,
            width=1528,
            height=330,
            value_key="pixel_count",
            percent_key="foreground_pixel_ratio",
            title="Pixel Count by Class, Excluding Background",
        )
    )
    parts.extend(_svg_summary_panel(data, 36, 1568, 748, 160))
    parts.extend(_svg_imbalance_panel(classes, 816, 1568, 748, 160))
    parts.extend(_svg_class_key(classes, 36, legend_y, 1528, legend_height))
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _svg_vertical_bar_chart(
    classes: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
    value_key: str,
    percent_key: str,
    title: str,
    value_transform=None,
    axis_value_formatter=None,
    subtitle: str | None = None,
) -> list[str]:
    transform = value_transform or (lambda value: value)
    axis_formatter = axis_value_formatter or _format_axis_value
    transformed = [transform(row[value_key]) for row in classes]
    max_value = max(transformed, default=0) or 1
    plot_left = x + 64
    axis_top = y + 70
    bar_top_limit = axis_top + 22
    plot_right = x + width - 28
    plot_bottom = y + height - 56
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - bar_top_limit
    tick_count = 4
    parts = [
        _svg_panel(x, y, width, height, "#ffffff"),
        f'<text x="{x + 18}" y="{y + 35}" class="body">{_svg_text(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{x + 18}" y="{y + 57}" class="small">'
            f"{_svg_text(subtitle)}</text>"
        )
    for tick in range(tick_count + 1):
        fraction = tick / tick_count
        grid_y = plot_bottom - int(fraction * plot_height)
        value = max_value * fraction
        parts.append(
            f'<line x1="{plot_left}" y1="{grid_y}" x2="{plot_right}" '
            'y2="{0}" stroke="#e2e8f0"/>'.format(grid_y)
        )
        parts.append(
            f'<text x="{x + 14}" y="{grid_y + 4}" class="small">'
            f"{_svg_text(axis_formatter(value))}</text>"
        )
    parts.append(
        f'<line x1="{plot_left}" y1="{axis_top}" x2="{plot_left}" '
        f'y2="{plot_bottom}" stroke="#cbd5e1"/>'
    )
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
        f'y2="{plot_bottom}" stroke="#cbd5e1"/>'
    )
    count = max(len(classes), 1)
    slot_width = plot_width / count
    bar_width = max(6, min(34, int(slot_width * 0.58)))
    for index, row in enumerate(classes):
        value = transform(row[value_key])
        bar_height = int((value / max_value) * plot_height) if max_value else 0
        center_x = int(plot_left + index * slot_width + slot_width / 2)
        left = center_x - bar_width // 2
        top = plot_bottom - bar_height
        color = _svg_color(row["color_rgb"])
        if bar_height > 0:
            parts.append(
                f'<rect x="{left}" y="{top}" width="{bar_width}" '
                f'height="{bar_height}" rx="5" fill="{color}"/>'
            )
        parts.append(
            f'<text x="{center_x}" y="{max(axis_top + 12, top - 5)}" '
            f'class="tiny" text-anchor="middle">'
            f"{_svg_text(_format_bar_percent(row[percent_key]))}</text>"
        )
        parts.append(
            f'<text x="{center_x}" y="{plot_bottom + 25}" class="small" '
            f'text-anchor="middle">{row["id"]}</text>'
        )
    return parts


def _svg_summary_panel(
    data: dict[str, Any], x: int, y: int, width: int, height: int
) -> list[str]:
    splits = (
        ", ".join(f"{name}: {count}" for name, count in data.get("splits", {}).items())
        or "none"
    )
    sizes = (
        ", ".join(
            f"{item['size']} ({item['count']})"
            for item in data.get("image_sizes", [])[:3]
        )
        or "none"
    )
    ignore = data["ignore_index"]
    lines = [
        f"Splits: {splits}",
        f"Top mask sizes: {sizes}",
        (
            f"Ignore {ignore['value']}: {_format_int(ignore['pixel_count'])} px "
            f"in {ignore['image_count']} masks"
        ),
    ]
    parts = [
        _svg_panel(x, y, width, height, "#f8fafc"),
        f'<text x="{x + 18}" y="{y + 38}" class="heading">Summary</text>',
    ]
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{x + 18}" y="{y + 72 + index * 26}" class="body">'
            f"{_svg_text(line)}</text>"
        )
    if data.get("validation_warnings"):
        parts.append(
            f'<text x="{x + 18}" y="{y + height - 18}" class="small" '
            f'fill="#b45309">Warnings: {len(data["validation_warnings"])}</text>'
        )
    return parts


def _svg_imbalance_panel(
    classes: list[dict[str, Any]], x: int, y: int, width: int, height: int
) -> list[str]:
    present = [row for row in classes if row["pixel_count"] > 0]
    rare = sorted(present, key=lambda row: row["pixel_ratio"])[:3]
    dominant = sorted(present, key=lambda row: row["pixel_ratio"], reverse=True)[:3]
    rare_text = (
        ", ".join(
            f"{row['name']} {_format_percent(row['pixel_ratio'])}" for row in rare
        )
        or "none"
    )
    dominant_text = (
        ", ".join(
            f"{row['name']} {_format_percent(row['pixel_ratio'])}" for row in dominant
        )
        or "none"
    )
    return [
        _svg_panel(x, y, width, height, "#f8fafc"),
        f'<text x="{x + 18}" y="{y + 38}" class="heading">Class Balance</text>',
        f'<text x="{x + 18}" y="{y + 70}" class="small">Rarest by pixels</text>',
        f'<text x="{x + 18}" y="{y + 96}" class="body">'
        f"{_svg_text(_truncate(rare_text, 82))}</text>",
        f'<text x="{x + 18}" y="{y + 126}" class="small">Largest by pixels</text>',
        f'<text x="{x + 18}" y="{y + 152}" class="body">'
        f"{_svg_text(_truncate(dominant_text, 82))}</text>",
    ]


def _svg_class_key(
    classes: list[dict[str, Any]], x: int, y: int, width: int, height: int
) -> list[str]:
    columns = 4
    rows_per_column = math.ceil(len(classes) / columns)
    column_width = width // columns
    parts = [_svg_panel(x, y, width, height, "#ffffff")]
    for index, row in enumerate(classes):
        col = index // rows_per_column
        row_index = index % rows_per_column
        item_x = x + 18 + col * column_width
        item_y = y + 14 + row_index * 21
        label = _truncate(f"{row['id']}: {row['name']}", 32)
        parts.append(
            f'<rect x="{item_x}" y="{item_y + 3}" width="13" height="13" '
            f'rx="3" fill="{_svg_color(row["color_rgb"])}" stroke="#cbd5e1"/>'
        )
        parts.append(
            f'<text x="{item_x + 20}" y="{item_y + 13}" class="small">'
            f"{_svg_text(label)}</text>"
        )
    return parts


def _svg_panel(x: int, y: int, width: int, height: int, fill: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" '
        f'fill="{fill}" stroke="#e2e8f0"/>'
    )


def _svg_color(color: list[int] | tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def _svg_text(text: str) -> str:
    return html.escape(text, quote=False)


def _draw_summary_panel(
    draw: ImageDraw.ImageDraw,
    data: dict[str, Any],
    x: int,
    y: int,
    width: int,
    height: int,
    heading_font: ImageFont.ImageFont,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    _panel(draw, x, y, width, height)
    draw.text((x + 18, y + 14), "Summary", fill=(20, 24, 30), font=heading_font)
    splits = (
        ", ".join(f"{name}: {count}" for name, count in data.get("splits", {}).items())
        or "none"
    )
    sizes = (
        ", ".join(
            f"{item['size']} ({item['count']})"
            for item in data.get("image_sizes", [])[:3]
        )
        or "none"
    )
    ignore = data["ignore_index"]
    lines = [
        f"Splits: {splits}",
        f"Top mask sizes: {sizes}",
        (
            f"Ignore {ignore['value']}: {_format_int(ignore['pixel_count'])} px "
            f"in {ignore['image_count']} masks"
        ),
    ]
    for index, line in enumerate(lines):
        draw.text((x + 18, y + 52 + index * 26), line, fill=(65, 72, 84), font=font)
    if data.get("validation_warnings"):
        draw.text(
            (x + 18, y + height - 26),
            f"Warnings: {len(data['validation_warnings'])}",
            fill=(180, 83, 9),
            font=small_font,
        )


def _draw_imbalance_panel(
    draw: ImageDraw.ImageDraw,
    classes: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
    heading_font: ImageFont.ImageFont,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    _panel(draw, x, y, width, height)
    draw.text((x + 18, y + 14), "Class Balance", fill=(20, 24, 30), font=heading_font)
    present = [row for row in classes if row["pixel_count"] > 0]
    rare = sorted(present, key=lambda row: row["pixel_ratio"])[:3]
    dominant = sorted(present, key=lambda row: row["pixel_ratio"], reverse=True)[:3]
    rare_text = (
        ", ".join(
            f"{row['name']} {_format_percent(row['pixel_ratio'])}" for row in rare
        )
        or "none"
    )
    dominant_text = (
        ", ".join(
            f"{row['name']} {_format_percent(row['pixel_ratio'])}" for row in dominant
        )
        or "none"
    )
    draw.text((x + 18, y + 52), "Rarest by pixels", fill=(65, 72, 84), font=small_font)
    draw.text((x + 18, y + 76), _truncate(rare_text, 82), fill=(42, 48, 58), font=font)
    draw.text(
        (x + 18, y + 106), "Largest by pixels", fill=(65, 72, 84), font=small_font
    )
    draw.text(
        (x + 18, y + 130),
        _truncate(dominant_text, 82),
        fill=(42, 48, 58),
        font=font,
    )


def _panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    fill: tuple[int, int, int] = (248, 250, 252),
) -> None:
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=8,
        fill=fill,
        outline=(226, 232, 240),
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_axis_value(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _format_bar_percent(value: float) -> str:
    percent = value * 100
    if percent >= 10:
        return f"{percent:.0f}%"
    if percent >= 1:
        return f"{percent:.1f}%"
    if percent > 0:
        return "<1%"
    return "0%"


def _format_percent(value: float) -> str:
    return f"{value * 100:.3f}%"


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def main() -> None:
    args = parse_args()
    card = write_dataset_card(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        ignore_index=args.ignore_index,
        include_background=args.include_background,
        card_name=args.card_name,
    )
    print(f"Wrote {card.svg_path}")
    print(f"Wrote {card.yaml_path}")


if __name__ == "__main__":
    main()
