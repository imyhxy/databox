"""Generate a compact dataset card for an MMSeg/VOC dataset with polygon/polyline branches.

Stats are reported per branch (main, polygon, polyline) and per split (train, val):
image / instance / pixel counts per class, OCNet class weights, and, for the polyline
branch, sigmoid-loss positive weights.
"""

import argparse
import html
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_FILE_STEMS = ("train", "val")
BRANCHES = ("main", "polygon", "polyline")
BRANCH_SUFFIXES = {"main": "", "polygon": "_polygon", "polyline": "_polyline"}
# Vertex txt files (one instance per line: "<global class id> x1 y1 ...") per branch.
INSTANCE_SUFFIXES = {
    "main": ("_polygon", "_polyline"),
    "polygon": ("_polygon",),
    "polyline": ("_polyline",),
}
POS_WEIGHT_CAP = 50.0
OCNET_WEIGHT_SOURCE = (
    "https://github.com/openseg-group/OCNet.pytorch/issues/14#issuecomment-528144988"
)
SERIES_COLORS = {"train": "#3b82f6", "val": "#f59e0b"}


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
    polygon_labelmap_path: Path
    polyline_labelmap_path: Path


@dataclass
class _Tally:
    """Raw counts for one (branch, split)."""

    mask_count: int = 0
    total_pixels: int = 0
    pixel: Counter[int] = field(default_factory=Counter)
    image: Counter[int] = field(default_factory=Counter)
    instance: Counter[int] = field(default_factory=Counter)
    unknown_instances: Counter[int] = field(default_factory=Counter)
    sizes: Counter[str] = field(default_factory=Counter)


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


def compute_sigmoid_pos_weights(
    pixel_counts: dict[int, int],
    valid_pixels: int,
    cap: float = POS_WEIGHT_CAP,
) -> dict[int, float]:
    """One-vs-rest sigmoid-loss positive weight: min(sqrt(neg / pos), cap).

    ``valid_pixels`` excludes ignored pixels; a class with no positives gets ``cap``.
    """
    weights = {}
    for class_id, pos in pixel_counts.items():
        if pos <= 0:
            weights[class_id] = cap
        else:
            weights[class_id] = min(math.sqrt((valid_pixels - pos) / pos), cap)
    return weights


def analyze_dataset(dataset_root: Path, ignore_index: int = 255) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    layout = detect_dataset_layout(dataset_root)
    branch_classes = {
        "main": parse_labelmap(layout.labelmap_path),
        "polygon": parse_labelmap(layout.polygon_labelmap_path),
        "polyline": parse_labelmap(layout.polyline_labelmap_path),
    }
    global_ids = {item.name: item.id for item in branch_classes["main"]}

    image_paths = _list_image_paths(layout.images_dir)
    mask_paths = _list_mask_paths(layout.masks_dir)
    image_stems = {path.stem for path in image_paths}
    mask_stems = {path.stem for path in mask_paths}
    split_stems = _read_split_stems(layout.split_dir)

    tallies: dict[str, dict[str, _Tally]] = {branch: {} for branch in BRANCHES}
    for branch, classes in branch_classes.items():
        # txt files always carry global (labelmap.txt) ids; map them to branch ids.
        to_branch_id = {
            global_ids[item.name]: item.id
            for item in classes
            if item.name in global_ids
        }
        for split in SPLIT_FILE_STEMS:
            stems = sorted(split_stems.get(split, set()) & mask_stems)
            tallies[branch][split] = _tally_branch(
                stems, layout.masks_dir, branch, to_branch_id, ignore_index
            )

    mask_sizes: Counter[str] = Counter()
    for tally in tallies["main"].values():
        mask_sizes.update(tally.sizes)

    branches = {
        branch: {
            split: _split_stats(
                tallies[branch][split],
                classes,
                ignore_index,
                with_pos_weights=branch == "polyline",
            )
            for split in SPLIT_FILE_STEMS
        }
        for branch, classes in branch_classes.items()
    }
    return {
        "dataset_root": str(dataset_root),
        "layout": layout.name,
        "image_count": len(image_paths),
        "mask_count": len(mask_paths),
        "splits": {
            split: len(split_stems.get(split, ())) for split in SPLIT_FILE_STEMS
        },
        "mask_sizes": [
            {"size": size, "count": count}
            for size, count in sorted(mask_sizes.items(), key=lambda i: (-i[1], i[0]))
        ],
        "ignore_index": ignore_index,
        "branches": branches,
        "ocnet_weight": {
            "formula": (
                "weight_i = n * (1 / log1p(pixel_count_i)) / "
                "sum_j(1 / log1p(pixel_count_j))"
            ),
            "source": OCNET_WEIGHT_SOURCE,
            "scope": (
                "per branch and split; n and the sum run over labelmap classes "
                "including background, excluding ignore; zero-pixel classes are 0"
            ),
        },
        "sigmoid_pos_weight": {
            "formula": "pos_weight_i = min(sqrt(neg_i / pos_i), cap)",
            "cap": POS_WEIGHT_CAP,
            "scope": (
                "polyline branch, per split, per foreground class (one-vs-rest); "
                "neg_i = non-ignore pixels - pos_i; zero-pixel classes get cap"
            ),
        },
        "validation_warnings": _make_warnings(
            image_stems=image_stems,
            mask_stems=mask_stems,
            split_stems=split_stems,
            tallies=tallies,
            branch_classes=branch_classes,
            ignore_index=ignore_index,
        ),
    }


def _tally_branch(
    stems: list[str],
    masks_dir: Path,
    branch: str,
    to_branch_id: dict[int, int],
    ignore_index: int,
) -> _Tally:
    tally = _Tally()
    suffix = BRANCH_SUFFIXES[branch]
    for stem in stems:
        mask_path = masks_dir / f"{stem}{suffix}.png"
        if not mask_path.exists():
            continue
        mask_counts, size = _read_mask_counts(mask_path)
        tally.mask_count += 1
        tally.total_pixels += sum(mask_counts.values())
        tally.sizes[f"{size[0]}x{size[1]}"] += 1
        for value, count in mask_counts.items():
            tally.pixel[value] += count
            tally.image[value] += 1
        for txt_suffix in INSTANCE_SUFFIXES[branch]:
            instances = _read_instance_counts(masks_dir / f"{stem}{txt_suffix}.txt")
            for global_id, count in instances.items():
                if global_id == ignore_index:
                    tally.instance[ignore_index] += count
                elif global_id in to_branch_id:
                    tally.instance[to_branch_id[global_id]] += count
                else:
                    tally.unknown_instances[global_id] += count
    return tally


def _split_stats(
    tally: _Tally,
    classes: list[ClassInfo],
    ignore_index: int,
    with_pos_weights: bool,
) -> dict[str, Any]:
    pixel_counts = {item.id: tally.pixel[item.id] for item in classes}
    ocnet = compute_ocnet_weights(pixel_counts)
    pos_weights: dict[int, float] = {}
    if with_pos_weights:
        pos_weights = compute_sigmoid_pos_weights(
            {item.id: pixel_counts[item.id] for item in classes[1:]},
            tally.total_pixels - tally.pixel[ignore_index],
        )

    rows = []
    for item in classes:
        row = _class_row(tally, item.id, item.name, item.color_rgb, has_instances=True)
        row["class_weight_ocnet"] = _round_float(ocnet[item.id])
        if with_pos_weights:
            row["pos_weight_sigmoid"] = _round_float(pos_weights.get(item.id))
        rows.append(row)
    # Background (first class) has no instances.
    rows[0]["instance_count"] = None
    ignore_row = _class_row(
        tally, ignore_index, "ignore", (128, 128, 128), has_instances=True
    )
    ignore_row["class_weight_ocnet"] = None
    if with_pos_weights:
        ignore_row["pos_weight_sigmoid"] = None
    rows.append(ignore_row)

    stats = {
        "mask_count": tally.mask_count,
        "instance_count": sum(
            count for key, count in tally.instance.items() if key != ignore_index
        ),
        "total_pixels": tally.total_pixels,
        "classes": rows,
        "class_weights_ocnet": FlowList(
            _round_float_four_decimals(ocnet[item.id]) for item in classes
        ),
    }
    if with_pos_weights:
        stats["pos_weights_sigmoid"] = FlowList(
            _round_float_four_decimals(pos_weights[item.id]) for item in classes[1:]
        )
    return stats


def _class_row(
    tally: _Tally,
    class_id: int,
    name: str,
    color_rgb: tuple[int, int, int],
    has_instances: bool,
) -> dict[str, Any]:
    return {
        "id": class_id,
        "name": name,
        "color_rgb": list(color_rgb),
        "image_count": tally.image[class_id],
        "instance_count": tally.instance[class_id] if has_instances else None,
        "pixel_count": tally.pixel[class_id],
        "image_ratio": _ratio(tally.image[class_id], tally.mask_count),
        "pixel_ratio": _ratio(tally.pixel[class_id], tally.total_pixels),
    }


def detect_dataset_layout(dataset_root: Path) -> DatasetLayout:
    dataset_root = dataset_root.resolve()
    labelmaps = {
        "labelmap_path": dataset_root / "labelmap.txt",
        "polygon_labelmap_path": dataset_root / "labelmap_polygon.txt",
        "polyline_labelmap_path": dataset_root / "labelmap_polyline.txt",
    }
    candidates = [
        DatasetLayout(
            name="mmseg",
            images_dir=dataset_root / "images",
            masks_dir=dataset_root / "annotations",
            split_dir=dataset_root,
            **labelmaps,
        ),
        DatasetLayout(
            name="voc",
            images_dir=dataset_root / "JPEGImages",
            masks_dir=dataset_root / "SegmentationClass",
            split_dir=dataset_root / "ImageSets" / "Segmentation",
            **labelmaps,
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


def _read_instance_counts(path: Path) -> Counter[int]:
    """Count instances per class id in a vertex txt (one instance per line)."""
    counts: Counter[int] = Counter()
    if not path.exists():
        return counts
    for line in path.read_text().splitlines():
        parts = line.split(None, 1)
        if parts:
            counts[int(parts[0])] += 1
    return counts


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


def _is_branch_mask_path(path: Path) -> bool:
    return any(
        path.stem.endswith(suffix) for suffix in BRANCH_SUFFIXES.values() if suffix
    )


def _read_split_stems(split_dir: Path) -> dict[str, set[str]]:
    split_stems = {}
    if not split_dir.exists():
        return split_stems
    for split in SPLIT_FILE_STEMS:
        path = split_dir / f"{split}.txt"
        if not path.exists():
            continue
        split_stems[split] = {
            line.strip() for line in path.read_text().splitlines() if line.strip()
        }
    return split_stems


def _make_warnings(
    image_stems: set[str],
    mask_stems: set[str],
    split_stems: dict[str, set[str]],
    tallies: dict[str, dict[str, _Tally]],
    branch_classes: dict[str, list[ClassInfo]],
    ignore_index: int,
) -> list[str]:
    warnings = []
    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)
    if missing_masks:
        warnings.append(f"{len(missing_masks)} images have no matching mask")
    if missing_images:
        warnings.append(f"{len(missing_images)} masks have no matching image")

    split_union = set()
    for split in SPLIT_FILE_STEMS:
        if split not in split_stems:
            warnings.append(f"{split}.txt not found")
            continue
        stems = split_stems[split]
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

    for branch, classes in branch_classes.items():
        class_ids = {item.id for item in classes}
        for split, tally in tallies[branch].items():
            prefix = f"{branch}/{split}"
            missing = tallies["main"][split].mask_count - tally.mask_count
            if missing:
                warnings.append(f"{prefix}: {missing} masks are missing")
            unknown = {
                value: count
                for value, count in sorted(tally.pixel.items())
                if value not in class_ids and value != ignore_index
            }
            if unknown:
                text = ", ".join(f"{v} ({c} px)" for v, c in unknown.items())
                warnings.append(f"{prefix}: unknown mask values found: {text}")
            if tally.unknown_instances:
                text = ", ".join(
                    f"{v} ({c})" for v, c in sorted(tally.unknown_instances.items())
                )
                warnings.append(f"{prefix}: instance class ids not in labelmap: {text}")
            empty = [item.name for item in classes if tally.pixel[item.id] == 0]
            if empty:
                warnings.append(f"{prefix}: classes with zero pixels: {', '.join(empty)}")
    return warnings


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


def _dump_dataset_card_yaml(data: dict[str, Any]) -> str:
    return yaml.dump(data, Dumper=DatasetCardYamlDumper, sort_keys=False)


def write_dataset_card(
    dataset_root: Path,
    output_dir: Path | None = None,
    ignore_index: int = 255,
    card_name: str = "dataset_card",
) -> DatasetCard:
    output_dir = (output_dir or dataset_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = analyze_dataset(dataset_root=dataset_root, ignore_index=ignore_index)
    yaml_path = output_dir / f"{card_name}.yaml"
    svg_path = output_dir / f"{card_name}.svg"

    yaml_path.write_text(_dump_dataset_card_yaml(data), encoding="utf-8")
    write_dataset_card_svg(data, svg_path)
    return DatasetCard(
        data=data,
        svg_path=svg_path,
        yaml_path=yaml_path,
    )


def _foreground_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Class rows without background (first) and ignore (last)."""
    return rows[1:-1]


def write_dataset_card_svg(data: dict[str, Any], output_path: Path) -> None:
    plotted = ("polygon", "polyline")
    key_columns = 2
    key_heights = {
        branch: _class_key_height(
            len(_foreground_rows(data["branches"][branch]["train"]["classes"])),
            key_columns,
        )
        for branch in plotted
    }
    width = 2400
    col_width = 570
    col_xs = (36, 622, 1208, 1794)
    row_ys = (112, 476)
    chart_height = 330
    summary_y = row_ys[-1] + chart_height + 34
    summary_height = 132
    key_y = summary_y + summary_height + 34
    height = key_y + max(key_heights.values()) + 46
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
        '<text x="36" y="50" class="title">Dataset Card</text>',
    ]
    subtitle = (
        f"{data['image_count']} images / {data['mask_count']} masks / "
        + " + ".join(f"{name} {count}" for name, count in data["splits"].items())
    )
    parts.append(f'<text x="38" y="84" class="body">{_svg_text(subtitle)}</text>')

    for row_y, branch in zip(row_ys, plotted):
        splits = data["branches"][branch]
        series = {
            split: _foreground_rows(splits[split]["classes"])
            for split in SPLIT_FILE_STEMS
        }
        mask_total = sum(splits[split]["mask_count"] for split in SPLIT_FILE_STEMS)
        instance_total = sum(
            splits[split]["instance_count"] for split in SPLIT_FILE_STEMS
        )
        pixel_total = sum(
            row["pixel_count"] for rows in series.values() for row in rows
        )
        charts = (
            ("image_count", "Images", mask_total, "% = share of masks with class"),
            ("instance_count", "Instances", instance_total, "% = share of instances"),
            (
                "pixel_count",
                "Pixels",
                pixel_total,
                "% = share of foreground pixels",
            ),
            (
                "pixel_count",
                "Pixels (All)",
                sum(splits[split]["total_pixels"] for split in SPLIT_FILE_STEMS),
                "% = share of all pixels incl. bg + ignore",
            ),
        )
        for col_x, (value_key, label, denominator, subtitle) in zip(col_xs, charts):
            parts.extend(
                _svg_stacked_bar_chart(
                    series=series,
                    x=col_x,
                    y=row_y,
                    width=col_width,
                    height=chart_height,
                    value_key=value_key,
                    denominator=denominator,
                    title=f"{branch.capitalize()}: {label} by Class",
                    subtitle=subtitle,
                )
            )

    parts.extend(_svg_summary_panel(data, 36, summary_y, 2328, summary_height))
    for key_x, branch in zip((36, 1208), plotted):
        rows = _foreground_rows(data["branches"][branch]["train"]["classes"])
        parts.extend(
            _svg_class_key(
                rows,
                key_x,
                key_y,
                1156,
                key_heights[branch],
                title=f"{branch.capitalize()} classes",
                columns=key_columns,
            )
        )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _svg_stacked_bar_chart(
    series: dict[str, list[dict[str, Any]]],
    x: int,
    y: int,
    width: int,
    height: int,
    value_key: str,
    denominator: int,
    title: str,
    subtitle: str,
) -> list[str]:
    """Stacked bars per class; ``series`` maps split name -> aligned class rows."""
    splits = list(series)
    count = len(series[splits[0]])
    totals = [sum(series[split][i][value_key] for split in splits) for i in range(count)]
    max_value = max(totals, default=0) or 1
    plot_left = x + 72
    axis_top = y + 88
    bar_top_limit = axis_top + 22
    plot_right = x + width - 28
    plot_bottom = y + height - 56
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - bar_top_limit
    tick_count = 4
    parts = [
        _svg_panel(x, y, width, height, "#ffffff"),
        f'<text x="{x + 18}" y="{y + 35}" class="body">{_svg_text(title)}</text>',
        f'<text x="{x + 18}" y="{y + 57}" class="small">{_svg_text(subtitle)}</text>',
    ]
    legend_x = x + 18
    for index, split in enumerate(splits):
        item_x = legend_x + index * 70
        parts.append(
            f'<rect x="{item_x}" y="{y + 66}" width="12" height="12" rx="2" '
            f'fill="{SERIES_COLORS[split]}"/>'
        )
        parts.append(
            f'<text x="{item_x + 17}" y="{y + 77}" class="small">{split}</text>'
        )
    for tick in range(tick_count + 1):
        fraction = tick / tick_count
        grid_y = plot_bottom - int(fraction * plot_height)
        parts.append(
            f'<line x1="{plot_left}" y1="{grid_y}" x2="{plot_right}" '
            f'y2="{grid_y}" stroke="#e2e8f0"/>'
        )
        parts.append(
            f'<text x="{x + 14}" y="{grid_y + 4}" class="small">'
            f"{_svg_text(_format_axis_value(max_value * fraction))}</text>"
        )
    parts.append(
        f'<line x1="{plot_left}" y1="{axis_top}" x2="{plot_left}" '
        f'y2="{plot_bottom}" stroke="#cbd5e1"/>'
    )
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
        f'y2="{plot_bottom}" stroke="#cbd5e1"/>'
    )
    slot_width = plot_width / max(count, 1)
    bar_width = max(6, min(34, int(slot_width * 0.58)))
    for index in range(count):
        center_x = int(plot_left + index * slot_width + slot_width / 2)
        left = center_x - bar_width // 2
        cumulative = 0
        boundary = plot_bottom
        for split in splits:
            cumulative += series[split][index][value_key]
            top = plot_bottom - int(cumulative / max_value * plot_height)
            if boundary > top:
                parts.append(
                    f'<rect x="{left}" y="{top}" width="{bar_width}" '
                    f'height="{boundary - top}" fill="{SERIES_COLORS[split]}"/>'
                )
            boundary = top
        percent = _format_bar_percent(_ratio(totals[index], denominator))
        parts.append(
            f'<text x="{center_x}" y="{max(axis_top + 12, boundary - 5)}" '
            f'class="tiny" text-anchor="middle">{_svg_text(percent)}</text>'
        )
        parts.append(
            f'<text x="{center_x}" y="{plot_bottom + 25}" class="small" '
            f'text-anchor="middle">{series[splits[0]][index]["id"]}</text>'
        )
    return parts


def _svg_summary_panel(
    data: dict[str, Any], x: int, y: int, width: int, height: int
) -> list[str]:
    splits = ", ".join(f"{name}: {count}" for name, count in data["splits"].items())
    sizes = (
        ", ".join(f"{item['size']} ({item['count']})" for item in data["mask_sizes"][:3])
        or "none"
    )
    lines = [f"Splits: {splits}", f"Top mask sizes: {sizes}"]
    parts = [
        _svg_panel(x, y, width, height, "#f8fafc"),
        f'<text x="{x + 18}" y="{y + 38}" class="heading">Summary</text>',
    ]
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{x + 18}" y="{y + 68 + index * 26}" class="body">'
            f"{_svg_text(line)}</text>"
        )
    if data["validation_warnings"]:
        parts.append(
            f'<text x="{x + 18}" y="{y + height - 14}" class="small" '
            f'fill="#b45309">Warnings: {len(data["validation_warnings"])}</text>'
        )
    return parts


def _class_key_height(class_count: int, columns: int) -> int:
    return 54 + max(1, math.ceil(class_count / columns)) * 21


def _svg_class_key(
    rows: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    columns: int,
) -> list[str]:
    rows_per_column = max(1, math.ceil(len(rows) / columns))
    column_width = width // columns
    parts = [
        _svg_panel(x, y, width, height, "#ffffff"),
        f'<text x="{x + 18}" y="{y + 24}" class="body">{_svg_text(title)}</text>',
    ]
    for index, row in enumerate(rows):
        col = index // rows_per_column
        row_index = index % rows_per_column
        item_x = x + 18 + col * column_width
        item_y = y + 40 + row_index * 21
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
        card_name=args.card_name,
    )
    print(f"Wrote {card.svg_path}")
    print(f"Wrote {card.yaml_path}")


if __name__ == "__main__":
    main()
