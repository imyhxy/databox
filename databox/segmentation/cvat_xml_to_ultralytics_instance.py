"""Convert a CVAT XML export into an Ultralytics instance-segmentation dataset.

The input root contains a CVAT ``annotations.xml`` file and the source images.
The output root has the layout expected by Ultralytics::

    <output-root>/images/<relative-image-path>  # relative symlink
    <output-root>/labels/<relative-image-path>.txt
    <output-root>/data.yaml

The CVAT export has no train/validation split, so both ``train`` and ``val``
in ``data.yaml`` point to ``images``.

CVAT polygons that have the same output class and ``group_id`` are one logical
instance.  Multiple polygons in that instance are joined with the same
``merge_multi_segment`` algorithm used by Ultralytics' COCO converter.  A
polygon without a group is its own instance.

Only polygon annotations are accepted.  Other CVAT shape types are rejected
instead of silently producing an incomplete instance dataset.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PARAM_NAME = "cvat_xml_to_ultralytics_instance"
ANNOTATIONS_FILENAME = "annotations.xml"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
CVAT_SHAPE_TAGS = {
    "box",
    "ellipse",
    "mask",
    "points",
    "polygon",
    "polyline",
    "skeleton",
}


@dataclass(frozen=True)
class Config:
    """Command-line/configuration values for one conversion."""

    input_root: Path
    output_root: Path
    categories: tuple[str, ...] | None
    category_map: dict[str, str]


@dataclass(frozen=True)
class ConversionStats:
    """Counters and class names produced by a conversion."""

    image_count: int
    instance_count: int
    filtered_polygon_count: int
    grouped_instance_count: int
    names: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    ``--input-root`` and ``--output-root`` are intentionally required CLI
    arguments.  All conversion choices can instead come from ``params.yaml``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert CVAT polygon annotations.xml into an Ultralytics "
            "instance-segmentation dataset."
        )
    )
    parser.add_argument(
        "--input-root",
        "--input",
        dest="input_root",
        type=Path,
        required=True,
        help="Input root containing annotations.xml and source images.",
    )
    parser.add_argument(
        "--output-root",
        "--output",
        dest="output_root",
        type=Path,
        required=True,
        help="Output root containing images/, labels/, and data.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional params YAML file.",
    )
    parser.add_argument(
        "--param-name",
        default=PARAM_NAME,
        help="YAML section to read when --config contains named sections.",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help=(
            "Source CVAT labels to keep, in output class order. If omitted, "
            "labels declared by CVAT are used."
        ),
    )
    parser.add_argument(
        "--category-map",
        nargs="+",
        metavar="SOURCE=TARGET",
        help="Optional source-to-output class remapping.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    """Resolve CLI-over-YAML precedence while keeping paths CLI-only."""
    values = _load_config_values(
        getattr(args, "config", None),
        getattr(args, "param_name", PARAM_NAME),
    )
    categories_value = (
        args.categories
        if getattr(args, "categories", None) is not None
        else values.get("categories")
    )
    category_map_value = (
        args.category_map
        if getattr(args, "category_map", None) is not None
        else values.get("category_map")
    )

    categories = _coerce_string_list(categories_value)
    if categories is not None:
        _validate_category_names(categories, "categories")

    input_value = getattr(args, "input_root", getattr(args, "input", None))
    output_value = getattr(args, "output_root", getattr(args, "output", None))
    if input_value is None or output_value is None:
        raise ValueError("input_root and output_root are required CLI paths")

    return Config(
        input_root=Path(input_value),
        output_root=Path(output_value),
        categories=tuple(categories) if categories is not None else None,
        category_map=_parse_category_map(category_map_value),
    )


def build_ultralytics_instance_dataset(
    input_root: Path,
    output_root: Path,
    categories: Sequence[str] | None = None,
    category_map: dict[str, str] | None = None,
) -> ConversionStats:
    """Build an Ultralytics instance dataset from a CVAT XML export.

    Args:
        input_root: Directory containing ``annotations.xml`` and images.  An
            annotations XML path is also accepted; its parent is then used as
            the image root.
        output_root: Destination containing ``images``, ``labels`` and
            ``data.yaml``.
        categories: Source labels to retain.  Omitted labels are discovered
            from CVAT metadata and image annotations.
        category_map: Source-label to output-name mapping.
    """
    input_path = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    annotations_path, image_root = _resolve_input(input_path)

    if _paths_overlap(input_path if input_path.is_dir() else image_root, output_root):
        raise ValueError(
            "Output root must be separate from the input root and its images"
        )

    tree = ET.parse(annotations_path)
    root = tree.getroot()
    images = root.findall("./image")
    if not images:
        raise ValueError(f"No CVAT images found in {annotations_path}")

    source_labels = _discover_labels(root, images)
    selected_categories = _resolve_categories(categories, source_labels)
    source_to_class_id, names = _build_class_lookup(
        selected_categories,
        category_map or {},
    )

    resolved_images = [
        _resolve_image_path(image_root, image.attrib.get("name"), annotations_path)
        for image in images
    ]
    output_relatives = [
        _output_image_relative_path(image_path, image_root)
        for image_path in resolved_images
    ]
    _validate_output_relatives(output_relatives, resolved_images)

    _clean_output(output_root)
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    instance_count = 0
    filtered_polygon_count = 0
    grouped_instance_count = 0

    for image, source_image, relative_image in zip(
        images,
        resolved_images,
        output_relatives,
        strict=True,
    ):
        width, height = _image_dimensions(image)
        output_image = output_images / relative_image
        relative_label = relative_image.with_suffix(".txt")
        output_label = output_labels / relative_label
        _relative_symlink(source_image, output_image)

        label_lines, filtered, grouped = _image_to_label_lines(
            image,
            width,
            height,
            source_to_class_id,
        )
        text = "\n".join(label_lines)
        if text:
            text += "\n"
        output_label.parent.mkdir(parents=True, exist_ok=True)
        output_label.write_text(text, encoding="utf-8")

        image_count += 1
        instance_count += len(label_lines)
        filtered_polygon_count += filtered
        grouped_instance_count += grouped

    _write_data_yaml(output_root, names)
    return ConversionStats(
        image_count=image_count,
        instance_count=instance_count,
        filtered_polygon_count=filtered_polygon_count,
        grouped_instance_count=grouped_instance_count,
        names=names,
    )


def convert_cvat_xml_to_ultralytics_instance(
    input_root: Path,
    output_root: Path,
    categories: Sequence[str] | None = None,
    category_map: dict[str, str] | None = None,
) -> ConversionStats:
    """Convert a CVAT XML export; public name matching the module's purpose."""
    return build_ultralytics_instance_dataset(
        input_root,
        output_root,
        categories=categories,
        category_map=category_map,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    stats = build_ultralytics_instance_dataset(
        config.input_root,
        config.output_root,
        categories=config.categories,
        category_map=config.category_map,
    )
    print(
        f"Wrote {stats.image_count} images and {stats.instance_count} instances "
        f"to {config.output_root} with classes [{', '.join(stats.names)}]"
    )


def _resolve_input(input_path: Path) -> tuple[Path, Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".xml":
            raise ValueError(f"Input file must be a CVAT XML file: {input_path}")
        return input_path, input_path.parent
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_path}")

    annotations_path = input_path / ANNOTATIONS_FILENAME
    if annotations_path.is_file():
        return annotations_path, input_path

    candidates = sorted(input_path.rglob(ANNOTATIONS_FILENAME))
    if not candidates:
        raise FileNotFoundError(
            f"CVAT annotations file not found under {input_path}: "
            f"expected {ANNOTATIONS_FILENAME}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple {ANNOTATIONS_FILENAME} files found under {input_path}: "
            f"{candidates}"
        )
    return candidates[0], input_path


def _paths_overlap(input_root: Path, output_root: Path) -> bool:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    return (
        input_root == output_root
        or _is_relative_to(output_root, input_root)
        or _is_relative_to(input_root, output_root)
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _discover_labels(root: ET.Element, images: Sequence[ET.Element]) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()

    for name_element in root.findall(".//labels/label/name"):
        name = (name_element.text or "").strip()
        if name and name not in seen:
            labels.append(name)
            seen.add(name)

    for image in images:
        for shape in image:
            if shape.tag != "polygon":
                continue
            label = shape.attrib.get("label", "").strip()
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
    return tuple(labels)


def _resolve_categories(
    categories: Sequence[str] | None,
    source_labels: Sequence[str],
) -> tuple[str, ...]:
    selected = list(source_labels if categories is None else categories)
    _validate_category_names(selected, "categories")
    if not selected:
        raise ValueError("No CVAT categories found; provide --categories")
    return tuple(selected)


def _validate_category_names(names: Sequence[str], field_name: str) -> None:
    stripped = [str(name).strip() for name in names]
    if any(not name for name in stripped):
        raise ValueError(f"{field_name} must not contain empty names")
    if len(set(stripped)) != len(stripped):
        raise ValueError(f"{field_name} must not contain duplicates")


def _build_class_lookup(
    categories: Sequence[str],
    category_map: dict[str, str],
) -> tuple[dict[str, int], tuple[str, ...]]:
    source_to_class_id: dict[str, int] = {}
    target_to_class_id: dict[str, int] = {}
    names: list[str] = []

    for source_name in categories:
        source_name = source_name.strip()
        target_name = category_map.get(source_name, source_name).strip()
        if not target_name:
            raise ValueError(
                f"category_map produced an empty target name for {source_name!r}"
            )
        if target_name not in target_to_class_id:
            target_to_class_id[target_name] = len(names)
            names.append(target_name)
        source_to_class_id[source_name] = target_to_class_id[target_name]

    return source_to_class_id, tuple(names)


def _resolve_image_path(
    image_root: Path,
    image_name: str | None,
    annotations_path: Path,
) -> Path:
    if not image_name or not image_name.strip():
        raise ValueError(f"CVAT image has no name in {annotations_path}")

    normalized_name = image_name.replace("\\", "/")
    name_path = Path(normalized_name)
    candidates: list[Path] = []
    if name_path.is_absolute():
        candidates.append(name_path)
    else:
        candidates.append(image_root / name_path)

        # CVAT exports commonly retain the original dataset prefix, e.g.
        # ``project/data/raw/batch/images/frame.jpg``.  Drop everything up to
        # the input root's final directory name when present.
        parts = name_path.parts
        for index, part in enumerate(parts):
            if part == image_root.name and index + 1 < len(parts):
                candidates.append(image_root / Path(*parts[index + 1 :]))

        if parts and parts[0] != "images":
            candidates.append(image_root / "images" / name_path)
        candidates.append(image_root / "images" / name_path.name)

    for candidate in _unique_paths(candidates):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate.resolve()

    # Absolute paths in an XML export may point to a machine that is not the
    # current one.  A unique basename is a safe final resolution only when it
    # is unambiguous.
    basename_matches = sorted(
        path.resolve()
        for path in image_root.rglob(name_path.name)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(basename_matches) == 1:
        return basename_matches[0]
    if not basename_matches:
        raise FileNotFoundError(
            f"CVAT image not found for {image_name!r} under {image_root}"
        )
    raise ValueError(
        f"CVAT image name {image_name!r} matches multiple files under "
        f"{image_root}: {basename_matches}"
    )


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(path)
    return unique


def _output_image_relative_path(source: Path, image_root: Path) -> Path:
    images_root = image_root / "images"
    if _is_relative_to(source, images_root):
        relative = source.relative_to(images_root)
    elif _is_relative_to(source, image_root):
        relative = source.relative_to(image_root)
    else:
        relative = Path(source.name)
    if not relative.parts or relative == Path("."):
        raise ValueError(f"Unable to derive output path for source image: {source}")
    return relative


def _validate_output_relatives(
    relatives: Sequence[Path],
    sources: Sequence[Path],
) -> None:
    seen_images: dict[Path, Path] = {}
    seen_labels: dict[Path, Path] = {}
    for relative, source in zip(relatives, sources, strict=True):
        if relative in seen_images:
            raise ValueError(
                f"Multiple CVAT images map to output path {relative}: "
                f"{seen_images[relative]}, {source}"
            )
        seen_images[relative] = source

        label_relative = relative.with_suffix(".txt")
        if label_relative in seen_labels:
            raise ValueError(
                f"Multiple CVAT images map to label path {label_relative}: "
                f"{seen_labels[label_relative]}, {source}"
            )
        seen_labels[label_relative] = source


def _image_dimensions(image: ET.Element) -> tuple[int, int]:
    try:
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"CVAT image has invalid dimensions: "
            f"width={image.attrib.get('width')!r}, "
            f"height={image.attrib.get('height')!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError(
            f"CVAT image dimensions must be positive: width={width}, height={height}"
        )
    return width, height


def _image_to_label_lines(
    image: ET.Element,
    width: int,
    height: int,
    source_to_class_id: dict[str, int],
) -> tuple[list[str], int, int]:
    groups: list[tuple[int, str | None, list[np.ndarray]]] = []
    group_indices: dict[tuple[int, str], int] = {}
    filtered_polygon_count = 0

    for shape in image:
        if shape.tag in CVAT_SHAPE_TAGS and shape.tag != "polygon":
            raise ValueError(
                f"Unsupported CVAT shape '{shape.tag}' in image "
                f"{image.attrib.get('name', '<unnamed>')}"
            )
        if shape.tag != "polygon":
            continue

        label = shape.attrib.get("label")
        if label is None or not label.strip():
            raise ValueError(
                f"CVAT polygon has no label in image "
                f"{image.attrib.get('name', '<unnamed>')}"
            )
        label = label.strip()
        class_id = source_to_class_id.get(label)
        if class_id is None:
            filtered_polygon_count += 1
            continue

        points = _parse_float_points(shape.attrib.get("points"))
        if len(points) < 3:
            raise ValueError(f"Polygon for label '{label}' must have at least 3 points")

        group_id = shape.attrib.get("group_id")
        group_id = group_id.strip() if group_id is not None else None
        if not group_id:
            groups.append((class_id, None, [points]))
            continue

        key = (class_id, group_id)
        index = group_indices.get(key)
        if index is None:
            group_indices[key] = len(groups)
            groups.append((class_id, group_id, [points]))
        else:
            groups[index][2].append(points)

    lines: list[str] = []
    grouped_instance_count = 0
    for class_id, _group_id, polygons in groups:
        if len(polygons) > 1:
            merged = merge_multi_segment(polygons)
            segment = np.concatenate(merged, axis=0)
            grouped_instance_count += 1
        else:
            segment = polygons[0]

        coordinates = (segment / np.array([width, height], dtype=np.float32)).reshape(
            -1
        )
        values = [f"{float(value):.6f}" for value in coordinates]
        lines.append(" ".join([str(class_id), *values]))

    return lines, filtered_polygon_count, grouped_instance_count


def _parse_float_points(points: str | None) -> np.ndarray:
    if points is None:
        raise ValueError("CVAT polygon is missing points")
    parsed: list[tuple[float, float]] = []
    for raw_point in points.split(";"):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        xy = [part.strip() for part in raw_point.split(",")]
        if len(xy) != 2:
            raise ValueError(f"Invalid CVAT point: {raw_point}")
        try:
            x, y = float(xy[0]), float(xy[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CVAT point: {raw_point}") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"CVAT point must be finite: {raw_point}")
        parsed.append((x, y))
    return np.asarray(parsed, dtype=np.float32)


def min_index(first: np.ndarray, second: np.ndarray) -> tuple[int, int]:
    """Return the closest point indexes, matching Ultralytics' helper."""
    distances = ((first[:, None, :] - second[None, :, :]) ** 2).sum(-1)
    return tuple(
        int(value) for value in np.unravel_index(np.argmin(distances), distances.shape)
    )


def merge_multi_segment(segments: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Merge polygons with Ultralytics' minimum-distance connection rule.

    This is the algorithm used by ``ultralytics.data.converter`` for COCO
    annotations containing multiple polygon parts for one instance.
    """
    merged_segments = [
        np.asarray(segment, dtype=np.float32).reshape(-1, 2) for segment in segments
    ]
    if not merged_segments:
        return []
    if len(merged_segments) == 1:
        return merged_segments

    result: list[np.ndarray] = []
    index_pairs: list[list[int]] = [[] for _ in merged_segments]
    for index in range(1, len(merged_segments)):
        first, second = min_index(merged_segments[index - 1], merged_segments[index])
        index_pairs[index - 1].append(first)
        index_pairs[index].append(second)

    for direction in range(2):
        if direction == 0:
            for index, pair in enumerate(index_pairs):
                if len(pair) == 2 and pair[0] > pair[1]:
                    pair = pair[::-1]
                    merged_segments[index] = merged_segments[index][::-1, :]
                merged_segments[index] = np.roll(
                    merged_segments[index], -pair[0], axis=0
                )
                merged_segments[index] = np.concatenate(
                    [merged_segments[index], merged_segments[index][:1]]
                )
                if index in {0, len(index_pairs) - 1}:
                    result.append(merged_segments[index])
                else:
                    pair = [0, pair[1] - pair[0]]
                    result.append(merged_segments[index][pair[0] : pair[1] + 1])
        else:
            for index in range(len(index_pairs) - 1, -1, -1):
                if index not in {0, len(index_pairs) - 1}:
                    pair = index_pairs[index]
                    distance = abs(pair[1] - pair[0])
                    result.append(merged_segments[index][distance:])
    return result


def _clean_output(output_root: Path) -> None:
    for name in ("images", "labels"):
        path = output_root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    data_path = output_root / "data.yaml"
    if data_path.exists() or data_path.is_symlink():
        data_path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def _relative_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            raise IsADirectoryError(f"Output image path is a directory: {destination}")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(Path(os.path.relpath(source, start=destination.parent)))


def _write_data_yaml(output_root: Path, names: Sequence[str]) -> None:
    data = {
        "path": ".",
        "train": "images",
        "val": "images",
        "nc": len(names),
        "names": list(names),
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _load_config_values(config_path: Path | None, param_name: str) -> dict[str, Any]:
    if config_path is None:
        return {}
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    if param_name in loaded:
        values = loaded[param_name]
        if not isinstance(values, dict):
            raise ValueError(
                f"Config entry {param_name!r} must contain a mapping in {config_path}"
            )
        return values
    return loaded


def _coerce_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or [value]
    if not isinstance(value, Sequence):
        raise ValueError(f"Expected a string or a sequence of strings, got: {value!r}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Expected all list items to be strings, got: {item!r}")
        if "," in item:
            items.extend(part.strip() for part in item.split(",") if part.strip())
        elif item.strip():
            items.append(item.strip())
    return items


def _parse_category_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        parsed: dict[str, str] = {}
        for source, target in value.items():
            source_name = str(source).strip()
            target_name = str(target).strip()
            if not source_name or not target_name:
                raise ValueError(
                    f"Invalid category_map entry: {source!r} -> {target!r}"
                )
            parsed[source_name] = target_name
        return parsed

    specs = _coerce_string_list(value)
    if specs is None:
        return {}
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid category_map entry {spec!r}; expected SOURCE=TARGET"
            )
        source, target = spec.split("=", 1)
        source_name = source.strip()
        target_name = target.strip()
        if not source_name or not target_name:
            raise ValueError(
                f"Invalid category_map entry {spec!r}; expected SOURCE=TARGET"
            )
        parsed[source_name] = target_name
    return parsed


if __name__ == "__main__":
    main()
