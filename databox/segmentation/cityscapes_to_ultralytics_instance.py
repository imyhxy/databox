"""Convert Cityscapes instance masks into an Ultralytics instance dataset.

The input root must contain the standard Cityscapes raw layout::

    <input-root>/gtFine/<split>/<city>/*_gtFine_instanceIds.png
    <input-root>/leftImg8bit/<split>/<city>/*_leftImg8bit.png

The output layout is::

    <output-root>/images/<split>/<city>/*.png   # relative symlinks
    <output-root>/labels/<split>/<city>/*.txt   # Ultralytics polygons
    <output-root>/data.yaml

By default the converter keeps the standard Cityscapes instance categories:
``person``, ``rider``, ``car``, ``truck``, ``bus``, ``train``,
``motorcycle``, and ``bicycle``.

Cityscapes encodes each instance pixel as ``class_id * 1000 + instance_id``.
The converter reads those visible masks directly, so occluded polygons are not
emitted as overlapping instances. Pixels with an ID below 1000 (stuff labels
and group annotations) are ignored.

``input_root`` and ``output_root`` are intentionally read from the command
line only, even when a YAML config is supplied. This keeps DVC stage
dependencies and outputs explicit at invocation time.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PARAM_NAME = "cityscapes_to_ultralytics_instance"
KNOWN_SPLITS = ("train", "val", "test")
DEFAULT_INSTANCE_CATEGORIES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# IDs from Cityscapes' ``labels.py``. Only labels with instance annotations are
# accepted because the source artifact is an instance-ID image.
CITYSCAPES_INSTANCE_CLASS_IDS = {
    "person": 24,
    "rider": 25,
    "car": 26,
    "truck": 27,
    "bus": 28,
    "caravan": 29,
    "trailer": 30,
    "train": 31,
    "motorcycle": 32,
    "bicycle": 33,
}


@dataclass(frozen=True)
class Config:
    input_root: Path
    output_root: Path
    splits: tuple[str, ...]
    categories: tuple[str, ...]
    category_map: dict[str, str]


@dataclass(frozen=True)
class ConversionStats:
    image_count: int
    instance_count: int
    skipped_instances: int
    splits: tuple[str, ...]
    names: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Cityscapes instance-ID masks into an Ultralytics instance "
            "segmentation dataset."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Cityscapes raw root containing gtFine/ and leftImg8bit/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output dataset root containing images/, labels/, and data.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional YAML config file. If --param-name exists inside it, "
            "that section is used."
        ),
    )
    parser.add_argument(
        "--param-name",
        type=str,
        default=PARAM_NAME,
        help="Key to read from the YAML config file when present.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=KNOWN_SPLITS,
        help="Splits to convert. Defaults to all splits with instance-ID masks.",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help=(
            "Source Cityscapes labels to keep. Defaults to the standard "
            "instance categories."
        ),
    )
    parser.add_argument(
        "--category-map",
        nargs="+",
        metavar="SOURCE=TARGET",
        help=(
            "Optional source-to-target class remapping, for example: "
            "car=vehicle truck=vehicle"
        ),
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    values = _load_config_values(args.config, args.param_name)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    categories_value = (
        args.categories if args.categories is not None else values.get("categories")
    )
    splits_value = args.splits if args.splits is not None else values.get("splits")
    category_map_value = (
        args.category_map
        if args.category_map is not None
        else values.get("category_map")
    )

    categories = tuple(
        _coerce_string_list(categories_value) or DEFAULT_INSTANCE_CATEGORIES
    )
    if not categories:
        raise ValueError("categories must not be empty")

    category_map = _parse_category_map(category_map_value)
    splits = _resolve_requested_splits(input_root, splits_value)

    return Config(
        input_root=input_root,
        output_root=output_root,
        splits=splits,
        categories=categories,
        category_map=category_map,
    )


def build_ultralytics_instance_dataset(
    input_root: Path,
    output_root: Path,
    categories: Sequence[str] | None = None,
    category_map: dict[str, str] | None = None,
    splits: Sequence[str] | None = None,
) -> ConversionStats:
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    if input_root == output_root:
        raise ValueError("Output root must differ from input root")

    _validate_input_root(input_root)

    selected_categories = tuple(categories or DEFAULT_INSTANCE_CATEGORIES)
    if not selected_categories:
        raise ValueError("categories must not be empty")

    requested_splits = tuple(splits) if splits is not None else None
    resolved_splits = _resolve_requested_splits(input_root, requested_splits)
    source_to_class_id, names = _build_class_lookup(
        selected_categories,
        category_map or {},
    )
    class_to_class_id = {
        CITYSCAPES_INSTANCE_CLASS_IDS[source_name]: class_id
        for source_name, class_id in source_to_class_id.items()
    }

    _clean_output(output_root)

    image_count = 0
    instance_count = 0
    skipped_instances = 0
    written_splits = []

    for split in resolved_splits:
        instance_paths = _collect_instance_paths(input_root, split)
        if not instance_paths:
            if requested_splits is None:
                continue
            raise ValueError(f"No instance-ID masks found for split: {split}")

        written_splits.append(split)
        for instance_path in instance_paths:
            image_path = _image_path_for_instance(input_root, instance_path)
            if not image_path.exists():
                raise FileNotFoundError(
                    "Missing Cityscapes image for instance mask "
                    f"{instance_path}: {image_path}"
                )

            relative_dir = instance_path.parent.relative_to(input_root / "gtFine")
            output_image = output_root / "images" / relative_dir / image_path.name
            output_label = (
                output_root / "labels" / relative_dir / f"{image_path.stem}.txt"
            )
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            _relative_symlink(image_path, output_image)

            instance_image = _read_instance_image(instance_path)
            label_lines, skipped = _instance_image_to_label_lines(
                instance_image,
                class_to_class_id,
            )
            text = "\n".join(label_lines)
            if text:
                text += "\n"
            output_label.write_text(text, encoding="utf-8")

            image_count += 1
            instance_count += len(label_lines)
            skipped_instances += skipped

    if not written_splits:
        raise ValueError(
            f"No Cityscapes instance-ID masks found under {input_root / 'gtFine'}"
        )

    _write_data_yaml(output_root, tuple(written_splits), names)
    return ConversionStats(
        image_count=image_count,
        instance_count=instance_count,
        skipped_instances=skipped_instances,
        splits=tuple(written_splits),
        names=names,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    stats = build_ultralytics_instance_dataset(
        config.input_root,
        config.output_root,
        categories=config.categories,
        category_map=config.category_map,
        splits=config.splits,
    )
    names = ", ".join(stats.names)
    splits = ", ".join(stats.splits)
    print(
        "Wrote "
        f"{stats.image_count} images, "
        f"{stats.instance_count} instances, "
        f"{stats.skipped_instances} skipped instances "
        f"to {config.output_root} "
        f"across splits [{splits}] with classes [{names}]"
    )


def _validate_input_root(input_root: Path) -> None:
    for dirname in ("gtFine", "leftImg8bit"):
        path = input_root / dirname
        if not path.exists():
            raise FileNotFoundError(f"Required Cityscapes path not found: {path}")


def _load_config_values(config_path: Path | None, param_name: str) -> dict[str, Any]:
    if config_path is None:
        return {}
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
        if items:
            return items
        return [value] if value else []
    if not isinstance(value, Sequence):
        raise ValueError(f"Expected a string or a sequence of strings, got: {value!r}")

    items = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Expected all list items to be strings, got: {item!r}")
        if "," in item:
            items.extend(part.strip() for part in item.split(",") if part.strip())
        else:
            stripped = item.strip()
            if stripped:
                items.append(stripped)
    return items


def _parse_category_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}

    if isinstance(value, dict):
        parsed = {}
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


def _resolve_requested_splits(
    input_root: Path,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested is None:
        discovered = [
            split for split in KNOWN_SPLITS if _has_annotated_split(input_root, split)
        ]
        if not discovered:
            raise ValueError(
                f"No Cityscapes instance-ID masks found under {input_root / 'gtFine'}"
            )
        return tuple(discovered)

    splits = tuple(_coerce_string_list(requested) or [])
    if not splits:
        raise ValueError("splits must not be empty")
    invalid = [split for split in splits if split not in KNOWN_SPLITS]
    if invalid:
        raise ValueError(f"Unsupported splits requested: {invalid}")
    for split in splits:
        gt_dir = input_root / "gtFine" / split
        image_dir = input_root / "leftImg8bit" / split
        if not gt_dir.exists() or not image_dir.exists():
            raise FileNotFoundError(
                f"Missing Cityscapes split directories for {split}: "
                f"{gt_dir}, {image_dir}"
            )
    return splits


def _has_annotated_split(input_root: Path, split: str) -> bool:
    gt_dir = input_root / "gtFine" / split
    image_dir = input_root / "leftImg8bit" / split
    if not gt_dir.exists() or not image_dir.exists():
        return False
    return any(gt_dir.rglob("*_gtFine_instanceIds.png"))


def _build_class_lookup(
    categories: Sequence[str],
    category_map: dict[str, str],
) -> tuple[dict[str, int], tuple[str, ...]]:
    source_to_class_id = {}
    names = []
    target_to_class_id = {}
    for source_name in categories:
        normalized_source = source_name.strip()
        if not normalized_source:
            raise ValueError("category names must not be empty")
        if normalized_source not in CITYSCAPES_INSTANCE_CLASS_IDS:
            supported = ", ".join(CITYSCAPES_INSTANCE_CLASS_IDS)
            raise ValueError(
                f"Unsupported Cityscapes instance category {normalized_source!r}; "
                f"expected one of: {supported}"
            )
        target_name = category_map.get(normalized_source, normalized_source).strip()
        if not target_name:
            raise ValueError(
                f"category_map produced an empty target name for {normalized_source!r}"
            )
        if target_name not in target_to_class_id:
            target_to_class_id[target_name] = len(names)
            names.append(target_name)
        source_to_class_id[normalized_source] = target_to_class_id[target_name]
    return source_to_class_id, tuple(names)


def _clean_output(output_root: Path) -> None:
    for name in ("images", "labels"):
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
    yaml_path = output_root / "data.yaml"
    if yaml_path.exists():
        yaml_path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def _collect_instance_paths(input_root: Path, split: str) -> list[Path]:
    gt_dir = input_root / "gtFine" / split
    if not gt_dir.exists():
        return []
    return sorted(gt_dir.rglob("*_gtFine_instanceIds.png"))


def _image_path_for_instance(input_root: Path, instance_path: Path) -> Path:
    relative = instance_path.relative_to(input_root / "gtFine")
    image_name = instance_path.name.replace(
        "_gtFine_instanceIds.png",
        "_leftImg8bit.png",
    )
    return input_root / "leftImg8bit" / relative.parent / image_name


def _read_instance_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read Cityscapes instance-ID mask: {path}")
    if image.ndim != 2:
        raise ValueError(f"Cityscapes instance-ID mask must be single-channel: {path}")
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(
            f"Cityscapes instance-ID mask must contain integer IDs: {path}"
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"Cityscapes instance-ID mask is empty: {path}")
    return image


def _instance_image_to_label_lines(
    instance_image: np.ndarray,
    class_to_class_id: dict[int, int],
) -> tuple[list[str], int]:
    if instance_image.ndim != 2:
        raise ValueError("Cityscapes instance-ID image must be single-channel")
    height, width = instance_image.shape
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in instance-ID image: {width}x{height}")

    lines = []
    skipped_instances = 0
    for instance_id in sorted(int(value) for value in np.unique(instance_image)):
        if instance_id < 1000:
            continue
        class_id = instance_id // 1000
        target_class_id = class_to_class_id.get(class_id)
        if target_class_id is None:
            continue

        binary_mask = np.asarray(instance_image == instance_id, dtype=np.uint8)
        segment = _mask_to_segment(binary_mask)
        if segment is None:
            skipped_instances += 1
            continue

        coordinates = (segment / np.array([width, height], dtype=np.float32)).reshape(
            -1
        )
        values = [f"{value:.6f}" for value in coordinates]
        lines.append(" ".join([str(target_class_id), *values]))
    return lines, skipped_instances


def _mask_to_segment(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        np.ascontiguousarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    segments = [
        contour.reshape(-1, 2).astype(np.float32)
        for contour in contours
        if len(contour) >= 3
    ]
    if not segments:
        return None
    if len(segments) == 1:
        return segments[0]
    return np.concatenate(_merge_multi_segment(segments), axis=0)


def _merge_multi_segment(segments: list[np.ndarray]) -> list[np.ndarray]:
    """Join disconnected contours using Ultralytics' COCO conversion rule."""
    normalized = [
        np.asarray(segment, dtype=np.float32).reshape(-1, 2) for segment in segments
    ]
    index_pairs: list[list[int]] = [[] for _ in normalized]

    for index in range(1, len(normalized)):
        first, second = _nearest_point_indices(normalized[index - 1], normalized[index])
        index_pairs[index - 1].append(first)
        index_pairs[index].append(second)

    merged: list[np.ndarray] = []
    for direction in range(2):
        if direction == 0:
            iterator = enumerate(index_pairs)
        else:
            iterator = reversed(list(enumerate(index_pairs)))

        for index, pair in iterator:
            if direction == 0:
                if len(pair) == 2 and pair[0] > pair[1]:
                    pair = pair[::-1]
                    normalized[index] = normalized[index][::-1, :]

                normalized[index] = np.roll(normalized[index], -pair[0], axis=0)
                normalized[index] = np.concatenate(
                    [normalized[index], normalized[index][:1]], axis=0
                )
                if index in {0, len(index_pairs) - 1}:
                    merged.append(normalized[index])
                else:
                    pair = [0, pair[1] - pair[0]]
                    merged.append(normalized[index][pair[0] : pair[1] + 1])
            elif index not in {0, len(index_pairs) - 1}:
                pair = index_pairs[index]
                distance = abs(pair[1] - pair[0])
                merged.append(normalized[index][distance:])
    return merged


def _nearest_point_indices(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[int, int]:
    distances = ((first[:, None, :] - second[None, :, :]) ** 2).sum(axis=-1)
    return tuple(
        int(value) for value in np.unravel_index(np.argmin(distances), distances.shape)
    )


def _relative_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = Path(os.path.relpath(source, start=destination.parent))
    destination.symlink_to(target)


def _write_data_yaml(
    output_root: Path,
    splits: Sequence[str],
    names: Sequence[str],
) -> None:
    data = {
        "path": ".",
        "nc": len(names),
        "names": list(names),
    }
    for split in splits:
        data[split] = f"images/{split}"
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
