"""Convert wheel contact annotations from CVAT XML to Ultralytics pose data.

The input directory contains ``annotations.xml`` and an ``images`` directory
(usually a relative symlink to the source dataset).  The output directory is
an Ultralytics pose dataset with relative image symlinks, one label file per
CVAT image, ``data.txt`` and ``data.yaml``.

The wheel-contact task has one ``wheel`` box and one ``wheel_pose`` skeleton
per ``group_id``.  Annotation-level pairing and point errors are reported as
warnings and skipped so a small amount of annotation noise does not discard
the rest of the dataset.  Broken XML or missing source images remain fatal.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

ANNOTATIONS_FILENAME = "annotations.xml"
BOX_LABEL = "wheel"
SKELETON_LABEL = "wheel_pose"
KEYPOINT_LABELS = ("center", "contact")
IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


@dataclass(frozen=True)
class ConversionStats:
    """Counts produced by one conversion."""

    image_count: int
    instance_count: int
    skipped_group_count: int
    skipped_shape_count: int


@dataclass(frozen=True)
class ImageInfo:
    """Validated source information for one CVAT image element."""

    element: ET.Element
    source: Path
    relative: Path
    width: int
    height: int


@dataclass
class GroupMembers:
    """Expected wheel box and skeleton members for one group."""

    boxes: list[ET.Element]
    skeletons: list[ET.Element]


@dataclass(frozen=True)
class Keypoint:
    """One parsed keypoint in source-image pixel coordinates."""

    x: float
    y: float
    visibility: int


class AnnotationIssue(ValueError):
    """An annotation can be skipped without invalidating the input dataset."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse converter arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert CVAT wheel box/skeleton annotations into an "
            "Ultralytics pose dataset."
        )
    )
    parser.add_argument(
        "--input-root",
        "--input",
        dest="input_root",
        type=Path,
        required=True,
        help="Input root containing annotations.xml and images.",
    )
    parser.add_argument(
        "--output-root",
        "--output",
        dest="output_root",
        type=Path,
        required=True,
        help="Output root for the Ultralytics pose dataset.",
    )
    return parser.parse_args(argv)


def convert_cvat_xml_to_ultralytics_pose(
    input_root: Path,
    output_root: Path,
) -> ConversionStats:
    """Convert one CVAT XML export into an Ultralytics pose dataset.

    Annotation-level issues are emitted as warnings and skipped.  Input and
    image-layout errors raise an exception before any output is replaced.
    """
    input_path = Path(input_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    annotations_path, image_root = _resolve_input(input_path)

    input_overlap_root = input_path if input_path.is_dir() else image_root
    if _paths_overlap(input_overlap_root, output_path):
        raise ValueError(
            "Output root must be separate from the input root and its images"
        )

    tree = ET.parse(annotations_path)
    root = tree.getroot()
    if root.tag != "annotations":
        raise ValueError(
            f"Expected CVAT annotations root, got <{root.tag}> in {annotations_path}"
        )

    image_elements = root.findall("./image")
    if not image_elements:
        raise ValueError(f"No CVAT images found in {annotations_path}")

    image_infos = _validate_images(image_elements, image_root, annotations_path)
    records: list[tuple[ImageInfo, list[str]]] = []
    skipped_group_count = 0
    skipped_shape_count = 0
    for info in image_infos:
        lines, skipped_groups, skipped_shapes = _image_to_label_lines(info)
        records.append((info, lines))
        skipped_group_count += skipped_groups
        skipped_shape_count += skipped_shapes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.tmp-",
            dir=output_path.parent,
        )
    )
    try:
        _write_dataset(staging_path, records)
        _replace_output(output_path, staging_path)
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    return ConversionStats(
        image_count=len(records),
        instance_count=sum(len(lines) for _, lines in records),
        skipped_group_count=skipped_group_count,
        skipped_shape_count=skipped_shape_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the converter CLI and return a process status."""
    args = parse_args(argv)
    try:
        stats = convert_cvat_xml_to_ultralytics_pose(
            args.input_root,
            args.output_root,
        )
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        f"Wrote {stats.image_count} images and {stats.instance_count} instances "
        f"to {Path(args.output_root).expanduser()} "
        f"(skipped groups: {stats.skipped_group_count}, "
        f"skipped shapes: {stats.skipped_shape_count})"
    )
    return 0


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


def _validate_images(
    image_elements: Sequence[ET.Element],
    image_root: Path,
    annotations_path: Path,
) -> list[ImageInfo]:
    infos: list[ImageInfo] = []
    seen_relatives: dict[Path, str] = {}
    seen_labels: dict[Path, str] = {}
    for image in image_elements:
        name = image.attrib.get("name", "").strip()
        if not name:
            raise ValueError(f"CVAT image has no name in {annotations_path}")

        width, height = _image_dimensions(image)
        source = _resolve_image_path(image_root, name)
        relative = _output_image_relative_path(source, image_root)
        _validate_relative_image_path(relative, source)

        previous = seen_relatives.get(relative)
        if previous is not None:
            raise ValueError(
                f"Multiple CVAT images map to output path {relative}: "
                f"{previous!r}, {name!r}"
            )
        seen_relatives[relative] = name

        label_relative = relative.with_suffix(".txt")
        previous = seen_labels.get(label_relative)
        if previous is not None:
            raise ValueError(
                f"Multiple CVAT images map to label path {label_relative}: "
                f"{previous!r}, {name!r}"
            )
        seen_labels[label_relative] = name
        infos.append(
            ImageInfo(
                element=image,
                source=source,
                relative=relative,
                width=width,
                height=height,
            )
        )
    return infos


def _image_dimensions(image: ET.Element) -> tuple[int, int]:
    try:
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "CVAT image has invalid dimensions: "
            f"width={image.attrib.get('width')!r}, "
            f"height={image.attrib.get('height')!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError(
            f"CVAT image dimensions must be positive: width={width}, height={height}"
        )
    return width, height


def _image_to_label_lines(
    info: ImageInfo,
) -> tuple[list[str], int, int]:
    image_name = info.element.attrib.get("name", "<unnamed>")
    groups: dict[int, GroupMembers] = {}
    skipped_group_count = 0
    skipped_shape_count = 0

    for shape in info.element:
        if shape.tag == "box":
            if shape.attrib.get("label", "").strip() != BOX_LABEL:
                _warn(
                    f"image={image_name}: unsupported box label "
                    f"{shape.attrib.get('label')!r}; skipped"
                )
                skipped_shape_count += 1
                continue
            group_id = _parse_group_id(shape.attrib.get("group_id"))
            if group_id is None:
                _warn(f"image={image_name}: wheel box has no valid group_id; skipped")
                skipped_shape_count += 1
                continue
            groups.setdefault(group_id, GroupMembers([], [])).boxes.append(shape)
        elif shape.tag == "skeleton":
            if shape.attrib.get("label", "").strip() != SKELETON_LABEL:
                _warn(
                    f"image={image_name}: unsupported skeleton label "
                    f"{shape.attrib.get('label')!r}; skipped"
                )
                skipped_shape_count += 1
                continue
            group_id = _parse_group_id(shape.attrib.get("group_id"))
            if group_id is None:
                _warn(
                    f"image={image_name}: wheel_pose skeleton has no valid "
                    "group_id; skipped"
                )
                skipped_shape_count += 1
                continue
            groups.setdefault(group_id, GroupMembers([], [])).skeletons.append(shape)
        else:
            _warn(f"image={image_name}: unsupported annotation <{shape.tag}>; skipped")
            skipped_shape_count += 1

    lines: list[str] = []
    for group_id, members in groups.items():
        if len(members.boxes) != 1 or len(members.skeletons) != 1:
            _warn(
                f"image={image_name} group={group_id}: expected exactly one "
                f"box and one skeleton, got {len(members.boxes)} box(es) and "
                f"{len(members.skeletons)} skeleton(s); skipped"
            )
            skipped_group_count += 1
            skipped_shape_count += len(members.boxes) + len(members.skeletons)
            continue

        try:
            lines.append(
                _pair_to_label_line(
                    members.boxes[0],
                    members.skeletons[0],
                    info.width,
                    info.height,
                )
            )
        except AnnotationIssue as error:
            _warn(f"image={image_name} group={group_id}: {error}; skipped")
            skipped_group_count += 1
            skipped_shape_count += 2

    return lines, skipped_group_count, skipped_shape_count


def _pair_to_label_line(
    box: ET.Element,
    skeleton: ET.Element,
    width: int,
    height: int,
) -> str:
    if _parse_bool(box.attrib.get("outside"), "box outside"):
        raise AnnotationIssue("box is outside")
    if _parse_bool(skeleton.attrib.get("outside"), "skeleton outside"):
        raise AnnotationIssue("skeleton is outside")
    _parse_bool(box.attrib.get("occluded"), "box occluded")
    _parse_bool(skeleton.attrib.get("occluded"), "skeleton occluded")

    coordinates = {
        name: _parse_float_attribute(box, name) for name in ("xtl", "ytl", "xbr", "ybr")
    }
    xtl, ytl = coordinates["xtl"], coordinates["ytl"]
    xbr, ybr = coordinates["xbr"], coordinates["ybr"]
    if not xtl < xbr or not ytl < ybr:
        raise AnnotationIssue("box coordinates are degenerate")
    if not (0 <= xtl <= width and 0 <= xbr <= width):
        raise AnnotationIssue("box x coordinates are outside the image")
    if not (0 <= ytl <= height and 0 <= ybr <= height):
        raise AnnotationIssue("box y coordinates are outside the image")

    keypoints = _parse_skeleton_points(skeleton, width, height)
    values = [
        0,
        (xtl + xbr) / 2 / width,
        (ytl + ybr) / 2 / height,
        (xbr - xtl) / width,
        (ybr - ytl) / height,
    ]
    for name in KEYPOINT_LABELS:
        point = keypoints[name]
        values.extend(
            [
                point.x / width,
                point.y / height,
                point.visibility,
            ]
        )
    return " ".join(
        str(value) if isinstance(value, int) else f"{value:.6f}" for value in values
    )


def _parse_skeleton_points(
    skeleton: ET.Element,
    width: int,
    height: int,
) -> dict[str, Keypoint]:
    points: dict[str, Keypoint] = {}
    for child in skeleton:
        if child.tag == "attribute":
            continue
        if child.tag != "points":
            raise AnnotationIssue(f"unsupported skeleton child <{child.tag}>")
        label = child.attrib.get("label", "").strip()
        if label not in KEYPOINT_LABELS:
            raise AnnotationIssue(f"unexpected keypoint label {label!r}")
        if label in points:
            raise AnnotationIssue(f"duplicate keypoint label {label!r}")
        x, y = _parse_point(child.attrib.get("points"))
        outside = _parse_bool(child.attrib.get("outside"), f"{label} outside")
        occluded = _parse_bool(child.attrib.get("occluded"), f"{label} occluded")
        if outside:
            points[label] = Keypoint(0.0, 0.0, 0)
            continue
        if not (0 <= x <= width and 0 <= y <= height):
            raise AnnotationIssue(f"{label} coordinates are outside the image")
        points[label] = Keypoint(x, y, 1 if occluded else 2)

    missing = [label for label in KEYPOINT_LABELS if label not in points]
    if missing:
        raise AnnotationIssue(f"missing keypoint label(s): {', '.join(missing)}")
    return points


def _parse_group_id(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        group_id = int(value)
    except ValueError:
        return None
    return group_id if group_id > 0 else None


def _parse_bool(value: str | None, name: str) -> bool:
    if value is None or not value.strip():
        return False
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no"}:
        return False
    if normalized in {"1", "true", "yes"}:
        return True
    raise AnnotationIssue(f"{name} must be 0/1, got {value!r}")


def _parse_float_attribute(element: ET.Element, name: str) -> float:
    value = element.attrib.get(name)
    if value is None:
        raise AnnotationIssue(f"box is missing {name}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise AnnotationIssue(f"box {name} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise AnnotationIssue(f"box {name} must be finite")
    return parsed


def _parse_point(value: str | None) -> tuple[float, float]:
    if value is None:
        raise AnnotationIssue("keypoint is missing points")
    chunks = [chunk.strip() for chunk in value.split(";") if chunk.strip()]
    if len(chunks) != 1:
        raise AnnotationIssue("each keypoint must contain exactly one coordinate")
    components = [component.strip() for component in chunks[0].split(",")]
    if len(components) != 2:
        raise AnnotationIssue(f"invalid keypoint coordinates: {value!r}")
    try:
        x, y = float(components[0]), float(components[1])
    except ValueError as exc:
        raise AnnotationIssue(f"invalid keypoint coordinates: {value!r}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise AnnotationIssue("keypoint coordinates must be finite")
    return x, y


def _resolve_image_path(image_root: Path, image_name: str) -> Path:
    normalized_name = image_name.replace("\\", "/")
    name_path = Path(normalized_name)
    candidates: list[Path] = []
    if name_path.is_absolute():
        candidates.append(name_path)
    else:
        candidates.extend(
            [
                image_root / name_path,
                image_root / "images" / name_path,
                image_root / "images" / name_path.name,
            ]
        )

    parts = name_path.parts
    image_indexes = [index for index, part in enumerate(parts) if part == "images"]
    if image_indexes:
        suffix = Path(*parts[image_indexes[-1] + 1 :])
        if suffix.parts:
            candidates.insert(0, image_root / "images" / suffix)

    for candidate in _unique_paths(candidates):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate.resolve()

    basename_matches = sorted(
        path.resolve()
        for path in (image_root / "images").rglob(name_path.name)
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
    source_resolved = source.resolve()
    images_root = (image_root / "images").resolve()
    image_root_resolved = image_root.resolve()
    if _is_relative_to(source_resolved, images_root):
        relative = source_resolved.relative_to(images_root)
    elif _is_relative_to(source_resolved, image_root_resolved):
        relative = source_resolved.relative_to(image_root_resolved)
    else:
        relative = Path(source.name)
    return relative


def _validate_relative_image_path(relative: Path, source: Path) -> None:
    if (
        relative.is_absolute()
        or not relative.parts
        or relative == Path(".")
        or ".." in relative.parts
        or relative.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        raise ValueError(f"Invalid relative image path for {source}: {relative}")


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return (
        first == second
        or _is_relative_to(first, second)
        or _is_relative_to(second, first)
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_dataset(
    output_root: Path,
    records: Sequence[tuple[ImageInfo, list[str]]],
) -> None:
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    data_lines: list[str] = []

    for info, lines in records:
        output_image = output_images / info.relative
        _relative_symlink(info.source, output_image)

        output_label = output_labels / info.relative.with_suffix(".txt")
        output_label.parent.mkdir(parents=True, exist_ok=True)
        output_label.write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )
        data_lines.append(f"./images/{info.relative.as_posix()}")

    (output_root / "data.txt").write_text(
        ("\n".join(data_lines) + "\n") if data_lines else "",
        encoding="utf-8",
    )
    data = {
        "train": "data.txt",
        "val": "data.txt",
        "names": {0: BOX_LABEL},
        "kpt_shape": [2, 3],
        "kpt_names": {0: list(KEYPOINT_LABELS)},
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _relative_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            raise IsADirectoryError(f"Output image path is a directory: {destination}")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, start=destination.parent))


def _replace_output(output_root: Path, staging_root: Path) -> None:
    if output_root.is_symlink() or output_root.is_file():
        output_root.unlink()
    elif output_root.exists():
        shutil.rmtree(output_root)
    staging_root.rename(output_root)


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
