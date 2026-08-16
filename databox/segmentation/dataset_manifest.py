"""Read, write, and validate per-sample segmentation dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_FILENAME = "manifest.jsonl"
MASK_PATH_KEYS = ("polygon", "polyline", "vehicle", "main")
REQUIRED_FIELDS = (
    "sample_id",
    "task_name",
    "split",
    "image_path",
    "mask_paths",
    "width",
    "height",
    "task_id",
    "job_id",
    "frame_id",
)
TEXT_FIELDS = ("sample_id", "task_name", "split", "image_path")
INTEGER_FIELDS = ("width", "height", "task_id", "job_id", "frame_id")
REMOVED_FIELDS = (
    "image_name",
    "gt_mask_path",
    "polygon_mask_path",
    "polyline_mask_path",
)


def read_manifest(
    dataset_root: Path, *, validate_paths: bool = True
) -> list[dict[str, Any]]:
    """Read and validate ``manifest.jsonl`` from a dataset root."""
    path = dataset_root / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record must be an object: {path}:{line_number}")
        records.append(record)

    validate_manifest_records(dataset_root, records, validate_paths=validate_paths)
    return records


def write_manifest(dataset_root: Path, records: list[dict[str, Any]]) -> Path:
    """Validate and deterministically write a dataset manifest."""
    validate_manifest_records(dataset_root, records, validate_paths=True)
    ordered = sorted(
        records, key=lambda record: (record["split"], record["image_path"])
    )
    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in ordered
    )
    path = dataset_root / MANIFEST_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def validate_manifest_records(
    dataset_root: Path,
    records: list[dict[str, Any]],
    *,
    validate_paths: bool,
) -> None:
    """Validate required fields, uniqueness, and referenced files."""
    if not records:
        raise ValueError(
            f"Dataset manifest must contain at least one record: {dataset_root}"
        )

    seen_values = {
        "sample_id": set(),
        "image_path": set(),
        **{f"mask_paths.{key}": set() for key in MASK_PATH_KEYS},
    }
    for index, record in enumerate(records, 1):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Manifest record {index} missing fields: {missing}")
        removed = [field for field in REMOVED_FIELDS if field in record]
        if removed:
            raise ValueError(
                f"Manifest record {index} contains removed fields: {removed}"
            )

        for field in TEXT_FIELDS:
            value = record[field]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"Manifest record {index} field {field!r} must be a "
                    "non-empty string"
                )
        for field in INTEGER_FIELDS:
            value = record[field]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Manifest record {index} field {field!r} must be an integer"
                )
            if value <= 0 and field in ("width", "height"):
                raise ValueError(
                    f"Manifest record {index} field {field!r} must be positive"
                )

        image_relative_path = _validate_relative_path(
            record["image_path"], "image_path", index
        )
        if validate_paths and not (
            dataset_root / Path(image_relative_path)
        ).is_file():
            raise FileNotFoundError(
                f"Manifest record {index} references missing image_path: "
                f"{dataset_root / Path(image_relative_path)}"
            )

        mask_paths = record["mask_paths"]
        if not isinstance(mask_paths, dict):
            raise ValueError(
                f"Manifest record {index} field 'mask_paths' must be an object"
            )
        missing_masks = [key for key in MASK_PATH_KEYS if key not in mask_paths]
        extra_masks = [key for key in mask_paths if key not in MASK_PATH_KEYS]
        if missing_masks or extra_masks:
            raise ValueError(
                f"Manifest record {index} field 'mask_paths' must contain exactly "
                f"{list(MASK_PATH_KEYS)}; missing={missing_masks}, extra={extra_masks}"
            )

        for key in MASK_PATH_KEYS:
            field = f"mask_paths.{key}"
            value = mask_paths[key]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"Manifest record {index} field {field!r} must be a "
                    "non-empty string"
                )
            relative_path = _validate_relative_path(value, field, index)
            if validate_paths and not (dataset_root / Path(relative_path)).is_file():
                raise FileNotFoundError(
                    f"Manifest record {index} references missing {field}: "
                    f"{dataset_root / Path(relative_path)}"
                )

        for field, values in seen_values.items():
            value = (
                record[field]
                if field != "image_path" and not field.startswith("mask_paths.")
                else (
                    record["image_path"]
                    if field == "image_path"
                    else mask_paths[field.removeprefix("mask_paths.")]
                )
            )
            if value in values:
                raise ValueError(f"Duplicate manifest {field}: {value!r}")
            values.add(value)


def _validate_relative_path(value: str, field: str, index: int) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(
            f"Manifest record {index} field {field!r} must be a relative POSIX path: "
            f"{value!r}"
        )
    return path
