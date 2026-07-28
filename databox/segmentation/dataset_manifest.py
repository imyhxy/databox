"""Read, write, and validate per-sample segmentation dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_FILENAME = "manifest.jsonl"
REQUIRED_FIELDS = (
    "sample_id",
    "task_name",
    "split",
    "image_name",
    "image_path",
    "gt_mask_path",
    "polygon_mask_path",
    "polyline_mask_path",
    "task_id",
    "job_id",
    "frame_id",
)
PATH_FIELDS = (
    "image_path",
    "gt_mask_path",
    "polygon_mask_path",
    "polyline_mask_path",
)
TEXT_FIELDS = ("sample_id", "task_name", "split", "image_name", *PATH_FIELDS)
INTEGER_FIELDS = ("task_id", "job_id", "frame_id")


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
        "gt_mask_path": set(),
        "polygon_mask_path": set(),
        "polyline_mask_path": set(),
    }
    for index, record in enumerate(records, 1):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Manifest record {index} missing fields: {missing}")

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

        image_path = PurePosixPath(record["image_path"])
        if image_path.name != record["image_name"]:
            raise ValueError(
                f"Manifest record {index} image_name does not match image_path: "
                f"{record['image_name']!r} != {image_path.name!r}"
            )

        for field in PATH_FIELDS:
            relative_path = _validate_relative_path(record[field], field, index)
            if validate_paths and not (dataset_root / Path(relative_path)).is_file():
                raise FileNotFoundError(
                    f"Manifest record {index} references missing {field}: "
                    f"{dataset_root / Path(relative_path)}"
                )

        for field, values in seen_values.items():
            value = record[field]
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
