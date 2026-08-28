"""Split an Ultralytics pose dataset into train and validation files.

The pose converter deliberately writes one complete ``data.txt`` manifest for
all CVAT images.  This stage is kept separate so changing the split ratio only
rewrites the split metadata and does not repeat annotation conversion.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SEED = 1234
PARAMS_SECTION = "wheel_contact_split"
IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse split arguments."""
    parser = argparse.ArgumentParser(
        description="Split an Ultralytics pose dataset into train and val sets."
    )
    parser.add_argument(
        "--dataset-root",
        "--input-root",
        "--input",
        dest="dataset_root",
        required=True,
        type=Path,
        help="Pose dataset root containing images, labels, and data.txt.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        help=(
            "YAML parameter file. The wheel_contact_split section supplies "
            "train and seed values."
        ),
    )
    parser.add_argument(
        "--train",
        "--train-ratio",
        dest="train_ratio",
        type=float,
        help="Fraction assigned to train; defaults to 0.8.",
    )
    parser.add_argument("--seed", type=int, help="Random seed; defaults to 1234.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the split CLI and return a process status."""
    args = parse_args(argv)
    try:
        train_ratio, seed = _resolve_options(args)
        stats = split_ultralytics_pose_dataset(
            args.dataset_root,
            train_ratio=train_ratio,
            seed=seed,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        f"Wrote {stats['train_count']} train and {stats['val_count']} val "
        f"entries to {Path(args.dataset_root).expanduser()}"
    )
    return 0


def split_ultralytics_pose_dataset(
    dataset_root: Path,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Write deterministic ``train.txt`` and ``val.txt``.

    Entries are read from the converter's ``data.txt`` manifest, rather than
    rediscovering images on disk.  That preserves CVAT images with empty
    labels and makes the split cover exactly the converted dataset.
    """
    root = Path(dataset_root).expanduser().resolve()
    _validate_train_ratio(train_ratio)

    if not root.is_dir():
        raise FileNotFoundError(f"Pose dataset root not found: {root}")
    manifest_path = root / "data.txt"
    images_root = root / "images"
    labels_root = root / "labels"
    for required_path in (manifest_path, images_root, labels_root):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required pose dataset path not found: {required_path}"
            )

    entries = _read_manifest(manifest_path)
    _validate_entries(root, entries)

    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    train_count = int(train_ratio * len(shuffled))
    train_entries = shuffled[:train_count]
    val_entries = shuffled[train_count:]

    _write_lines(root / "train.txt", train_entries)
    _write_lines(root / "val.txt", val_entries)
    return {"train_count": len(train_entries), "val_count": len(val_entries)}


def _resolve_options(args: argparse.Namespace) -> tuple[float, int]:
    train_ratio = args.train_ratio
    seed = args.seed
    if args.params is not None:
        with args.params.expanduser().open(encoding="utf-8") as file:
            params = yaml.safe_load(file)
        if not isinstance(params, dict):
            raise ValueError(f"Parameter file root must be a mapping: {args.params}")
        section = params.get(PARAMS_SECTION)
        if not isinstance(section, dict):
            raise ValueError(
                f"Parameter file must contain a '{PARAMS_SECTION}' mapping: "
                f"{args.params}"
            )
        if train_ratio is None:
            train_ratio = section.get("train", DEFAULT_TRAIN_RATIO)
        if seed is None:
            seed = section.get("seed", DEFAULT_SEED)

    if train_ratio is None:
        train_ratio = DEFAULT_TRAIN_RATIO
    if seed is None:
        seed = DEFAULT_SEED
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if isinstance(train_ratio, bool):
        raise ValueError(f"train ratio must be a number, got {train_ratio!r}")
    try:
        train_ratio = float(train_ratio)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"train ratio must be a number, got {train_ratio!r}"
        ) from error
    return train_ratio, seed


def _validate_train_ratio(train_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError(
            f"train ratio must be strictly between 0 and 1, got {train_ratio}"
        )


def _read_manifest(path: Path) -> list[str]:
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    entries = [entry for entry in entries if entry]
    if not entries:
        raise ValueError(f"Pose data manifest is empty: {path}")
    if len(set(entries)) != len(entries):
        raise ValueError(f"Pose data manifest contains duplicate entries: {path}")
    return entries


def _validate_entries(root: Path, entries: Sequence[str]) -> None:
    seen_labels: set[Path] = set()
    for entry in entries:
        relative = Path(entry.removeprefix("./"))
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.parts[0] != "images"
            or relative.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            raise ValueError(f"Invalid pose image path in data.txt: {entry!r}")

        image_path = root / relative
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Pose image listed in data.txt not found: {image_path}"
            )

        label_relative = relative.relative_to("images").with_suffix(".txt")
        if label_relative in seen_labels:
            raise ValueError(f"Multiple images map to label path: {label_relative}")
        seen_labels.add(label_relative)
        label_path = root / "labels" / label_relative
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Pose label listed image has no label file: {label_path}"
            )


def _write_lines(path: Path, entries: Sequence[str]) -> None:
    text = "\n".join(entries)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
