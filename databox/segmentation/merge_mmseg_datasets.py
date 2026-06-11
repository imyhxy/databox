"""Merge MMSeg/Pascal VOC-style segmentation datasets."""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge datasets with JPEGImages, SegmentationClass, "
            "ImageSets/Segmentation, and labelmap.txt into one dataset."
        )
    )
    parser.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        type=Path,
        help="Input dataset roots.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_datasets(args.inputs, args.output)


def merge_datasets(input_roots: list[Path], output_root: Path) -> None:
    input_roots = [path.resolve() for path in input_roots]
    output_root = output_root.resolve()

    if output_root in input_roots:
        raise ValueError("Output root must not be one of the input roots")

    _validate_inputs(input_roots)
    labelmap = _read_common_labelmap(input_roots)

    if output_root.exists():
        shutil.rmtree(output_root)

    output_images_dir = output_root / "JPEGImages"
    output_masks_dir = output_root / "SegmentationClass"
    output_splits_dir = output_root / "ImageSets" / "Segmentation"
    output_images_dir.mkdir(parents=True)
    output_masks_dir.mkdir(parents=True)
    output_splits_dir.mkdir(parents=True)

    merged_splits: dict[str, list[str]] = defaultdict(list)
    used_stems: set[str] = set()

    for input_root in input_roots:
        prefix = input_root.name
        image_paths = _collect_by_stem(input_root / "JPEGImages", IMAGE_EXTENSIONS)
        mask_paths = _collect_by_stem(input_root / "SegmentationClass", MASK_EXTENSIONS)
        split_entries = _read_split_entries(input_root / "ImageSets" / "Segmentation")
        split_stems = {stem for stems in split_entries.values() for stem in stems}

        missing_images = sorted(split_stems - set(image_paths))
        missing_masks = sorted(split_stems - set(mask_paths))
        if missing_images or missing_masks:
            raise ValueError(
                f"{input_root} split files reference missing images or masks: "
                f"images={missing_images[:5]}, masks={missing_masks[:5]}"
            )

        for stem, image_path in sorted(image_paths.items()):
            if stem not in mask_paths:
                raise ValueError(f"Missing mask for image stem {stem}: {input_root}")
            merged_stem = f"{prefix}__{stem}"
            if merged_stem in used_stems:
                raise ValueError(f"Duplicate merged stem: {merged_stem}")
            used_stems.add(merged_stem)

            shutil.copy2(
                image_path,
                output_images_dir / f"{merged_stem}{image_path.suffix}",
            )
            mask_path = mask_paths[stem]
            shutil.copy2(
                mask_path,
                output_masks_dir / f"{merged_stem}{mask_path.suffix}",
            )

        for split_name, stems in split_entries.items():
            merged_splits[split_name].extend(f"{prefix}__{stem}" for stem in stems)

    (output_root / "labelmap.txt").write_text(labelmap)
    for split_name, stems in sorted(merged_splits.items()):
        (output_splits_dir / f"{split_name}.txt").write_text("\n".join(stems) + "\n")


def _validate_inputs(input_roots: list[Path]) -> None:
    if not input_roots:
        raise ValueError("At least one input dataset is required")
    for input_root in input_roots:
        for required_path in (
            input_root / "JPEGImages",
            input_root / "SegmentationClass",
            input_root / "ImageSets" / "Segmentation",
            input_root / "labelmap.txt",
        ):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required path does not exist: {required_path}"
                )


def _read_common_labelmap(input_roots: list[Path]) -> str:
    first_labelmap = (input_roots[0] / "labelmap.txt").read_text()
    for input_root in input_roots[1:]:
        labelmap = (input_root / "labelmap.txt").read_text()
        if labelmap != first_labelmap:
            raise ValueError(
                f"labelmap.txt differs between {input_roots[0]} and {input_root}"
            )
    return first_labelmap


def _collect_by_stem(root: Path, extensions: set[str]) -> dict[str, Path]:
    paths = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in paths:
            raise ValueError(f"Duplicate stem in {root}: {path.stem}")
        paths[path.stem] = path
    return paths


def _read_split_entries(split_dir: Path) -> dict[str, list[str]]:
    entries = {}
    for path in sorted(split_dir.glob("*.txt")):
        stems = []
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            stems.append(Path(line).stem)
        entries[path.stem] = stems
    return entries


if __name__ == "__main__":
    main()
