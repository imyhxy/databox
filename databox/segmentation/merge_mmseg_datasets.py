"""Merge MMSeg/Pascal VOC-style segmentation datasets."""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

try:
    from databox.segmentation.dataset_manifest import read_manifest, write_manifest
except ModuleNotFoundError:
    from dataset_manifest import read_manifest, write_manifest

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
BRANCH_MASK_SUFFIXES = ("_polygon", "_polyline")
POLYLINE_ANNOTATION_SUFFIX = "_polyline.txt"
LABELMAP_FILENAMES = (
    "labelmap.txt",
    "labelmap_polygon.txt",
    "labelmap_polyline.txt",
)


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
    parser.add_argument(
        "--prefixes",
        nargs="+",
        help=(
            "Output stem prefix for each input, in the same order as --inputs. "
            "Defaults to each input directory name."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_datasets(args.inputs, args.output, args.prefixes)


def merge_datasets(
    input_roots: list[Path],
    output_root: Path,
    prefixes: list[str] | None = None,
) -> None:
    input_roots = [path.resolve() for path in input_roots]
    output_root = output_root.resolve()
    prefixes = prefixes or [path.name for path in input_roots]

    if output_root in input_roots:
        raise ValueError("Output root must not be one of the input roots")
    if len(prefixes) != len(input_roots):
        raise ValueError("--prefixes must contain exactly one value per input")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"Input prefixes must be unique: {prefixes}")
    if any(not prefix or Path(prefix).name != prefix for prefix in prefixes):
        raise ValueError(
            f"Input prefixes must be non-empty file-name components: {prefixes}"
        )

    _validate_inputs(input_roots)
    labelmaps = _read_common_labelmaps(input_roots)

    if output_root.exists():
        shutil.rmtree(output_root)

    output_images_dir = output_root / "JPEGImages"
    output_masks_dir = output_root / "SegmentationClass"
    output_splits_dir = output_root / "ImageSets" / "Segmentation"
    output_images_dir.mkdir(parents=True)
    output_masks_dir.mkdir(parents=True)
    output_splits_dir.mkdir(parents=True)

    merged_splits: dict[str, list[str]] = defaultdict(list)
    merged_manifest = []
    used_stems: set[str] = set()

    for input_root, prefix in zip(input_roots, prefixes, strict=True):
        image_paths = _collect_by_stem(input_root / "JPEGImages", IMAGE_EXTENSIONS)
        split_entries = _read_split_entries(input_root / "ImageSets" / "Segmentation")
        split_stems = {stem for stems in split_entries.values() for stem in stems}
        mask_paths = _collect_masks(input_root / "SegmentationClass", split_stems)
        manifest_by_stem = _manifest_by_stem(input_root, split_entries)

        missing_images = sorted(split_stems - set(image_paths))
        extra_images = sorted(set(image_paths) - split_stems)
        missing_masks = sorted(split_stems - set(mask_paths))
        missing_polyline_annotations = sorted(
            stem
            for stem in split_stems
            if not (
                input_root / "SegmentationClass" / f"{stem}{POLYLINE_ANNOTATION_SUFFIX}"
            ).exists()
        )
        if (
            missing_images
            or extra_images
            or missing_masks
            or missing_polyline_annotations
        ):
            raise ValueError(
                f"{input_root} split files reference missing images, masks, or "
                "polyline annotations: "
                f"images={missing_images[:5]}, extra_images={extra_images[:5]}, "
                f"masks={missing_masks[:5]}, "
                f"polylines={missing_polyline_annotations[:5]}"
            )

        for stem, image_path in sorted(image_paths.items()):
            if stem not in mask_paths:
                raise ValueError(f"Missing mask for image stem {stem}: {input_root}")
            merged_stem = f"{prefix}__{stem}"
            if merged_stem in used_stems:
                raise ValueError(f"Duplicate merged stem: {merged_stem}")
            _check_merged_mask_name_collisions(merged_stem, used_stems)
            used_stems.add(merged_stem)

            shutil.copy2(
                image_path,
                output_images_dir / f"{merged_stem}{image_path.suffix}",
            )
            for suffix, mask_path in mask_paths[stem].items():
                shutil.copy2(
                    mask_path,
                    output_masks_dir / f"{merged_stem}{suffix}{mask_path.suffix}",
                )
            polyline_annotation_path = (
                input_root / "SegmentationClass" / f"{stem}{POLYLINE_ANNOTATION_SUFFIX}"
            )
            shutil.copy2(
                polyline_annotation_path,
                output_masks_dir / f"{merged_stem}{POLYLINE_ANNOTATION_SUFFIX}",
            )
            record = dict(manifest_by_stem[stem])
            output_image = output_images_dir / f"{merged_stem}{image_path.suffix}"
            record.update(
                {
                    "image_path": output_image.relative_to(output_root).as_posix(),
                    "mask_paths": {
                        "polygon": (
                            output_masks_dir / f"{merged_stem}_polygon.png"
                        )
                        .relative_to(output_root)
                        .as_posix(),
                        "polyline": (
                            output_masks_dir / f"{merged_stem}_polyline.png"
                        )
                        .relative_to(output_root)
                        .as_posix(),
                        "main": (output_masks_dir / f"{merged_stem}.png")
                        .relative_to(output_root)
                        .as_posix(),
                    },
                }
            )
            merged_manifest.append(record)

        for split_name, stems in split_entries.items():
            merged_splits[split_name].extend(f"{prefix}__{stem}" for stem in stems)

    for filename, text in labelmaps.items():
        (output_root / filename).write_text(text)
    for split_name, stems in sorted(merged_splits.items()):
        (output_splits_dir / f"{split_name}.txt").write_text("\n".join(stems) + "\n")
    write_manifest(output_root, merged_manifest)


def _validate_inputs(input_roots: list[Path]) -> None:
    if not input_roots:
        raise ValueError("At least one input dataset is required")
    for input_root in input_roots:
        for required_path in (
            input_root / "JPEGImages",
            input_root / "SegmentationClass",
            input_root / "ImageSets" / "Segmentation",
            *[input_root / filename for filename in LABELMAP_FILENAMES],
        ):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required path does not exist: {required_path}"
                )


def _read_common_labelmaps(input_roots: list[Path]) -> dict[str, str]:
    first_root = input_roots[0]
    first_labelmaps = {
        filename: (first_root / filename).read_text()
        for filename in LABELMAP_FILENAMES
    }
    for input_root in input_roots[1:]:
        for filename in LABELMAP_FILENAMES:
            labelmap = (input_root / filename).read_text()
            if labelmap != first_labelmaps[filename]:
                raise ValueError(
                    f"{filename} differs between {first_root} and {input_root}"
                )
    return first_labelmaps


def _collect_by_stem(root: Path, extensions: set[str]) -> dict[str, Path]:
    paths = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in paths:
            raise ValueError(f"Duplicate stem in {root}: {path.stem}")
        paths[path.stem] = path
    return paths


def _collect_masks(root: Path, expected_stems: set[str]) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    used_paths: dict[Path, str] = {}
    for stem in sorted(expected_stems):
        grouped[stem] = {}
        for suffix in ("", *BRANCH_MASK_SUFFIXES):
            mask_path = root / f"{stem}{suffix}.png"
            if not mask_path.exists():
                raise ValueError(
                    f"Missing branch masks for stem {stem}: {[mask_path.name]}"
                )
            if mask_path in used_paths:
                raise ValueError(
                    "Mask file would be reused for multiple stems: "
                    f"{used_paths[mask_path]!r} and {stem!r} both need "
                    f"{mask_path.name!r}"
                )
            used_paths[mask_path] = stem
            grouped[stem][suffix] = mask_path
    return grouped


def _check_merged_mask_name_collisions(merged_stem: str, used_stems: set[str]) -> None:
    output_mask_names = {
        f"{merged_stem}.png",
        f"{merged_stem}_polygon.png",
        f"{merged_stem}_polyline.png",
        f"{merged_stem}{POLYLINE_ANNOTATION_SUFFIX}",
    }
    for used_stem in used_stems:
        used_mask_names = {
            f"{used_stem}.png",
            f"{used_stem}_polygon.png",
            f"{used_stem}_polyline.png",
            f"{used_stem}{POLYLINE_ANNOTATION_SUFFIX}",
        }
        overlap = sorted(output_mask_names & used_mask_names)
        if overlap:
            raise ValueError(
                "Merged stems would overwrite masks: "
                f"{used_stem!r} and {merged_stem!r} both write {overlap}"
            )


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


def _manifest_by_stem(
    input_root: Path, split_entries: dict[str, list[str]]
) -> dict[str, dict]:
    records = read_manifest(input_root)
    by_stem = {}
    for record in records:
        stem = Path(record["image_path"]).stem
        if stem in by_stem:
            raise ValueError(f"Duplicate manifest image stem in {input_root}: {stem}")
        by_stem[stem] = record

    stem_splits: dict[str, list[str]] = defaultdict(list)
    for split, stems in split_entries.items():
        for stem in stems:
            stem_splits[stem].append(split)
    ambiguous = {
        stem: splits for stem, splits in stem_splits.items() if len(splits) != 1
    }
    if ambiguous:
        raise ValueError(f"Dataset stems must belong to exactly one split: {ambiguous}")

    expected_stems = set(stem_splits)
    if set(by_stem) != expected_stems:
        raise ValueError(
            f"{input_root} manifest and split stems differ: "
            f"missing={sorted(expected_stems - set(by_stem))[:5]}, "
            f"extra={sorted(set(by_stem) - expected_stems)[:5]}"
        )
    mismatched = sorted(
        stem
        for stem, record in by_stem.items()
        if record["split"] != stem_splits[stem][0]
    )
    if mismatched:
        raise ValueError(
            f"{input_root} manifest split differs from split files: {mismatched[:5]}"
        )
    return by_stem


if __name__ == "__main__":
    main()
