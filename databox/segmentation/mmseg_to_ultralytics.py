"""Build an Ultralytics-style semantic segmentation shadow dataset."""

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Ultralytics-style shadow dataset from an MMSeg/Pascal "
            "VOC-style segmentation dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Input dataset root with JPEGImages, SegmentationClass, and splits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output dataset root. Defaults to a sibling directory named "
            "<dataset-root>-ultralytics."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.dataset_root.with_name(f"{args.dataset_root.name}-ultralytics")
    count = build_ultralytics_dataset(args.dataset_root, output)
    print(f"Wrote {count} samples to {output}")


def build_ultralytics_dataset(dataset_root: Path, output_root: Path) -> int:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()

    if dataset_root == output_root:
        raise ValueError("Output root must differ from dataset root")

    image_paths = _collect_images(dataset_root / "JPEGImages")
    mask_dir = dataset_root / "SegmentationClass"
    split_dir = dataset_root / "ImageSets" / "Segmentation"
    _validate_required_paths(dataset_root, mask_dir, split_dir)

    split_stems = {split: _read_split(split_dir / f"{split}.txt") for split in SPLITS}
    needed_stems = {stem for stems in split_stems.values() for stem in stems}
    _validate_split_stems(dataset_root, needed_stems, image_paths, mask_dir)

    _clean_output(output_root)
    output_images_dir = output_root / "images"
    output_labels_dir = output_root / "labels"
    output_images_dir.mkdir(parents=True)
    output_labels_dir.mkdir(parents=True)

    for stem in sorted(needed_stems):
        image_path = image_paths[stem]
        output_image = output_images_dir / image_path.name
        _relative_symlink(image_path, output_image)
        _save_plain_mask(mask_dir / f"{stem}.png", output_labels_dir / f"{stem}.png")

    for split, stems in split_stems.items():
        lines = [f"./images/{image_paths[stem].name}" for stem in stems]
        text = "\n".join(lines)
        if text:
            text += "\n"
        (output_root / f"{split}.txt").write_text(text)

    return len(needed_stems)


def _validate_required_paths(
    dataset_root: Path, mask_dir: Path, split_dir: Path
) -> None:
    for required_path in (dataset_root / "JPEGImages", mask_dir, split_dir):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path does not exist: {required_path}")


def _collect_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Required path does not exist: {image_dir}")

    paths = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in paths:
            raise ValueError(f"Duplicate image stem in {image_dir}: {path.stem}")
        paths[path.stem] = path
    return paths


def _read_split(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [
        Path(line.strip()).stem
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _validate_split_stems(
    dataset_root: Path,
    stems: set[str],
    image_paths: dict[str, Path],
    mask_dir: Path,
) -> None:
    missing_images = sorted(stems - set(image_paths))
    missing_masks = sorted(
        stem for stem in stems if not (mask_dir / f"{stem}.png").exists()
    )
    if missing_images or missing_masks:
        raise ValueError(
            f"{dataset_root} split files reference missing images or masks: "
            f"images={missing_images[:5]}, masks={missing_masks[:5]}"
        )


def _clean_output(output_root: Path) -> None:
    for dirname in ("images", "labels"):
        path = output_root / dirname
        if path.exists():
            shutil.rmtree(path)
    for split in SPLITS:
        path = output_root / f"{split}.txt"
        if path.exists():
            path.unlink()
    if output_root.exists() and not any(output_root.iterdir()):
        output_root.rmdir()


def _relative_symlink(source: Path, destination: Path) -> None:
    target = Path(os.path.relpath(source, start=destination.parent))
    destination.symlink_to(target)


def _save_plain_mask(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        if image.mode == "P":
            plain = Image.frombytes("L", image.size, image.tobytes())
        elif image.mode == "L":
            plain = image.copy()
        else:
            plain = image.convert("L")
        plain.save(destination)


if __name__ == "__main__":
    main()
