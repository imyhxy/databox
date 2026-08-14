"""Build a VOC segmentation dataset from matched slave images."""

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

try:
    from databox.segmentation.dataset_manifest import (
        MANIFEST_FILENAME,
        read_manifest,
        write_manifest,
    )
except ModuleNotFoundError:
    from dataset_manifest import MANIFEST_FILENAME, read_manifest, write_manifest

BRIGHTNESS_SUFFIX = re.compile(r"_0G_\d{3}$")
SPLITS = ("train", "val")
ANNOTATION_SUFFIXES = ("_polygon.txt", "_polyline.txt")


@dataclass(frozen=True)
class MasterItem:
    stem: str
    split: str
    mask_paths: tuple[Path, ...]
    manifest_record: dict


def scene_key(stem: str) -> str:
    return BRIGHTNESS_SUFFIX.sub("", stem)


def read_split(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def clean_output(output: Path) -> None:
    for dirname in ("JPEGImages", "SegmentationClass"):
        path = output / dirname
        if path.exists():
            shutil.rmtree(path)
    split_dir = output / "ImageSets" / "Segmentation"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    image_sets = output / "ImageSets"
    if image_sets.exists() and not any(image_sets.iterdir()):
        image_sets.rmdir()
    for filename in (
        "labelmap.txt",
        "labelmap_polygon.txt",
        "labelmap_polyline.txt",
    ):
        labelmap = output / filename
        if labelmap.exists():
            labelmap.unlink()
    manifest = output / MANIFEST_FILENAME
    if manifest.exists():
        manifest.unlink()


def build_master_index(master: Path) -> dict[str, MasterItem]:
    split_dir = master / "ImageSets" / "Segmentation"
    mask_dir = master / "SegmentationClass"
    index = {}
    manifest_records = read_manifest(master)
    manifest_by_stem = {
        Path(record["image_path"]).stem: record for record in manifest_records
    }
    if len(manifest_by_stem) != len(manifest_records):
        raise ValueError(f"Duplicate master manifest image stems: {master}")
    split_stems = set()

    for split in SPLITS:
        for stem in read_split(split_dir / f"{split}.txt"):
            split_stems.add(stem)
            key = scene_key(stem)
            if key in index:
                existing = index[key]
                raise ValueError(
                    "Duplicate master scene key "
                    f"{key!r}: {existing.stem!r} in {existing.split}, "
                    f"{stem!r} in {split}"
                )

            mask_paths = (
                mask_dir / f"{stem}.png",
                mask_dir / f"{stem}_polygon.png",
                mask_dir / f"{stem}_polyline.png",
                *(mask_dir / f"{stem}{suffix}" for suffix in ANNOTATION_SUFFIXES),
            )
            for mask_path in mask_paths:
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Master segmentation file not found: {mask_path}"
                    )
            if stem not in manifest_by_stem:
                raise ValueError(f"Master manifest missing image stem: {stem}")
            manifest_record = manifest_by_stem[stem]
            if manifest_record["split"] != split:
                raise ValueError(
                    f"Master manifest split differs for {stem}: "
                    f"{manifest_record['split']!r} != {split!r}"
                )
            index[key] = MasterItem(
                stem=stem,
                split=split,
                mask_paths=mask_paths,
                manifest_record=manifest_record,
            )

    extra_manifest_stems = sorted(set(manifest_by_stem) - split_stems)
    if extra_manifest_stems:
        raise ValueError(
            f"Master manifest contains stems absent from splits: "
            f"{extra_manifest_stems[:5]}"
        )
    return index


def iter_slave_images(slave_raw: Path) -> list[Path]:
    if not slave_raw.exists():
        raise FileNotFoundError(f"Slave raw directory not found: {slave_raw}")
    return sorted(
        path
        for path in slave_raw.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def build_slave_voc_dataset(master: Path, slave_raw: Path, output: Path) -> int:
    master_index = build_master_index(master)
    labelmaps = [
        master / "labelmap.txt",
        master / "labelmap_polygon.txt",
        master / "labelmap_polyline.txt",
    ]
    for labelmap in labelmaps:
        if not labelmap.exists():
            raise FileNotFoundError(f"Master labelmap not found: {labelmap}")

    matched = []
    seen_output_stems = set()
    seen_mask_names = {}
    for slave_image in iter_slave_images(slave_raw):
        item = master_index.get(scene_key(slave_image.stem))
        if item is None:
            continue
        if slave_image.stem in seen_output_stems:
            raise ValueError(f"Duplicate slave output stem: {slave_image.stem!r}")
        seen_output_stems.add(slave_image.stem)
        for mask_name in (
            f"{slave_image.stem}.png",
            f"{slave_image.stem}_polygon.png",
            f"{slave_image.stem}_polyline.png",
            *(f"{slave_image.stem}{suffix}" for suffix in ANNOTATION_SUFFIXES),
        ):
            if mask_name in seen_mask_names:
                existing = seen_mask_names[mask_name]
                raise ValueError(
                    "Slave output stems would overwrite masks: "
                    f"{existing!r} and {slave_image.stem!r} both write {mask_name!r}"
                )
            seen_mask_names[mask_name] = slave_image.stem
        matched.append((slave_image, item))

    if not matched:
        raise ValueError(
            f"No slave images under {slave_raw} matched master dataset {master}"
        )

    clean_output(output)
    image_dir = output / "JPEGImages"
    mask_dir = output / "SegmentationClass"
    split_dir = output / "ImageSets" / "Segmentation"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    split_stems = {split: [] for split in SPLITS}
    manifest_records = []
    for slave_image, item in matched:
        dst_stem = slave_image.stem
        dst_image = image_dir / f"{dst_stem}.jpg"
        shutil.copy2(slave_image, dst_image)
        with Image.open(dst_image) as image:
            width, height = image.size
        for mask_path in item.mask_paths:
            suffix = mask_path.stem.removeprefix(item.stem)
            if mask_path.suffix == ".txt":
                shutil.copy2(mask_path, mask_dir / f"{dst_stem}{suffix}.txt")
            else:
                shutil.copy2(mask_path, mask_dir / f"{dst_stem}{suffix}.png")
        split_stems[item.split].append(dst_stem)
        source_record = item.manifest_record
        manifest_records.append(
            {
                "sample_id": (
                    f"{source_record['sample_id']}_derived_{dst_stem}"
                ),
                "task_name": source_record["task_name"],
                "split": item.split,
                "image_path": dst_image.relative_to(output).as_posix(),
                "mask_paths": {
                    "polygon": (
                        mask_dir / f"{dst_stem}_polygon.png"
                    ).relative_to(output).as_posix(),
                    "polyline": (
                        mask_dir / f"{dst_stem}_polyline.png"
                    ).relative_to(output).as_posix(),
                    "main": (mask_dir / f"{dst_stem}.png")
                    .relative_to(output)
                    .as_posix(),
                },
                "width": width,
                "height": height,
                "task_id": source_record["task_id"],
                "job_id": source_record["job_id"],
                "frame_id": source_record["frame_id"],
            }
        )

    for split in SPLITS:
        text = "\n".join(split_stems[split])
        if text:
            text += "\n"
        (split_dir / f"{split}.txt").write_text(text)

    for labelmap in labelmaps:
        shutil.copy2(labelmap, output / labelmap.name)
    write_manifest(output, manifest_records)
    return len(matched)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--slave-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    count = build_slave_voc_dataset(args.master, args.slave_raw, args.output)
    print(f"Wrote {count} matched slave samples to {args.output}")


if __name__ == "__main__":
    main()
