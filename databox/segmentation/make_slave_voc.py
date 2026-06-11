"""Build a VOC segmentation dataset from matched slave images."""

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BRIGHTNESS_SUFFIX = re.compile(r"_0G_\d{3}$")
SPLITS = ("train", "val")


@dataclass(frozen=True)
class MasterItem:
    stem: str
    split: str
    mask_path: Path


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
    labelmap = output / "labelmap.txt"
    if labelmap.exists():
        labelmap.unlink()


def build_master_index(master: Path) -> dict[str, MasterItem]:
    split_dir = master / "ImageSets" / "Segmentation"
    mask_dir = master / "SegmentationClass"
    index = {}

    for split in SPLITS:
        for stem in read_split(split_dir / f"{split}.txt"):
            key = scene_key(stem)
            if key in index:
                existing = index[key]
                raise ValueError(
                    "Duplicate master scene key "
                    f"{key!r}: {existing.stem!r} in {existing.split}, "
                    f"{stem!r} in {split}"
                )

            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Master mask not found: {mask_path}")
            index[key] = MasterItem(stem=stem, split=split, mask_path=mask_path)

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
    labelmap = master / "labelmap.txt"
    if not labelmap.exists():
        raise FileNotFoundError(f"Master labelmap not found: {labelmap}")

    matched = []
    seen_output_stems = set()
    for slave_image in iter_slave_images(slave_raw):
        item = master_index.get(scene_key(slave_image.stem))
        if item is None:
            continue
        if slave_image.stem in seen_output_stems:
            raise ValueError(f"Duplicate slave output stem: {slave_image.stem!r}")
        seen_output_stems.add(slave_image.stem)
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
    for slave_image, item in matched:
        dst_stem = slave_image.stem
        shutil.copy2(slave_image, image_dir / f"{dst_stem}.jpg")
        shutil.copy2(item.mask_path, mask_dir / f"{dst_stem}.png")
        split_stems[item.split].append(dst_stem)

    for split in SPLITS:
        text = "\n".join(split_stems[split])
        if text:
            text += "\n"
        (split_dir / f"{split}.txt").write_text(text)

    shutil.copy2(labelmap, output / "labelmap.txt")
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
