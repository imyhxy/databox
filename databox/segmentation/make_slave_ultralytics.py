"""Build an Ultralytics instance dataset from slave images of a master dataset.

Slave images are the same scenes as the master at other brightness levels
(``<scene>_0G_<level>``).  Each matched slave image is symlinked into
``<output>/images``, reuses the master label, and inherits the master split.
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from databox.segmentation.make_slave_voc import scene_key
except ModuleNotFoundError:
    from make_slave_voc import scene_key

SPLITS = ("train", "val")


def build_master_index(master: Path) -> dict[str, tuple[str, str]]:
    """Map scene key -> (master stem, split) from master train/val lists."""
    index = {}
    for split in SPLITS:
        for line in (master / f"{split}.txt").read_text().split():
            stem = Path(line).stem
            key = scene_key(stem)
            if key in index:
                raise ValueError(f"Duplicate master scene key {key!r}: {stem!r}")
            index[key] = (stem, split)
    return index


def build_slave_dataset(master: Path, slave_images: Path, output: Path) -> int:
    index = build_master_index(master)
    if not slave_images.is_dir():
        raise FileNotFoundError(f"Slave image directory not found: {slave_images}")

    matched = {}
    for image in sorted(slave_images.rglob("*.jpg")):
        item = index.get(scene_key(image.stem))
        if item is None:
            continue
        if image.stem in matched:
            raise ValueError(f"Duplicate slave image stem: {image.stem!r}")
        matched[image.stem] = (image, *item)
    if not matched:
        raise ValueError(f"No slave images under {slave_images} matched {master}")

    image_dir = output / "images"
    label_dir = output / "labels"
    for path in (image_dir, label_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    splits = {split: [] for split in SPLITS}
    for stem, (image, master_stem, split) in matched.items():
        dst = image_dir / f"{stem}.jpg"
        dst.symlink_to(os.path.relpath(os.path.abspath(image), os.path.abspath(image_dir)))
        shutil.copy2(master / "labels" / f"{master_stem}.txt", label_dir / f"{stem}.txt")
        splits[split].append(f"./images/{stem}.jpg")

    for split, lines in splits.items():
        (output / f"{split}.txt").write_text("\n".join(lines))
    shutil.copy2(master / "data.yaml", output / "data.yaml")
    return len(matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--slave-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = build_slave_dataset(args.master, args.slave_images, args.output)
    print(f"Wrote {count} matched slave samples to {args.output}")


if __name__ == "__main__":
    main()
