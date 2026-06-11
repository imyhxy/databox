import argparse
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert P mode images under an input directory to grayscale images."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where converted grayscale images will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def convert_p_mode_images(input_dir: Path, output_dir: Path, overwrite: bool = False):
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    converted = 0
    skipped = 0

    for image_path in sorted(path for path in input_dir.rglob("*") if path.is_file()):
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        try:
            with Image.open(image_path) as image:
                if image.mode != "P":
                    skipped += 1
                    continue

                if output_path.exists() and not overwrite:
                    skipped += 1
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                image = np.asarray(image)
                image = Image.fromarray(image, mode="L")
                image.save(output_path)
                converted += 1
        except UnidentifiedImageError:
            skipped += 1

    return converted, skipped


def main():
    opt = parse_args()
    converted, skipped = convert_p_mode_images(
        opt.input_dir, opt.output_dir, overwrite=opt.overwrite
    )
    print(f"Converted {converted} image(s), skipped {skipped} file(s).")


if __name__ == "__main__":
    main()
