"""Resize images to Nano Banana supported output resolutions."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class TargetResolution:
    resolution: str
    aspect_ratio: str
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def ratio(self) -> float:
        return self.width / self.height


def load_supported_resolutions(path: Path) -> dict[str, dict[str, TargetResolution]]:
    with path.open() as f:
        config = yaml.safe_load(f)

    try:
        raw_resolutions = config["google_nano_banana_pro_output_resolutions"][
            "resolutions"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid resolution YAML: {path}") from exc

    supported: dict[str, dict[str, TargetResolution]] = {}
    for resolution, aspect_map in raw_resolutions.items():
        supported[resolution] = {}
        for aspect_ratio, size in aspect_map.items():
            supported[resolution][aspect_ratio] = TargetResolution(
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                width=int(size["width"]),
                height=int(size["height"]),
            )

    if not supported:
        raise ValueError(f"No supported resolutions found in {path}")
    return supported


def choose_target_resolution(
    width: int,
    height: int,
    supported: dict[str, dict[str, TargetResolution]],
    lock_resolution: str | None = None,
) -> TargetResolution:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")

    if lock_resolution is not None and lock_resolution not in supported:
        available = ", ".join(supported)
        raise ValueError(
            f"Unsupported locked resolution {lock_resolution!r}; available: {available}"
        )

    candidates = _flatten_targets(supported, lock_resolution)
    source_ratio = width / height
    source_area = width * height

    best_aspect_ratio = min(
        {target.aspect_ratio for target in candidates},
        key=lambda aspect_ratio: (
            min(
                _relative_difference(source_ratio, target.ratio)
                for target in candidates
                if target.aspect_ratio == aspect_ratio
            ),
            aspect_ratio,
        ),
    )
    aspect_candidates = [
        target for target in candidates if target.aspect_ratio == best_aspect_ratio
    ]

    if lock_resolution is not None:
        return aspect_candidates[0]

    return min(
        aspect_candidates,
        key=lambda target: (
            _relative_difference(source_area, target.area),
            target.resolution,
        ),
    )


def resize_batch(
    input_path: Path,
    output_dir: Path,
    supported: dict[str, dict[str, TargetResolution]],
    lock_resolution: str | None = None,
    background: tuple[int, ...] | None = None,
    show_progress: bool = True,
) -> tuple[int, int]:
    image_paths = list(iter_image_paths(input_path))
    processed = 0
    skipped = 0
    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = (
            image_path.name
            if input_path.is_file()
            else image_path.relative_to(input_path)
        )
        output_path = output_dir / relative_path

        try:
            resize_one(
                image_path,
                output_path,
                supported=supported,
                lock_resolution=lock_resolution,
                background=background,
            )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            if show_progress:
                _clear_progress_line()
            print(f"warning: skipped {image_path}: {exc}", file=sys.stderr)
            skipped += 1
        else:
            processed += 1

        if show_progress:
            _write_progress(index, total, processed, skipped)

    if show_progress and total > 0:
        print(file=sys.stderr, flush=True)

    return processed, skipped


def resize_one(
    input_path: Path,
    output_path: Path,
    supported: dict[str, dict[str, TargetResolution]],
    lock_resolution: str | None = None,
    background: tuple[int, ...] | None = None,
) -> TargetResolution:
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        target = choose_target_resolution(
            image.width,
            image.height,
            supported=supported,
            lock_resolution=lock_resolution,
        )
        resized = contain_and_pad(image, target, background)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {}
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            resized = resized.convert("RGB")
            save_kwargs["quality"] = 95
        resized.save(output_path, **save_kwargs)
        return target


def contain_and_pad(
    image: Image.Image,
    target: TargetResolution,
    background: tuple[int, ...] | None = None,
) -> Image.Image:
    has_alpha = image.mode in {"LA", "RGBA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    mode = "RGBA" if has_alpha else "RGB"
    fill = _normalize_background(
        _default_background(mode) if background is None else background,
        mode,
    )

    source = image.convert(mode)
    resized_size = _contained_size(source.size, (target.width, target.height))
    resized = source.resize(resized_size, _resampling_filter())

    canvas = Image.new(mode, (target.width, target.height), fill)
    offset = (
        (target.width - resized.width) // 2,
        (target.height - resized.height) // 2,
    )
    if mode == "RGBA":
        canvas.alpha_composite(resized, offset)
    else:
        canvas.paste(resized, offset)
    return canvas


def _contained_size(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = min(target_width / source_width, target_height / source_height)

    return (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )


def iter_image_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield input_path
        return

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def parse_background(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("background must be 'R,G,B' or 'R,G,B,A'")

    try:
        channels = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "background channels must be integers"
        ) from exc

    if any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError("background channels must be 0..255")
    return channels


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Resize images to the closest Nano Banana supported resolution."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input image or directory",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--resolution-yaml",
        type=Path,
        default=script_dir / "ai_studio_resolution.yaml",
        help="Path to ai_studio_resolution.yaml",
    )
    parser.add_argument(
        "--lock-resolution",
        choices=("1K", "2K"),
        default=None,
        help="Limit target selection to one Nano Banana resolution tier",
    )
    parser.add_argument(
        "--background",
        type=parse_background,
        default=None,
        help="Padding color as R,G,B or R,G,B,A. Defaults to black.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the batch progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supported = load_supported_resolutions(args.resolution_yaml)
    processed, skipped = resize_batch(
        input_path=args.input,
        output_dir=args.output,
        supported=supported,
        lock_resolution=args.lock_resolution,
        background=args.background,
        show_progress=not args.no_progress,
    )
    print(f"processed={processed} skipped={skipped}", flush=True)


def _flatten_targets(
    supported: dict[str, dict[str, TargetResolution]],
    lock_resolution: str | None,
) -> list[TargetResolution]:
    if lock_resolution is not None:
        return list(supported[lock_resolution].values())

    return [
        target for aspect_map in supported.values() for target in aspect_map.values()
    ]


def _relative_difference(source: float, target: float) -> float:
    return abs(source - target) / source


def _write_progress(index: int, total: int, processed: int, skipped: int) -> None:
    if total <= 0:
        return

    width = 30
    filled = round(width * index / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100 * index / total
    message = (
        f"\rResizing [{bar}] {index}/{total} "
        f"{percent:5.1f}% processed={processed} skipped={skipped}"
    )
    print(message, end="", file=sys.stderr, flush=True)


def _clear_progress_line() -> None:
    print("\r" + " " * 100 + "\r", end="", file=sys.stderr, flush=True)


def _default_background(mode: str) -> tuple[int, ...]:
    if mode == "RGBA":
        return (0, 0, 0, 0)
    return (0, 0, 0)


def _normalize_background(background: tuple[int, ...], mode: str) -> tuple[int, ...]:
    if mode == "RGBA" and len(background) == 3:
        return (*background, 255)
    if mode == "RGB" and len(background) == 4:
        return background[:3]
    return background


def _resampling_filter() -> int:
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


if __name__ == "__main__":
    main()
