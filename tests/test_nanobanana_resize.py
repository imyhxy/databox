from pathlib import Path
from unittest import TestCase

from PIL import Image

from databox.nanobanana.resize_to_supported_resolution import (
    choose_target_resolution,
    load_supported_resolutions,
    resize_one,
)

RESOLUTION_YAML = (
    Path(__file__).resolve().parents[1]
    / "databox"
    / "nanobanana"
    / "ai_studio_resolution.yaml"
)


class TestNanoBananaResize(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supported = load_supported_resolutions(RESOLUTION_YAML)

    def test_square_image_maps_to_square_aspect_ratio(self):
        target = choose_target_resolution(1500, 1500, self.supported)

        self.assertEqual("1:1", target.aspect_ratio)

    def test_landscape_image_maps_to_16_9_aspect_ratio(self):
        target = choose_target_resolution(1920, 1080, self.supported)

        self.assertEqual("16:9", target.aspect_ratio)

    def test_portrait_image_maps_to_9_16_aspect_ratio(self):
        target = choose_target_resolution(1080, 1920, self.supported)

        self.assertEqual("9:16", target.aspect_ratio)

    def test_unlocked_mode_chooses_closest_resolution_tier_by_area(self):
        small = choose_target_resolution(1000, 1000, self.supported)
        large = choose_target_resolution(2100, 2100, self.supported)

        self.assertEqual("1K", small.resolution)
        self.assertEqual("2K", large.resolution)

    def test_locked_mode_never_returns_other_resolution_tier(self):
        target = choose_target_resolution(
            2100,
            2100,
            self.supported,
            lock_resolution="1K",
        )

        self.assertEqual("1K", target.resolution)
        self.assertEqual("1:1", target.aspect_ratio)

    def test_resize_one_writes_supported_size_and_keeps_original_unchanged(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.png"
            output_path = root / "outputs" / "input.png"
            Image.new("RGB", (1200, 800), (10, 20, 30)).save(input_path)

            target = resize_one(input_path, output_path, supported=self.supported)

            with Image.open(output_path) as output_image:
                self.assertEqual((target.width, target.height), output_image.size)

            with Image.open(input_path) as original_image:
                self.assertEqual((1200, 800), original_image.size)

    def test_resize_one_upscales_to_target_edge_before_padding(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "small.png"
            output_path = root / "outputs" / "small.png"
            Image.new("RGB", (160, 90), (255, 0, 0)).save(input_path)

            target = resize_one(
                input_path,
                output_path,
                supported=self.supported,
                lock_resolution="1K",
            )

            with Image.open(output_path) as output_image:
                self.assertEqual((1376, 768), output_image.size)
                self.assertEqual((1376, 768), (target.width, target.height))
                self.assertEqual((255, 0, 0), output_image.getpixel((688, 0)))
