# Author: imyhxy
# File: test_create_missing_empty.py
# Date: 2/6/24
import tempfile
from pathlib import Path
from unittest import TestCase

from databox.yolo.create_missing_empty import create_missing_empty


class TestCreateMissingEmpty(TestCase):
    @staticmethod
    def _create_file(path):
        with path.open("a") as _:
            pass

    def test_create_missing_empty(self):
        with tempfile.TemporaryDirectory() as p:
            root = Path(p)
            train_image_dir = root / "images" / "train"
            test_image_dir = root / "images" / "test"
            train_image_dir.mkdir(parents=True)
            test_image_dir.mkdir(parents=True)
            train_label_dir = root / "labels" / "train"
            test_label_dir = root / "labels" / "test"
            train_label_dir.mkdir(parents=True)
            test_label_dir.mkdir(parents=True)
            self._create_file(train_image_dir / "test01.png")
            self._create_file(train_image_dir / "test02.jpg")
            self._create_file(test_image_dir / "test03.png")
            self._create_file(test_image_dir / "test04.jpg")

            with (test_label_dir / "test04.txt").open("w") as f:
                f.write("Hello Test")

            create_missing_empty(str(root))

            self.assertTrue((root / "labels/train/test01.txt").is_file())
            self.assertTrue((root / "labels/train/test02.txt").is_file())
            self.assertTrue((root / "labels/test/test03.txt").is_file())
            self.assertTrue((root / "labels/test/test04.txt").is_file())

            with (test_label_dir / "test04.txt").open() as f:
                self.assertEqual("Hello Test", f.read())
