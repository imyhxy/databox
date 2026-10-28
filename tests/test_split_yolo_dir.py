import subprocess
import sys
from pathlib import Path


def test_split_includes_common_image_formats(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    names = ["a.jpg", "b.JPEG", "c.png", "d.BMP", "e.tif", "f.TIFF", "g.webp"]
    for name in names + ["ignored.txt"]:
        (images / name).touch()
    (images / "folder.jpg").mkdir()
    script = Path(__file__).resolve().parents[1] / "databox" / "split_yolo_dir.py"
    subprocess.run([sys.executable, str(script), "--seed", "1234", "--inputs", str(tmp_path)], check=True)
    train = (tmp_path / "train.txt").read_text().splitlines()
    val = (tmp_path / "val.txt").read_text().splitlines()
    assert len(train) == 5
    assert len(val) == 2
    assert set(train + val) == {f"./images/{name}" for name in names}
