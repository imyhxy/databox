from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pytest
from PIL import Image

from databox.segmentation.cvat_xml_to_mmseg import (
    Config,
    config_from_args,
    convert_cvat_xml_to_mmseg,
    rasterize_image,
    validate_config,
)


def _config(**kwargs):
    config = dict(
        annotations=Path("annotations.xml"),
        output=Path("out"),
        seed=0,
        train=0.8,
        categories=["background", "object"],
        ignore_categories=[],
        ignore_index=255,
        ignore_palette=(128, 128, 128),
        polyline_width=5,
        strict_categories=False,
        palette=[(0, 0, 0), (255, 255, 255)],
    )
    config.update(kwargs)
    return Config(**config)


def _image(xml: str):
    return ET.fromstring(xml)


def test_yaml_mode_reads_input_and_output_from_cli(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        """cvat_xml_to_mmseg:
  input: yaml_annotations.xml
  output: yaml_out
  seed: 7
  train: 0.6
  layout: voc
  categories:
    - background
    - object
  palette:
    - [0, 0, 0]
    - [255, 255, 255]
  ignore_palette: [128, 128, 128]
"""
    )

    config = config_from_args(
        SimpleNamespace(
            yaml=True,
            config=config_path,
            param_name="cvat_xml_to_mmseg",
            input="cli_annotations.xml",
            output="cli_out",
            palette=None,
        )
    )

    assert config.annotations == Path("cli_annotations.xml")
    assert config.output == Path("cli_out")
    assert config.seed == 7
    assert config.layout == "voc"
    assert config.palette == [(0, 0, 0), (255, 255, 255)]
    assert config.ignore_palette == (128, 128, 128)


def test_cli_mode_requires_palette():
    with pytest.raises(ValueError, match="--palette"):
        config_from_args(
            SimpleNamespace(
                yaml=False,
                input="annotations.xml",
                output="out",
                seed=0,
                train=0.8,
                categories=["background", "object"],
                palette=None,
                ignore_categories=[],
                ignore_index=255,
                ignore_palette="128,128,128",
                polyline_width=5,
                strict_categories=False,
            )
        )


def test_cli_mode_requires_ignore_palette():
    with pytest.raises(ValueError, match="--ignore-palette"):
        config_from_args(
            SimpleNamespace(
                yaml=False,
                input="annotations.xml",
                output="out",
                seed=0,
                train=0.8,
                categories=["background", "object"],
                palette=["0,0,0", "255,255,255"],
                ignore_categories=[],
                ignore_index=255,
                ignore_palette=None,
                polyline_width=5,
                strict_categories=False,
            )
        )


def test_cli_mode_parses_palette():
    config = config_from_args(
        SimpleNamespace(
            yaml=False,
            input="annotations.xml",
            output="out",
            seed=0,
            train=0.8,
            categories=["background", "object"],
            palette=["0,0,0", "255,255,255"],
            ignore_categories=[],
            ignore_index=255,
            ignore_palette="128,128,128",
            polyline_width=5,
            strict_categories=False,
            layout="voc",
        )
    )

    assert config.palette == [(0, 0, 0), (255, 255, 255)]
    assert config.ignore_palette == (128, 128, 128)
    assert config.layout == "voc"


def test_polygon_fill_uses_category_index():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <polygon label="object" points="1,1;5,1;5,5;1,5" />
        </image>"""
    )

    mask = rasterize_image(image, ["background", "object"], [])

    assert mask[3, 3] == 1
    assert mask[0, 0] == 0


def test_later_category_overlays_earlier_category():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <polygon label="low" points="1,1;7,1;7,7;1,7" />
          <polygon label="high" points="3,3;9,3;9,9;3,9" />
        </image>"""
    )

    mask = rasterize_image(image, ["low", "high"], [])

    assert mask[2, 2] == 0
    assert mask[4, 4] == 1


def test_ignore_label_writes_ignore_index_and_overrides_classes():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <polygon label="object" points="1,1;8,1;8,8;1,8" />
          <polygon label="ignore" points="3,3;6,3;6,6;3,6" />
        </image>"""
    )

    mask = rasterize_image(
        image,
        ["background", "object"],
        ["ignore"],
        ignore_index=255,
    )

    assert mask[2, 2] == 1
    assert mask[4, 4] == 255


def test_polyline_width_validation():
    validate_config(_config(categories=["background", "line"], polyline_width=5))

    with pytest.raises(ValueError, match="polyline_width"):
        validate_config(_config(categories=["background", "line"], polyline_width=21))


def test_unsupported_shape_raises():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <box label="object" xtl="1" ytl="1" xbr="5" ybr="5" />
        </image>"""
    )

    with pytest.raises(ValueError, match="Unsupported CVAT shape"):
        rasterize_image(image, ["background", "object"], [])


def test_convert_writes_mmseg_layout(tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.jpg"
    cv2.imwrite(str(img1), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((8, 8, 3), dtype=np.uint8))
    annotations = tmp_path / "annotations.xml"
    annotations.write_text(
        """<annotations>
          <meta>
            <task>
              <labels>
                <label><name>background</name></label>
                <label><name>object</name></label>
              </labels>
            </task>
          </meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
          <image id="1" name="two.jpg" width="8" height="8">
            <polyline label="object" points="1,1;6,6" />
          </image>
        </annotations>"""
    )

    out = tmp_path / "prepared"
    convert_cvat_xml_to_mmseg(
        _config(
            annotations=annotations,
            output=out,
            train=0.5,
            categories=["background", "object"],
            polyline_width=3,
        )
    )

    copied_images = list((out / "images").glob("*"))
    masks = list((out / "annotations").glob("*.png"))
    assert len(copied_images) == 2
    assert len(masks) == 2
    assert {p.stem for p in copied_images} == {p.stem for p in masks}
    assert set((out / "train.txt").read_text().splitlines()) | set(
        (out / "val.txt").read_text().splitlines()
    ) == {"one", "two"}
    assert (out / "labelmap.txt").read_text().splitlines() == [
        "# label:color_rgb:parts:actions",
        "background:0,0,0::",
        "object:255,255,255::",
    ]
    assert not (out / "test.txt").exists()
    assert not (out / "JPEGImages").exists()
    assert not (out / "SegmentationClass").exists()
    assert not (out / "ImageSets" / "Segmentation").exists()
    with Image.open(masks[0]) as mask:
        assert mask.mode == "P"
        palette = mask.getpalette()
        assert palette[:6] == [0, 0, 0, 255, 255, 255]
        assert palette[255 * 3 : 255 * 3 + 3] == [128, 128, 128]


def test_convert_writes_voc_layout_and_cleans_stale_mmseg_outputs(tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.png"
    Image.new("RGB", (8, 8)).save(img1)
    Image.new("RGB", (8, 8)).save(img2)
    annotations = tmp_path / "annotations.xml"
    annotations.write_text(
        """<annotations>
          <meta>
            <task>
              <labels>
                <label><name>background</name></label>
                <label><name>object</name></label>
                <label><name>ignore</name></label>
              </labels>
            </task>
          </meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
            <polygon label="ignore" points="3,3;4,3;4,4;3,4" />
          </image>
          <image id="1" name="two.png" width="8" height="8">
            <polyline label="object" points="1,1;6,6" />
          </image>
        </annotations>"""
    )

    out = tmp_path / "prepared"
    (out / "images").mkdir(parents=True)
    (out / "annotations").mkdir()
    (out / "train.txt").write_text("stale\n")
    (out / "val.txt").write_text("stale\n")
    (out / "images" / "stale.jpg").write_text("stale")
    (out / "annotations" / "stale.png").write_text("stale")

    convert_cvat_xml_to_mmseg(
        _config(
            annotations=annotations,
            output=out,
            train=0.5,
            categories=["background", "object"],
            ignore_categories=["ignore"],
            polyline_width=3,
            layout="voc",
        )
    )

    copied_images = sorted(path.name for path in (out / "JPEGImages").glob("*"))
    masks = sorted((out / "SegmentationClass").glob("*.png"))
    assert copied_images == ["one.jpg", "two.png"]
    assert [path.name for path in masks] == ["one.png", "two.png"]
    assert set(
        (out / "ImageSets" / "Segmentation" / "train.txt").read_text().splitlines()
    ) | set(
        (out / "ImageSets" / "Segmentation" / "val.txt").read_text().splitlines()
    ) == {
        "one",
        "two",
    }
    assert (out / "labelmap.txt").read_text().splitlines() == [
        "# label:color_rgb:parts:actions",
        "background:0,0,0::",
        "object:255,255,255::",
    ]
    assert not (out / "images").exists()
    assert not (out / "annotations").exists()
    assert not (out / "train.txt").exists()
    assert not (out / "val.txt").exists()

    with Image.open(out / "SegmentationClass" / "one.png") as mask:
        assert mask.mode == "P"
        values = np.array(mask)
        palette = mask.getpalette()
        assert values[2, 2] == 1
        assert values[3, 3] == 255
        assert palette[:6] == [0, 0, 0, 255, 255, 255]
        assert palette[255 * 3 : 255 * 3 + 3] == [128, 128, 128]


def test_strict_categories_allows_background_not_in_cvat(tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.jpg"
    cv2.imwrite(str(img1), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((8, 8, 3), dtype=np.uint8))
    annotations = tmp_path / "annotations.xml"
    annotations.write_text(
        """<annotations>
          <meta><task><labels><label><name>object</name></label></labels></task></meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
          <image id="1" name="two.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
        </annotations>"""
    )

    convert_cvat_xml_to_mmseg(
        _config(
            annotations=annotations,
            output=tmp_path / "prepared",
            train=0.5,
            categories=["background", "object"],
            strict_categories=True,
        )
    )

    mask = cv2.imread(
        str(next((tmp_path / "prepared" / "annotations").glob("*.png"))),
        cv2.IMREAD_UNCHANGED,
    )
    assert 0 in np.unique(mask)


def test_strict_categories_rejects_extra_non_background_label(tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.jpg"
    cv2.imwrite(str(img1), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((8, 8, 3), dtype=np.uint8))
    annotations = tmp_path / "annotations.xml"
    annotations.write_text(
        """<annotations>
          <meta><task><labels><label><name>object</name></label></labels></task></meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
          <image id="1" name="two.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
        </annotations>"""
    )

    with pytest.raises(ValueError, match="Config labels missing from CVAT"):
        convert_cvat_xml_to_mmseg(
            _config(
                annotations=annotations,
                output=tmp_path / "prepared",
                train=0.5,
                categories=["background", "object", "extra"],
                strict_categories=True,
            )
        )
