from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pytest

from databox.segmentation.cvat_xml_to_mmseg import (
    Config,
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
        polyline_width=5,
        strict_categories=False,
    )
    config.update(kwargs)
    return Config(**config)


def _image(xml: str):
    return ET.fromstring(xml)


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
    assert not (out / "test.txt").exists()


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
