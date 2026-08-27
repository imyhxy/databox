import os

import pytest
import yaml
from databox.pose.cvat_xml_to_ultralytics_pose import (
    convert_cvat_xml_to_ultralytics_pose,
)


def _write_export(root, image_xml: str, *, extra_images: str = ""):
    (root / "images").mkdir(parents=True)
    (root / "images" / "frame.jpg").write_bytes(b"frame")
    (root / "images" / "empty.jpg").write_bytes(b"empty")
    (root / "images" / "deleted.jpg").write_bytes(b"deleted")
    (root / "annotations.xml").write_text(
        f"""<annotations>
          <meta><task><labels>
            <label><name>wheel</name><type>bbox</type></label>
            <label><name>wheel_pose</name><type>skeleton</type></label>
            <label><name>center</name><type>points</type></label>
            <label><name>contact</name><type>points</type></label>
          </labels></task></meta>
          {image_xml}
          {extra_images}
        </annotations>""",
        encoding="utf-8",
    )


def test_converts_visibility_and_keeps_empty_xml_images(tmp_path):
    root = tmp_path / "raw"
    _write_export(
        root,
        """<image id="0" name="images/frame.jpg" width="100" height="50">
          <box label="wheel" group_id="1" xtl="10" ytl="5" xbr="30"
               ybr="25" occluded="0" />
          <skeleton label="wheel_pose" group_id="1" outside="0">
            <points label="center" points="20,20" occluded="1" outside="0" />
            <points label="contact" points="25,25" occluded="0" outside="1" />
          </skeleton>
        </image>
        <image id="1" name="images/empty.jpg" width="100" height="50" />""",
    )

    output = tmp_path / "pose"
    stats = convert_cvat_xml_to_ultralytics_pose(root, output)

    assert stats.image_count == 2
    assert stats.instance_count == 1
    assert stats.skipped_group_count == 0
    assert stats.skipped_shape_count == 0

    output_image = output / "images" / "frame.jpg"
    assert output_image.is_symlink()
    assert not os.readlink(output_image).startswith("/")
    assert output_image.resolve() == (root / "images" / "frame.jpg").resolve()

    assert (output / "data.txt").read_text() == (
        "./images/frame.jpg\n./images/empty.jpg\n"
    )
    assert not (output / "data.txt").read_text().endswith("deleted.jpg\n")

    assert (output / "labels" / "frame.txt").read_text() == (
        "0 0.200000 0.300000 0.200000 0.400000 "
        "0.200000 0.400000 1 0.000000 0.000000 0\n"
    )
    assert (output / "labels" / "empty.txt").read_text() == ""
    assert yaml.safe_load((output / "data.yaml").read_text()) == {
        "train": "data.txt",
        "val": "data.txt",
        "names": {0: "wheel"},
        "kpt_shape": [2, 3],
        "kpt_names": {0: ["center", "contact"]},
    }


def test_non_one_to_one_groups_are_warned_and_skipped(tmp_path, capsys):
    root = tmp_path / "raw"
    _write_export(
        root,
        """<image id="0" name="images/frame.jpg" width="100" height="50">
          <box label="wheel" group_id="1" xtl="10" ytl="5" xbr="30" ybr="25" />
          <skeleton label="wheel_pose" group_id="1">
            <points label="center" points="20,20" />
            <points label="contact" points="25,25" />
          </skeleton>
          <box label="wheel" group_id="2" xtl="40" ytl="5" xbr="60" ybr="25" />
          <skeleton label="wheel_pose" group_id="3">
            <points label="center" points="50,20" />
            <points label="contact" points="55,25" />
          </skeleton>
          <box label="wheel" group_id="4" xtl="65" ytl="5" xbr="80" ybr="25" />
          <box label="wheel" group_id="4" xtl="5" ytl="5" xbr="15" ybr="25" />
          <skeleton label="wheel_pose" group_id="4">
            <points label="center" points="70,20" />
            <points label="contact" points="75,25" />
          </skeleton>
        </image>""",
    )

    output = tmp_path / "pose"
    stats = convert_cvat_xml_to_ultralytics_pose(root, output)

    assert stats.instance_count == 1
    assert stats.skipped_group_count == 3
    assert stats.skipped_shape_count == 5
    warnings = capsys.readouterr().err
    assert "group=2" in warnings
    assert "group=3" in warnings
    assert "group=4" in warnings
    assert "expected exactly one box and one skeleton" in warnings
    assert len((output / "labels" / "frame.txt").read_text().splitlines()) == 1


def test_image_with_only_skipped_groups_still_has_empty_label(tmp_path, capsys):
    root = tmp_path / "raw"
    _write_export(
        root,
        """<image id="0" name="images/frame.jpg" width="100" height="50">
          <box label="wheel" group_id="9" xtl="10" ytl="5" xbr="30" ybr="25" />
        </image>""",
    )

    output = tmp_path / "pose"
    stats = convert_cvat_xml_to_ultralytics_pose(root, output)

    assert stats.instance_count == 0
    assert stats.skipped_group_count == 1
    assert (output / "data.txt").read_text() == "./images/frame.jpg\n"
    assert (output / "labels" / "frame.txt").read_text() == ""
    assert "group=9" in capsys.readouterr().err


def test_paired_group_with_invalid_points_is_skipped(tmp_path, capsys):
    root = tmp_path / "raw"
    _write_export(
        root,
        """<image id="0" name="images/frame.jpg" width="100" height="50">
          <box label="wheel" group_id="1" xtl="10" ytl="5" xbr="30" ybr="25" />
          <skeleton label="wheel_pose" group_id="1">
            <points label="center" points="20,20" />
            <points label="unknown" points="25,25" />
          </skeleton>
        </image>""",
    )

    output = tmp_path / "pose"
    stats = convert_cvat_xml_to_ultralytics_pose(root, output)

    assert stats.instance_count == 0
    assert stats.skipped_group_count == 1
    assert (output / "labels" / "frame.txt").read_text() == ""
    assert "unexpected keypoint label" in capsys.readouterr().err


def test_invalid_source_image_is_fatal_and_does_not_replace_output(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "images").mkdir()
    (root / "annotations.xml").write_text(
        """<annotations>
          <image id="0" name="images/missing.jpg" width="10" height="10" />
        </annotations>""",
        encoding="utf-8",
    )
    output = tmp_path / "pose"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        convert_cvat_xml_to_ultralytics_pose(root, output)

    assert sentinel.read_text() == "keep"
