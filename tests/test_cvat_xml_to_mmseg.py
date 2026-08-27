from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pytest
from databox.segmentation.cvat_shape_repair import (
    deduplicate_points,
    repair_shape,
)
from databox.segmentation.cvat_xml_to_mmseg import (
    Config,
    config_from_args,
    convert_cvat_xml_to_mmseg,
    job_id_for_frame,
    parse_cvat_task_metadata,
    polygon_annotation_lines,
    polyline_annotation_lines,
    rasterize_image,
    rasterize_shape_branch,
    validate_config,
)
from databox.segmentation.dataset_manifest import read_manifest
from PIL import Image


def _config(**kwargs):
    config = {
        "annotations": Path("annotations.xml"),
        "output": Path("out"),
        "seed": 0,
        "train": 0.8,
        "categories": ["background", "object", "line"],
        "ignore_categories": [],
        "ignore_index": 255,
        "ignore_palette": (128, 128, 128),
        "polyline_width": 5,
        "repair_polyline_width": 1,
        "strict_categories": False,
        "palette": [(0, 0, 0), (255, 255, 255), (0, 255, 0)],
        "polygon_categories": ["object"],
        "polyline_categories": ["line"],
        "vehicle_label_dir": Path("vehicle"),
    }
    config.update(kwargs)
    return Config(**config)


def _image(xml: str):
    return ET.fromstring(xml)


def _write_annotations(path: Path, xml_text: str) -> None:
    metadata = """<task>
              <id>104</id>
              <name>batch_260618</name>
              <segments>
                <segment><id>208</id><start>0</start><stop>0</stop></segment>
                <segment><id>209</id><start>1</start><stop>999</stop></segment>
              </segments>"""
    path.write_text(xml_text.replace("<task>", metadata, 1))


def _write_vehicle_labels(root: Path, *stems: str) -> Path:
    directory = root / "vehicle"
    directory.mkdir(exist_ok=True)
    for stem in stems:
        Image.new("L", (8, 8), 0).save(directory / f"{stem}_vehicle.png")
        (directory / f"{stem}_vehicle.txt").write_text("")
    return directory


def test_parse_cvat_task_metadata_maps_segment_boundaries():
    root = ET.fromstring(
        """<annotations><meta><task>
          <id>104</id><name>batch_260618</name>
          <segments>
            <segment><id>208</id><start>0</start><stop>315</stop></segment>
            <segment><id>209</id><start>316</start><stop>500</stop></segment>
          </segments>
        </task></meta></annotations>"""
    )

    metadata = parse_cvat_task_metadata(root)

    assert metadata.task_id == 104
    assert metadata.task_name == "batch_260618"
    assert job_id_for_frame(metadata, 315) == 208
    assert job_id_for_frame(metadata, 316) == 209
    with pytest.raises(ValueError, match="exactly one job"):
        job_id_for_frame(metadata, 501)


def test_parse_cvat_task_metadata_rejects_overlapping_segments():
    root = ET.fromstring(
        """<annotations><meta><task>
          <id>104</id><name>batch</name>
          <segments>
            <segment><id>208</id><start>0</start><stop>10</stop></segment>
            <segment><id>209</id><start>10</start><stop>20</stop></segment>
          </segments>
        </task></meta></annotations>"""
    )

    with pytest.raises(ValueError, match="segments overlap"):
        parse_cvat_task_metadata(root)


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
    - line
  polygon_categories:
    - object
  polyline_categories:
    - line
  palette:
    - [0, 0, 0]
    - [255, 255, 255]
    - [0, 255, 0]
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
    assert config.palette == [(0, 0, 0), (255, 255, 255), (0, 255, 0)]
    assert config.ignore_palette == (128, 128, 128)
    assert config.polygon_categories == ["object"]
    assert config.polyline_categories == ["line"]
    assert config.repair_self_intersections is False
    assert config.self_intersection_threshold == 10
    assert config.polyline_diff_threshold == 10
    assert config.repair_polyline_width == 1


def test_yaml_self_intersection_options_can_be_overridden_by_cli(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        """cvat_xml_to_mmseg:
  seed: 7
  categories: [background, object, line]
  polygon_categories: [object]
  polyline_categories: [line]
  palette: [[0, 0, 0], [255, 255, 255], [0, 255, 0]]
  ignore_palette: [128, 128, 128]
  repair_self_intersections: true
  self_intersection_threshold: 12
  polyline_diff_threshold: 14
  repair_polyline_width: 2
"""
    )

    yaml_config = config_from_args(
        SimpleNamespace(
            yaml=True,
            config=config_path,
            param_name="cvat_xml_to_mmseg",
            input="annotations.xml",
            output="out",
            repair_self_intersections=None,
            self_intersection_threshold=None,
            polyline_diff_threshold=None,
            repair_polyline_width=None,
        )
    )

    assert yaml_config.repair_self_intersections is True
    assert yaml_config.self_intersection_threshold == 12
    assert yaml_config.polyline_diff_threshold == 14
    assert yaml_config.repair_polyline_width == 2

    config = config_from_args(
        SimpleNamespace(
            yaml=True,
            config=config_path,
            param_name="cvat_xml_to_mmseg",
            input="annotations.xml",
            output="out",
            repair_self_intersections=False,
            self_intersection_threshold=3,
            polyline_diff_threshold=4,
            repair_polyline_width=3,
        )
    )

    assert config.repair_self_intersections is False
    assert config.self_intersection_threshold == 3
    assert config.polyline_diff_threshold == 4
    assert config.repair_polyline_width == 3


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
                polygon_categories=["object"],
                polyline_categories=["line"],
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
                polygon_categories=["object"],
                polyline_categories=["line"],
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
            categories=["background", "object", "line"],
            palette=["0,0,0", "255,255,255", "0,255,0"],
            ignore_categories=[],
            ignore_index=255,
            ignore_palette="128,128,128",
            polyline_width=5,
            strict_categories=False,
            layout="voc",
            polygon_categories=["object"],
            polyline_categories=["line"],
        )
    )

    assert config.palette == [(0, 0, 0), (255, 255, 255), (0, 255, 0)]
    assert config.ignore_palette == (128, 128, 128)
    assert config.layout == "voc"
    assert config.polygon_categories == ["object"]
    assert config.polyline_categories == ["line"]


def test_cli_mode_requires_branch_categories():
    with pytest.raises(ValueError, match="--polygon-categories"):
        config_from_args(
            SimpleNamespace(
                yaml=False,
                input="annotations.xml",
                output="out",
                seed=0,
                train=0.8,
                categories=["background", "object", "line"],
                palette=["0,0,0", "255,255,255", "0,255,0"],
                ignore_categories=[],
                ignore_index=255,
                ignore_palette="128,128,128",
                polyline_width=5,
                strict_categories=False,
                layout="voc",
                polygon_categories=None,
                polyline_categories=["line"],
            )
        )


def test_yaml_mode_requires_branch_categories(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        """cvat_xml_to_mmseg:
  seed: 7
  categories:
    - background
    - object
    - line
  palette:
    - [0, 0, 0]
    - [255, 255, 255]
    - [0, 255, 0]
  ignore_palette: [128, 128, 128]
"""
    )

    with pytest.raises(ValueError, match="polygon_categories"):
        config_from_args(
            SimpleNamespace(
                yaml=True,
                config=config_path,
                param_name="cvat_xml_to_mmseg",
                input="cli_annotations.xml",
                output="cli_out",
                palette=None,
            )
        )


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
    validate_config(_config(polyline_width=5))

    with pytest.raises(ValueError, match="polyline_width"):
        validate_config(_config(polyline_width=21))


def test_self_intersection_threshold_validation():
    validate_config(_config(self_intersection_threshold=0, polyline_diff_threshold=0))

    with pytest.raises(ValueError, match="self_intersection_threshold"):
        validate_config(_config(self_intersection_threshold=-1))

    with pytest.raises(ValueError, match="polyline_diff_threshold"):
        validate_config(_config(polyline_diff_threshold=-1))

    with pytest.raises(ValueError, match="repair_polyline_width"):
        validate_config(_config(repair_polyline_width=0))

    with pytest.raises(ValueError, match="repair_polyline_width"):
        validate_config(_config(repair_polyline_width=21))


def test_branch_category_validation():
    with pytest.raises(ValueError, match="polygon_categories missing from categories"):
        validate_config(_config(polygon_categories=["missing"]))

    with pytest.raises(
        ValueError, match="polygon_categories and polyline_categories overlap"
    ):
        validate_config(
            _config(polygon_categories=["object"], polyline_categories=["object"])
        )


def test_polygon_branch_uses_one_based_branch_index():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <polygon label="late" points="1,1;5,1;5,5;1,5" />
        </image>"""
    )

    mask = rasterize_shape_branch(
        image,
        "polygon",
        ["late"],
        [],
    )

    assert mask[3, 3] == 1


def test_polyline_branch_uses_one_based_branch_index():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <polyline label="late_line" points="1,1;6,6" />
        </image>"""
    )

    mask = rasterize_shape_branch(
        image,
        "polyline",
        ["late_line"],
        [],
        polyline_width=3,
    )

    assert mask[3, 3] == 1


def test_polyline_annotation_lines_keep_original_float_points():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <polyline label="line" points="1.25,1.5;6.75,6.125" />
          <polyline label="ignored_line" points="0,0;1,1" />
        </image>"""
    )

    lines = polyline_annotation_lines(
        image,
        ["background", "object", "line", "ignored_line"],
        ["line"],
    )

    assert lines == ["2 1.25 1.5 6.75 6.125"]


def test_polygon_annotation_lines_keep_original_float_points():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <polygon label="object" points="1.25,1.5;6.75,1.125;6.5,6.25" />
          <polygon label="ignored_object" points="0,0;1,0;1,1" />
        </image>"""
    )

    lines = polygon_annotation_lines(
        image,
        ["background", "object", "line", "ignored_object"],
        ["object"],
    )

    assert lines == ["1 1.25 1.5 6.75 1.125 6.5 6.25"]


def test_repair_shape_deduplicates_points_before_detection():
    points = np.array(
        [
            [1.0, 1.0],
            [6.0, 1.0],
            [6.0, 6.0],
            [1.0, 6.0],
            [1.0, 6.0],
            [1.0, 1.0],
        ]
    )

    result = repair_shape(points, "polygon", (8, 8), threshold=0)

    assert result.points.tolist() == [
        [1.0, 1.0],
        [6.0, 1.0],
        [6.0, 6.0],
        [1.0, 6.0],
    ]
    assert result.intersections == ()
    assert result.diff_pixels == 0
    assert result.needs_review is False


def test_deduplicate_points_preserves_order_and_removes_all_repeats():
    points = [(1, 1), (2, 2), (1, 1), (3, 3), (2, 2)]

    assert deduplicate_points(points).tolist() == [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]


def test_annotation_lines_export_without_duplicate_points():
    image = _image(
        """<image id="0" name="foo.jpg" width="8" height="8">
          <polygon label="object" points="1,1;6,1;6,6;1,6;1,6;1,1" />
          <polyline label="line" points="1,1;3,3;3,3;1,1" />
        </image>"""
    )

    polygon_lines = polygon_annotation_lines(
        image,
        ["background", "object", "line"],
        ["object"],
    )
    polyline_lines = polyline_annotation_lines(
        image,
        ["background", "object", "line"],
        ["line"],
    )

    assert polygon_lines == ["1 1 1 6 1 6 6 1 6"]
    assert polyline_lines == ["2 1 1 3 3"]


def test_annotation_lines_keep_original_points_when_repair_is_disabled():
    image = _image(
        """<image id="0" name="foo.jpg" width="8" height="8">
          <polygon label="object" points="1,1;6,6;1,6;6,1" />
          <polyline label="line" points="1,1;3,3;1,3;3,1" />
        </image>"""
    )

    polygon_lines = polygon_annotation_lines(
        image,
        ["background", "object", "line"],
        ["object"],
        repair_self_intersections=False,
        self_intersection_threshold=0,
    )
    polyline_lines = polyline_annotation_lines(
        image,
        ["background", "object", "line"],
        ["line"],
        repair_self_intersections=False,
        polyline_diff_threshold=0,
        repair_polyline_width=1,
    )

    assert polygon_lines == ["1 1 1 6 6 1 6 6 1"]
    assert polyline_lines == ["2 1 1 3 3 1 3 3 1"]


def test_polygon_annotation_lines_can_repair_a_small_self_intersection():
    image = _image(
        """<image id="0" name="foo.jpg" width="8" height="8">
          <polygon label="object" points="1,1;6,6;1,6;6,1" />
        </image>"""
    )

    lines = polygon_annotation_lines(
        image,
        ["background", "object"],
        ["object"],
        repair_self_intersections=True,
        self_intersection_threshold=10,
    )

    assert lines == ["1 1 1 3.5 3.5 6 1"]


def test_self_intersection_repair_keeps_large_changes_for_manual_review(caplog):
    image = _image(
        """<image id="0" name="foo.jpg" width="100" height="100">
          <polygon label="object" points="1,1;90,90;1,90;90,1" />
        </image>"""
    )

    lines = polygon_annotation_lines(
        image,
        ["background", "object"],
        ["object"],
        repair_self_intersections=True,
        self_intersection_threshold=10,
    )

    assert lines == ["1 1 1 90 90 1 90 90 1"]
    assert "manual review is required" in caplog.text


def test_polyline_annotation_lines_can_repair_with_line_pixel_threshold():
    image = _image(
        """<image id="0" name="foo.jpg" width="8" height="8">
          <polyline label="line" points="1,1;3,3;1,3;3,1" />
        </image>"""
    )

    lines = polyline_annotation_lines(
        image,
        ["background", "line"],
        ["line"],
        repair_self_intersections=True,
        polyline_diff_threshold=10,
        repair_polyline_width=1,
    )

    assert lines == ["1 1 1 1 3 3 3 3 1"]


def test_polyline_diff_threshold_controls_acceptance_separately_from_width():
    image = _image(
        """<image id="0" name="foo.jpg" width="8" height="8">
          <polyline label="line" points="1,1;3,3;1,3;3,1" />
        </image>"""
    )

    kept = polyline_annotation_lines(
        image,
        ["background", "line"],
        ["line"],
        repair_self_intersections=True,
        polyline_diff_threshold=2,
        repair_polyline_width=1,
    )
    repaired = polyline_annotation_lines(
        image,
        ["background", "line"],
        ["line"],
        repair_self_intersections=True,
        polyline_diff_threshold=3,
        repair_polyline_width=1,
    )

    assert kept == ["1 1 1 3 3 1 3 3 1"]
    assert repaired == ["1 1 1 1 3 3 3 3 1"]


def test_conversion_repairs_txt_sidecars_without_changing_masks(tmp_path):
    image_one = tmp_path / "one.jpg"
    image_two = tmp_path / "two.jpg"
    cv2.imwrite(str(image_one), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(image_two), np.zeros((8, 8, 3), dtype=np.uint8))
    annotations = tmp_path / "annotations.xml"
    _write_annotations(
        annotations,
        """<annotations>
          <meta>
            <task>
              <labels>
                <label><name>background</name></label>
                <label><name>object</name></label>
                <label><name>line</name></label>
              </labels>
            </task>
          </meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,6;1,6;6,1" />
          </image>
          <image id="1" name="two.jpg" width="8" height="8">
            <polyline label="line" points="1,1;3,3;1,3;3,1" />
          </image>
        </annotations>""",
    )
    vehicle_labels = _write_vehicle_labels(tmp_path, "one", "two")
    output = tmp_path / "prepared"

    convert_cvat_xml_to_mmseg(
        _config(
            annotations=annotations,
            output=output,
            train=0.5,
            vehicle_label_dir=vehicle_labels,
            repair_self_intersections=True,
            self_intersection_threshold=10,
            polyline_diff_threshold=10,
            repair_polyline_width=1,
            polyline_width=5,
        )
    )

    original_one = _image(
        '<image id="0" name="one.jpg" width="8" height="8">'
        '<polygon label="object" points="1,1;6,6;1,6;6,1" />'
        "</image>"
    )
    original_mask = rasterize_shape_branch(
        original_one,
        "polygon",
        ["object"],
        [],
    )
    with Image.open(output / "annotations" / "one_polygon.png") as mask:
        assert np.array_equal(np.asarray(mask), original_mask)
    assert (output / "annotations" / "one_polygon.txt").read_text() == (
        "1 1 1 3.5 3.5 6 1\n"
    )

    original_two = _image(
        '<image id="1" name="two.jpg" width="8" height="8">'
        '<polyline label="line" points="1,1;3,3;1,3;3,1" />'
        "</image>"
    )
    original_polyline_mask = rasterize_shape_branch(
        original_two,
        "polyline",
        ["line"],
        [],
        polyline_width=5,
    )
    with Image.open(output / "annotations" / "two_polyline.png") as mask:
        assert np.array_equal(np.asarray(mask), original_polyline_mask)
    assert (output / "annotations" / "two_polyline.txt").read_text() == (
        "2 1 1 1 3 3 3 3 1\n"
    )


def test_branch_masks_share_global_ignore_shapes():
    image = _image(
        """<image id="0" name="foo.jpg" width="10" height="10">
          <polygon label="object" points="1,1;8,1;8,8;1,8" />
          <polyline label="line" points="1,8;8,1" />
          <polygon label="ignore" points="2,2;4,2;4,4;2,4" />
          <polyline label="ignore" points="6,1;6,8" />
        </image>"""
    )

    polygon_mask = rasterize_shape_branch(
        image,
        "polygon",
        ["object"],
        ["ignore"],
        ignore_index=255,
        polyline_width=3,
    )
    polyline_mask = rasterize_shape_branch(
        image,
        "polyline",
        ["line"],
        ["ignore"],
        ignore_index=255,
        polyline_width=3,
    )

    assert polygon_mask[3, 3] == 255
    assert polygon_mask[4, 6] == 255
    assert polyline_mask[3, 3] == 255
    assert polyline_mask[4, 6] == 255


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
    _write_annotations(
        annotations,
        """<annotations>
          <meta>
            <task>
              <labels>
                <label><name>background</name></label>
                <label><name>object</name></label>
                <label><name>line</name></label>
              </labels>
            </task>
          </meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
          <image id="1" name="two.jpg" width="8" height="8">
            <polyline label="line" points="1.25,1.5;6.75,6.125" />
          </image>
        </annotations>""",
    )

    out = tmp_path / "prepared"
    vehicle_labels = _write_vehicle_labels(tmp_path, "one", "two")
    convert_cvat_xml_to_mmseg(
        _config(
            annotations=annotations,
            output=out,
            train=0.5,
            categories=["background", "object", "line"],
            polyline_width=3,
            vehicle_label_dir=vehicle_labels,
        )
    )

    copied_images = list((out / "images").glob("*"))
    masks = sorted(path.name for path in (out / "annotations").glob("*.png"))
    shape_txts = sorted(path.name for path in (out / "annotations").glob("*.txt"))
    assert len(copied_images) == 2
    assert masks == [
        "one.png",
        "one_polygon.png",
        "one_polyline.png",
        "one_vehicle.png",
        "two.png",
        "two_polygon.png",
        "two_polyline.png",
        "two_vehicle.png",
    ]
    assert shape_txts == [
        "one_polygon.txt",
        "one_polyline.txt",
        "one_vehicle.txt",
        "two_polygon.txt",
        "two_polyline.txt",
        "two_vehicle.txt",
    ]
    assert (
        out / "annotations" / "one_polygon.txt"
    ).read_text() == "1 1 1 6 1 6 6 1 6\n"
    assert (out / "annotations" / "two_polygon.txt").read_text() == ""
    assert (out / "annotations" / "one_polyline.txt").read_text() == ""
    assert (
        out / "annotations" / "two_polyline.txt"
    ).read_text() == "2 1.25 1.5 6.75 6.125\n"
    assert set((out / "train.txt").read_text().splitlines()) | set(
        (out / "val.txt").read_text().splitlines()
    ) == {"one", "two"}
    assert (out / "labelmap.txt").read_text().splitlines() == [
        "# label:color_rgb:parts:actions",
        "background:0,0,0::",
        "object:255,255,255::",
        "line:0,255,0::",
    ]
    assert (out / "labelmap_polygon.txt").read_text().splitlines() == [
        "# label:color_rgb:parts:actions",
        "background:0,0,0::",
        "object:255,255,255::",
    ]
    assert (out / "labelmap_polyline.txt").read_text().splitlines() == [
        "# label:color_rgb:parts:actions",
        "background:0,0,0::",
        "line:0,255,0::",
    ]
    assert (out / "labelmap_vehicle.txt").read_text().splitlines()[1:] == [
        "background:0,0,0::",
        "vehicle:255,255,255::",
        "ignore:128,128,128::",
    ]
    assert not (out / "test.txt").exists()
    assert not (out / "JPEGImages").exists()
    assert not (out / "SegmentationClass").exists()
    assert not (out / "ImageSets" / "Segmentation").exists()
    manifest = {
        Path(record["image_path"]).name: record for record in read_manifest(out)
    }
    assert manifest["one.jpg"]["sample_id"] == "cvat_104_208_0"
    assert manifest["two.jpg"]["sample_id"] == "cvat_104_209_1"
    assert manifest["one.jpg"]["task_name"] == "batch_260618"
    assert manifest["one.jpg"]["image_path"] == "images/one.jpg"
    assert manifest["one.jpg"]["mask_paths"] == {
        "polygon": "annotations/one_polygon.png",
        "polyline": "annotations/one_polyline.png",
        "vehicle": "annotations/one_vehicle.png",
        "main": "annotations/one.png",
    }
    assert manifest["one.jpg"]["width"] == 8
    assert manifest["one.jpg"]["height"] == 8
    assert "image_name" not in manifest["one.jpg"]
    with Image.open(out / "annotations" / "one.png") as mask:
        assert mask.mode == "P"
        palette = mask.getpalette()
        assert palette[:6] == [0, 0, 0, 255, 255, 255]
        assert palette[255 * 3 : 255 * 3 + 3] == [128, 128, 128]
    with Image.open(out / "annotations" / "one_polygon.png") as mask:
        assert np.array(mask)[2, 2] == 1
        assert mask.getpalette()[:6] == [0, 0, 0, 255, 255, 255]
    with Image.open(out / "annotations" / "two_polyline.png") as mask:
        assert np.array(mask)[3, 3] == 1
        assert mask.getpalette()[:6] == [0, 0, 0, 0, 255, 0]


def test_convert_writes_voc_layout_and_cleans_stale_mmseg_outputs(tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.png"
    Image.new("RGB", (8, 8)).save(img1)
    Image.new("RGB", (8, 8)).save(img2)
    annotations = tmp_path / "annotations.xml"
    _write_annotations(
        annotations,
        """<annotations>
          <meta>
            <task>
              <labels>
                <label><name>background</name></label>
                <label><name>object</name></label>
                <label><name>line</name></label>
                <label><name>ignore</name></label>
              </labels>
            </task>
          </meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
            <polygon label="ignore" points="3,3;4,3;4,4;3,4" />
          </image>
          <image id="1" name="two.png" width="8" height="8">
            <polyline label="line" points="1.25,1.5;6.75,6.125" />
          </image>
        </annotations>""",
    )

    out = tmp_path / "prepared"
    vehicle_labels = _write_vehicle_labels(tmp_path, "one", "two")
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
            categories=["background", "object", "line"],
            ignore_categories=["ignore"],
            polyline_width=3,
            layout="voc",
            vehicle_label_dir=vehicle_labels,
        )
    )

    copied_images = sorted(path.name for path in (out / "JPEGImages").glob("*"))
    masks = sorted(path.name for path in (out / "SegmentationClass").glob("*.png"))
    shape_txts = sorted(path.name for path in (out / "SegmentationClass").glob("*.txt"))
    assert copied_images == ["one.jpg", "two.jpg"]
    assert masks == [
        "one.png",
        "one_polygon.png",
        "one_polyline.png",
        "one_vehicle.png",
        "two.png",
        "two_polygon.png",
        "two_polyline.png",
        "two_vehicle.png",
    ]
    assert shape_txts == [
        "one_polygon.txt",
        "one_polyline.txt",
        "one_vehicle.txt",
        "two_polygon.txt",
        "two_polyline.txt",
        "two_vehicle.txt",
    ]
    assert (
        out / "SegmentationClass" / "one_polygon.txt"
    ).read_text() == "1 1 1 6 1 6 6 1 6\n"
    assert (out / "SegmentationClass" / "two_polygon.txt").read_text() == ""
    assert (out / "SegmentationClass" / "one_polyline.txt").read_text() == ""
    assert (
        out / "SegmentationClass" / "two_polyline.txt"
    ).read_text() == "2 1.25 1.5 6.75 6.125\n"
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
        "line:0,255,0::",
    ]
    assert (out / "labelmap_polygon.txt").read_text().splitlines() == [
        "# label:color_rgb:parts:actions",
        "background:0,0,0::",
        "object:255,255,255::",
    ]
    assert (out / "labelmap_polyline.txt").read_text().splitlines() == [
        "# label:color_rgb:parts:actions",
        "background:0,0,0::",
        "line:0,255,0::",
    ]
    assert not (out / "images").exists()
    assert not (out / "annotations").exists()
    assert not (out / "train.txt").exists()
    assert not (out / "val.txt").exists()
    manifest = {
        Path(record["image_path"]).name: record for record in read_manifest(out)
    }
    assert manifest["one.jpg"]["image_path"] == "JPEGImages/one.jpg"
    assert manifest["one.jpg"]["mask_paths"] == {
        "polygon": "SegmentationClass/one_polygon.png",
        "polyline": "SegmentationClass/one_polyline.png",
        "vehicle": "SegmentationClass/one_vehicle.png",
        "main": "SegmentationClass/one.png",
    }
    assert manifest["one.jpg"]["width"] == 8
    assert manifest["one.jpg"]["height"] == 8

    with Image.open(out / "SegmentationClass" / "one.png") as mask:
        assert mask.mode == "P"
        values = np.array(mask)
        palette = mask.getpalette()
        assert values[2, 2] == 1
        assert values[3, 3] == 255
        assert palette[:6] == [0, 0, 0, 255, 255, 255]
        assert palette[255 * 3 : 255 * 3 + 3] == [128, 128, 128]
    with Image.open(out / "SegmentationClass" / "one_polygon.png") as mask:
        assert np.array(mask)[2, 2] == 1
        assert np.array(mask)[3, 3] == 255
    with Image.open(out / "SegmentationClass" / "one_polyline.png") as mask:
        assert np.array(mask)[3, 3] == 255
    with Image.open(out / "SegmentationClass" / "two_polyline.png") as mask:
        assert np.array(mask)[3, 3] == 1
        assert mask.getpalette()[:6] == [0, 0, 0, 0, 255, 0]


def test_convert_rejects_branch_mask_name_collisions(tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "one_polygon.jpg"
    cv2.imwrite(str(img1), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((8, 8, 3), dtype=np.uint8))
    annotations = tmp_path / "annotations.xml"
    _write_annotations(
        annotations,
        """<annotations>
          <meta><task><labels>
            <label><name>object</name></label>
            <label><name>line</name></label>
          </labels></task></meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
          <image id="1" name="one_polygon.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
        </annotations>""",
    )

    with pytest.raises(ValueError, match="overwrite branch masks"):
        convert_cvat_xml_to_mmseg(
            _config(annotations=annotations, output=tmp_path / "prepared", train=0.5)
        )


def test_strict_categories_allows_background_not_in_cvat(tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.jpg"
    cv2.imwrite(str(img1), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((8, 8, 3), dtype=np.uint8))
    annotations = tmp_path / "annotations.xml"
    _write_annotations(
        annotations,
        """<annotations>
          <meta><task><labels>
            <label><name>object</name></label>
            <label><name>line</name></label>
          </labels></task></meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
          <image id="1" name="two.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
        </annotations>""",
    )

    vehicle_labels = _write_vehicle_labels(tmp_path, "one", "two")
    convert_cvat_xml_to_mmseg(
        _config(
            annotations=annotations,
            output=tmp_path / "prepared",
            train=0.5,
            categories=["background", "object", "line"],
            strict_categories=True,
            vehicle_label_dir=vehicle_labels,
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
    _write_annotations(
        annotations,
        """<annotations>
          <meta><task><labels>
            <label><name>object</name></label>
            <label><name>line</name></label>
          </labels></task></meta>
          <image id="0" name="one.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
          <image id="1" name="two.jpg" width="8" height="8">
            <polygon label="object" points="1,1;6,1;6,6;1,6" />
          </image>
        </annotations>""",
    )

    with pytest.raises(ValueError, match="Config labels missing from CVAT"):
        convert_cvat_xml_to_mmseg(
            _config(
                annotations=annotations,
                output=tmp_path / "prepared",
                train=0.5,
                categories=["background", "object", "line", "extra"],
                palette=[(0, 0, 0), (255, 255, 255), (0, 255, 0), (255, 0, 0)],
                strict_categories=True,
            )
        )
