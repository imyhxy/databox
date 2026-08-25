import os

import yaml
from databox.segmentation.cvat_xml_to_ultralytics_instance import (
    PARAM_NAME,
    build_ultralytics_instance_dataset,
    config_from_args,
    parse_args,
)


def _write_cvat_export(root, xml_body):
    image = root / "images" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    (root / "annotations.xml").write_text(
        f"""<annotations>
          <meta><task><labels>
            <label><name>car</name></label>
            <label><name>truck</name></label>
          </labels></task></meta>
          {xml_body}
        </annotations>""",
        encoding="utf-8",
    )


def test_grouped_parts_are_one_ultralytics_instance_and_images_are_relative_links(
    tmp_path,
):
    root = tmp_path / "raw"
    _write_cvat_export(
        root,
        """<image id="0" name="images/frame.jpg" width="100" height="50">
          <polygon label="car" group_id="7" points="0,0;20,0;20,10" />
          <polygon label="car" group_id="7" points="80,40;100,40;100,50" />
          <polygon label="truck" points="50,10;70,10;70,20" />
        </image>""",
    )

    output = tmp_path / "ultralytics"
    stats = build_ultralytics_instance_dataset(
        root,
        output,
        categories=["car", "truck"],
        category_map={"car": "vehicle", "truck": "vehicle"},
    )

    assert stats.image_count == 1
    assert stats.instance_count == 2
    assert stats.grouped_instance_count == 1
    assert stats.names == ("vehicle",)

    output_image = output / "images" / "frame.jpg"
    assert output_image.is_symlink()
    assert not os.readlink(output_image).startswith("/")
    assert output_image.resolve() == (root / "images" / "frame.jpg").resolve()

    lines = (output / "labels" / "frame.txt").read_text().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("0 ") for line in lines)
    assert len(lines[0].split()) > 1 + 3 * 2
    assert yaml.safe_load((output / "data.yaml").read_text()) == {
        "path": ".",
        "train": "images",
        "val": "images",
        "nc": 1,
        "names": ["vehicle"],
    }


def test_category_filter_drops_annotations_and_keeps_empty_label_file(tmp_path):
    root = tmp_path / "raw"
    _write_cvat_export(
        root,
        """<image id="0" name="images/frame.jpg" width="10" height="10">
          <polygon label="truck" points="1,1;5,1;5,5" />
        </image>""",
    )

    output = tmp_path / "ultralytics"
    stats = build_ultralytics_instance_dataset(root, output, categories=["car"])

    assert stats.instance_count == 0
    assert stats.filtered_polygon_count == 1
    assert stats.names == ("car",)
    assert (output / "labels" / "frame.txt").read_text() == ""


def test_config_reads_non_path_values_and_cli_wins(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                PARAM_NAME: {
                    "input_root": "ignored-input",
                    "output_root": "ignored-output",
                    "categories": ["car", "truck"],
                    "category_map": {"car": "vehicle", "truck": "vehicle"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--input-root",
            str(tmp_path / "cli-input"),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--config",
            str(config_path),
            "--categories",
            "car",
            "--category-map",
            "car=road_vehicle",
        ]
    )
    config = config_from_args(args)

    assert config.input_root == tmp_path / "cli-input"
    assert config.output_root == tmp_path / "cli-output"
    assert config.categories == ("car",)
    assert config.category_map == {"car": "road_vehicle"}
