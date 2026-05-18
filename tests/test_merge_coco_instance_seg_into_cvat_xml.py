from unittest import TestCase
from xml.etree import ElementTree as ET

from databox.cvat.merge_coco_instance_seg_into_cvat_xml import (
    merge_coco_instance_segmentation_into_cvat_xml,
)


def _root():
    return ET.fromstring(
        """<annotations>
  <meta>
    <task>
      <labels>
        <label><name>door</name></label>
        <label><name>wheel</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="dataset/images/foo.jpg" width="100" height="50">
    <polyline label="existing_line" source="manual" occluded="0" points="1.00,1.00;2.00,2.00" z_order="0">
    </polyline>
  </image>
</annotations>"""
    )


class TestMergeCocoInstanceSegIntoCvatXml(TestCase):
    def test_appends_polygons_without_removing_existing_shapes(self):
        coco = {
            "images": [{"id": 10, "file_name": "foo.jpg"}],
            "categories": [{"id": 2, "name": "door"}],
            "annotations": [
                {
                    "id": 100,
                    "image_id": 10,
                    "category_id": 2,
                    "segmentation": [[1, 2, 3, 2, 3, 4, 1, 4]],
                }
            ],
        }
        root = _root()

        stats = merge_coco_instance_segmentation_into_cvat_xml(coco, root)

        image = root.find("./image")
        self.assertEqual(
            {
                "annotations": 1,
                "polygons": 1,
                "skipped_images": 0,
                "skipped_labels": 0,
            },
            stats,
        )
        self.assertEqual("polyline", image[0].tag)
        polygon = image.find("./polygon")
        self.assertIsNotNone(polygon)
        self.assertEqual("door", polygon.get("label"))
        self.assertEqual(
            "1.00,2.00;3.00,2.00;3.00,4.00;1.00,4.00",
            polygon.get("points"),
        )

    def test_uses_one_group_id_for_coco_instance_sub_parts(self):
        coco = {
            "images": [{"id": 10, "file_name": "foo.jpg"}],
            "categories": [{"id": 2, "name": "wheel"}],
            "annotations": [
                {
                    "id": 100,
                    "image_id": 10,
                    "category_id": 2,
                    "segmentation": [
                        [1, 1, 2, 1, 2, 2],
                        [3, 3, 4, 3, 4, 4],
                    ],
                }
            ],
        }
        root = _root()

        merge_coco_instance_segmentation_into_cvat_xml(coco, root)

        polygons = root.findall("./image/polygon")
        self.assertEqual(2, len(polygons))
        self.assertEqual({"1"}, {polygon.get("group_id") for polygon in polygons})

    def test_starts_after_existing_group_ids(self):
        root = _root()
        root.find("./image/polyline").set("group_id", "7")
        coco = {
            "images": [{"id": 10, "file_name": "foo.jpg"}],
            "categories": [{"id": 2, "name": "door"}],
            "annotations": [
                {
                    "id": 100,
                    "image_id": 10,
                    "category_id": 2,
                    "segmentation": [[1, 1, 2, 1, 2, 2]],
                }
            ],
        }

        merge_coco_instance_segmentation_into_cvat_xml(coco, root)

        self.assertEqual("8", root.find("./image/polygon").get("group_id"))

    def test_rejects_missing_cvat_label_by_default(self):
        coco = {
            "images": [{"id": 10, "file_name": "foo.jpg"}],
            "categories": [{"id": 2, "name": "hood"}],
            "annotations": [
                {
                    "id": 100,
                    "image_id": 10,
                    "category_id": 2,
                    "segmentation": [[1, 1, 2, 1, 2, 2]],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "missing label"):
            merge_coco_instance_segmentation_into_cvat_xml(coco, _root())
