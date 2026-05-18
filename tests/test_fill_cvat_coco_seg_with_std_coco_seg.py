from unittest import TestCase

from databox.cvat.fill_cvat_coco_seg_with_std_coco_seg import migrate_coco_annotations


class TestCocoToCvat(TestCase):
    def test_migrate_annotations_by_basename_and_category_name(self):
        standard = {
            "images": [
                {"id": 10, "file_name": "foo.jpg", "width": 100, "height": 50},
                {
                    "id": 11,
                    "file_name": "nested/bar.jpg",
                    "width": 100,
                    "height": 50,
                },
            ],
            "annotations": [
                {
                    "id": 100,
                    "image_id": 10,
                    "category_id": 2,
                    "bbox": [1, 2, 3, 4],
                    "segmentation": [[1, 2, 3, 2, 3, 4, 1, 4]],
                    "area": 6,
                    "iscrowd": 0,
                },
                {
                    "id": 101,
                    "image_id": 11,
                    "category_id": 1,
                    "bbox": [5, 6, 7, 8],
                    "segmentation": [[5, 6, 12, 6, 12, 14, 5, 14]],
                    "area": 56,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 1, "name": "door", "supercategory": "parts"},
                {"id": 2, "name": "hood", "supercategory": "parts"},
            ],
        }
        cvat = {
            "licenses": [],
            "info": {"description": "cvat export"},
            "categories": [
                {"id": 7, "name": "hood", "supercategory": ""},
                {"id": 3, "name": "roof", "supercategory": ""},
                {"id": 9, "name": "door", "supercategory": ""},
            ],
            "images": [
                {
                    "id": 501,
                    "file_name": "task/images/bar.jpg",
                    "width": 100,
                    "height": 50,
                },
                {
                    "id": 500,
                    "file_name": "task/images/foo.jpg",
                    "width": 100,
                    "height": 50,
                },
            ],
            "annotations": [],
        }

        migrated = migrate_coco_annotations(standard, cvat)

        self.assertEqual(cvat["images"], migrated["images"])
        self.assertEqual(cvat["categories"], migrated["categories"])
        self.assertEqual(2, len(migrated["annotations"]))
        self.assertEqual(1, migrated["annotations"][0]["id"])
        self.assertEqual(500, migrated["annotations"][0]["image_id"])
        self.assertEqual(7, migrated["annotations"][0]["category_id"])
        self.assertEqual(2, migrated["annotations"][1]["id"])
        self.assertEqual(501, migrated["annotations"][1]["image_id"])
        self.assertEqual(9, migrated["annotations"][1]["category_id"])
        self.assertEqual(
            [[5, 6, 12, 6, 12, 14, 5, 14]],
            migrated["annotations"][1]["segmentation"],
        )

    def test_rejects_existing_target_annotations_without_opt_in(self):
        standard = {
            "images": [{"id": 1, "file_name": "foo.jpg"}],
            "annotations": [],
            "categories": [{"id": 1, "name": "hood"}],
        }
        cvat = {
            "images": [{"id": 1, "file_name": "task/foo.jpg"}],
            "annotations": [{"id": 1}],
            "categories": [{"id": 1, "name": "hood"}],
        }

        with self.assertRaisesRegex(ValueError, "Target CVAT annotations are not empty"):
            migrate_coco_annotations(standard, cvat)

    def test_rejects_duplicate_cvat_image_basenames(self):
        standard = {
            "images": [{"id": 1, "file_name": "foo.jpg"}],
            "annotations": [],
            "categories": [{"id": 1, "name": "hood"}],
        }
        cvat = {
            "images": [
                {"id": 1, "file_name": "a/foo.jpg"},
                {"id": 2, "file_name": "b/foo.jpg"},
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "hood"}],
        }

        with self.assertRaisesRegex(ValueError, "duplicate image basenames"):
            migrate_coco_annotations(standard, cvat)
