import argparse
import copy
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Migrate annotations from a standard COCO JSON into a CVAT-exported "
            "COCO JSON by matching images on basename and categories on name."
        )
    )
    parser.add_argument(
        "standard_annotations",
        type=str,
        help="Path to the source COCO annotations JSON",
    )
    parser.add_argument(
        "cvat_annotations",
        type=str,
        help="Path to the target CVAT COCO annotations JSON",
    )
    parser.add_argument("output", type=str, help="Path to write the migrated JSON")
    parser.add_argument(
        "--allow-existing-annotations",
        action="store_true",
        help="Replace non-empty annotations already present in the CVAT JSON",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level used when writing JSON output",
    )
    return parser.parse_args()


def _load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def _index_images_by_basename(images, dataset_name):
    indexed = {}
    duplicates = set()
    for image in images:
        basename = Path(image["file_name"]).name
        if basename in indexed:
            duplicates.add(basename)
            continue
        indexed[basename] = image

    if duplicates:
        duplicates_str = ", ".join(sorted(duplicates))
        raise ValueError(
            f"{dataset_name} contains duplicate image basenames: {duplicates_str}"
        )

    return indexed


def _index_categories_by_name(categories):
    return {category["name"]: category for category in categories}


def migrate_coco_annotations(
    standard_data, cvat_data, allow_existing_annotations=False
):
    if cvat_data.get("annotations") and not allow_existing_annotations:
        raise ValueError(
            "Target CVAT annotations are not empty. "
            "Use --allow-existing-annotations to replace them."
        )

    standard_images_by_id = {image["id"]: image for image in standard_data["images"]}
    cvat_images_by_basename = _index_images_by_basename(
        cvat_data["images"], "CVAT annotations"
    )

    standard_categories_by_id = {
        category["id"]: category["name"] for category in standard_data["categories"]
    }
    cvat_categories_by_name = _index_categories_by_name(cvat_data["categories"])

    missing_categories = sorted(
        {
            category_name
            for category_name in standard_categories_by_id.values()
            if category_name not in cvat_categories_by_name
        }
    )
    if missing_categories:
        missing_categories_str = ", ".join(missing_categories)
        raise ValueError(
            "Target CVAT annotations are missing categories required by the "
            f"source annotations: {missing_categories_str}"
        )

    migrated_annotations = []
    for index, annotation in enumerate(standard_data["annotations"], start=1):
        image = standard_images_by_id.get(annotation["image_id"])
        if image is None:
            raise ValueError(
                f"Source annotation {annotation['id']} references unknown image_id "
                f"{annotation['image_id']}"
            )

        category_name = standard_categories_by_id.get(annotation["category_id"])
        if category_name is None:
            raise ValueError(
                f"Source annotation {annotation['id']} references unknown category_id "
                f"{annotation['category_id']}"
            )

        basename = Path(image["file_name"]).name
        cvat_image = cvat_images_by_basename.get(basename)
        if cvat_image is None:
            raise ValueError(
                f"Could not find matching CVAT image for source image '{basename}'"
            )

        migrated = copy.deepcopy(annotation)
        migrated["id"] = index
        migrated["image_id"] = cvat_image["id"]
        migrated["category_id"] = cvat_categories_by_name[category_name]["id"]
        migrated_annotations.append(migrated)

    migrated_data = copy.deepcopy(cvat_data)
    migrated_data["annotations"] = migrated_annotations
    return migrated_data


def main():
    args = parse_args()
    standard_data = _load_json(args.standard_annotations)
    cvat_data = _load_json(args.cvat_annotations)
    migrated = migrate_coco_annotations(
        standard_data,
        cvat_data,
        allow_existing_annotations=args.allow_existing_annotations,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(migrated, f, indent=args.indent)
        f.write("\n")


if __name__ == "__main__":
    main()
