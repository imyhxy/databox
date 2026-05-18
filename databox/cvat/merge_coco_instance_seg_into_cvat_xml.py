import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Append COCO instance segmentation polygons into a CVAT XML export. "
            "Existing CVAT shapes are preserved."
        )
    )
    parser.add_argument("coco_annotations", type=str, help="Source COCO JSON file")
    parser.add_argument("cvat_annotations", type=str, help="Target CVAT XML file")
    parser.add_argument("output", type=str, help="Path to write the merged CVAT XML")
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip COCO annotations whose image basename is absent from the CVAT XML",
    )
    parser.add_argument(
        "--skip-missing-labels",
        action="store_true",
        help="Skip COCO annotations whose category name is absent from the CVAT XML",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="manual",
        help="CVAT source attribute for appended polygons",
    )
    parser.add_argument(
        "--z-order",
        type=str,
        default="0",
        help="CVAT z_order attribute for appended polygons",
    )
    return parser.parse_args()


def _load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def _index_by_basename(items, name_key, dataset_name):
    indexed = {}
    duplicates = set()
    for item in items:
        name = item.get(name_key) if hasattr(item, "get") else item[name_key]
        basename = Path(name).name
        if basename in indexed:
            duplicates.add(basename)
            continue
        indexed[basename] = item

    if duplicates:
        duplicates_str = ", ".join(sorted(duplicates))
        raise ValueError(
            f"{dataset_name} contains duplicate basenames: {duplicates_str}"
        )

    return indexed


def _format_number(value):
    return f"{float(value):.2f}"


def _format_points(points):
    if len(points) < 6 or len(points) % 2 != 0:
        raise ValueError(
            "COCO polygon segmentation must contain an even number of at least 6 values"
        )

    return ";".join(
        f"{_format_number(points[index])},{_format_number(points[index + 1])}"
        for index in range(0, len(points), 2)
    )


def _next_group_id(root):
    group_ids = []
    for image in root.findall("./image"):
        for shape in image:
            group_id = shape.get("group_id")
            if group_id is not None:
                group_ids.append(int(group_id))

    return max(group_ids, default=0) + 1


def _label_names(root):
    return {
        label_name.text
        for label_name in root.findall("./meta/task/labels/label/name")
        if label_name.text
    }


def _append_polygon(image_element, label, points, group_id, source, z_order):
    polygon = ET.SubElement(
        image_element,
        "polygon",
        {
            "label": label,
            "source": source,
            "occluded": "0",
            "points": points,
            "z_order": str(z_order),
            "group_id": str(group_id),
        },
    )
    polygon.text = "\n    "
    polygon.tail = "\n  "
    return polygon


def merge_coco_instance_segmentation_into_cvat_xml(
    coco_data,
    cvat_root,
    skip_missing_images=False,
    skip_missing_labels=False,
    source="manual",
    z_order="0",
):
    coco_images_by_id = {image["id"]: image for image in coco_data["images"]}
    coco_categories_by_id = {
        category["id"]: category["name"] for category in coco_data["categories"]
    }
    cvat_images_by_basename = _index_by_basename(
        cvat_root.findall("./image"), "name", "CVAT XML"
    )
    cvat_labels = _label_names(cvat_root)
    next_group_id = _next_group_id(cvat_root)
    stats = {
        "annotations": 0,
        "polygons": 0,
        "skipped_images": 0,
        "skipped_labels": 0,
    }

    for annotation in coco_data["annotations"]:
        image = coco_images_by_id.get(annotation["image_id"])
        if image is None:
            raise ValueError(
                f"COCO annotation {annotation['id']} references unknown image_id "
                f"{annotation['image_id']}"
            )

        category_name = coco_categories_by_id.get(annotation["category_id"])
        if category_name is None:
            raise ValueError(
                f"COCO annotation {annotation['id']} references unknown category_id "
                f"{annotation['category_id']}"
            )

        if category_name not in cvat_labels:
            if skip_missing_labels:
                stats["skipped_labels"] += 1
                continue
            raise ValueError(
                f"CVAT XML is missing label required by COCO: {category_name}"
            )

        basename = Path(image["file_name"]).name
        image_element = cvat_images_by_basename.get(basename)
        if image_element is None:
            if skip_missing_images:
                stats["skipped_images"] += 1
                continue
            raise ValueError(
                f"Could not find matching CVAT image for COCO image: {basename}"
            )

        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError(
                f"COCO annotation {annotation['id']} has unsupported RLE segmentation"
            )

        polygon_count = 0
        group_id = next_group_id
        for polygon_points in segmentation:
            points = _format_points(polygon_points)
            _append_polygon(
                image_element, category_name, points, group_id, source, z_order
            )
            polygon_count += 1

        if polygon_count:
            stats["annotations"] += 1
            stats["polygons"] += polygon_count
            next_group_id += 1

    return stats


def _write_xml(tree, output):
    root = tree.getroot()
    ET.indent(tree, space="  ")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_text = ET.tostring(
        root,
        encoding="unicode",
        short_empty_elements=False,
        xml_declaration=True,
    )
    with output_path.open("w") as f:
        f.write(xml_text.replace("'", '"'))
        f.write("\n")


def main():
    args = parse_args()
    coco_data = _load_json(args.coco_annotations)
    cvat_tree = ET.parse(args.cvat_annotations)
    stats = merge_coco_instance_segmentation_into_cvat_xml(
        coco_data,
        cvat_tree.getroot(),
        skip_missing_images=args.skip_missing_images,
        skip_missing_labels=args.skip_missing_labels,
        source=args.source,
        z_order=args.z_order,
    )
    _write_xml(cvat_tree, args.output)
    print(
        "Merged "
        f"{stats['annotations']} COCO annotations as {stats['polygons']} CVAT polygons "
        f"({stats['skipped_images']} skipped missing images, "
        f"{stats['skipped_labels']} skipped missing labels)."
    )


if __name__ == "__main__":
    main()
