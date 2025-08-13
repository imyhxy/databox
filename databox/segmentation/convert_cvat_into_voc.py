# Author: fkwong
# File: convert_cvat_into_voc.py
# Date: 8/13/25
import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import yaml
from datumaro import (
    AnnotationType,
    CliPlugin,
    Image,
    ItemTransform,
    Mask,
    TQDMProgressReporter,
)
from datumaro.components.dataset import Dataset


class AddEmptyMask(ItemTransform, CliPlugin):
    _allowed_types = {AnnotationType.mask}

    def transform_item(self, item):
        mask_annos = [
            anno for anno in item.annotations if anno.type in self._allowed_types
        ]

        if len(mask_annos) == 0:
            if not isinstance(item.media, Image):
                raise Exception("Image info is required for this transform")
            h, w = item.media.size
            mask_annos.append(Mask(image=np.zeros((h, w), dtype=np.uint8)))

        return self.wrap_item(item, annotations=item.annotations + mask_annos)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pipeline",
        type=str,
        choices=("cvat_polygon_to_voc_mask",),
        help="Pipeline to be performed",
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to input dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output dataset"
    )

    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument(
        "--param-name",
        type=str,
        default=None,
        help="Name of the parameter to use from params.yaml",
    )
    exclusive_group.add_argument(
        "--seed", default=None, type=int, help="Random seed for splitting"
    )

    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    return parser.parse_args()


def cvat_polygon_to_voc_mask(
    input_dir: str, output_dir: str, train, val, seed=None
) -> None:
    if Path(output_dir).exists():
        logging.info(f"Remove existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    ds = Dataset.import_from(input_dir, "cvat")
    ds.transform("polygons_to_masks")
    ds.transform(AddEmptyMask)
    ds.transform(
        "split", task="segmentation", splits=[("train", train), ("val", val)], seed=seed
    )
    ds.export(
        output_dir,
        format="voc_segmentation",
        apply_colormap=0,
        save_media=True,
        progress_reporter=TQDMProgressReporter(),
    )


def main():
    args = parse_args()

    if args.param_name is not None:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)[args.param_name]
            seed = params["seed"]
            train = params["train"]
            assert 0 < train < 1, "Train split must be between 0 and 1"
            val = 1 - train
    else:
        seed = args.seed
        train = args.train
        assert 0 < train < 1, "Train split must be between 0 and 1"
        val = 1 - train

    if args.pipeline == "cvat_polygon_to_voc_mask":
        cvat_polygon_to_voc_mask(args.input_dir, args.output_dir, train, val, seed)
    else:
        raise AssertionError


if __name__ == "__main__":
    main()
