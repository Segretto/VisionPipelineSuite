#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from vision_suite.data import converters, splitting, processing, filtering

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def convert_command(args):
    converters.coco2yolo(args.dataset_path, args.mode, args.custom_yaml_data_path)


def split_command(args):
    splitting.split_data(
        images_dir=args.images_dir,
        coco_json_path=args.coco_json_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        ablation=args.ablation,
        k=args.k,
        rename_images=args.rename_images,
        classes=args.classes,
    )


def resize_command(args):
    processing.process_images_and_annotations(args.image_folder, args.annotation_path)


def filter_command(args):
    filtering.filter_coco_annotations(
        input_json_path=Path(args.input_json),
        output_json_path=Path(args.output_json),
        classes_to_keep=args.classes_to_keep,
    )


def deduplicate_command(args):
    duplicates = filtering.scan_for_duplicates(args.folder_path)
    if duplicates:
        logger.info(f"Found {len(duplicates)} sets of duplicates. Processing...")
        filtering.delete_duplicates(duplicates, args.coco_file)
        logger.info("Duplicates removed and COCO JSON updated.")
    else:
        logger.info("No duplicate images found.")


def main():
    parser = argparse.ArgumentParser(
        description="Vision Pipeline Suite - Data Preparation Tools"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert Subcommand
    parser_convert = subparsers.add_parser(
        "convert", help="Convert COCO annotations to YOLO/KITTI"
    )
    parser_convert.add_argument(
        "dataset_path", help="Path to the root directory of the dataset"
    )
    parser_convert.add_argument(
        "--mode",
        choices=["detection", "segmentation", "od_kitti", "pose_detection"],
        default="detection",
        help="Processing mode",
    )
    parser_convert.add_argument(
        "--custom_yaml_data_path", help="Custom data path for YAML file"
    )
    parser_convert.set_defaults(func=convert_command)

    # Split Subcommand
    parser_split = subparsers.add_parser("split", help="Split COCO dataset")
    parser_split.add_argument("images_dir", help="Path to input images directory")
    parser_split.add_argument("coco_json_path", help="Path to COCO JSON file")
    parser_split.add_argument("output_dir", help="Path to output directory")
    parser_split.add_argument(
        "--k", type=int, default=0, help="Number of folds for k-fold CV"
    )
    parser_split.add_argument(
        "--train_ratio", type=float, default=0.75, help="Training ratio"
    )
    parser_split.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation ratio"
    )
    parser_split.add_argument("--ablation", type=int, default=0, help="Ablation chunks")
    parser_split.add_argument(
        "--rename_images", action="store_true", default=True, help="Rename images"
    )
    parser_split.add_argument(
        "--classes", nargs="+", default=[], help="Filter by classes"
    )
    parser_split.set_defaults(func=split_command)

    # Resize Subcommand
    parser_resize = subparsers.add_parser(
        "resize", help="Resize images and update annotations"
    )
    parser_resize.add_argument("image_folder", help="Path to images folder")
    parser_resize.add_argument("annotation_path", help="Path to COCO annotation file")
    parser_resize.set_defaults(func=resize_command)

    # Filter Subcommand
    parser_filter = subparsers.add_parser(
        "filter", help="Filter COCO annotations by class"
    )
    parser_filter.add_argument("--input-json", required=True, help="Input COCO JSON")
    parser_filter.add_argument("--output-json", required=True, help="Output COCO JSON")
    parser_filter.add_argument(
        "--classes-to-keep", nargs="+", required=True, help="Classes to keep"
    )
    parser_filter.set_defaults(func=filter_command)

    # Deduplicate Subcommand
    parser_dedup = subparsers.add_parser("deduplicate", help="Remove duplicate images")
    parser_dedup.add_argument("folder_path", help="Path to images folder")
    parser_dedup.add_argument("coco_file", help="Path to COCO JSON file")
    parser_dedup.set_defaults(func=deduplicate_command)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
