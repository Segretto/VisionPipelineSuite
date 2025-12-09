#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_suite.data.converters import yolo2coco

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO JSON format.")
    parser.add_argument("--images_dir", required=True, help="Directory containing images.")
    parser.add_argument("--labels_dir", required=True, help="Directory containing YOLO txt label files.")
    parser.add_argument("--yaml_path", required=True, help="Path to the YOLO YAML file defining classes.")
    parser.add_argument("--output_path", help="Path to save the output COCO JSON file. Defaults to 'coco.json' in the parent of images_dir.")
    parser.add_argument("--mode", choices=["detection", "segmentation"], default="detection", help="Conversion mode: 'detection' or 'segmentation'.")
    parser.add_argument("--rle", action="store_true", help="Convert segmentation masks to RLE format (only for segmentation mode).")

    args = parser.parse_args()

    # Determine output path if not provided
    if not args.output_path:
        images_path = Path(args.images_dir)
        # Try to put it in the parent directory of images_dir, typically dataset root
        args.output_path = str(images_path.parent / "coco.json")
        print(f"Output path not provided, defaulting to: {args.output_path}")

    # Call the conversion function
    print(f"Starting conversion from YOLO to COCO...")
    print(f"Images: {args.images_dir}")
    print(f"Labels: {args.labels_dir}")
    print(f"Mode: {args.mode}")
    
    yolo2coco(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        yaml_path=args.yaml_path,
        output_json_path=args.output_path,
        mode=args.mode,
        rle_option=args.rle
    )
    
    print("Conversion completed.")

if __name__ == "__main__":
    main()
