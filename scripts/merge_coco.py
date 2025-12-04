#!/usr/bin/env python3
import argparse
import sys
import os

# Add src to path so we can import vision_suite
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_suite.utils.coco import merge_coco_jsons

def main():
    parser = argparse.ArgumentParser(description="Merge multiple COCO JSON files (intersection of images).")
    parser.add_argument("--inputs", nargs='+', required=True, help="Paths to input COCO JSON files.")
    parser.add_argument("--output", required=True, help="Path to output merged JSON file.")
    args = parser.parse_args()

    for p in args.inputs:
        if not os.path.exists(p):
            print(f"Error: Input file not found: {p}")
            sys.exit(1)

    print(f"Merging {len(args.inputs)} files into {args.output}...")
    try:
        merge_coco_jsons(args.inputs, args.output)
        print("✅ Merge completed successfully.")
    except Exception as e:
        print(f"❌ Merge failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
