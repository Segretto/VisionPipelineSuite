#!/usr/bin/env python3
import argparse
import sys
import os

# Add src to path so we can import vision_suite
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_suite.utils.coco import validate_coco_json

def main():
    parser = argparse.ArgumentParser(description="Validate a COCO JSON file.")
    parser.add_argument("input", help="Path to the COCO JSON file to validate.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    print(f"Validating {args.input}...")
    results = validate_coco_json(args.input)

    if results["valid"]:
        print("✅ Validation PASSED")
    else:
        print("❌ Validation FAILED")

    if results["errors"]:
        print("\nErrors:")
        for err in results["errors"]:
            print(f"  - {err}")

    if results["warnings"]:
        print("\nWarnings:")
        for warn in results["warnings"]:
            print(f"  - {warn}")

    if results["stats"]:
        print("\nStatistics:")
        for k, v in results["stats"].items():
            print(f"  - {k}: {v}")

    if not results["valid"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
