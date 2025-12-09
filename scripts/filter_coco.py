#!/usr/bin/env python3
import json
import argparse
import logging
from vision_suite.utils.coco import filter_coco_duplicates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Filter duplicate annotations from a COCO dataset.")
    parser.add_argument("input_json", help="Path to the input COCO JSON file.")
    parser.add_argument("output_json", help="Path to save the filtered COCO JSON file.")
    parser.add_argument("--iou_threshold", type=float, default=0.95, help="IoU threshold for duplicate detection (default: 0.95).")
    
    args = parser.parse_args()
    
    try:
        with open(args.input_json, 'r') as f:
            data = json.load(f)
            
        filtered_data = filter_coco_duplicates(data, iou_threshold=args.iou_threshold)
        
        with open(args.output_json, 'w') as f:
            json.dump(filtered_data, f, indent=2)
            
        logger.info(f"Filtered dataset saved to {args.output_json}")
        
    except Exception as e:
        logger.error(f"Failed to process dataset: {e}")
        exit(1)

if __name__ == "__main__":
    main()
