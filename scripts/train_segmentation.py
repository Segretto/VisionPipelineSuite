#!/usr/bin/env python3
"""
Train DINOv3 for Segmentation (Linear Head)

This script is a wrapper around the DINOv3 segmentation training module.
It currently supports training a Linear Head on top of a frozen DINOv3 backbone.

Usage:
    python scripts/train_segmentation.py \
        config=dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
        datasets.root=/path/to/dataset \
        output_dir=/path/to/output

Note:
    - Full Mask2Former training is not currently supported by the DINOv3 repository's training loop.
    - This script defaults to Linear Probing.
"""

import sys
import os

# Add the dinov3 directory to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
dinov3_path = os.path.join(os.path.dirname(current_dir), "dinov3")
if dinov3_path not in sys.path:
    sys.path.append(dinov3_path)

from dinov3.eval.segmentation.run import main as dinov3_segmentation_main
import dinov3.data.loaders as loaders
from vision_suite.data.coco_dataset import CocoSegmentation

# Monkey-patch _parse_dataset_str to support CocoSegmentation
original_parse_dataset_str = loaders._parse_dataset_str

def custom_parse_dataset_str(dataset_str: str):
    if dataset_str.startswith("CocoSegmentation"):
        tokens = dataset_str.split(":")
        kwargs = {}
        for token in tokens[1:]:
            key, value = token.split("=")
            if key == "split":
                kwargs["split"] = CocoSegmentation.Split[value]
            elif key == "root" and not value and "root" in kwargs:
                # Don't overwrite existing root with empty string (appended by train.py)
                continue
            else:
                kwargs[key] = value
        return CocoSegmentation, kwargs
    return original_parse_dataset_str(dataset_str)

loaders._parse_dataset_str = custom_parse_dataset_str

if __name__ == "__main__":
    print("🚀 Starting DINOv3 Segmentation Training (Linear Head)...")
    print("⚠️ Note: Mask2Former training is not fully supported in this version. Using Linear Head.")
    print("ℹ️  Custom COCO Dataset support enabled via 'CocoSegmentation' prefix.")
    # Filter out empty arguments that might cause OmegaConf to fail
    sys.argv = [arg for arg in sys.argv if arg.strip()]
    
    print(f"DEBUG: sys.argv after filtering: {sys.argv}")
    sys.exit(dinov3_segmentation_main())
