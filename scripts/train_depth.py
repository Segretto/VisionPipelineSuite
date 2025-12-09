#!/usr/bin/env python3
"""
Train DINOv3 for Depth Estimation (DPT Head)

This script is a wrapper around the DINOv3 depth training module.
It supports training a Depther (DPT) head on top of a frozen DINOv3 backbone.

Usage:
    python scripts/train_depth.py \
        config=dinov3/eval/depth/configs/config-nyu.yaml \
        datasets.root=/path/to/dataset \
        output_dir=/path/to/output
"""

import sys
import os

# Add the dinov3 directory to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
dinov3_path = os.path.join(os.path.dirname(current_dir), "dinov3")
if dinov3_path not in sys.path:
    sys.path.append(dinov3_path)

from dinov3.eval.depth.run import main as dinov3_depth_main

if __name__ == "__main__":
    print("🚀 Starting DINOv3 Depth Estimation Training...")
    sys.exit(dinov3_depth_main())
