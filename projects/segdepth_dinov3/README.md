# Unified DINOv3 Depth and Segmentation

This folder contains scripts to merge and run a unified model with a shared DINOv3 backbone and two heads (Depth and Segmentation).

## Structure
- `backbone.py`: Wrapper for DINOv3 backbone.
- `depth_head.py`: Depth estimation head definition.
- `seg_head.py`: Semantic segmentation head definition.
- `model.py`: Unified model class (`UnifiedDINOv3`).
- `merge_weights.py`: Script to merge separate checkpoints into one.
- `inference.py`: Script to run inference on an image.

## Usage

### 1. Merge Weights
Run this once to create `weights/unified_model.pth`.
```bash
python merge_weights.py
```

### 2. Run Inference
```bash
python inference.py \
  --image /path/to/image.jpg \
  --backbone_weights /path/to/dinov3_backbone.pth \
  --unified_weights weights/unified_model.pth
```

**Arguments:**
- `--image`: Path to input image (required).
- `--backbone_weights`: Path to the pre-trained DINOv3 backbone weights (required).
- `--unified_weights`: Path to the merged checkpoint (default: `weights/unified_model.pth`).
- `--dino_dir`: Path to the dinov3 code directory (default: `../dinov3`).
- `--class_names`: Path to class names text file (default: `../semantic_segmentation_dinov3/src/class_names.txt`).
