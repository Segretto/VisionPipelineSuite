# VisionPipelineSuite

A workbench of computer vision tools for data preprocessing, training, and inference.

## Structure

The repository has been restructured into a Python package `vision_suite` located in `src/`.

- **`src/vision_suite/`**: Core library code.
  - **`data/`**: Data preparation tools (COCO to YOLO, splitting, resizing, filtering).
  - **`visualization/`**: Visualization tools (bounding boxes, segmentation masks).
  - **`inference/`**: Inference pipelines (Jetson/DeepStream).
  - **`core/`**: Shared utilities.

- **`scripts/`**: CLI entry points.
  - **`prepare_data.py`**: Unified script for data prep.
    - `python scripts/prepare_data.py convert ...`
    - `python scripts/prepare_data.py split ...`
    - `python scripts/prepare_data.py resize ...`
    - `python scripts/prepare_data.py filter ...`
  - **`visualize.py`**: Visualization script.
  - **`run_inference.py`**: Inference runner.

- **`legacy/`**: Original scripts and folders.

## Installation

To install the package in editable mode:

```bash
pip install -e .
```

## Usage

### Data Preparation

```bash
# Convert COCO to YOLO
python scripts/prepare_data.py convert /path/to/dataset --mode detection

# Split Dataset
python scripts/prepare_data.py split /path/to/images /path/to/coco.json /path/to/output
```

### Inference

```bash
python scripts/run_inference.py /dev/video0
```
