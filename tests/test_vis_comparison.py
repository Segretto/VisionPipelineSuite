import json
import os
import numpy as np
import subprocess
import sys
from PIL import Image

def create_dummy_data():
    os.makedirs("tests/data", exist_ok=True)
    
    # Create dummy image using PIL
    img = Image.new('RGB', (100, 100), color = 'black')
    img.save("tests/data/test_img.jpg")
    
    # GT Data
    gt_data = {
        "images": [{"id": 1, "file_name": "test_img.jpg", "height": 100, "width": 100}],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            # GT 1: Cat at 10,10,20,20 (Will be TP if matched with Pred Cat ID 10 and Image 99)
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            # GT 2: Dog at 50,50,20,20 (Will be FN - missed)
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0}
        ]
    }
    
    # Pred Data
    # Note: Using different IDs AND different file extensions!
    # Image ID 99 -> "test_img.jpeg" (Matches GT "test_img.jpg" by stem)
    # Pred Cat ID 10 -> "cat" (Matches GT ID 1)
    pred_data = {
        "images": [{"id": 99, "file_name": "test_img.jpeg", "height": 100, "width": 100}],
        "categories": [{"id": 10, "name": "cat"}, {"id": 20, "name": "dog"}],
        "annotations": [
            # Pred 1: Cat (ID 10) - Exact match (TP)
            {"image_id": 99, "category_id": 10, "bbox": [12, 12, 20, 20], "score": 0.9},
            
            # Pred 2: Dog (ID 20) - Mismatched IoU but High IoP?
            # GT Dog is at 50,50,20,20. Area=400.
            # Let's make a Pred that is mostly INSIDE the GT, but IoU is low?
            # Or IoP >= 0.9 logic: (Inter / PredArea) >= 0.9.
            # Make Pred small and inside GT.
            # Pred at 55,55,10,10. Area=100. Inter=100. IoP = 100/100 = 1.0.
            # IoU = 100 / (400 + 100 - 100) = 100/400 = 0.25 (Low IoU).
            # This should be accepted as TP by "Partial" logic.
            {"image_id": 99, "category_id": 20, "bbox": [55, 55, 10, 10], "score": 0.8},
            
            # Pred 3: False Dog at 80,80 (FP)
            {"image_id": 99, "category_id": 20, "bbox": [80, 80, 10, 10], "score": 0.7}
        ]
    }
    
    with open("tests/data/gt.json", "w") as f:
        json.dump(gt_data, f)
        
    with open("tests/data/preds.json", "w") as f:
        json.dump(pred_data, f)

def run_test():
    create_dummy_data()
    # Create Duplicate predictions case
    pred_data_dup = {
        "images": [{"id": 100, "file_name": "test_img.jpeg", "height": 100, "width": 100}],
        "categories": [{"id": 10, "name": "cat"}],
        "annotations": [
            # Two duplicate Cats matching the single GT Cat (at 10,10,20,20)
            {"image_id": 100, "category_id": 10, "bbox": [10, 10, 20, 20], "score": 0.95},
            {"image_id": 100, "category_id": 10, "bbox": [10, 10, 20, 20], "score": 0.90}
        ]
    }
    with open("tests/data/preds_dup.json", "w") as f:
        json.dump(pred_data_dup, f)

    print("Running visualize_coco.py with --match_n 2...")
    subprocess.run([
        "python3", "scripts/visualize_coco.py",
        "tests/data/gt.json",
        "--image_dir", "tests/data",
        "--compare_json", "tests/data/preds_dup.json",
        "--output_dir", "tests/output",
        "--iou_threshold", "0.5",
        "--match_n", "2"
    ], check=True)
    
    if os.path.exists("tests/output/vis_test_img.jpg"):
        print("Output for match_n verified.")
    print("Running visualize_coco.py with comparison plotting...")
    result = subprocess.run([
        "python3", "scripts/visualize_coco.py",
        "tests/data/gt.json",
        "--image_dir", "tests/data",
        "--compare_json", "tests/data/preds.json",
        "--output_dir", "tests/output",
        "--iou_threshold", "0.5",
        "--plot_comparison"
    ], capture_output=True, text=True, check=False) # check=False because we want to print stdout/stderr even on error
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print("Test FAILED")
        sys.exit(1)
    if os.path.exists("tests/output/vis_test_img.jpg"):
        print("Output image generated successfully.")
    else:
        print("Output image NOT found.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
