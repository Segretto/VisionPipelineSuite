#!/usr/bin/env python3
import argparse
import sys
import os
import json
import glob
from typing import List, Dict, Any
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import SAM3
try:
    import sam3
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False

def run_sam3_inference(
    image_dir: str,
    prompts: List[str],
    output_path: str,
    checkpoint: str,
    model_type: str = "sam3_h" 
) -> None:
    
    if not SAM3_AVAILABLE:
        print("❌ Error: 'sam3' module not found. Please install it first.")
        print("  git clone https://github.com/facebookresearch/sam3")
        print("  pip install -e sam3")
        sys.exit(1)

    print(f"Loading SAM3 model from {checkpoint}...")
    try:
        # Using API from official README
        # Assuming build_sam3_image_model accepts checkpoint path. 
        # If not, it might need to be loaded differently or it loads default.
        # We will try passing it as an argument.
        model = build_sam3_image_model(checkpoint=checkpoint, model_type=model_type)
        processor = Sam3Processor(model)
    except TypeError:
        # Fallback if arguments are different
        print("⚠️ Warning: build_sam3_image_model signature mismatch. Trying without arguments...")
        try:
            model = build_sam3_image_model()
            # If checkpoint was provided but not used, we might need to load it manually
            if checkpoint:
                print(f"Loading checkpoint {checkpoint} manually...")
                checkpoint_state = torch.load(checkpoint, map_location="cpu")
                model.load_state_dict(checkpoint_state, strict=False) # strict=False to be safe
            processor = Sam3Processor(model)
        except Exception as e:
             print(f"❌ Failed to load model: {e}")
             sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_paths:
        print(f"❌ No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Processing with {len(prompts)} prompts...")

    # Prepare COCO structure
    coco_output = {
        "info": {"description": "SAM3 Auto-Annotation"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories from prompts
    prompt_to_cat_id = {}
    for idx, prompt in enumerate(prompts):
        cat_id = idx + 1
        coco_output["categories"].append({
            "id": cat_id,
            "name": prompt,
            "supercategory": "object"
        })
        prompt_to_cat_id[prompt] = cat_id

    ann_id = 1
    
    for img_idx, img_path in enumerate(tqdm(image_paths)):
        image_name = os.path.basename(img_path)
        
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not read {img_path}: {e}")
            continue
            
        width, height = pil_image.size
        image_id = img_idx + 1
        
        coco_output["images"].append({
            "id": image_id,
            "file_name": image_name,
            "height": height,
            "width": width
        })

        # Set image for processor
        try:
            inference_state = processor.set_image(pil_image)
        except Exception as e:
             print(f"Error setting image {image_name}: {e}")
             continue

        # Run inference for each prompt
        for prompt in prompts:
            try:
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                
                # Output contains masks, boxes, scores
                masks = output["masks"] # Likely tensor [N, H, W]
                scores = output["scores"]
                
                # Convert to numpy
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                
                for mask_idx, mask in enumerate(masks):
                    # Mask is likely boolean or 0/1
                    mask_uint8 = (mask > 0).astype(np.uint8)
                    
                    # Find contours for bbox
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                        
                    # Get largest contour or all? Let's take all valid ones
                    for contour in contours:
                        if len(contour) < 3:
                            continue
                            
                        x, y, w, h = cv2.boundingRect(contour)
                        area = cv2.contourArea(contour)
                        
                        # Polygon format
                        polygon = contour.flatten().tolist()
                        
                        coco_output["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": prompt_to_cat_id[prompt],
                            "bbox": [x, y, w, h],
                            "area": area,
                            "iscrowd": 0,
                            "segmentation": [polygon],
                            "score": float(scores[mask_idx]) if mask_idx < len(scores) else 1.0
                        })
                        ann_id += 1
                    
            except Exception as e:
                print(f"Error processing {image_name} with prompt '{prompt}': {e}")

    # Save output
    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
        
    print(f"✅ Saved annotations to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run SAM3 on images using text prompts.")
    parser.add_argument("--images", required=True, help="Directory containing images.")
    parser.add_argument("--prompts", nargs='+', help="List of text prompts (e.g. 'dog' 'cat').")
    parser.add_argument("--prompts_file", help="Path to a text file containing prompts (one per line).")
    parser.add_argument("--output", required=True, help="Path to save COCO JSON output.")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM3 model checkpoint.")
    parser.add_argument("--model-type", default="sam3_h", help="Model type (default: sam3_h).")
    
    args = parser.parse_args()
    
    prompts = []
    if args.prompts:
        prompts.extend(args.prompts)
        
    if args.prompts_file:
        if not os.path.exists(args.prompts_file):
            print(f"❌ Error: Prompts file not found: {args.prompts_file}")
            sys.exit(1)
        with open(args.prompts_file, 'r') as f:
            file_prompts = [line.strip() for line in f if line.strip()]
            prompts.extend(file_prompts)
            
    if not prompts:
        print("❌ Error: No prompts provided. Use --prompts or --prompts_file.")
        sys.exit(1)
        
    # Remove duplicates while preserving order
    prompts = list(dict.fromkeys(prompts))
    
    run_sam3_inference(
        args.images,
        prompts,
        args.output,
        args.checkpoint,
        args.model_type
    )

if __name__ == "__main__":
    main()
