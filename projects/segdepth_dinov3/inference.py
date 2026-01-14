import argparse
import os
import sys
import torch
import cv2
import numpy as np
import time
import glob
from tqdm import tqdm

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import UnifiedDINOv3
from common import image_to_tensor
from utils import visualize_combined, outputs_to_maps, save_combined_visualization

# Constants from original configs
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 640

def main():
    parser = argparse.ArgumentParser(description="Unified Inference for Depth and Segmentation with DINOv3")
    parser.add_argument('--image', type=str, required=True, help='Path to input image or directory of images')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results (required if input is directory)')
    parser.add_argument('--unified_weights', type=str, default='weights/unified_model.pth', help='Path to merged checkpoint')
    parser.add_argument('--backbone_weights', type=str, required=True, help='Path to DINOv3 backbone weights (.pth)')
    parser.add_argument('--dino_dir', type=str, default='../dinov3', help='Path to local dinov3 code directory')
    parser.add_argument('--dino_model_type', type=str, default='dinov3_vits16plus', help='DINOv3 model type')
    parser.add_argument('--class_names', type=str, default='../semantic_segmentation_dinov3/src/class_names.txt', help='Path to class names file')
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Class Names
    if not os.path.exists(args.class_names):
        print(f"Error: Class names file not found at {args.class_names}")
        return
    # Determine input type
    input_path = args.image
    if os.path.isdir(input_path):
        is_directory = True
        if args.output_dir is None:
            print("Error: --output_dir is required when input is a directory.")
            return
        image_paths = sorted(glob.glob(os.path.join(input_path, '*')))
        # Filter for images
        image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"Found {len(image_paths)} images in directory.")
        if len(image_paths) == 0:
            print("No images found.")
            return
    else:
        is_directory = False
        image_paths = [input_path]

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Class Names (as fallback)
    fallback_classes = 150
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names, 'r') as f:
            class_names = [line.strip() for line in f]
        fallback_classes = len(class_names)
    
    # 2. Load DINOv3 Backbone
    # Resolve dino_dir relative to script if it's the default or relative path
    if not os.path.exists(args.dino_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, args.dino_dir)
        if os.path.exists(candidate):
            args.dino_dir = candidate

    print(f"Loading DINOv3 backbone from {args.dino_dir} with weights {args.backbone_weights}...")
    if not os.path.exists(args.dino_dir):
        print(f"Error: DINO directory not found at {args.dino_dir}")
        return
    if not os.path.exists(args.backbone_weights):
        print(f"Error: Backbone weights not found at {args.backbone_weights}")
        return

    try:
        dino_model = torch.hub.load(
            repo_or_dir=args.dino_dir,
            model=args.dino_model_type,
            source="local",
            weights=args.backbone_weights
        )
    except Exception as e:
        print(f"Failed to load DINO model: {e}")
        return

    # 3. Load Unified Weights & Detect Classes
    # Resolve unified_weights relative to script if it's the default or relative path
    if not os.path.exists(args.unified_weights):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, args.unified_weights)
        if os.path.exists(candidate):
            args.unified_weights = candidate

    print(f"Loading unified weights from {args.unified_weights}...")
    if not os.path.exists(args.unified_weights):
        print(f"Error: Unified weights not found at {args.unified_weights}")
        return
    
    checkpoint = torch.load(args.unified_weights, map_location=device)
    
    # Auto-detect num_classes from checkpoint
    num_classes = fallback_classes
    if 'seg_head' in checkpoint and 'classifier.weight' in checkpoint['seg_head']:
        detected_classes = checkpoint['seg_head']['classifier.weight'].shape[0]
        print(f"Detected {detected_classes} classes from checkpoint.")
        num_classes = detected_classes
    else:
        print(f"Using {num_classes} classes (fallback).")

    # 3. Initialize Unified Model
    embed_dim = 384
    if 'vitb' in args.dino_model_type: embed_dim = 768
    elif 'vitl' in args.dino_model_type: embed_dim = 1024
    
    # Instantiate model
    model = UnifiedDINOv3(
        dino_model=dino_model,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth_out_size=(IMG_SIZE, IMG_SIZE),
        seg_target_size=(320, 320) # Based on original config
    ).to(device)

    # Load state dicts
    if 'depth_head' in checkpoint and 'seg_head' in checkpoint:
        model.depth_head.load_state_dict(checkpoint['depth_head'])
        model.seg_head.load_state_dict(checkpoint['seg_head'])
    else:
        print("Error: Checkpoint does not contain 'depth_head' and 'seg_head' keys.")
        return

    model.eval()

    # 4. Process Images
    mean_tensor = torch.tensor(IMG_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(IMG_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    for img_path in tqdm(image_paths, desc="Processing images"):
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Failed to read image {img_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # To Tensor (optimized)
        img_tensor = torch.from_numpy(image_resized).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = (img_tensor - mean_tensor) / std_tensor

        # 6. Inference
        with torch.no_grad():
            start = time.time()
            depth_pred, seg_logits = model(img_tensor)
            end = time.time()
            print(f"Inference time for {os.path.basename(img_path)}: {(end-start)*1000:.2f} ms")

        # 7. Visualization / Saving
        # Depth
        depth_map = depth_pred.squeeze().detach().cpu().numpy()
        
        # Seg
        seg_map = outputs_to_maps(seg_logits, (IMG_SIZE, IMG_SIZE))

        if args.output_dir:
            fname = os.path.basename(img_path)
            # Change extension to png for result or keep same?
            # Let's append _result.png
            fname_no_ext = os.path.splitext(fname)[0]
            out_path = os.path.join(args.output_dir, f"{fname_no_ext}_result.png")
            save_combined_visualization(image_resized, depth_map, seg_map, out_path, class_names=class_names)
            if not is_directory:
                print(f"Saved result to {out_path}")
        else:
            print("Visualizing results...")
            visualize_combined(image_resized, depth_map, seg_map, class_names=class_names)


if __name__ == "__main__":
    main()
