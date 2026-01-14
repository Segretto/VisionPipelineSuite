import argparse
import os
import sys
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import UnifiedDINOv3
from common import image_to_tensor
from utils import depth_to_colormap

# Constants
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class Visualizer:
    """Helper to handle fast coloring of segmentation maps"""
    def __init__(self, num_classes=256):
        # Pre-generate a static color palette
        # Index 0 is background (black), others are random
        np.random.seed(42) # Consistent colors
        self.colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0] 
        self.num_classes = num_classes

    def overlay_mask(self, frame, segmentation):
        # segmentation: (H, W) int32 or int64
        # Ensure mask isn't larger than our color palette
        mask_mapped = segmentation % self.num_classes
        
        # Vectorized lookup: (H, W) -> (H, W, 3)
        colored_mask = self.colors[mask_mapped]

        # Resize to match frame if necessary (normally not needed if we resize seg to frame earlier)
        if colored_mask.shape[:2] != frame.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create binary mask for blending (where class_id > 0)
        # Assuming class 0 is background
        is_foreground = (segmentation > 0)
        
        # Fast OpenCV blending using numpy slicing where possible is faster than whole image ops if sparse,
        # but for semantic segmentation (dense), whole image ops are fine.
        # We'll use addWeighted on the whole image for simplicity and consistent speed.
        
        # However, we only want to color the foreground.
        # colored_mask has color for FG, and is black for BG (since colors[0] is black).
        
        # Make a copy of frame to blend
        # frame is uint8, colored_mask is uint8
        
        alpha = 0.6
        beta = 1.0 - alpha
        
        # Blend: output = alpha*frame + beta*colored_mask
        # But only where segmentation > 0.
        # Where segmentation == 0, we want pure frame.
        
        # A fast way:
        # 1. Convert everything to float or keep uint8 with cv2.addWeighted
        blended = cv2.addWeighted(frame, alpha, colored_mask, beta, 0.0)
        
        # 2. Selectively replace foreground pixels in the original frame with the blended pixels
        # Using numpy boolean indexing is reasonably fast.
        # We need to broadcast is_foreground to 3 channels: (H, W) -> (H, W, 3)
        fg_mask_3d = np.repeat(is_foreground[:, :, np.newaxis], 3, axis=2)
        
        # Final output buffer
        final = frame.copy()
        final[fg_mask_3d] = blended[fg_mask_3d]
        
        return final

def load_model(args, device):
    # Resolve dino_dir
    if not os.path.exists(args.dino_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, args.dino_dir)
        if os.path.exists(candidate):
            args.dino_dir = candidate

    print(f"Loading DINOv3 backbone from {args.dino_dir} with weights {args.backbone_weights}...")
    if not os.path.exists(args.dino_dir):
        raise FileNotFoundError(f"DINO directory not found at {args.dino_dir}")
    if not os.path.exists(args.backbone_weights):
        raise FileNotFoundError(f"Backbone weights not found at {args.backbone_weights}")

    dino_model = torch.hub.load(
        repo_or_dir=args.dino_dir,
        model=args.dino_model_type,
        source="local",
        weights=args.backbone_weights
    )

    # Determine config
    embed_dim = 384
    if 'vitb' in args.dino_model_type: embed_dim = 768
    elif 'vitl' in args.dino_model_type: embed_dim = 1024
    
    # Instantiate
    # Note: We can pass num_classes=256 or simply enough to cover the max ID.
    # If the checkpoint has 134 classes, we must match it.
    # We'll guess num_classes from the unified checkpoint later if possible, 
    # but for now we need it for initialization.
    # Let's peek at the class_names file or assume a default.
    # Ideally should pass class_names arg.
    
    # Load Unified Weights first to detect num_classes
    if not os.path.exists(args.unified_weights):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, args.unified_weights)
        if os.path.exists(candidate):
            args.unified_weights = candidate

    print(f"Loading unified weights from {args.unified_weights}...")
    if not os.path.exists(args.unified_weights):
        raise FileNotFoundError(f"Unified weights not found at {args.unified_weights}")
    
    checkpoint = torch.load(args.unified_weights, map_location=device)
    
    # Auto-detect num_classes from checkpoint
    if 'seg_head' in checkpoint and 'classifier.weight' in checkpoint['seg_head']:
        detected_classes = checkpoint['seg_head']['classifier.weight'].shape[0]
        print(f"Detected {detected_classes} classes from checkpoint.")
        num_classes = detected_classes
    else: 
        # Fallback to file or default
        num_classes = 150 
        if args.class_names and os.path.exists(args.class_names):
            with open(args.class_names, 'r') as f:
                lines = f.readlines()
                num_classes = len(lines)
                print(f"Read {num_classes} classes from file.")
        print(f"Using {num_classes} classes (fallback).")

    model = UnifiedDINOv3(
        dino_model=dino_model,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth_out_size=(args.image_size, args.image_size),
        seg_target_size=(args.image_size // 4, args.image_size // 4) 
    ).to(device)

    model.depth_head.load_state_dict(checkpoint['depth_head'])
    model.seg_head.load_state_dict(checkpoint['seg_head'])
    
    model.eval()
    
    if args.fp16 and device == "cuda":
        print("Enabling FP16 inference")
        model = model.half()
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Optimized Video Inference for Unified DINOv3")
    parser.add_argument('--camera_id', type=str, default="0", help='Camera ID (int) or Video Path (str)')
    parser.add_argument('--unified_weights', type=str, default='weights/unified_model.pth', help='Path to merged checkpoint')
    parser.add_argument('--backbone_weights', type=str, required=True, help='Path to DINOv3 backbone weights')
    parser.add_argument('--dino_dir', type=str, default='../dinov3', help='Path to local dinov3 code')
    parser.add_argument('--dino_model_type', type=str, default='dinov3_vits16plus', help='DINOv3 model type')
    parser.add_argument('--class_names', type=str, default='../semantic_segmentation_dinov3/src/class_names.txt', help='Path to class names')
    parser.add_argument('--image_size', type=int, default=640, help='Input inference size')
    parser.add_argument('--fp16', action='store_true', help='Use Half Precision (FP16)')
    
    args = parser.parse_args()

    # Handle camera_id (str to int if digit)
    if args.camera_id.isdigit():
        camera_source = int(args.camera_id)
    else:
        camera_source = args.camera_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        model = load_model(args, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    vis = Visualizer(num_classes=300) # Ensure enough colors

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting Stream. Press 'q' to exit.")
    
    # Pre-allocate mean/std on device for speed
    # Shape: (1, 3, 1, 1)
    mean_tensor = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(IMG_STD, device=device).view(1, 3, 1, 1)
    if args.fp16 and device == "cuda":
        mean_tensor = mean_tensor.half()
        std_tensor = std_tensor.half()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Preprocess
            # Resize strict
            input_frame = cv2.resize(frame, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
            
            # To tensor
            # Normalize on GPU is often faster if we just upload uint8 first
            # But let's stick to simple logic first: standard convert
            # img: (H, W, 3) BGR
            img_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).to(device) # (H, W, 3) uint8
            
            # Permute to (1, 3, H, W)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Convert to float/half and normalize
            if args.fp16 and device == "cuda":
                img_tensor = img_tensor.half()
            else:
                img_tensor = img_tensor.float()
                
            img_tensor = img_tensor / 255.0
            img_tensor = (img_tensor - mean_tensor) / std_tensor

            # 2. Inference
            with torch.inference_mode():
                depth_pred, seg_logits = model(img_tensor)
                
                # 3. Post-Process
                # Depth
                # depth_pred is (1, 1, H, W)
                depth_map = depth_pred.squeeze().float().cpu().numpy() # Transfer back to CPU for vis
                
                # Seg
                # seg_logits: (1, C, H', W')
                # Argmax
                seg_idx = torch.argmax(seg_logits, dim=1).squeeze() # (H', W')
                
                # Resize seg mask to input frame size for overlay
                # Using nearest neighbor via torch interpolate before CPU transfer might be cleaner
                if seg_idx.shape != (args.image_size, args.image_size):
                    # We need the batch dim and channel dim for interpolate to work on "mask"
                    # But argmax kills channel.
                    # Actually, better to interpolate logits then argmax? Or just interpolate the mask (nearest).
                    # Interpolating logits is more expensive (C channels).
                    # Interpolating mask:
                     seg_idx = F.interpolate(seg_idx.view(1, 1, seg_idx.shape[0], seg_idx.shape[1]).float(), 
                                             size=(args.image_size, args.image_size), 
                                             mode='nearest').squeeze().long()
                
                seg_map = seg_idx.cpu().numpy().astype(np.int32)

            # 4. Visualize
            # Depth colormap
            depth_vis = depth_to_colormap(depth_map, bgr=True)
            
            # Segmentation overlay on original input_frame
            seg_vis = vis.overlay_mask(input_frame, seg_map)
            
            # Concatenate side-by-side
            combined = np.hstack((seg_vis, depth_vis))
            
            cv2.imshow("Unified DINOv3 (Left: Seg, Right: Depth)", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
