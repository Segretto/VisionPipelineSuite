import torch
import os

def merge_weights():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    depth_checkpoint_path = os.path.join(script_dir, "../depth_dinov3/weights/model.pth")
    seg_checkpoint_path = os.path.join(script_dir, "../semantic_segmentation_dinov3/weights/model.pth")
    output_path = os.path.join(script_dir, "weights/unified_model.pth")

    print(f"Loading depth checkpoint from {depth_checkpoint_path}...")
    depth_state = torch.load(depth_checkpoint_path, map_location='cpu')

    print(f"Loading segmentation checkpoint from {seg_checkpoint_path}...")
    seg_state = torch.load(seg_checkpoint_path, map_location='cpu')

    # Create merged dictionary
    merged_state = {
        'depth_head': depth_state,
        'seg_head': seg_state
    }

    print(f"Saving merged checkpoint to {output_path}...")
    torch.save(merged_state, output_path)
    print("Done!")

if __name__ == "__main__":
    merge_weights()
