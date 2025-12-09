#!/usr/bin/env python3
"""
Run DINOv3 Segmentation Inference on USB Camera using ONNX

Usage:
    python scripts/inference_usb_camera.py --model dinov3_seg_coco.onnx --camera 0
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import time

def get_args():
    parser = argparse.ArgumentParser(description="DINOv3 ONNX Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha")
    return parser.parse_args()

def preprocess(frame, mean, std):
    # Resize to a multiple of 14 (patch size) if needed, or just use raw size if model supports it
    # DINOv3 ViT usually works best with patch size multiples.
    # But let's just resize to 512x512 for consistency with training, or keep aspect ratio.
    # For simplicity, let's resize to 512x512 for inference input, then resize mask back.
    
    input_size = (512, 512)
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    # Normalize
    img = (img - mean) / std
    
    # CHW
    img = img.transpose(2, 0, 1)
    
    # Batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def main():
    args = get_args()
    
    # Load ONNX model
    print(f"Loading model from {args.model}...")
    # Use CUDA if available, else CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(args.model, providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # ImageNet mean/std (0-255)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    # Open Camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting inference. Press 'q' to quit.")
    
    # Random colors for segmentation masks
    np.random.seed(42)
    colors = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Preprocess
        input_tensor = preprocess(frame, mean, std)
        
        # Inference
        outputs = session.run([output_name], {input_name: input_tensor})
        logits = outputs[0] # (B, C, H, W)
        
        # Postprocess
        # Logits are likely 512x512 (or whatever input size was)
        # We need to resize back to original frame size
        
        # Argmax to get class indices
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
        
        # Resize mask to frame size
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Colorize
        colored_mask = colors[mask_resized]
        
        # Overlay
        overlay = cv2.addWeighted(frame, 1 - args.alpha, colored_mask, args.alpha, 0)
        
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("DINOv3 Segmentation", overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
