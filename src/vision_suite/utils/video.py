import cv2
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frames(video_path: str, output_dir: str, fps: Optional[float] = None, prefix: str = "") -> int:
    """
    Extracts frames from a video file.
    
    Args:
        video_path: Path to the input video.
        output_dir: Directory to save extracted frames.
        fps: Optional frame rate to sample at. If None, extracts all frames.
        prefix: Optional prefix for the saved frame filenames.
        
    Returns:
        Number of frames extracted.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1
    if fps:
        frame_interval = max(1, int(round(video_fps / fps)))
        
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            frame_name = f"{prefix}{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        count += 1
        
    cap.release()
    logger.info(f"Extracted {saved_count} frames from {video_path} to {output_dir}")
    return saved_count
