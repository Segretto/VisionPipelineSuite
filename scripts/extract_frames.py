#!/usr/bin/env python3
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_suite.utils.video import extract_frames

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file or a directory of videos.")
    parser.add_argument("input_path", help="Path to the input video file or directory.")
    parser.add_argument("output_dir", help="Directory to save extracted frames.")
    parser.add_argument("--fps", type=float, help="Frame rate to sample at (default: all frames).")
    args = parser.parse_args()

    # Common video extensions
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'}

    input_path = args.input_path
    output_dir = args.output_dir

    videos_to_process = []

    if os.path.isdir(input_path):
        print(f"📂 Scanning directory {input_path} for videos...")
        for root, _, files in os.walk(input_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in VIDEO_EXTENSIONS:
                    videos_to_process.append(os.path.join(root, file))
    elif os.path.isfile(input_path):
        videos_to_process.append(input_path)
    else:
        print(f"❌ Error: Input path not found: {input_path}")
        sys.exit(1)

    if not videos_to_process:
        print("❌ No video files found.")
        sys.exit(1)

    print(f"Found {len(videos_to_process)} videos to process.")

    total_extracted = 0
    errors = 0

    for video_path in videos_to_process:
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Use output_dir directly, but add video name as prefix to avoid collisions
            prefix = f"{video_name}_"
            
            print(f"🎬 Processing {os.path.basename(video_path)} -> {output_dir} (prefix: {prefix}) ...")
            count = extract_frames(video_path, output_dir, args.fps, prefix=prefix)
            total_extracted += count
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            errors += 1

    print(f"\n✅ Batch processing complete.")
    print(f"Total frames extracted: {total_extracted}")
    if errors > 0:
        print(f"Errors encountered: {errors}")

if __name__ == "__main__":
    main()
