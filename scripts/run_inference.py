#!/usr/bin/env python3
import argparse
import logging
from vision_suite.inference import jetson_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Vision Pipeline Suite - Inference Tools"
    )
    parser.add_argument(
        "device_path",
        nargs="?",
        default="/dev/video0",
        help="Path to video device (default: /dev/video0)",
    )

    args = parser.parse_args()

    try:
        jetson_detector.run_pipeline(args.device_path)
    except Exception as e:
        logger.error(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
