import json
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def process_images_and_annotations(image_folder, annotation_path):
    image_folder = Path(image_folder)
    annotation_path = Path(annotation_path)

    min_width, min_height = find_min_resolution(image_folder)
    target_size = (min_width, min_height)

    for image_path in image_folder.glob("*.*"):
        resize_image(image_path, target_size)

    update_annotations(annotation_path, target_size, (min_width, min_height))


def find_min_resolution(image_folder):
    min_width = float("inf")
    min_height = float("inf")
    for image_path in image_folder.glob("*.*"):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width < min_width:
                    min_width = width
                if height < min_height:
                    min_height = height
        except IOError as e:
            logger.error(f"Error opening image {image_path}: {e}")
    return min_width, min_height


def resize_image(image_path, target_size):
    try:
        with Image.open(image_path) as img:
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(image_path)
            logger.info(f"Resized image {image_path} to {target_size}")
    except IOError as e:
        logger.error(f"Error processing image {image_path}: {e}")


def update_annotations(annotation_path, target_size, original_size):
    try:
        with open(annotation_path) as f:
            data = json.load(f)

        for ann in data["annotations"]:
            image_id = ann["image_id"]
            image_info = next(img for img in data["images"] if img["id"] == image_id)
            original_width, original_height = original_size
            target_width, target_height = target_size

            ann["bbox"][0] = ann["bbox"][0] * target_width / original_width
            ann["bbox"][1] = ann["bbox"][1] * target_height / original_height
            ann["bbox"][2] = ann["bbox"][2] * target_width / original_width
            ann["bbox"][3] = ann["bbox"][3] * target_height / original_height

        with open(annotation_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Updated annotations in {annotation_path}")

    except IOError as e:
        logger.error(f"Error processing annotation file {annotation_path}: {e}")
