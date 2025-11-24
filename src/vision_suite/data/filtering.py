import json
import logging
import hashlib
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def filter_coco_annotations(
    input_json_path: Path, output_json_path: Path, classes_to_keep: list
):
    """
    Reads a COCO annotations JSON from `input_json_path`, filters out annotations
    to keep only the specified `classes_to_keep`, reassigns category and annotation
    IDs sequentially, and saves the filtered dataset to `output_json_path`.
    """

    logger.info(f"Reading input annotations from: {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build a mapping of category_name -> category_info
    name_to_cat = {cat["name"]: cat for cat in coco_data["categories"]}

    # Determine which category IDs we should keep
    keep_category_ids = []
    for cls in classes_to_keep:
        if cls not in name_to_cat:
            logger.warning(f"Class '{cls}' not found in 'categories'. Skipping.")
        else:
            keep_category_ids.append(name_to_cat[cls]["id"])

    keep_category_ids = set(keep_category_ids)
    logger.debug(f"Category IDs to keep: {keep_category_ids}")

    # Filter annotations
    original_anns_count = len(coco_data["annotations"])
    filtered_annotations = [
        ann
        for ann in coco_data["annotations"]
        if ann["category_id"] in keep_category_ids
    ]
    logger.info(
        f"Number of original annotations: {original_anns_count}. "
        f"Number after filtering: {len(filtered_annotations)}"
    )

    # Filter categories based on keep_category_ids
    filtered_categories = [
        cat for cat in coco_data["categories"] if cat["id"] in keep_category_ids
    ]

    # Filter images to keep only those referenced by the filtered annotations
    keep_image_ids = {ann["image_id"] for ann in filtered_annotations}
    filtered_images = [
        img for img in coco_data["images"] if img["id"] in keep_image_ids
    ]
    logger.info(
        f"Number of original images: {len(coco_data['images'])}. "
        f"Number after filtering: {len(filtered_images)}"
    )

    # --- Reassign Category IDs Sequentially ---
    cat_id_map = {}
    for i, cat in enumerate(filtered_categories, start=1):
        old_id = cat["id"]
        cat_id_map[old_id] = i

    for cat in filtered_categories:
        old_id = cat["id"]
        cat["id"] = cat_id_map[old_id]

    # --- Reassign Annotation IDs and Category IDs ---
    for i, ann in enumerate(filtered_annotations, start=1):
        ann["id"] = i
        old_cat_id = ann["category_id"]
        ann["category_id"] = cat_id_map[old_cat_id]

    # Build the new COCO dataset
    filtered_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories,
    }

    logger.info(f"Writing filtered annotations to: {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_coco, f, ensure_ascii=False, indent=2)


def hash_image(file_path):
    """Generate a hash for an image file to check for duplicates."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def scan_for_duplicates(folder_path):
    """Scan a folder for image duplicates and return a dictionary mapping unique IDs to lists of duplicate filenames."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder path {folder_path} does not exist.")

    image_hashes = defaultdict(list)

    for file_path in folder_path.rglob("*"):
        if file_path.is_file():
            try:
                file_hash = hash_image(file_path)
                image_hashes[file_hash].append(file_path)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

    duplicate_dict = {}
    for idx, (hash_key, file_paths) in enumerate(image_hashes.items()):
        if len(file_paths) > 1:
            unique_id = f"{idx:04d}"
            duplicate_dict[unique_id] = file_paths

    return duplicate_dict


def get_annotation_count(coco_data, image_id):
    """Get the number of annotations for a given image ID in the COCO JSON data."""
    return sum(1 for ann in coco_data["annotations"] if ann["image_id"] == image_id)


def _update_coco_json(coco_data, files_to_keep):
    """Update the COCO JSON file by removing references to deleted images and reassigning image IDs."""
    image_info = {img["file_name"]: img for img in coco_data["images"]}
    updated_images = []
    updated_annotations = []

    new_image_id_map = {}
    for new_id, file_name in enumerate(sorted(files_to_keep), start=1):
        if file_name in image_info:
            image = image_info[file_name]
            new_image_id_map[image["id"]] = new_id
            image["id"] = new_id
            updated_images.append(image)

    for annotation in coco_data["annotations"]:
        if annotation["image_id"] in new_image_id_map:
            annotation["image_id"] = new_image_id_map[annotation["image_id"]]
            updated_annotations.append(annotation)

    coco_data["images"] = updated_images
    coco_data["annotations"] = updated_annotations

    return coco_data


def delete_duplicates(duplicates, coco_file):
    """Delete duplicate images and update the COCO JSON file."""

    files_to_keep = set()

    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    # Add images without duplicates to used_file_names
    duplicate_files = {path.name for paths in duplicates.values() for path in paths}
    for image in coco_data["images"]:
        if image["file_name"] not in duplicate_files:
            files_to_keep.add(image["file_name"])

    # Find images to keep from duplicates
    for unique_id, file_paths in duplicates.items():
        if len(file_paths) < 2:
            continue

        # We need to find the image ID corresponding to the file path to count annotations
        # This part is a bit tricky because we need to map file path to image ID from coco_data
        # Assuming file_name in coco_data matches path.name

        # Helper to find image ID by filename
        def find_id_by_name(name):
            for img in coco_data["images"]:
                if img["file_name"] == name:
                    return img["id"]
            return None

        annotations_count = []
        for path in file_paths:
            img_id = find_id_by_name(path.name)
            count = get_annotation_count(coco_data, img_id) if img_id is not None else 0
            annotations_count.append((path, count))

        annotations_count.sort(key=lambda x: x[1], reverse=True)
        file_to_keep = annotations_count[0][0]
        files_to_keep.add(file_to_keep.name)

        for file_path, _ in annotations_count[1:]:
            try:
                file_path.unlink()
                logger.info(f"Deleted duplicate: {file_path}")
            except OSError as e:
                logger.error(f"Error deleting {file_path}: {e}")

    new_coco_data = _update_coco_json(coco_data, files_to_keep)

    with open(coco_file, "w") as f:
        json.dump(new_coco_data, f, indent=4)
