import json
import logging
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as maskUtils
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def coco2yolo(dataset_path, mode, custom_yaml_data_path=None):
    """Process COCO annotations and generate YOLO or KITTI dataset files.

    Args:
        dataset_path (str): Path to the root directory of the dataset.
        mode (str): Processing mode ('detection', 'segmentation', 'od_kitti', or 'pose_detection').
        custom_yaml_data_path (str, optional): Custom path for the YAML data file.
    """
    dataset_root = Path(dataset_path)

    labels_dir = dataset_root / "labels"
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return

    splits = [f.name for f in labels_dir.iterdir() if f.is_dir()]

    for split in splits:
        coco_path = labels_dir / split / "coco.json"

        if not coco_path.exists():
            logger.error(f"File not found: {coco_path}")
            continue

        with open(coco_path) as f:
            data = json.load(f)

        image_info = {img["id"]: img for img in data["images"]}

        annotations = process_annotations_parallel(image_info, data, mode)

        create_annotation_files(annotations, coco_path.parent)

        if split not in ["val", "test"]:
            create_yaml_file(dataset_root, custom_yaml_data_path, data, mode, split)


def convert_bounding_boxes(size, box, category_id):
    """Convert COCO bounding box format to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return f"{category_id} {x} {y} {w} {h}"


def convert_pose_keypoints(size, box, keypoints, category_id):
    """Convert COCO pose keypoints to YOLO format."""
    yolo_bbox = convert_bounding_boxes(size, box, category_id)
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    yolo_keypoints = [
        (keypoints[i] * dw, keypoints[i + 1] * dh, keypoints[i + 2])
        for i in range(0, len(keypoints), 3)
    ]
    keypoints_str = " ".join([f"{kp[0]} {kp[1]} {kp[2]}" for kp in yolo_keypoints])
    return f"{yolo_bbox} {keypoints_str}"


def convert_segmentation_masks_direct(
    size, segmentation_mask, category_id, min_pixels=20
):
    """Convert COCO segmentation masks (RLE or polygon) to YOLO format string, optimized for speed."""
    width, height = size
    annotation_line = f"{category_id}"

    if (
        isinstance(segmentation_mask, dict) and "counts" in segmentation_mask
    ):  # RLE mask
        # Decode RLE into binary mask
        rle = maskUtils.frPyObjects(
            segmentation_mask,
            segmentation_mask["size"][0],
            segmentation_mask["size"][1],
        )
        binary_mask = maskUtils.decode(rle)
        binary_mask = filter_small_regions(binary_mask, 50)

        # Find all non-zero pixels in the mask
        rows, cols = np.nonzero(binary_mask)  # Faster than np.argwhere

        norm_coords = np.vstack((cols / width, rows / height)).T.flatten()
        annotation_line += " " + " ".join(map(str, norm_coords))

    elif isinstance(segmentation_mask, list):  # Polygon format
        for polygon in segmentation_mask:
            poly_array = np.array(polygon).reshape(-1, 2)  # Reshape into a 2D array
            if len(poly_array) <= min_pixels:  # Filter small polygons
                continue

            # Normalize polygon coordinates directly
            norm_coords = []
            for x, y in poly_array:
                norm_coords.append(round(x / width, 5))  # Normalize x
                norm_coords.append(round(y / height, 5))  # Normalize y

            # Add normalized coordinates to the annotation line
            annotation_line += " " + " ".join(map(str, norm_coords))

    return annotation_line


def convert_coco_to_kitti(size, box, category_name):
    """Convert COCO bounding box format to KITTI format."""
    x1, y1 = box[0], box[1]
    x2, y2 = box[0] + box[2], box[1] + box[3]
    return f"{category_name} 0 0 0 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0"


def process_annotations_parallel(image_info, data, mode, n_jobs=-1):
    """Process COCO annotations into the desired format based on mode, with parallelization."""
    # Process all annotations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_annotation)(ann, image_info, data, mode)
        for ann in data["annotations"]
    )

    # Group annotations by image filename
    annotations_by_image = {}
    for img_filename, annotation_line in results:
        if img_filename not in annotations_by_image:
            annotations_by_image[img_filename] = []
        annotations_by_image[img_filename].append(annotation_line)

    return annotations_by_image


def process_single_annotation(ann, image_info, data, mode):
    """Process a single COCO annotation based on mode."""
    img_id = ann["image_id"]
    coco_bbox = ann["bbox"]
    category_id = ann["category_id"] - 1
    category_name = next(
        (cat["name"] for cat in data["categories"] if cat["id"] == ann["category_id"]),
        "unknown",
    )
    img_filename = Path(image_info[img_id]["file_name"])
    img_size = (image_info[img_id]["width"], image_info[img_id]["height"])

    match mode:
        case "detection" | "pose_detection":
            if mode.startswith("pose") and "keypoints" in ann:
                annotation_line = convert_pose_keypoints(
                    img_size, coco_bbox, ann["keypoints"], category_id
                )
            else:
                annotation_line = convert_bounding_boxes(
                    img_size, coco_bbox, category_id
                )

        case "segmentation":
            annotation_line = convert_segmentation_masks_direct(
                img_size, ann["segmentation"], category_id
            )

        case "od_kitti":
            annotation_line = convert_coco_to_kitti(img_size, coco_bbox, category_name)

    return img_filename, annotation_line


def filter_small_regions(binary_mask, min_pixels):
    """Remove small blob regions from a binary mask."""
    # Perform connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Create an empty mask to store the filtered result
    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Iterate over each region, skipping the background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]  # Get the area of the region
        if area > min_pixels:  # Retain regions larger than the threshold
            filtered_mask[labels == i] = 1

    return filtered_mask


def create_annotation_files(annotations_by_image, output_dir):
    """Write annotation files for each image."""
    for img_filename, annotations in annotations_by_image.items():
        txt_path = output_dir / (img_filename.stem + ".txt")
        try:
            with open(txt_path, "w") as file:
                file.write("\n".join(annotations) + "\n")
            logger.debug(f"Processed annotation for image: {img_filename}")
        except IOError as e:
            logger.error(f"Error writing to file {txt_path}: {e}")


def create_yaml_file(dataset_path, custom_yaml_data_path, data, mode, split=None):
    """Generate a YAML configuration file for the dataset."""

    yaml_path = dataset_path / (split + ".yaml")
    train_path = "images/" + split
    val_path = "images/val"

    class_names = {
        category["id"] - 1: category["name"] for category in data["categories"]
    }
    sorted_class_names = sorted(class_names.items())
    class_entries = "\n".join([f"  {id}: {name}" for id, name in sorted_class_names])

    if custom_yaml_data_path:
        dataset_path = Path(custom_yaml_data_path)

    yaml_content = f"""path: {dataset_path.absolute()}  # dataset root dir
train: {train_path}  # train images (relative to 'path')
val: {val_path}  # val images (relative to 'path')
test:  # test images (optional)

# Classes
names:
{class_entries}
    """

    if mode.startswith("pose"):
        categories = data["categories"]
        keypoints_info = categories[0].get("keypoints", [])
        kpt_shape = [len(keypoints_info), 3]

        yaml_content += f"\n\n# Keypoints\nkpt_shape: {kpt_shape}"

    try:
        with open(yaml_path, "w") as file:
            file.write(yaml_content.strip())
        logger.info(f"YAML file created at {yaml_path}")
    except IOError as e:
        logger.error(f"Error writing to file {yaml_path}: {e}")
