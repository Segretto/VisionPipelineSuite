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


def yolo2coco(
    images_dir,
    labels_dir,
    yaml_path,
    output_json_path,
    mode="detection",
    rle_option=False,
):
    """Convert YOLO annotations to COCO JSON format.

    Args:
        images_dir (str): Path to the directory containing images.
        labels_dir (str): Path to the directory containing YOLO txt labels.
        yaml_path (str): Path to the YOLO YAML file defining classes.
        output_json_path (str): Path to save the generated COCO JSON.
        mode (str): 'detection' or 'segmentation'.
        rle_option (bool): If True, convert segmentation polygons to RLE.
    """
    import yaml
    import os

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_json_path)

    # 1. Load Classes from YAML
    try:
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        names = yaml_data.get("names", {})
        # Ensure names is a dict id->name. If it's a list, convert it.
        if isinstance(names, list):
            categories = [{"id": i, "name": n} for i, n in enumerate(names)]
        else:
            categories = [{"id": int(k), "name": v} for k, v in names.items()]
    except Exception as e:
        logger.error(f"Error loading YAML file: {e}")
        return

    # 2. Initialize COCO JSON structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    # 3. Iterate over images
    annotation_id = 1
    image_id = 1

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = [
        f for f in images_path.iterdir() if f.suffix.lower() in valid_extensions
    ]
    image_files.sort()

    for img_file in image_files:
        # Image info
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Could not read image: {img_file}")
                continue
            height, width = img.shape[:2]
        except Exception as e:
            logger.error(f"Error reading image {img_file}: {e}")
            continue

        # Relative path for portability
        # We want the path relative to the JSON file's directory
        try:
            file_name = os.path.relpath(str(img_file), str(output_path.parent))
        except ValueError:
            # Fallback if paths are on different drives or something weird
            file_name = img_file.name

        image_info = {
            "id": image_id,
            "file_name": file_name,
            "height": height,
            "width": width,
        }
        coco_data["images"].append(image_info)

        # Corresponding Label File
        label_file = labels_path / (img_file.stem + ".txt")
        if label_file.exists():
            with open(label_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                # Check for Confidence
                # Detection: class x y w h [conf] -> 5 or 6 items
                # Segmentation: class x1 y1 ... [conf] -> odd number if conf exists (2N + 1 + 1 = even) vs (2N + 1 = odd)?
                # Actually:
                # Detection: 5 args (class, x, y, w, h) -> if 6, last is conf
                # Segmentation: 2N args (coords) + 1 (class). Total 2N+1. If 2N+2, last is conf.
                
                confidence = None
                coords = [float(x) for x in parts[1:]]

                if mode == "detection":
                    if len(coords) == 5:
                        confidence = coords[-1]
                        coords = coords[:-1]
                    
                    x_c, y_c, w_n, h_n = coords
                    # Convert to Top-Left x, y, w, h (absolute)
                    w = w_n * width
                    h = h_n * height
                    x = (x_c * width) - (w / 2)
                    y = (y_c * height) - (h / 2)
                    bbox = [x, y, w, h]
                    segmentation = []
                    area = w * h

                elif mode == "segmentation":
                    # Check if last element is confidence
                    # Standard segmentation line: class x1 y1 x2 y2 ...
                    # Number of coords should be even (x, y pairs).
                    if len(coords) % 2 != 0:
                        # Odd number of coords means we have an extra value, likely confidence
                         confidence = coords[-1]
                         coords = coords[:-1]

                    # Denormalize
                    poly_coords = []
                    for i in range(0, len(coords), 2):
                        px = coords[i] * width
                        py = coords[i+1] * height
                        poly_coords.extend([px, py])

                    # Create BBox from Polygon
                    x_coords = poly_coords[0::2]
                    y_coords = poly_coords[1::2]
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    bound_w = max(x_coords) - x_min
                    bound_h = max(y_coords) - y_min
                    bbox = [x_min, y_min, bound_w, bound_h]

                    if rle_option:
                         rles = maskUtils.frPyObjects([poly_coords], height, width)
                         rle = maskUtils.merge(rles)
                         rle["counts"] = rle["counts"].decode("utf-8")
                         segmentation = rle
                         area = float(maskUtils.area(rle))
                    else:
                        segmentation = [poly_coords]
                         # Calculate Polygon Area (Shoelace formula approx or similar, but simplified here)
                         # Use pycocotools for accurate area
                        rles = maskUtils.frPyObjects([poly_coords], height, width)
                        area = float(maskUtils.area(rles))

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id, # YOLO 0-indexed matches or mapped via YAML? YAML is source of truth.
                    # Usually COCO uses 1-based but if we define categories from 0 in YAML, we keep it consistent.
                    # Standard COCO often starts at 1, but we'll stick to what the YAML says.
                    "bbox": bbox,
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
                if confidence is not None:
                    annotation["score"] = confidence

                coco_data["annotations"].append(annotation)
                annotation_id += 1

        image_id += 1

    # 4. Save JSON
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coco_data, f, indent=4)
        logger.info(f"COCO JSON converted to: {output_path}")
    except IOError as e:
        logger.error(f"Error writing to file {output_path}: {e}")
