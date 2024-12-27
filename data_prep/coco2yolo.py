import json
import argparse
import logging
import cv2
import numpy as np

from pathlib import Path
from pycocotools import mask as maskUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(dataset_path, dataset_splits, pose_estimation, mode):
    dataset_root = Path(dataset_path)
    
    for split in dataset_splits:
        label_path = dataset_root / "labels" / split / 'coco.json'
        if not label_path.exists():
            logger.warning(f"File not found: {label_path}")
            continue

        with open(label_path) as f:
            data = json.load(f)
        
        image_info = {img['id']: img for img in data['images']}
        
        annotations = process_annotations(image_info, data, pose_estimation, mode)

        create_annotation_files(annotations, label_path.parent)

    create_yaml_file(dataset_root, dataset_splits[0], pose_estimation)

def convert_bounding_boxes(size, box, category_id):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return f"{category_id} {x} {y} {w} {h}"

def convert_pose_keypoints(size, box, keypoints, category_id):
    yolo_bbox = convert_bounding_boxes(size, box, category_id)
    dw = 1. / size[0]
    dh = 1. / size[1]
    yolo_keypoints = [(keypoints[i] * dw, keypoints[i + 1] * dh, keypoints[i + 2]) for i in range(0, len(keypoints), 3)]
    keypoints_str = ' '.join([f"{kp[0]} {kp[1]} {kp[2]}" for kp in yolo_keypoints])
    return f"{yolo_bbox} {keypoints_str}"


def convert_segmentation_masks(size, segmentation_mask, category_id):

    # Checks if mask is in RLE format
    if isinstance(segmentation_mask, dict) and 'counts' in segmentation_mask:
       
        # Decode RLE to a binary mask, then convert to polygons
        rle = maskUtils.frPyObjects(segmentation_mask,
                segmentation_mask['size'][0],
                segmentation_mask['size'][1])
        
        binary_mask = maskUtils.decode(rle)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        norm_coords = contours[0].flatten()
        norm_coords = norm_coords.astype(np.float32)

        norm_coords[0::2] = np.round(norm_coords[0::2] / size[0], 5)
        norm_coords[1::2] = np.round(norm_coords[1::2] / size[1], 5)

    # Otherwise, assume polygon format (alternative cases)
    else:
        norm_coords = np.array(segmentation_mask).astype(np.float32)
        norm_coords[0::2] = np.round(norm_coords[0::2] / size[0], 5)
        norm_coords[1::2] = np.round(norm_coords[1::2] / size[1], 5)

    return f"{category_id} {' '.join(map(str, norm_coords))}"

def convert_coco_to_kitti(size, box, category_name):
    x1, y1 = box[0], box[1]
    x2, y2 = box[0] + box[2], box[1] + box[3]
    return f"{category_name} 0 0 0 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0"

def decode_rle(rle, height, width):
    """
    Decode RLE format to a binary mask.
    """
    # Initialize an empty binary mask
    mask = np.zeros(height * width, dtype=np.uint8)

    # Decode the RLE to create the mask
    idx = 0
    for i, length in enumerate(rle):
        if i % 2 == 0:
            idx += length
        else:
            mask[idx:idx + length] = 1
            idx += length

    return mask.reshape((height, width))

def process_annotations(image_info, data, is_pose_estimation=False, mode="detection"):
    annotations_by_image = {}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        coco_bbox = ann['bbox']
        category_id = ann['category_id'] - 1
        category_name = next((cat['name'] for cat in data['categories'] if cat['id'] == ann['category_id']), "unknown")
        img_filename = Path(image_info[img_id]['file_name'])
        img_size = (image_info[img_id]['width'], image_info[img_id]['height'])

        if img_filename not in annotations_by_image:
            annotations_by_image[img_filename] = []

        match mode:
            case "detection":
                if is_pose_estimation and 'keypoints' in ann:
                    annotation_line = convert_pose_keypoints(img_size, coco_bbox, ann['keypoints'], category_id)
                else:
                    annotation_line = convert_bounding_boxes(img_size, coco_bbox, category_id)        

            case "segmentation":
                annotation_line = convert_segmentation_masks(img_size, ann["segmentation"], category_id)

            case "od_kitti":
                annotation_line = convert_coco_to_kitti(img_size, coco_bbox, category_name)
        
        annotations_by_image[img_filename].append(annotation_line)
    
    return annotations_by_image

def create_annotation_files(annotations_by_image, output_dir):
    for img_filename, annotations in annotations_by_image.items():
        
        txt_path = output_dir / (img_filename.stem + '.txt')
        try:
            with open(txt_path, 'w') as file:
                file.write("\n".join(annotations) + "\n")
            logger.debug(f"Processed annotation for image: {img_filename}")
        except IOError as e:
            logger.error(f"Error writing to file {txt_path}: {e}")

def create_yaml_file(dataset_path, data_split, is_pose_estimation=False):
    label_path = dataset_path / "labels" / data_split / 'coco.json'
    if not label_path.exists():
        logger.error(f"File not found: {label_path}")
        return

    with open(label_path) as f:
        data = json.load(f)
    class_names = {category['id'] - 1: category['name'] for category in data['categories']}
    sorted_class_names = sorted(class_names.items())
    class_entries = "\n".join([f"  {id}: {name}" for id, name in sorted_class_names])

    # TODO: Make the contents of the data.yml file to be read from a template
    yaml_content = f"""path: {dataset_path.absolute()}  # dataset root dir
train: images/{data_split}  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
names:
{class_entries}
    """

    if is_pose_estimation:
        categories = data['categories']
        keypoints_info = categories[0].get('keypoints', [])
        kpt_shape = [len(keypoints_info), 3]

        yaml_content += f"\n\n# Keypoints\nkpt_shape: {kpt_shape}"

    yaml_path = dataset_path / 'data.yaml'
    try:
        with open(yaml_path, 'w') as file:
            file.write(yaml_content.strip())
        logger.info(f"YAML file created at {yaml_path}")
    except IOError as e:
        logger.error(f"Error writing to file {yaml_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process COCO annotations and create YOLO or KITTI dataset.")
    parser.add_argument("dataset_path", help="Path to the root directory of the dataset.")
    parser.add_argument("--pose_estimation", action='store_true', help="Flag to indicate if the dataset is for pose estimation")
    parser.add_argument("--dataset_splits", nargs='+', default=['train', 'val', 'test'], help="Custom dataset split for ablation studies")
    parser.add_argument("--mode", choices=["detection", "segmentation", "od_kitti"], default="detection",
                        help="Choose processing mode: 'detection' for bounding boxes, 'segmentation' for segmentation masks.")
    # parser.add_argument("--output_format", choices=['yolo', 'kitti'], default='yolo', help="Output format for annotations")
   
    args = parser.parse_args()
   
    main(**vars(args))

