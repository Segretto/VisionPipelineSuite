import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_coco_to_yolo(size, box, category_id):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return f"{category_id} {x} {y} {w} {h}"

def convert_keypoints_to_yolo(size, box, keypoints, category_id):
    yolo_bbox = convert_coco_to_yolo(size, box, category_id)
    dw = 1. / size[0]
    dh = 1. / size[1]
    yolo_keypoints = [(keypoints[i] * dw, keypoints[i + 1] * dh, keypoints[i + 2]) for i in range(0, len(keypoints), 3)]
    keypoints_str = ' '.join([f"{kp[0]} {kp[1]} {kp[2]}" for kp in yolo_keypoints])
    return f"{yolo_bbox} {keypoints_str}"

def convert_coco_to_kitti(size, box, category_name):
    x1, y1 = box[0], box[1]
    x2, y2 = box[0] + box[2], box[1] + box[3]
    return f"{category_name} 0 0 0 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0"

def process_annotations(image_info, data, is_pose_estimation=False, output_format="yolo"):
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

        if output_format == "yolo":
            if is_pose_estimation and 'keypoints' in ann:
                annotation_line = convert_keypoints_to_yolo(img_size, coco_bbox, ann['keypoints'], category_id)
            else:
                annotation_line = convert_coco_to_yolo(img_size, coco_bbox, category_id)
        elif output_format == "kitti":
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

def create_yaml_file(dataset_path, is_pose_estimation=False):
    label_path = dataset_path / "labels" / 'train' / 'coco.json'
    if not label_path.exists():
        logger.error(f"File not found: {label_path}")
        return

    with open(label_path) as f:
        data = json.load(f)
    class_names = {category['id'] - 1: category['name'] for category in data['categories']}
    sorted_class_names = sorted(class_names.items())
    class_entries = "\n".join([f"  {id}: {name}" for id, name in sorted_class_names])

    yaml_content = f"""path: {dataset_path.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
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

def main():
    parser = argparse.ArgumentParser(description="Process COCO annotations and create YOLO or KITTI dataset.")
    parser.add_argument("dataset_path", help="Path to the root directory of the dataset.")
    parser.add_argument("--pose_estimation", action='store_true', help="Flag to indicate if the dataset is for pose estimation")
    parser.add_argument("--output_format", choices=['yolo', 'kitti'], default='yolo', help="Output format for annotations")
    parser.add_argument("--custom_dataset_split", nargs='+', default=['train', 'val', 'test'], help="Custom dataset split for ablation studies")
    
    args = parser.parse_args()
    dataset_root = Path(args.dataset_path)
    
    for split in args.custom_dataset_split:
        label_path = dataset_root / "labels" / split / 'coco.json'
        if not label_path.exists():
            logger.warning(f"File not found: {label_path}")
            continue

        with open(label_path) as f:
            data = json.load(f)
        
        image_info = {img['id']: img for img in data['images']}
        
        annotations = process_annotations(image_info, data, args.pose_estimation, args.output_format)
        create_annotation_files(annotations, label_path.parent)

    
    if args.output_format == "yolo":
        create_yaml_file(dataset_root, args.pose_estimation)

if __name__ == "__main__":
    main()
