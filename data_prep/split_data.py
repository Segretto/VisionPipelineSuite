import json
import shutil
import random
import argparse
import logging
import logging.config

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(images_dir, coco_json_path, output_dir, train_ratio, val_ratio, ablation, pose_estimation):
    if ablation < 0:
        logger.error("Ablation must be a positive integer.")
        return
    
    elif ablation > 0:
        logger.info(f"Ablation mode enabled with {ablation} dataset chunks.")
   
    split_data(images_dir, coco_json_path, output_dir, train_ratio, val_ratio, ablation, pose_estimation, ablation)

def copy_images(images, src_dir, dest_dir, rename_images=False):
    """
    Copy selected images to a specified directory, and optionally rename them with new numerical IDs.
    Update the images' filenames in the dataset metadata if renamed.
    """
    total_images = len(images)
    id_format = f"{{:0{len(str(total_images + 2))}d}}"  # Adds two orders of magnitude to total_images for new IDs

    updated_images = []  # To keep track of updated image information

    for idx, image in enumerate(images, start=1):
        try:
            # Ensure image file extension is '.jpeg'
            image_name = Path(image['file_name'])
            if image_name.suffix in ['.jpg', '.jpeg']:
                new_file_name = image_name.stem + '.jpeg'
            else:
                new_file_name = image_name.name

            # If rename_images is True, rename files using numerical IDs
            if rename_images:
                new_file_name = id_format.format(idx) + '.jpeg'

            # Update the image metadata to reflect the new filename
            updated_image = image.copy()
            updated_image['file_name'] = new_file_name
            updated_images.append(updated_image)

            src_path = Path(src_dir) / image['file_name']
            dest_path = Path(dest_dir) / new_file_name
            shutil.copy(src_path, dest_path)
            logger.info(f"Successfully copied {src_path} to {dest_path}")
        
        except Exception as e:
            logger.error(f"Failed to copy {src_path} to {dest_path}: {e}")

    return updated_images

def filter_annotations(images_set, annotations, is_pose_estimation):
    image_ids = {image['id'] for image in images_set}
    if is_pose_estimation:
        return [annotation for annotation in annotations if annotation['image_id'] in image_ids and 'keypoints' in annotation]
    else:
        return [annotation for annotation in annotations if annotation['image_id'] in image_ids]
    
def filter_coco_by_classes(coco_data, classes):
    """
    Filters the COCO data to only include specified classes.
    """
    if classes == "all":
        return coco_data  # No filtering needed

    # Build mapping from category names to IDs
    category_name_to_id = {category['name']: category['id'] for category in coco_data['categories']}
    # Get category IDs for specified class names
    selected_category_ids = [category_name_to_id[name] for name in classes if name in category_name_to_id]

    if not selected_category_ids:
        logger.error(f"No matching categories found for classes: {classes}")
        raise ValueError(f"No matching categories found for classes: {classes}")

    # Filter annotations
    filtered_annotations = [annotation for annotation in coco_data['annotations'] if annotation['category_id'] in selected_category_ids]

    # Get image IDs corresponding to filtered annotations
    image_ids = {annotation['image_id'] for annotation in filtered_annotations}

    # Filter images
    filtered_images = [image for image in coco_data['images'] if image['id'] in image_ids]

    # Filter categories
    filtered_categories = [category for category in coco_data['categories'] if category['id'] in selected_category_ids]

    # Build the new coco_data
    filtered_coco_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }

    return filtered_coco_data

def create_coco_subset(images, annotations, categories):
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

def log_object_count_per_class(coco_data):
    """
    Logs the total number of objects for each class in the COCO dataset.
    """
    category_counts = {category['name']: 0 for category in coco_data['categories']}
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), None)
        if category_name:
            category_counts[category_name] += 1

    logger.info("Object counts per class:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count}")
    return category_counts

def split_for_ablation(images, annotations, categories, images_dir, 
                       output_dir, ablation, is_pose_estimation, 
                       rename_images):
    
    
    ablation_chunks = [int(len(images) * (i + 1) / ablation) for i in range(ablation)]
    logger.info(f"Ablation chunk sizes (number of images): {ablation_chunks}")
    
    for _, chunk_size in enumerate(ablation_chunks, start=1):
        chunk_images = images[:chunk_size]

        # Create folder for this ablation chunk
        ablation_output_path = output_dir / f"{int((chunk_size / len(images)) * 100)}_percent"
        ablation_output_path.mkdir(parents=True, exist_ok=True)

        images_output_path = ablation_output_path / "images"
        images_output_path.mkdir(parents=True, exist_ok=True)

        labels_output_path = ablation_output_path / "labels"
        labels_output_path.mkdir(parents=True, exist_ok=True)

        # Copy images for this chunk
        updated_chunk_images = copy_images(chunk_images, images_dir, images_output_path, rename_images=rename_images)

        # Filter annotations for this chunk
        filtered_annotations = filter_annotations(updated_chunk_images, annotations, is_pose_estimation)
        coco_chunk = create_coco_subset(chunk_images, filtered_annotations, categories)

        # Save COCO JSON for this chunk
        with open(labels_output_path / "coco.json", 'w') as file:
            json.dump(coco_chunk, file, indent=4)
        
        # Log object counts for this chunk
        chunk_category_counts = log_object_count_per_class(coco_chunk)

        # Save metadata for this chunk
        meta_file_path = ablation_output_path / "meta.txt"
        with open(meta_file_path, 'w') as meta_file:
            meta_file.write("Class-wise Object Counts:\n")
            for category, count in chunk_category_counts.items():
                meta_file.write(f"{category}: {count}\n")
        
        logger.info(f"Ablation dataset for {int((chunk_size / len(images)) * 100)}% created successfully.")

def split_train_test_val(images, annotations, categories, images_dir, 
                         output_dir, train_ratio, val_ratio, 
                         is_pose_estimation, rename_images):

    # Validate ratios
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio <= 1):
        logger.error("Invalid training/validation ratios.")
        return

    random.shuffle(images)
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for type, images_set in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        try:
            images_output_path = output_dir / "images" / type
            images_output_path.mkdir(parents=True, exist_ok=True)

            labels_output_path = output_dir / "labels" / type
            labels_output_path.mkdir(parents=True, exist_ok=True)

            updated_images_set = copy_images(images_set, images_dir, images_output_path, rename_images=rename_images)

            filtered_annotations = filter_annotations(updated_images_set, annotations, is_pose_estimation)
            coco_file = create_coco_subset(images_set, filtered_annotations, categories)
            with open(labels_output_path / "coco.json", 'w') as file:
                json.dump(coco_file, file, indent=4)
            logger.info(f"Dataset for {type} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to process data for {type}: {e}")

def split_data(images_dir, labels_json_path, output_dir, 
                  train_ratio=0.75, val_ratio=0.1, ablation=0, is_pose_estimation=False, 
                  rename_images=False, classes="all"):
    """
    Splits a COCO dataset into training, validation, and testing sets based on given ratios.
    """

    logger.info("Loading COCO annotations...")
    with open(labels_json_path, 'r') as f:
        coco_data = json.load(f)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    if not images_dir.exists():
        logger.error(f"Images directory does not exist: {images_dir}")
        return

    coco_data = filter_coco_by_classes(coco_data, classes)

    # Extract image and annotation details
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])

    # Log the count of annotated images and objects
    logger.info(f"Total annotated images: {len(images)}")
    logger.info(f"Total annotated objects: {len(annotations)}")

    # Log the total amount of images in the images folder
    total_images_in_folder = len(list(images_dir.glob('*')))
    logger.info(f"Total images in the folder {images_dir}: {total_images_in_folder}")

    if ablation > 0:
        split_for_ablation(images, annotations, categories, images_dir, 
                         output_dir, train_ratio, val_ratio, 
                         is_pose_estimation, rename_images)

    else:
        split_train_test_val(images, annotations, categories, images_dir, 
                         output_dir, train_ratio, val_ratio, 
                         is_pose_estimation, rename_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split COCO dataset into training, validation, and testing sets.")
    parser.add_argument("images_dir", help="Path to the input directory containing images.")
    parser.add_argument("coco_json_path", help="Path to the COCO JSON file containing annotations.")
    parser.add_argument("output_dir", help="Path to the root output directory for training, validation, and testing sets.")
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Proportion of images for training (default: 0.75)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion of images for validation (default: 0.1)")
    parser.add_argument("--pose_estimation", action='store_true', help="Flag to indicate if the dataset is for pose estimation")
    parser.add_argument("--rename_images", action="store_true", help="Assign new numerical IDs to image file names")
    parser.add_argument("--classes", nargs='+', help="List of class names to process (default: all classes)")
    parser.add_argument("--ablation", type=int, default=0, help="Number of dataset chunks for ablation testing (default: 0, no ablation)")


    args = parser.parse_args()

    main(**vars(args))
