import json
import os
from typing import List, Dict, Any, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_coco_json(json_path: str) -> Dict[str, Any]:
    """
    Validates a COCO JSON file structure.
    
    Args:
        json_path: Path to the COCO JSON file.
        
    Returns:
        A dictionary containing validation results:
        - valid: Boolean indicating overall validity.
        - errors: List of critical error messages.
        - warnings: List of warning messages.
        - stats: Dictionary of statistics (num_images, num_annotations, etc.)
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to load JSON: {str(e)}")
        return results

    # Check required top-level keys
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in data:
            results["valid"] = False
            results["errors"].append(f"Missing required key: '{key}'")
    
    if not results["valid"]:
        return results

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    results["stats"] = {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories)
    }

    # Validate Images
    image_ids = set()
    for img in images:
        if "id" not in img:
            results["errors"].append("Image missing 'id' field")
            results["valid"] = False
        else:
            if img["id"] in image_ids:
                results["errors"].append(f"Duplicate image ID found: {img['id']}")
                results["valid"] = False
            image_ids.add(img["id"])
        
        if "file_name" not in img:
            results["errors"].append(f"Image {img.get('id', 'unknown')} missing 'file_name'")
            results["valid"] = False

    # Validate Categories
    cat_ids = set()
    for cat in categories:
        if "id" not in cat:
            results["errors"].append("Category missing 'id' field")
            results["valid"] = False
        else:
            if cat["id"] in cat_ids:
                results["errors"].append(f"Duplicate category ID found: {cat['id']}")
                results["valid"] = False
            cat_ids.add(cat["id"])
        
        if "name" not in cat:
            results["errors"].append(f"Category {cat.get('id', 'unknown')} missing 'name'")
            results["valid"] = False

    # Validate Annotations
    ann_ids = set()
    for ann in annotations:
        if "id" not in ann:
            results["errors"].append("Annotation missing 'id' field")
            results["valid"] = False
        else:
            if ann["id"] in ann_ids:
                results["errors"].append(f"Duplicate annotation ID found: {ann['id']}")
                results["valid"] = False
            ann_ids.add(ann["id"])
        
        if "image_id" not in ann:
            results["errors"].append(f"Annotation {ann.get('id', 'unknown')} missing 'image_id'")
            results["valid"] = False
        elif ann["image_id"] not in image_ids:
            results["errors"].append(f"Annotation {ann.get('id', 'unknown')} references non-existent image_id: {ann['image_id']}")
            results["valid"] = False
            
        if "category_id" not in ann:
            results["errors"].append(f"Annotation {ann.get('id', 'unknown')} missing 'category_id'")
            results["valid"] = False
        elif ann["category_id"] not in cat_ids:
            results["errors"].append(f"Annotation {ann.get('id', 'unknown')} references non-existent category_id: {ann['category_id']}")
            results["valid"] = False

    return results

def merge_coco_jsons(json_paths: List[str], output_path: str) -> None:
    """
    Merges multiple COCO JSON files, keeping only images present in ALL files.
    
    Args:
        json_paths: List of paths to COCO JSON files.
        output_path: Path to save the merged JSON.
    """
    if not json_paths:
        logger.warning("No input files provided.")
        return

    loaded_data = []
    for p in json_paths:
        try:
            with open(p, 'r') as f:
                loaded_data.append(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load {p}: {e}")
            raise

    # 1. Find intersection of images (by file_name)
    # Map file_name -> image info for each dataset
    # We need to ensure we are talking about the same images.
    
    # sets of file_names
    file_name_sets = []
    for data in loaded_data:
        fnames = {img["file_name"] for img in data.get("images", [])}
        file_name_sets.append(fnames)
    
    common_file_names = set.intersection(*file_name_sets)
    logger.info(f"Found {len(common_file_names)} common images across {len(json_paths)} files.")

    if not common_file_names:
        logger.warning("No common images found. Output will be empty.")
        
    # 2. Build Unified Categories
    # We assume categories with same name are the same.
    # We will create a new category map: name -> new_id
    unified_categories = {}
    next_cat_id = 1
    
    # We also need to map (dataset_index, old_cat_id) -> new_cat_id
    cat_id_mapping = {} # (dataset_idx, old_id) -> new_id

    for idx, data in enumerate(loaded_data):
        for cat in data.get("categories", []):
            name = cat["name"]
            if name not in unified_categories:
                unified_categories[name] = {
                    "id": next_cat_id,
                    "name": name,
                    "supercategory": cat.get("supercategory", "")
                }
                next_cat_id += 1
            
            cat_id_mapping[(idx, cat["id"])] = unified_categories[name]["id"]

    # 3. Construct Merged Data
    merged_images = []
    merged_annotations = []
    
    # We need to re-index images and annotations to avoid ID collisions
    next_img_id = 1
    next_ann_id = 1
    
    # Map file_name -> new_image_id
    fname_to_new_id = {}
    
    # Process images
    # We take image info from the first dataset that has it (they should be identical mostly)
    # But we only include if it is in common_file_names
    
    # To ensure deterministic order, sort common_file_names
    sorted_fnames = sorted(list(common_file_names))
    
    for fname in sorted_fnames:
        # Find the image info in the first dataset
        # (We could check consistency across datasets, but let's assume first is truth for metadata like height/width)
        img_info = None
        for img in loaded_data[0]["images"]:
            if img["file_name"] == fname:
                img_info = img
                break
        
        if img_info:
            new_img = img_info.copy()
            new_img["id"] = next_img_id
            fname_to_new_id[fname] = next_img_id
            merged_images.append(new_img)
            next_img_id += 1

    # Process annotations
    for idx, data in enumerate(loaded_data):
        # Map old_img_id -> file_name for this dataset
        old_img_id_to_fname = {img["id"]: img["file_name"] for img in data.get("images", [])}
        
        for ann in data.get("annotations", []):
            old_img_id = ann["image_id"]
            if old_img_id not in old_img_id_to_fname:
                continue # Orphan annotation
                
            fname = old_img_id_to_fname[old_img_id]
            if fname in common_file_names:
                # This annotation belongs to a common image
                new_ann = ann.copy()
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = fname_to_new_id[fname]
                
                # Update category ID
                old_cat_id = ann["category_id"]
                if (idx, old_cat_id) in cat_id_mapping:
                    new_ann["category_id"] = cat_id_mapping[(idx, old_cat_id)]
                else:
                    # Should not happen if categories are consistent
                    logger.warning(f"Skipping annotation with unknown category {old_cat_id} in dataset {idx}")
                    continue
                
                merged_annotations.append(new_ann)
                next_ann_id += 1

    final_json = {
        "info": {"description": "Merged COCO dataset"},
        "licenses": [],
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": list(unified_categories.values())
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_json, f, indent=2)
    
    logger.info(f"Stats: {len(merged_images)} images, {len(merged_annotations)} annotations, {len(unified_categories)} categories")

def filter_coco_duplicates(coco_data: Dict[str, Any], iou_threshold: float = 0.95) -> Dict[str, Any]:
    """
    Filters duplicate annotations in a COCO dataset based on IoU threshold.
    If IoU > threshold for two annotations of the same category in the same image,
    the one with the smaller area is removed.

    Args:
        coco_data: The COCO dataset dictionary.
        iou_threshold: IoU threshold for considering two annotations as duplicates.

    Returns:
        The filtered COCO dataset dictionary.
    """
    try:
        from pycocotools import mask as maskUtils
    except ImportError:
        logger.error("pycocotools is required for filtering duplicates.")
        raise

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    
    # Organize annotations by image_id
    img_to_anns = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    anns_to_remove = set()
    
    logger.info(f"Processing {len(img_to_anns)} images for duplicate detection...")
    
    for img_id, img_anns in img_to_anns.items():
        # Group by category
        cat_to_anns = {}
        for ann in img_anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_to_anns:
                cat_to_anns[cat_id] = []
            cat_to_anns[cat_id].append(ann)
            
        for cat_id, anns in cat_to_anns.items():
            if len(anns) < 2:
                continue
            
            # Prepare RLEs or BBoxes for IoU calculation
            # We prefer mask IoU if segmentation is available
            
            # Check if majority have segmentation
            has_segmentation = any("segmentation" in ann and ann["segmentation"] for ann in anns)
            
            iscrowd_list = [ann.get("iscrowd", 0) for ann in anns]
            
            if has_segmentation:
                # Convert to RLEs
                rles = []
                for ann in anns:
                    seg = ann.get("segmentation", [])
                    if not seg:
                        # Fallback to bbox if missing segmentation? 
                        # Or treat as empty mask? 
                        # Construct from bbox
                        x, y, w, h = ann["bbox"]
                        # rle from bbox (not exact but okay if mixed? No, better consistency)
                        # Actually pycocotools frPyObjects handles both if we format correctly
                        # But simpler: if no seg, use empty? No, use bbox.
                        # For consistency let's stick to what availability dictates.
                        # If mixed, this is tricky. 
                        # Let's assume consistent format within dataset usually.
                        # If segmentation is available, use it.
                        if "segmentation" in ann:
                            if isinstance(ann["segmentation"], list):
                                # Polygon
                                # getting image height/width is needed for frPyObjects
                                # We need to look up image size.
                                # This is slow if we do it for every ann.
                                pass
                            elif isinstance(ann["segmentation"], dict) and "counts" in ann["segmentation"]:
                                # RLE
                                pass
                        pass

                # Optimized approach:
                # We need image dims for polygon -> RLE
                # Find image info
                img_info = None
                for img in images:
                    if img["id"] == img_id:
                        img_info = img
                        break
                if not img_info:
                    # skip if image info missing
                    continue
                
                h, w = img_info["height"], img_info["width"]
                
                encoded_masks = []
                for ann in anns:
                    if "segmentation" in ann and ann["segmentation"]:
                        seg = ann["segmentation"]
                        if isinstance(seg, list):
                            # Polygon
                            rle = maskUtils.frPyObjects(seg, h, w)
                            rle = maskUtils.merge(rle) # merge parts into one RLE
                            encoded_masks.append(rle)
                        elif isinstance(seg, dict) and "counts" in seg:
                            # RLE (could be uncompressed)
                            if isinstance(seg["counts"], list):
                                # Uncompressed RLE -> Convert to compressed (encoded) RLE
                                encoded = maskUtils.frPyObjects([seg], h, w)[0]
                                encoded_masks.append(encoded)
                            else:
                                # Already compressed (bytes)
                                encoded_masks.append(seg)
                        else:
                            # Empty or invalid
                            encoded_masks.append(maskUtils.frPyObjects([[0,0,0,0]], h, w)[0]) # dummy
                    else:
                        # Fallback to bbox
                        rect = [ann["bbox"]] # [x,y,w,h]
                        rle = maskUtils.frPyObjects(rect, h, w)[0]
                        encoded_masks.append(rle)
                
                # Compute IoU matrix
                ious = maskUtils.iou(encoded_masks, encoded_masks, iscrowd_list)
                
            else:
                # Use BBoxes
                bboxes = [ann["bbox"] for ann in anns]
                try:
                    ious = maskUtils.iou(bboxes, bboxes, iscrowd_list)
                except Exception as e:
                    logger.error(f"BBox IoU calculation failed for image {img_id}, category {cat_id}")
                    logger.error(f"BBoxes type: {type(bboxes)}")
                    if bboxes:
                        logger.error(f"First bbox: {bboxes[0]}")
                    raise e
            
            # Find pairs > threshold
            # ious is N x N matrix
            # check upper triangle
            num = len(anns)
            for i in range(num):
                if anns[i]["id"] in anns_to_remove:
                    continue
                for j in range(i + 1, num):
                    if anns[j]["id"] in anns_to_remove:
                        continue
                    
                    if ious[i, j] > iou_threshold:
                        # Check areas
                        area_i = anns[i].get("area", 0)
                        area_j = anns[j].get("area", 0)
                        
                        # Remove smaller
                        if area_i < area_j:
                            anns_to_remove.add(anns[i]["id"])
                            break # i is removed, stop checking i against others
                        else:
                            anns_to_remove.add(anns[j]["id"])
    
    if not anns_to_remove:
        logger.info("No duplicate annotations found.")
        return coco_data
        
    logger.info(f"Removing {len(anns_to_remove)} duplicate annotations.")
    
    new_annotations = [ann for ann in annotations if ann["id"] not in anns_to_remove]
    
    coco_data["annotations"] = new_annotations
    return coco_data
