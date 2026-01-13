import json
import os
from typing import List, Dict, Any, Set, Tuple
import logging

import logging
import numpy as np
from collections import defaultdict
import pycocotools.mask as mask_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_coco(json_path: str) -> Dict[str, Any]:
    """
    Load a COCO JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_iou_boxes(box1, box2):
    """
    Compute IoU between two bounding boxes [x, y, w, h].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

def compute_iop_boxes(pred_box, gt_box):
    """
    Compute Intersection over Prediction Area.
    pred_box: [x, y, w, h]
    gt_box: [x, y, w, h]
    """
    x1, y1, w1, h1 = pred_box
    x2, y2, w2, h2 = gt_box
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    pred_area = w1 * h1
    
    if pred_area == 0: return 0
    return inter_area / pred_area

def ann_to_rle(ann, h, w):
    """
    Convert annotation (Target or Pred) to RLE.
    Handles Polygon, RLE (dict), and RLE (bytes).
    """
    seg = ann.get("segmentation")
    if not seg:
        # Fallback to bbox if available, else empty
        if "bbox" in ann:
            return mask_utils.frPyObjects([ann["bbox"]], h, w)[0]
        return mask_utils.frPyObjects([[0,0,0,0]], h, w)[0]

    if isinstance(seg, list):
        # Polygon - list of lists
        # Filter empty polys
        valid_polys = [p for p in seg if len(p) >= 6]
        if not valid_polys:
             return mask_utils.frPyObjects([[0,0,0,0]], h, w)[0]
        rles = mask_utils.frPyObjects(valid_polys, h, w)
        return mask_utils.merge(rles)
    elif isinstance(seg, dict) and "counts" in seg:
        # RLE
        if isinstance(seg["counts"], list):
            # Uncompressed RLE
            return mask_utils.frPyObjects([seg], h, w)[0]
        else:
            # Compressed RLE
            return seg
    return mask_utils.frPyObjects([[0,0,0,0]], h, w)[0]

def compute_iou_masks(ann1, ann2, h, w, iscrowd=0):
    rle1 = ann_to_rle(ann1, h, w)
    rle2 = ann_to_rle(ann2, h, w)
    
    # iou returns M x N array, we have 1 vs 1
    # iscrowd list for ground truth (ann2 usually)
    # mask_utils.iou takes list of RLEs
    
    # Note: mask_utils.iou expects iscrowd as list of 0/1 for dt (detection) or gt (ground truth)?
    # Documentation: iou(dt, gt, iscrowd)
    # dt: list of RLEs
    # gt: list of RLEs
    # iscrowd: list of 0/1 (length of gt)
    
    val = mask_utils.iou([rle1], [rle2], [iscrowd])
    return val[0][0]

def match_annotations(gt_anns, pred_anns, pred_id_to_gt_id=None, iou_threshold=0.5, use_segmentation=False, match_n=1, img_height=None, img_width=None):
    """
    Match ground truth and predicted annotations.
    match_n: Number of times a GT can be matched (default 1).
    """
    tp_preds = []
    fp_preds = []
    
    matched_gt_counts = defaultdict(int)
    
    pred_anns_sorted = sorted(pred_anns, key=lambda x: x.get("score", 0), reverse=True)
    
    for pred in pred_anns_sorted:
        best_metric = 0 # Can be IoU or IoP
        best_gt_idx = -1
        match_type = "none"
        
        pred_cat = pred["category_id"]
        if pred_id_to_gt_id:
            gt_cat_target = pred_id_to_gt_id.get(pred_cat)
        else:
            gt_cat_target = pred_cat
            
        for i, gt in enumerate(gt_anns):
            # Check if this GT has been matched 'n' times already
            if matched_gt_counts[gt["id"]] >= match_n:
                continue
            if gt["category_id"] != gt_cat_target:
                continue
                
            # Default IoU check
            iou = 0
            
            if use_segmentation and img_height and img_width:
                 try:
                     iou = compute_iou_masks(pred, gt, img_height, img_width, gt.get("iscrowd", 0))
                     # logger.info(f"Computed Mask IoU: {iou}") 
                 except Exception as e:
                     logger.warning(f"Mask IoU failed: {e}. Falling back to Box.")
                     if "bbox" in pred and "bbox" in gt:
                         iou = compute_iou_boxes(pred["bbox"], gt["bbox"])
            else:
                if use_segmentation:
                    logger.warning(f"Segmentation requested but missing img dims: h={img_height}, w={img_width}")
                
                # Box IoU
                if "bbox" in pred and "bbox" in gt:
                    iou = compute_iou_boxes(pred["bbox"], gt["bbox"])
            
            # IoP Check (Intersection over Prediction) - ONLY FOR BOX currently
            iop = 0
            if not use_segmentation and "bbox" in pred and "bbox" in gt:
                iop = compute_iop_boxes(pred["bbox"], gt["bbox"])
            
            # Priority: High IoU > High IoP?
            if iou >= iou_threshold:
                if iou > best_metric: # Track best IoU
                    best_metric = iou
                    best_gt_idx = i
                    match_type = "iou"
            elif iop >= 0.9:
                 if match_type != "iou":
                    if iop > best_metric:
                        best_metric = iop
                        best_gt_idx = i
                        match_type = "iop"
        
        if best_gt_idx != -1:
            tp_preds.append((pred, gt_anns[best_gt_idx]))
            matched_gt_counts[gt_anns[best_gt_idx]["id"]] += 1
        else:
            fp_preds.append(pred)
            
    # False Negatives are those that were NEVER matched (count == 0)
    fn_gts = [gt for gt in gt_anns if matched_gt_counts[gt["id"]] == 0]
    return tp_preds, fp_preds, fn_gts

def calculate_confusion_matrix(tp_preds, fp_preds, fn_gts, categories):
    """
    Calculate the confusion matrix.
    Rows: Ground Truth
    Columns: Predictions
    
    Returns:
       matrix: sklearn-style confusion matrix (or dict representation)
       labels: list of class names corresponding to indices
    """
    # Categories include "Background" typically for FP/FN?
    # Or we can just do Class vs Class.
    # TP: GT Class X matched with Pred Class X (or mapped)
    # FP: Pred Class X unmatched (GT Background)
    # FN: GT Class X unmatched (Pred Background)
    # Misclassification: GT Class X matched with Pred Class Y? match_annotations enforces category match currently if we strictly filter.
    # But `match_annotations` strictly filters by category: `if gt["category_id"] != gt_cat_target: continue`.
    # So we don't handle cross-class confusion in that function. 
    # That function assumes we only match same-class.
    # So "Confusion Matrix" here is diagonal + FP col + FN row.
    
    # Let's organize by class ID
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    sorted_cat_ids = sorted(cat_id_to_name.keys())
    
    stats = {}
    for cid in sorted_cat_ids:
        cname = cat_id_to_name[cid]
        stats[cname] = {"TP": 0, "FP": 0, "FN": 0}
        
    for pred, gt in tp_preds:
        # Assuming pred category maps to gt category correctly
        cid = gt["category_id"]
        if cid in cat_id_to_name:
            stats[cat_id_to_name[cid]]["TP"] += 1
            
    for pred in fp_preds:
        cid = pred["category_id"]
        # If we have a map, we might need to map it back? 
        # But usually we report in terms of GT classes or Pred classes?
        # Usually Precision/Recall is per class.
        if cid in cat_id_to_name:
             stats[cat_id_to_name[cid]]["FP"] += 1
        
    for gt in fn_gts:
        cid = gt["category_id"]
        if cid in cat_id_to_name:
            stats[cat_id_to_name[cid]]["FN"] += 1
            
    return stats

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
