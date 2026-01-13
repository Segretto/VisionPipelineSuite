#!/usr/bin/env python3
import argparse
import sys
import os
import json
import cv2
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_suite.visualization.drawing import draw_bounding_boxes, draw_segmentation_masks
from vision_suite.utils.coco import load_coco, match_annotations



def get_class_map(coco_data):
    class_map = {}
    # Define a palette of colors
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255)
    ]
    
    for i, cat in enumerate(coco_data.get("categories", [])):
        color = palette[i % len(palette)]
        class_map[cat["id"]] = {
            "name": cat["name"],
            "color": color
        }
    return class_map



def main():
    parser = argparse.ArgumentParser(description="Visualize COCO annotations.")
    parser.add_argument("coco_json", help="Path to the COCO JSON file (Ground Truth).")
    parser.add_argument("--image_dir", help="Directory containing images.")
    parser.add_argument("--output_dir", help="Directory to save visualized images.")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of images.")
    parser.add_argument("--segmentation", action="store_true", help="Render segmentation masks.")
    
    parser.add_argument("--compare_json", help="Path to the Predictions JSON file.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument("--hide_correct", action="store_true", help="Hide correct predictions.")
    parser.add_argument("--plot_comparison", action="store_true", help="Generate a 3-panel comparison (Errors, Preds, GT).")
    parser.add_argument("--match_n", type=int, default=1, help="Number of times a GT can be matched (default: 1).")
    
    args = parser.parse_args()

    if not os.path.exists(args.coco_json):
        print(f"Error: File not found: {args.coco_json}")
        sys.exit(1)

    print("Loading GT COCO data...")
    gt_data = load_coco(args.coco_json)
    class_map = get_class_map(gt_data)
    gt_name_to_id = {cat["name"]: cat["id"] for cat in gt_data.get("categories", [])}
    
    gt_img_to_anns = {}
    for ann in gt_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in gt_img_to_anns: gt_img_to_anns[img_id] = []
        gt_img_to_anns[img_id].append(ann)

    pred_img_to_anns = {}
    pred_id_to_gt_id = {}
    pred_stem_to_id = {}
    
    if args.compare_json:
        print(f"Loading Predictions from {args.compare_json}...")
        pred_data = load_coco(args.compare_json)
        if isinstance(pred_data, dict) and "categories" in pred_data:
            for cat in pred_data["categories"]:
                p_name = cat["name"]
                p_id = cat["id"]
                if p_name in gt_name_to_id:
                    pred_id_to_gt_id[p_id] = gt_name_to_id[p_name]
        
        if isinstance(pred_data, dict) and "images" in pred_data:
             for img in pred_data["images"]:
                 fname = img["file_name"]
                 stem = os.path.splitext(os.path.basename(fname))[0]
                 pred_stem_to_id[stem] = img["id"]

        anns_list = pred_data if isinstance(pred_data, list) else pred_data.get("annotations", [])
        for ann in anns_list:
            img_id = ann["image_id"]
            if img_id not in pred_img_to_anns: pred_img_to_anns[img_id] = []
            pred_img_to_anns[img_id].append(ann)
    
    images = gt_data.get("images", [])
    if not images: sys.exit(0)

    # Helper to generate spectrum colors
    def get_spectrum_color(index, base_type):
        np.random.seed(index)
        if base_type == "red":
            r = np.random.randint(200, 256)
            g = np.random.randint(0, 100)
            b = np.random.randint(0, 100)
            return (r, g, b)
        elif base_type == "cyan_white":
            r = np.random.randint(0, 256)
            g = np.random.randint(200, 256)
            b = np.random.randint(200, 256)
            return (r, g, b)
        return (0, 0, 0)
    
    # Legend Maps
    # 1. Standard Map (for GT and Preds view)
    standard_vis_map = class_map
    
    # 2. Comparison Map (Errors view) - has FP/FN entries
    # "Even if ... not present ... render ... all four"
    comparison_vis_map = {}
    if args.compare_json:
        for cid, info in class_map.items():
            name = info["name"]
            # Correct (Standard)
            comparison_vis_map[cid] = info
            # False
            f_color = get_spectrum_color(cid, "red")
            comparison_vis_map[-2000 - cid] = {"name": f"False {name}", "color": f_color}
            # Missed
            m_color = get_spectrum_color(cid, "cyan_white")
            comparison_vis_map[-3000 - cid] = {"name": f"Missed {name}", "color": m_color}

    limit = args.limit if args.limit != -1 else len(images)
    if args.output_dir:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        print(f"Saving {limit} visualized images...")
    else: sys.exit(0)

    # Drawing Helpers
    def get_boxes_masks_standard(anns, img_cv_shape, mapping=None):
        boxes = []
        masks = []
        h_img, w_img, _ = img_cv_shape
        for ann in anns:
            cat_id = ann["category_id"]
            # Map if provided
            if mapping and cat_id in mapping: cat_id = mapping[cat_id]
            
            if cat_id not in class_map: continue
            color = class_map[cat_id]["color"]
            
            if not args.segmentation and "bbox" in ann:
                x, y, w, h = ann["bbox"]
                boxes.append({"box": (x, y, x + w, y + h), "color": color, "text": ""})
            elif args.segmentation and "segmentation" in ann:
                seg = ann["segmentation"]
                polygons = []
                if isinstance(seg, list): polygons.extend(seg)
                elif isinstance(seg, dict) and "counts" in seg:
                     try:
                        rle = seg
                        if isinstance(rle['counts'], list):
                             mask = mask_utils.decode(mask_utils.frPyObjects([rle], rle['size'][0], rle['size'][1]))
                        else:
                             mask = mask_utils.decode(rle)
                        if len(mask.shape) == 3: mask = mask[:, :, 0]
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour) >= 3: polygons.append(contour.flatten().tolist())
                     except: pass
                
                for poly in polygons:
                    poly_np = np.array(poly).reshape(-1, 2)
                    poly_norm = poly_np / [w_img, h_img]
                    masks.append({"polygon": poly_norm.tolist(), "color": color, "text": ""})
        return boxes, masks
    
    def get_boxes_masks_errors(gt_anns, pred_anns, img_cv_shape, hide_correct_override=False):
        boxes = []
        masks = []
        
        tp_preds, fp_preds, fn_gts = match_annotations(gt_anns, pred_anns, pred_id_to_gt_id, args.iou_threshold, args.segmentation, args.match_n)
        
        h_img, w_img, _ = img_cv_shape

        def add_items(anns, item_type="tp", is_tp=False):
            for ann in anns:
                item = ann[0] if is_tp else ann
                cat_id = item["category_id"]
                target_cat_id = cat_id
                
                if item_type == "tp":
                     if pred_id_to_gt_id and cat_id in pred_id_to_gt_id:
                        target_cat_id = pred_id_to_gt_id[cat_id]
                     if target_cat_id not in class_map: continue
                     color = class_map[target_cat_id]["color"]
                elif item_type == "fp":
                     if pred_id_to_gt_id and cat_id in pred_id_to_gt_id:
                        target_cat_id = pred_id_to_gt_id[cat_id]
                     if target_cat_id not in class_map: continue
                     color = get_spectrum_color(target_cat_id, "red")
                elif item_type == "fn":
                     color = get_spectrum_color(cat_id, "cyan_white")
                else: continue
                
                if not args.segmentation and "bbox" in item:
                    x, y, w, h = item["bbox"]
                    boxes.append({"box": (x, y, x + w, y + h), "color": color, "text": ""})
                elif args.segmentation and "segmentation" in item:
                    seg = item["segmentation"]
                    polygons = []
                    # ... reuse rle decode ...
                    if isinstance(seg, list): polygons.extend(seg)
                    elif isinstance(seg, dict) and "counts" in seg:
                            try:
                                rle = seg
                                if isinstance(rle['counts'], list):
                                    mask = mask_utils.decode(mask_utils.frPyObjects([rle], rle['size'][0], rle['size'][1]))
                                else:
                                    mask = mask_utils.decode(rle)
                                if len(mask.shape) == 3: mask = mask[:, :, 0]
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    if len(contour) >= 3: polygons.append(contour.flatten().tolist())
                            except: pass
                    for poly in polygons:
                        poly_np = np.array(poly).reshape(-1, 2)
                        poly_norm = poly_np / [w_img, h_img]
                        masks.append({"polygon": poly_norm.tolist(), "color": color, "text": ""})

        hide = hide_correct_override or args.hide_correct
        if not hide: add_items(tp_preds, item_type="tp", is_tp=True)
        add_items(fp_preds, item_type="fp")
        add_items(fn_gts, item_type="fn")
        return boxes, masks

    def render_image(img_cv, boxes, masks, legend_map):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if boxes:
            img_pil = draw_bounding_boxes(img_pil, boxes, legend_map)
        if args.segmentation or masks:
            img_pil = draw_segmentation_masks(img_pil, masks, legend_map)
        return img_pil

    for i, img_info in enumerate(images[:limit]):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        
        if args.image_dir: img_path = os.path.join(args.image_dir, file_name)
        else:
            img_path = os.path.join(os.path.dirname(args.coco_json), file_name)
            if not os.path.exists(img_path): img_path = file_name

        if not os.path.exists(img_path): continue
        img_cv = cv2.imread(img_path)
        if img_cv is None: continue

        gt_anns = gt_img_to_anns.get(img_id, [])
        pred_anns = []
        if args.compare_json:
            gt_stem = os.path.splitext(os.path.basename(file_name))[0]
            if gt_stem in pred_stem_to_id:
                pred_anns = pred_img_to_anns.get(pred_stem_to_id[gt_stem], [])
            else:
                pred_anns = pred_img_to_anns.get(img_id, [])

        if args.plot_comparison:
            # 1. Error View (No Correct)
            boxes_err, masks_err = get_boxes_masks_errors(gt_anns, pred_anns, img_cv.shape, hide_correct_override=True)
            # Remove Correct from legend for this view?
            # User said "without the correct predictions". 
            # Implies hiding TPs. Legend? "render the legend with all four of them" was previous request.
            # But the "Errors" view specifically focuses on errors.
            # I will pass comparison_vis_map which has all of them.
            img_err = render_image(img_cv.copy(), boxes_err, masks_err, comparison_vis_map)
            
            # 2. All Preds View (Standard)
            boxes_pred, masks_pred = get_boxes_masks_standard(pred_anns, img_cv.shape, mapping=pred_id_to_gt_id)
            img_pred = render_image(img_cv.copy(), boxes_pred, masks_pred, standard_vis_map)
            
            # 3. GT View (Standard)
            boxes_gt, masks_gt = get_boxes_masks_standard(gt_anns, img_cv.shape)
            img_gt = render_image(img_cv.copy(), boxes_gt, masks_gt, standard_vis_map)
            
            # Stitch
            w, h = img_err.size
            composite = Image.new('RGB', (w * 3, h))
            composite.paste(img_err, (0, 0))
            composite.paste(img_pred, (w, 0))
            composite.paste(img_gt, (w * 2, 0))
            
            img_cv_final = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)
            
        elif args.compare_json:
            # Standard comparison view
            boxes, masks = get_boxes_masks_errors(gt_anns, pred_anns, img_cv.shape)
            img_pil = render_image(img_cv, boxes, masks, comparison_vis_map)
            img_cv_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # Standard GT view
            boxes, masks = get_boxes_masks_standard(gt_anns, img_cv.shape)
            img_pil = render_image(img_cv, boxes, masks, standard_vis_map)
            img_cv_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        output_path = os.path.join(args.output_dir, f"vis_{os.path.basename(file_name)}")
        cv2.imwrite(output_path, img_cv_final)
        print(f"Saved {output_path}")

    print("Done.")

if __name__ == "__main__":
    main()
