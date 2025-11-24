import argparse
import cv2

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def bbox_maker(images_folder, labels_folder, output_folder, resize, gt, falses, gt_labels_folder=None):
    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)
    output_folder = Path(output_folder)
    
    if gt_labels_folder:
        gt_labels_folder = Path(gt_labels_folder)

    if not images_folder.exists():
        print(f"Images folder {images_folder} does not exist.")
        return
    if not labels_folder.exists():
        print(f"Labels folder {labels_folder} does not exist.")
        return
    if falses and not gt_labels_folder.exists():
        print(f"Ground truth labels folder {gt_labels_folder} does not exist.")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    resize_dims = parse_resize_arg(resize)
    process_images(images_folder, labels_folder, output_folder, resize_dims, gt, falses, gt_labels_folder)

def process_images(images_folder, labels_folder, output_folder, resize_dims, gt, falses, gt_labels_folder=None):
    # Hardcoded color/class mapping for two classes
    class_map = {
        '0': {'name': 'soy', 'color': (252, 236, 3)},
        '1': {'name': 'cotton', 'color': (201, 14, 230)},
        '-1': {'name': 'wrong predictions', 'color': (255, 0, 0)},
        '-2': {'name': 'missed predictions', 'color': (255, 255, 255)}
    }

    image_extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in images_folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {images_folder}.")
        return

    for image_path in images:
        
        img = Image.open(image_path)
        if resize_dims is not None:
            w, h = resize_dims
            img = img.resize((w, h), Image.LANCZOS)

        pred_label_file = labels_folder / (image_path.stem + '.txt')
        if not pred_label_file.exists():
            print(f"Label file {pred_label_file} does not exist for image {image_path.name}. Skipping this image.")
            continue

        # Read prediction labels (when gt=True, labels_folder is ground truth; otherwise, predictions)
        pred_labels = read_labels(pred_label_file, gt=False if not gt else True)

        if falses:
            gt_label_file = gt_labels_folder / (image_path.stem + '.txt')
            if not gt_label_file.exists():
                print(f"Ground truth label file {gt_label_file} does not exist for image {image_path.name}. Skipping this image.")
                continue
            gt_labels = read_labels(gt_label_file, gt=True)
            # Classify boxes when --falses is active
            boxes_to_render = classify_boxes(pred_labels, gt_labels, class_map, img.size)
        else:
            # Prepare boxes based on whether labels are ground truth or predictions
            img = Image.open(image_path)
            if resize_dims is not None:
                w, h = resize_dims
                img = img.resize((w, h), Image.LANCZOS)
            img_width, img_height = img.size
            boxes_to_render = []
            for label in pred_labels:
                cls_id = label['class_id']
                if cls_id not in class_map:
                    continue
                color = class_map[cls_id]['color']
                box = label_to_box(label, img_width, img_height)
                text = f"{label['confidence']:.2f}" if not gt else None
                boxes_to_render.append({
                    'box': box,
                    'color': color,
                    'text': text
                })

        # Load and resize image, then draw boxes

        img_with_boxes = draw_bounding_boxes(img, boxes_to_render, class_map)
        img_with_boxes = img_with_boxes.convert('RGB')

        output_path = output_folder / image_path.name
        img_with_boxes.save(output_path)
        print(f"Saved image with bounding boxes to {output_path}")

def parse_resize_arg(resize_arg):
    """
    Parse the '--resize' argument.
    If the argument is 'None', return None.
    Otherwise, expect the format 'widthxheight' and return (width, height) as integers.
    """
    if resize_arg is None or resize_arg.lower() == "none":
        return None
    
    try:
        w, h = resize_arg.lower().split("x")
        return (int(w), int(h))
    except ValueError:
        print(f"Invalid format for --resize: '{resize_arg}'. Use 'widthxheight' or 'None'.")
        return None


def read_labels(label_file, gt):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6 and not gt:

                cls, x_center, y_center, width, height, confidence = parts
                labels.append({
                    'class_id': cls,
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height),
                    'confidence': float(confidence)
                })
            
            elif len(parts) == 5 and gt:
                cls, x_center, y_center, width, height = parts
                labels.append({
                    'class_id': cls,
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height),
                })

            else:
                print(f"Invalid label format in {label_file}: {line}")


    return labels

def draw_bounding_boxes(img, boxes_to_render, class_map):
    img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    overlay_error = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw_error = ImageDraw.Draw(overlay_error)

    try:
        font = ImageFont.truetype("DroidSerif-Regular.ttf", size=32)
    except IOError:
        font = ImageFont.load_default(size=30)

    boxes = []
    texts = []

    # Collect box and text information
    for box_info in boxes_to_render:
        coords = box_info['box']
        color = box_info['color']
        text = box_info.get('text')

        fill_opacity = 0.161  # ~16.1% opacity
        alpha = int(255 * fill_opacity)
        fill_color = color + (alpha,)
        outline_color = color + (255,)

        boxes.append({
            'coords': coords,
            'fill_color': fill_color,
            'outline_color': outline_color,
            'type': box_info['type']
        })

        if text:
            x_min, y_min, x_max, y_max = coords
            x_left, y_top, x_right, y_bottom = font.getbbox(text)
            text_width = abs(x_right - x_left)
            text_height = abs(y_bottom - y_top)
            bbox_xmid = (x_max - x_min) / 2
            text_position = (x_min + bbox_xmid - text_width // 2, y_min - 2 * text_height)
            if text_position[1] < 0:
                text_position = (x_min + bbox_xmid - text_width // 2, y_max)
            shadow_position = (text_position[0] + 1, text_position[1] + 1)
            texts.append({
                'text': text,
                'position': text_position,
                'shadow_position': shadow_position
            })

    # Draw fills
    for box in boxes:         
        draw_rounded_rectangle(
            draw,
            box['coords'],
            radius=10,
            fill=box['fill_color'],
            outline=None
        )

    # Draw outlines
    for box in boxes:
        draw_rounded_rectangle(
            draw,
            box['coords'],
            radius=10,
            fill=None,
            outline=box['outline_color'],
            width=3
        )

    # Step 4: Draw all text
    # shadow_color = (0, 0, 0, 128)  # Semi-transparent black

    colors = [cl['color'][::-1] for cl in class_map.values()]
    for text_info in texts:
        draw.text(text_info['shadow_position'], text_info['text'], font=font, fill=colors[1])
        draw.text(text_info['position'], text_info['text'], font=font, fill=colors[0])

    # Composite overlay onto the image
    draw_legend(draw, class_map, font, *img.size)
    img = Image.alpha_composite(img, overlay)
    return img

def label_to_box(label, img_width, img_height):
    x_center = label['x_center'] * img_width
    y_center = label['y_center'] * img_height
    width = label['width'] * img_width
    height = label['height'] * img_height
    x_min = max(0, x_center - width / 2)
    y_min = max(0, y_center - height / 2)
    x_max = min(img_width, x_center + width / 2)
    y_max = min(img_height, y_center + height / 2)
    return [x_min, y_min, x_max, y_max]

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def classify_boxes(pred_labels, gt_labels, class_map, img_size, iou_threshold=0.3):
    img_width, img_height = img_size
    pred_boxes = []
    for label in pred_labels:
        box = label_to_box(label, img_width, img_height)
        pred_boxes.append({
            'box': box,
            'class_id': label['class_id'],
            'confidence': label['confidence']
        })
    gt_boxes = []
    for label in gt_labels:
        box = label_to_box(label, img_width, img_height)
        gt_boxes.append({
            'box': box,
            'class_id': label['class_id']
        })

    # Sort predictions by confidence
    pred_boxes.sort(key=lambda x: x['confidence'], reverse=True)

    matched_gt = set()
    boxes_to_render = []

    # Match predictions to ground truth
    for pred in pred_boxes:
        pred_box = pred['box']
        pred_class = pred['class_id']
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            current_iou = iou(pred_box, gt['box'])
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold and gt_boxes[best_gt_idx]['class_id'] != pred['box']:
            # True Positive
            color = class_map[pred_class]['color']
            boxes_to_render.append({
                'box': pred_box,
                'color': color,
                'text': None,
                # 'text': f"{pred['confidence']:.2f}",
                'type':'tp'
            })
            matched_gt.add(best_gt_idx)
        else:
            # False Positive
            boxes_to_render.append({
                'box': pred_box,
                'color': (255, 0, 0),  # Red
                'text': None,
                'type':'fp'
            })

    for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                boxes_to_render.append({
                    'box': gt['box'],
                    'color': (255, 255, 255),  # White
                    'text': None,
                    'type':'fn'
                })

    # Add False Negatives
    # for gt_idx, gt in enumerate(gt_boxes):
    #     best_iou = 0
    #     best_pred_idx = -1
    #     # if gt_idx not in matched_gt:
 
    #     for pred_idx, pred in enumerate(pred_boxes):
    #         if pred['class_id'] != gt['class_id']:
    #             continue
    #         current_iou = iou(pred['box'], gt['box'])
    #         if current_iou > best_iou:
    #             best_iou = current_iou
    #             best_pred_idx = pred_idx
        
    #     if best_iou < iou_threshold:
    #         boxes_to_render.append({
    #             'box': gt['box'],
    #             'color': (255, 255, 255),  # White
    #             'text': None,
    #             'type': 'fn'
    #         })


    return boxes_to_render

def draw_rounded_rectangle(draw, xy, radius=5, fill=None, outline=None, width=1):
    
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

def draw_legend(draw, class_map, font, img_width, img_height, radius=10):

    legend_x = 10  # Padding from the left edge
    legend_y = 10  # Padding from the top edge

    y_text_offset = 5
    x_text_offset = 5

    max_text_width = 0
    total_text_height = 0
    entries = []

    for cls_id, class_info in class_map.items():
        class_name = class_info['name'].capitalize()
        color = class_info['color']
        text = class_name

        # Get text size
        x_left, y_top, x_right, y_bottom = font.getbbox(text)
        text_width = abs(x_right - x_left)
        text_height = abs(y_bottom - y_top)

        max_text_width = max(max_text_width, text_width)
        total_text_height += text_height + 5  # Adding spacing between entries

        entries.append({
            'text': text,
            'text_width': text_width,
            'text_height': text_height,
            'color': color
        })

    # Square size is 80% of text heightimages_folder/
    square_size = int(0.6 * entries[-1]["text_height"])

    # Background for the legend (optional)
    legend_width =  square_size + max_text_width + 5*x_text_offset  
    legend_height = total_text_height + 3*y_text_offset 
    legend_background = [
        (legend_x, legend_y),
        (legend_x + legend_width, legend_y + legend_height)
    ]
    draw.rounded_rectangle(legend_background, radius=radius, fill=(50, 50, 50, 180))

    # Draw each legend entry
    current_y = legend_y + y_text_offset  # Starting y position with padding
    for entry in entries:
        text = entry['text']
        text_width = entry['text_width']
        text_height = entry['text_height']
        color = entry['color']

        square_offset = abs(text_height + 2*y_text_offset - square_size)/2
        square_y = current_y + square_offset

        # Draw the color square
        square_coords = [
            legend_x + x_text_offset,
            square_y,
            legend_x + x_text_offset + square_size,
            square_y + square_size
        ]
        draw.rounded_rectangle(square_coords, radius=radius*0.1, fill=color + (255,), outline=None)

        text_position = (legend_x + x_text_offset*3 + square_size, current_y)
        draw.text(
            text_position,
            text,
            fill=(255, 255, 255, 255), 
            font=font
        )

        current_y += text_height + y_text_offset 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render bounding boxes on images.")
    parser.add_argument("images_folder", help="Path to the folder containing images.")
    parser.add_argument("labels_folder", help="Path to the folder containing label files.")
    parser.add_argument("output_folder", help="Path to the folder where output images will be saved.")
    parser.add_argument("--gt", action='store_true', default=False, help="Ground truth masks flag.")
    parser.add_argument(
        "--resize",
        default="None",
        help="Resize images to 'widthxheight'. Use 'None' to not resize."
    )
    parser.add_argument(
        "--falses",
        action='store_true',
        default=False,
        help="Render false positives and false negatives."
    )
    parser.add_argument(
        "--gt_labels_folder",
        help="Path to the folder containing ground truth label files. Required if --falses is used."
    )

    args = parser.parse_args()

    # Validate arguments
    if args.falses:
        if args.gt:
            parser.error("--falses cannot be used when --gt is True.")
        if not args.gt_labels_folder:
            parser.error("--gt_labels_folder is required when --falses is used.")
    elif args.gt_labels_folder:
        print("Warning: --gt_labels_folder is provided but --falses is not used. Ignoring --gt_labels_folder.")

    bbox_maker(**vars(args))
