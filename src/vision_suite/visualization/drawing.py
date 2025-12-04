import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bounding_boxes(img, boxes_to_render, class_map):
    """
    Draw bounding boxes on a PIL Image.
    """
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("DroidSerif-Regular.ttf", size=32)
    except IOError:
        font = ImageFont.load_default()

    boxes = []
    texts = []

    # Collect box and text information
    for box_info in boxes_to_render:
        coords = box_info["box"]
        color = box_info["color"]
        text = box_info.get("text")

        fill_opacity = 0.161  # ~16.1% opacity
        alpha = int(255 * fill_opacity)
        fill_color = color + (alpha,)
        outline_color = color + (255,)

        boxes.append(
            {
                "coords": coords,
                "fill_color": fill_color,
                "outline_color": outline_color,
            }
        )

        if text:
            x_min, y_min, x_max, y_max = coords
            x_left, y_top, x_right, y_bottom = font.getbbox(text)
            text_width = abs(x_right - x_left)
            text_height = abs(y_bottom - y_top)
            bbox_xmid = (x_max - x_min) / 2
            text_position = (
                x_min + bbox_xmid - text_width // 2,
                y_min - 2 * text_height,
            )
            if text_position[1] < 0:
                text_position = (x_min + bbox_xmid - text_width // 2, y_max)
            shadow_position = (text_position[0] + 1, text_position[1] + 1)
            texts.append(
                {
                    "text": text,
                    "position": text_position,
                    "shadow_position": shadow_position,
                }
            )

    # Draw fills
    for box in boxes:
        draw_rounded_rectangle(
            draw, box["coords"], radius=10, fill=box["fill_color"], outline=None
        )

    # Draw outlines
    for box in boxes:
        draw_rounded_rectangle(
            draw,
            box["coords"],
            radius=10,
            fill=None,
            outline=box["outline_color"],
            width=3,
        )

    # Draw text
    colors = [cl["color"][::-1] for cl in class_map.values()]
    # Fallback if class_map doesn't have enough colors or specific structure
    # Ideally pass text color explicitly

    for text_info in texts:
        # Simple white text with shadow for now
        draw.text(
            text_info["shadow_position"],
            text_info["text"],
            font=font,
            fill=(0, 0, 0, 128),
        )
        draw.text(
            text_info["position"],
            text_info["text"],
            font=font,
            fill=(255, 255, 255, 255),
        )

    # Composite overlay onto the image
    draw_legend(draw, class_map, font, *img.size)
    img = Image.alpha_composite(img, overlay)
    return img


def draw_segmentation_masks(img, masks_to_render, class_map):
    """
    Draw segmentation masks on a PIL Image.
    """
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("DroidSerif-Regular.ttf", size=26)
    except IOError:
        font = ImageFont.load_default()

    img_width, img_height = img.size

    for mask_info in masks_to_render:
        polygon = mask_info["polygon"] # Expecting list of (x, y) tuples or list of lists
        color = mask_info["color"]
        text = mask_info.get("text")
        
        fill_opacity = 0.4
        alpha = int(255 * fill_opacity)
        fill_color = color + (alpha,)
        outline_color = color + (255,)

        # Convert normalized coordinates to absolute if necessary
        # Assuming input is normalized if max value <= 1.0
        # But visualize_coco.py was passing normalized.
        # Let's handle list of lists or flattened list
        
        abs_polygon = []
        if isinstance(polygon[0], (list, tuple)):
             for pt in polygon:
                 abs_polygon.append((int(pt[0] * img_width), int(pt[1] * img_height)))
        else:
             # Flattened list [x1, y1, x2, y2, ...]
             for i in range(0, len(polygon), 2):
                 abs_polygon.append((int(polygon[i] * img_width), int(polygon[i+1] * img_height)))

        if len(abs_polygon) < 2:
            continue

        draw.polygon(abs_polygon, fill=fill_color, outline=outline_color)
        
        if text:
             # Draw text at centroid or first point
             text_position = abs_polygon[0]
             draw.text(text_position, text, font=font, fill=(255, 255, 255, 255))

    # Composite overlay onto the image
    draw_legend(draw, class_map, font, *img.size)
    img = Image.alpha_composite(img, overlay)
    return img


def draw_rounded_rectangle(draw, xy, radius=5, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_legend(draw, class_map, font, img_width, img_height, radius=10):
    legend_x = 10
    legend_y = 10
    y_text_offset = 5
    x_text_offset = 5

    max_text_width = 0
    total_text_height = 0
    entries = []

    for cls_id, class_info in class_map.items():
        class_name = class_info["name"].capitalize()
        color = class_info["color"]
        text = class_name

        x_left, y_top, x_right, y_bottom = font.getbbox(text)
        text_width = abs(x_right - x_left)
        text_height = abs(y_bottom - y_top)

        max_text_width = max(max_text_width, text_width)
        total_text_height += text_height + 5

        entries.append(
            {
                "text": text,
                "text_width": text_width,
                "text_height": text_height,
                "color": color,
            }
        )

    if not entries:
        return

    square_size = int(0.6 * entries[-1]["text_height"])
    legend_width = square_size + max_text_width + 5 * x_text_offset
    legend_height = total_text_height + 3 * y_text_offset
    legend_background = [
        (legend_x, legend_y),
        (legend_x + legend_width, legend_y + legend_height),
    ]
    draw.rounded_rectangle(legend_background, radius=radius, fill=(50, 50, 50, 180))

    current_y = legend_y + y_text_offset
    for entry in entries:
        text = entry["text"]
        text_height = entry["text_height"]
        color = entry["color"]

        square_offset = abs(text_height + 2 * y_text_offset - square_size) / 2
        square_y = current_y + square_offset

        square_coords = [
            legend_x + x_text_offset,
            square_y,
            legend_x + x_text_offset + square_size,
            square_y + square_size,
        ]
        draw.rounded_rectangle(
            square_coords, radius=radius * 0.1, fill=color + (255,), outline=None
        )

        text_position = (legend_x + x_text_offset * 3 + square_size, current_y)
        draw.text(text_position, text, fill=(255, 255, 255, 255), font=font)

        current_y += text_height + y_text_offset


def draw_confidence_values(
    draw, class_map, labels, img_width, img_height, font, conf_threshold
):
    for label in labels:
        cls_id = label["class_id"]
        if cls_id not in class_map:
            continue

        confidence = label["confidence"]
        if confidence < conf_threshold:
            continue

        # Simple text drawing at centroid or top-left
        # Simplified for brevity, original logic was complex
        # Assuming polygon exists
        if "polygon" in label and label["polygon"]:
            # Just take the first point for simplicity in this migration
            x, y = label["polygon"][0]
            text_position = (int(x * img_width), int(y * img_height))
            draw.text(
                text_position, f"{confidence:.2f}", font=font, fill=(255, 255, 255)
            )
