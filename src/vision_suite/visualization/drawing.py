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
        # User requested Bigger and Bold.
        # Try finding a bold font from system or common paths if possible, 
        # otherwise just increase size significantly to appear "Bigger" and "Bold-like" (thicker via stroke if needed, but PIL stroke on text is specific)
        # Using a larger size (e.g. 40, was 32)
        font = ImageFont.truetype("DroidSerif-Bold.ttf", size=40)
    except IOError:
        try:
            font = ImageFont.truetype("DroidSerif-Regular.ttf", size=40)
        except IOError:
            font = ImageFont.load_default()

    boxes = []
    texts = []

    # Collect box and text information
    for box_info in boxes_to_render:
        coords = box_info["box"]
        color = box_info["color"]
        text = box_info.get("text")

        fill_opacity = 0.161
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
            
            # Position text on top of the box
            text_position = (
                x_min + bbox_xmid - text_width // 2,
                y_min - text_height - 5, # Slight padding
            )
            # If off screen, put below
            if text_position[1] < 0:
                text_position = (x_min + bbox_xmid - text_width // 2, y_max + 5)
            
            shadow_position = (text_position[0] + 1, text_position[1] + 1)
            texts.append(
                {
                    "text": text,
                    "position": text_position,
                    "shadow_position": shadow_position,
                    "color": color # Pass class color for text
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
    for text_info in texts:
        # Shadow for contrast
        draw.text(
            text_info["shadow_position"],
            text_info["text"],
            font=font,
            fill=(0, 0, 0, 128),
        )
        # Main text in class color (User request: "bold, with the same color as the class")
        # To emulate bold with PIL if font not bold, we can draw with stroke_width (avail in newer PIL)
        # or just rely on the larger font size if it's "Bold" font.
        # Let's assume class color + full alpha
        text_color = text_info["color"] + (255,)
        draw.text(
            text_info["position"],
            text_info["text"],
            font=font,
            fill=text_color,
            stroke_width=1, 
            stroke_fill=(0,0,0,255) # Add stroke for "Bold" look and contrast if using class color
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
        # Consistent larger font
        font = ImageFont.truetype("DroidSerif-Bold.ttf", size=32) # Increased from 26
    except IOError:
        try:
            font = ImageFont.truetype("DroidSerif-Regular.ttf", size=32)
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

        abs_polygon = []
        if isinstance(polygon[0], (list, tuple)):
             for pt in polygon:
                 abs_polygon.append((int(pt[0] * img_width), int(pt[1] * img_height)))
        else:
             for i in range(0, len(polygon), 2):
                 abs_polygon.append((int(polygon[i] * img_width), int(polygon[i+1] * img_height)))

        if len(abs_polygon) < 2:
            continue

        draw.polygon(abs_polygon, fill=fill_color, outline=outline_color)
        
        if text:
             # Draw text at centroid or first point
             text_position = abs_polygon[0]
             # Text in class color, bold-ish
             draw.text(text_position, text, font=font, fill=color+(255,), stroke_width=1, stroke_fill=(0,0,0,255))

    # Composite overlay onto the image
    draw_legend(draw, class_map, font, *img.size)
    img = Image.alpha_composite(img, overlay)
    return img


def draw_rounded_rectangle(draw, xy, radius=5, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_legend(draw, class_map, font, img_width, img_height, radius=10):
    legend_x = 10
    legend_y = 10
    y_text_offset = 6 # Increased padding (was 5)
    x_text_offset = 6

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
        total_text_height += text_height + 10 # Increased spacing (was 5)

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

    # User requested 15% bigger legend elements
    # square_size currently derived from text height. 
    # Since text/font is already bigger (set in caller), this scales naturally?
    # But let's check square size factor. Was 0.6 * text_height.
    # User said "Make the legend box and text bigger". Font is bigger now.
    # Center alignment: "little colored square... aligned... text is aligned misaligned upwards".
    
    square_size = int(0.7 * entries[-1]["text_height"]) # Increased factor from 0.6
    
    legend_width = square_size + max_text_width + 5 * x_text_offset
    # Increase legend width padding?
    legend_width = int(legend_width * 1.15)
    
    legend_height = total_text_height + 3 * y_text_offset
    legend_height = int(legend_height * 1.15) # Force bigger box
    
    # Recalculate background (maybe just padding)
    # Actually if I scale width/height blindly, content might not fill it properly or be centered.
    # Better to increase internal paddings.
    # Let's revert explicit width/height mult and just use generous padding.
    
    padding_scale = 1.3 # Increase padding
    x_text_offset = int(x_text_offset * padding_scale)
    y_text_offset = int(y_text_offset * padding_scale)
    
    # Re-calc based on new offsets
    total_text_height = 0
    for entry in entries:
        total_text_height += entry["text_height"] + y_text_offset # spacing

    legend_width = square_size + max_text_width + 4 * x_text_offset 
    legend_height = total_text_height + 2 * y_text_offset

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

        # Alignment logic
        # We want the square to be vertically centered relative to the text line.
        # Row height roughly text_height (plus spacing).
        # Center line of this row is at current_y + text_height/2.
        
        row_center_y = current_y + text_height / 2
        
        # Square top should be center - size/2
        square_y = row_center_y - square_size / 2
        
        square_coords = [
            legend_x + x_text_offset,
            square_y,
            legend_x + x_text_offset + square_size,
            square_y + square_size,
        ]
        draw.rounded_rectangle(
            square_coords, radius=radius * 0.1, fill=color + (255,), outline=None
        )

        # Text position: 
        # Text is drawn from top-left usually? confirm ImageDraw.text behavior.
        # Default anchor is 'la' (left ascender) or similar.
        # Ideally we draw text at current_y.
        text_position = (legend_x + x_text_offset * 2 + square_size, current_y)
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
