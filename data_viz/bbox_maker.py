import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Render bounding boxes on images.")
    parser.add_argument("images_folder", help="Path to the folder containing images.")
    parser.add_argument("labels_folder", help="Path to the folder containing label files.")
    parser.add_argument("output_folder", help="Path to the folder where output images will be saved.")
    args = parser.parse_args()

    images_folder = Path(args.images_folder)
    labels_folder = Path(args.labels_folder)
    output_folder = Path(args.output_folder)

    # Verify that the input folders exist
    if not images_folder.exists():
        print(f"Images folder {images_folder} does not exist.")
        return
    if not labels_folder.exists():
        print(f"Labels folder {labels_folder} does not exist.")
        return

    # Create the output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process images
    process_images(images_folder, labels_folder, output_folder)

def process_images(images_folder, labels_folder, output_folder):
    # Map class IDs to labels and colors
    class_map = {
        '0': {'name': 'soy', 'color': (252, 35, 97)},        # Bright red color for soy
        '1': {'name': 'cotton', 'color': (7, 234, 250)}    # Bright cyan color for cotton
    }

    # Get list of images in the images folder
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in images_folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {images_folder}.")
        return

    # For each image
    for image_path in images:
        # Corresponding label file
        label_file = labels_folder / (image_path.stem + '.txt')
        if not label_file.exists():
            print(f"Label file {label_file} does not exist for image {image_path.name}. Skipping this image.")
            continue
        # Read the label file
        labels = read_labels(label_file)
        # Open the image
        with Image.open(image_path) as img:
            # Draw the bounding boxes
            img_with_boxes = draw_bounding_boxes(img, labels, class_map)
            # Save the image
            output_path = output_folder / (image_path.name)
            img_with_boxes.save(output_path)
            print(f"Saved image with bounding boxes to {output_path}")

def read_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"Invalid label format in {label_file}: {line}")
                continue
            cls, x_center, y_center, width, height, confidence = parts
            labels.append({
                'class_id': cls,
                'x_center': float(x_center),
                'y_center': float(y_center),
                'width': float(width),
                'height': float(height),
                'confidence': float(confidence)
            })
    return labels

def draw_bounding_boxes(img, labels, class_map):
    # Convert the image to RGBA to support transparency
    img = img.convert('RGBA')
    img_width, img_height = img.size

    # Create a transparent overlay image
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Load font (Adjust the font path if necessary)
    try:
        font = ImageFont.truetype("DroidSerif-Regular.ttf", size=26)
        legend_font = ImageFont.truetype("DroidSerif-Regular.ttf", size=36)
    except IOError:
        font = ImageFont.load_default()

    for label in labels:
        cls_id = label['class_id']
        if cls_id not in class_map:
            continue
        class_info = class_map[cls_id]
        class_name = class_info['name']
        color = class_info['color']
        # Convert normalized coordinates to absolute pixel coordinates
        x_center = label['x_center'] * img_width
        y_center = label['y_center'] * img_height
        width = label['width'] * img_width
        height = label['height'] * img_height
        confidence = label['confidence']

        # Calculate bounding box coordinates
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        # Ensure bounding box is within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)

        # Create a semi-transparent fill color
        fill_opacity = 0.161  # Opacity level (10%)
        alpha = int(255 * fill_opacity)
        fill_color = color + (alpha,)          # Semi-transparent fill color
        outline_color = color + (255,)         # Fully opaque outline color

        # Draw rounded rectangle on the overlay
        draw_rounded_rectangle(
            draw,
            [x_min, y_min, x_max, y_max],
            radius=10,
            fill=fill_color,
            outline=outline_color,
            width=3
        )

        # Prepare text
        text = f"{confidence:.2f}"

        # Use font.getbbox() to get the size of the text
        x_left, y_top, x_right, y_bottom = font.getbbox(text)
        text_width = abs(x_right - x_left)
        text_height = abs(y_bottom - y_top)

        bbox_xmid = (x_max - x_min)/2

        # text_position = (x_min + bbox_xmid - text_width/2, y_min - text_height*1.161)
        text_position = (x_min + bbox_xmid - text_width//2, y_min - text_height - 6)
        text_position_back = (x_min, y_min - text_height - 6)

        # Ensure text is within image bounds
        if text_position[1] < 0:
            text_position = (x_min, y_max + 4)

        # draw.text(
        #     text_position_back,
        #     text,
        #     fill=(0, 0, 0, 255),  # Bright white color with full opacity
        #     font=font_back
        # )

        # Define shadow offset and color
        shadow_offset = (1, 1)  # (x_offset, y_offset)
        shadow_color = (0, 0, 0, 128)  # Semi-transparent black

        # Draw shadow text on the overlay
        shadow_position = (text_position[0] + shadow_offset[0], text_position[1] + shadow_offset[1])
        draw.text(
            shadow_position,
            text,
            font=font,
            fill=color
        )

        # Draw text on the overlay
        draw.text(
            text_position,
            text,
            fill=(239, 235, 245, 255),  # Bright white color with full opacity
            font=font
        )

    # Draw the legend
    draw_legend(draw, class_map, legend_font, img_width, img_height)

    # Composite the overlay onto the original image
    img = Image.alpha_composite(img, overlay)

    # Convert back to RGB mode if desired
    return img.convert('RGB')

def draw_rounded_rectangle(draw, xy, radius=5, fill=None, outline=None, width=1):
    # Draw a rounded rectangle using PIL.ImageDraw.Draw
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

def draw_legend(draw, class_map, font, img_width, img_height, radius=10):
    # Legend position (choose one of the corners)
    legend_x = 10  # Padding from the left edge
    legend_y = 10  # Padding from the top edge

    y_text_offset = 5
    x_text_offset = 5

    # Calculate maximum text width and height
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

        # Update maximum text width and total text height
        max_text_width = max(max_text_width, text_width)
        total_text_height += text_height + 5  # Adding spacing between entries

        # Store the entry details
        entries.append({
            'text': text,
            'text_width': text_width,
            'text_height': text_height,
            'color': color
        })

    # Square size is 80% of text height
    square_size = int(0.6 * entries[-1]["text_height"])

    # Background for the legend (optional)
    legend_width =  square_size + max_text_width + 5*x_text_offset  # Padding and spacing
    legend_height = total_text_height + 3*y_text_offset  # Padding
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

        # Center the square vertically with the text
        square_offset = abs(text_height + 2*y_text_offset - square_size)/2
        square_y = current_y + square_offset

        # Draw the color square
        square_coords = [
            legend_x + x_text_offset,  # Padding from the left edge
            square_y,
            legend_x + x_text_offset + square_size,
            square_y + square_size
        ]
        draw.rounded_rectangle(square_coords, radius=radius*0.1, fill=color + (255,), outline=None)

        # Draw the text next to the square
        text_position = (legend_x + x_text_offset*3 + square_size, current_y)
        draw.text(
            text_position,
            text,
            fill=(255, 255, 255, 255),  # White color
            font=font
        )

        current_y += text_height + y_text_offset  # Move to the next entry





if __name__ == "__main__":
    main()
