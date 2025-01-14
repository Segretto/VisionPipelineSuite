import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main(images_folder, labels_folder, output_folder):
    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)
    output_folder = Path(output_folder)

    if not images_folder.exists():
        print(f"Images folder {images_folder} does not exist.")
        return
    if not labels_folder.exists():
        print(f"Labels folder {labels_folder} does not exist.")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    process_images(images_folder, labels_folder, output_folder)

def process_images(images_folder, labels_folder, output_folder):
    class_map = {
        '0': {'name': 'soy', 'color': (252, 35, 97)},       
        '1': {'name': 'cotton', 'color': (7, 234, 250)}   
    }

    image_extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in images_folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {images_folder}.")
        return

    for image_path in images:
        label_file = labels_folder / (image_path.stem + '.txt')
        if not label_file.exists():
            print(f"Label file {label_file} does not exist for image {image_path.name}. Skipping this image.")
            continue

        labels = read_labels(label_file)

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load image {image_path}.")
            continue

        img_with_masks = draw_segmentation_masks(img, labels, class_map)

        output_path = output_folder / image_path.name
        cv2.imwrite(str(output_path), img_with_masks)
        print(f"Saved image with segmentation masks to {output_path}")

def read_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                print(f"Invalid label format in {label_file}: {line}")
                continue

            cls_id = parts[0]
            confidence = float(parts[-1])
            coordinates = parts[1:-1]

            if len(coordinates) % 2 != 0:
                print(f"Invalid number of coordinates in {label_file}: {line}")
                continue

            num_points = len(coordinates) // 2
            polygon_coords = []

            for i in range(num_points):
                x = float(coordinates[2 * i])
                y = float(coordinates[2 * i + 1])
                polygon_coords.append((x, y))

            labels.append({
                'class_id': cls_id,
                'polygon': polygon_coords,
                'confidence': confidence
            })

    return labels

def draw_segmentation_masks(img, labels, class_map):
    img_height, img_width, _ = img.shape
    overlay = img.copy()

    try:
        font = ImageFont.truetype("DroidSerif-Regular.ttf", size=26)
        legend_font = ImageFont.truetype("DroidSerif-Regular.ttf", size=36)
    except IOError:
        font = ImageFont.load_default(size=26)
        legend_font = ImageFont.load_default(size=36)

    for label in labels:
        cls_id = label['class_id']
        if cls_id not in class_map:
            continue

        class_info = class_map[cls_id]
        color = tuple(class_info['color'])
        confidence = label['confidence']

        # Convert normalized coordinates to absolute pixel coordinates
        abs_polygon = np.array(
            [[int(x * img_width), int(y * img_height)] for x, y in label['polygon']],
            dtype=np.int32
        )

        # Draw the mask using OpenCV
        cv2.fillPoly(overlay, [abs_polygon], color[::-1])

        # # Draw confidence value above the bounding box
        # text = f"{confidence:.2f}"
        # x_left, y_top, x_right, y_bottom = font.getbbox(text)
        # text_width = abs(x_right - x_left)
        # text_height = abs(y_bottom - y_top)

        # text_x = int(x_min + (x_max - x_min) / 2 - text_width / 2)
        # text_y = max(0, y_min - text_height - 6)

        # shadow_offset = (1, 1)
        # shadow_color = (0, 0, 0, 128)
        # cv2.putText(
        #     overlay,
        #     text,
        #     (text_x + shadow_offset[0], text_y + shadow_offset[1]),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     shadow_color,
        #     thickness=1,
        #     lineType=cv2.LINE_AA
        # )
        # cv2.putText(
        #     overlay,
        #     text,
        #     (text_x, text_y),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (255, 255, 255),
        #     thickness=1,
        #     lineType=cv2.LINE_AA
        # )

    # Blend the overlay with the original image
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
    draw = ImageDraw.Draw(pil_img)

    # Add legend
    draw_confidence_values(draw, class_map, labels, img_width, img_height, font)
    draw_legend(draw, class_map, legend_font, img_width, img_height)

    img_with_legend = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)

    return img_with_legend

def draw_confidence_values(draw, class_map, labels, img_width, img_height, font):

    for label in labels:
        cls_id = label['class_id']
        if cls_id not in class_map:
            continue
        class_info = class_map[cls_id]
    #     class_name = class_info['name']
        color = class_info['color']
    #     # Convert normalized coordinates to absolute pixel coordinates
    #     x_center = label['x_center'] * img_width
    #     y_center = label['y_center'] * img_height
    #     width = label['width'] * img_width
    #     height = label['height'] * img_height
        confidence = label['confidence']

    #     # Calculate bounding box coordinates
    #     x_min = x_center - width / 2
    #     y_min = y_center - height / 2
    #     x_max = x_center + width / 2
    #     y_max = y_center + height / 2

        abs_polygon = np.array(
            [[int(x * img_width), int(y * img_height)] for x, y in label['polygon']],
            dtype=np.int32
        )

        # Calculate bounding box for the polygon
        x_min = np.min(abs_polygon[:, 0])
        y_min = np.min(abs_polygon[:, 1])
        x_max = np.max(abs_polygon[:, 0])
        y_max = np.max(abs_polygon[:, 1])


        # Prepare text
        text = f"{confidence:.2f}"

        # Use font.getbbox() to get the size of the text
        x_left, y_top, x_right, y_bottom = font.getbbox(text)
        text_width = abs(x_right - x_left)
        text_height = abs(y_bottom - y_top)

        bbox_xmid = (x_max - x_min)/2
        bbox_ymid = (y_max - y_min)/2

        # text_position = (x_min + bbox_xmid - text_width/2, y_min - text_height*1.161)
        text_position = (x_min + bbox_xmid - text_width//2, y_min + bbox_ymid - text_height//2)
        text_position_back = (x_min, y_min - text_height - 6)

        # Ensure text is within image bounds
        if text_position[1] < 0:
            text_position = (x_min, y_max + 4)

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

    # img_with_legend = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
    # return img_with_legend

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render segmentation masks on images using OpenCV.")
    parser.add_argument("images_folder", help="Path to the folder containing images.")
    parser.add_argument("labels_folder", help="Path to the folder containing label files.")
    parser.add_argument("output_folder", help="Path to the folder where output images will be saved.")
    args = parser.parse_args()

    main(**vars(args))
