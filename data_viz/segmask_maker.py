import argparse
import numpy as np

from pathlib import Path
from shapely.geometry import Polygon
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

def B_spline_smoothing(polygon):
    from scipy import interpolate

    x = [pt[0] for pt in polygon]
    y = [pt[1] for pt in polygon]

    # Compute the B-spline representation of the polygon
    # s=0 for interpolation; adjust s for smoothing
    tck, u = interpolate.splprep([x, y], s=27, per=False)
    out = interpolate.splev(u, tck)

    x_new, y_new = out[0], out[1]

    return list(zip(x_new, y_new))

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

        with Image.open(image_path) as img:

            img_with_masks = draw_segmentation_masks(img, labels, class_map)

            output_path = output_folder / (image_path.name)
            img_with_masks.save(output_path)
            print(f"Saved image with segmentation masks to {output_path}")

def polygon_area(polygon):
    n = len(polygon)
    area = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def filter_good_masks(labels, iou_threshold=0.6):
    filtered_labels = []
    used_indices = set()
    for i, label_i in enumerate(labels):
        if i in used_indices:
            continue
        cls_id_i = label_i['class_id']
        polygon_i = label_i['polygon']
        area_i = label_i['area']
        n_vertices_i = label_i['n_vertices']
        is_best_mask = True
        for j, label_j in enumerate(labels):
            if i == j or j in used_indices:
                continue
            cls_id_j = label_j['class_id']
            if cls_id_i != cls_id_j:
                continue
            polygon_j = label_j['polygon']
            area_j = label_j['area']
            n_vertices_j = label_j['n_vertices']

            intersection_area = polygon_i.intersection(polygon_j).area
            union_area = polygon_i.union(polygon_j).area
            iou = intersection_area / union_area if union_area != 0 else 0

            if iou > iou_threshold: 
                if n_vertices_i >= n_vertices_j:
                    used_indices.add(j)
                else:
                    is_best_mask = False
                    used_indices.add(i)
                    break
        if is_best_mask:
            filtered_labels.append(label_i)
    return filtered_labels


def read_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"Invalid label format in {label_file}: {line}")
                continue
            cls_id = parts[0]
            
            # Assuming the rest are x and y coordinates in pairs
            coordinates = parts[1:]
            if len(coordinates) % 2 != 0:
                print(f"Invalid number of coordinates in {label_file}: {line}")
                continue
            num_points = len(coordinates) // 2

            polygon_coords = []
            for i in range(num_points):
                x = float(coordinates[2*i])
                y = float(coordinates[2*i + 1])
                polygon_coords.append((x, y))
            
            # Create a Shapely Polygon
            polygon = Polygon(polygon_coords)
            if not polygon.is_valid:
                print(f"Invalid polygon in {label_file}: {line}")
                continue
            area = polygon.area
            n_vertices = len(polygon.exterior.coords)
            labels.append({
                'class_id': cls_id,
                'polygon': polygon,
                'area': area,
                'n_vertices': n_vertices
            })

    labels = filter_good_masks(labels)
    return labels

def draw_segmentation_masks(img, labels, class_map):
    # Convert the image to RGBA to support transparency
    img = img.convert('RGBA')
    img_width, img_height = img.size

    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("DroidSerif-Regular.ttf", size=26)
        legend_font = ImageFont.truetype("DroidSerif-Regular.ttf", size=36)
    except IOError:
        font = ImageFont.load_default()
        legend_font = font

    for label in labels:
        cls_id = label['class_id']
        if cls_id not in class_map:
            continue
        class_info = class_map[cls_id]
        color = class_info['color']

        polygon = label['polygon']

        normalized_coords = list(polygon.exterior.coords)
        
        abs_polygon = [(x * img_width, y * img_height) for x, y in normalized_coords]
        abs_polygon = B_spline_smoothing(abs_polygon)

        # Create a semi-transparent fill color
        fill_opacity = 0.2
        alpha = int(255 * fill_opacity)
        fill_color = color + (alpha,)          # Semi-transparent fill color
        outline_color = color + (255,)         # Fully opaque outline color

        draw.polygon(abs_polygon, fill=fill_color, outline=outline_color, width=2)


    draw_legend(draw, class_map, legend_font, img_width, img_height)

    img = Image.alpha_composite(img, overlay)

    return img.convert('RGB')

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

        # Update maximum text width and total text height
        max_text_width = max(max_text_width, text_width)
        total_text_height += text_height + 5  # Adding spacing between entries

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
            fill=(255, 255, 255, 255),
            font=font
        )

        current_y += text_height + y_text_offset  # Move to the next entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render segmentation masks on images.")
    parser.add_argument("images_folder", help="Path to the folder containing images.")
    parser.add_argument("labels_folder", help="Path to the folder containing label files.")
    parser.add_argument("output_folder", help="Path to the folder where output images will be saved.")
    args = parser.parse_args()
    
    main(**vars(args))
