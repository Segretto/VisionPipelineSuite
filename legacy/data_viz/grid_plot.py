#%%
from pathlib import Path
import shutil
import pandas as pd
import string
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

#%% Helper function to add rounded edges and labels
def plot_with_rounded_edges(ax, image, label=None, size=1200):
    image = image.convert("RGBA")
    image = image.resize((size, size))  # Resize image to 1200x1200
    width, height = image.size

    # Create a mask for slightly rounded edges
    mask = Image.new("L", (width, height), 0)
    corner_radius = min(width, height) // 100  # Slightly rounded corners
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width, height), radius=corner_radius, fill=255)

    # Apply mask
    image.putalpha(mask)
    ax.imshow(image)
    ax.axis("off")

    # Add label if provided
    # box_scale = 0.8
    box_size = 0.05
    if label:
        label_box = patches.FancyBboxPatch(
            (0.98 - box_size*1.02,  0.98 - box_size*1.02), box_size, box_size, boxstyle="round,pad=0.02",
            linewidth=1, edgecolor="black", facecolor="white", transform=ax.transAxes
        )
        ax.add_patch(label_box)
        ax.text(0.98 - box_size/2, 0.98 - box_size/2, label, transform=ax.transAxes, ha="center", va="center", fontsize=10, color="black", fontweight="bold")

#%% Function to parse CSV and organize images using pandas
def parse_csv_and_organize_images(csv_file, image_paths):
    df = pd.read_csv(csv_file, delim_whitespace=True, dtype=str)
    image_dict = {"early": {}, "normal": {}, "dense": {}}
    
    for category in image_dict.keys():
        for image_id in df[category].dropna():
            image_id = image_id.zfill(5)  # Ensure IDs are zero-padded
            image_dict[category][image_id] = {
                "PLAIN": Path(image_paths["PLAIN"]) / f"{image_id}.jpeg",
                "ODGT": Path(image_paths["ODGT"]) / f"{image_id}.jpeg",
                "ODPRED": Path(image_paths["ODPRED"]) / f"{image_id}.jpeg",
                "SEGGT": Path(image_paths["SEGGT"]) / f"{image_id}.jpeg",
                "SEGPRED": Path(image_paths["SEGPRED"]) / f"{image_id}.jpeg"
            }
    return image_dict

#%% Function to copy and resize images to output folder
def save_images_to_output(image_dict, output_folder, size=1200):
    output_folder = Path(output_folder)
    
    for category, images in image_dict.items():
        category_folder = output_folder / category
        category_folder.mkdir(parents=True, exist_ok=True)
        
        for image_id, image_files in images.items():
            for tag, image_path in image_files.items():
                tag_folder = output_folder / tag
                tag_folder.mkdir(parents=True, exist_ok=True)

                if image_path.exists():
                    new_image_name = f"{image_id}-{tag}.jpeg"
                    image = Image.open(image_path).resize((size, size))  # Resize to 1200x1200
                    image.save(category_folder / new_image_name)
                    image.save(tag_folder/ new_image_name)

#%% Example Usage
# Paths
plain_path = Path("/home/segreto/Documents/Data/SoyCotton-FinalSplit-OD/images/test")
od_gt_path = Path("/home/segreto/Documents/Analytics/SoyCotton-OD/gt")
od_pred_path = Path("/home/segreto/Documents/Analytics/SoyCotton-OD/bboxes")
seg_gt_path = Path("/home/segreto/Documents/Analytics/SoyCotton-SEG/gt")
seg_pred_path = Path("/home/segreto/Documents/Analytics/SoyCotton-SEG/segmasks")

csv_file = Path("/home/segreto/Documents/Papers/SoyCotton-Data/selected_images.txt")
output_folder = Path("/home/segreto/Documents/Papers/SoyCotton-Data/output")
results_folder = Path("/home/segreto/Documents/Papers/SoyCotton-Data/results")

image_paths = {
    "PLAIN": plain_path,
    "ODGT": od_gt_path,
    "ODPRED": od_pred_path,
    "SEGGT": seg_gt_path,
    "SEGPRED": seg_pred_path
}

# Parse CSV and organize images
image_dict = parse_csv_and_organize_images(csv_file, image_paths)

results_folder.mkdir(parents=True, exist_ok=True)

# Save images to output folder
save_images_to_output(image_dict, output_folder)



#%% Function to plot 3x3 grid for normal images
def plot_3x3_grid(image_dict):
    alphabet = string.ascii_lowercase
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), gridspec_kw={"wspace": 0.01, "hspace": 0.01})
    
    for row, category in enumerate(["early", "normal", "dense"]):
        images = list(image_dict[category].keys())[:3]  # Get first 3 images per category
        
        for col in range(3):
            ax = axes[row, col]
            if col < len(images):
                image_path = image_dict[category][images[col]]["PLAIN"]
                image = Image.open(image_path)
                label = alphabet[3*row + col]  # Add label for all rows
                plot_with_rounded_edges(ax, image, label=label)
            else:
                ax.axis("off")
    plt.tight_layout()
    plt.show()

    fig.savefig(results_folder / "3x3grid.png", dpi=300)

#%% Cell 3: Plot 3x3 grid
plot_3x3_grid(image_dict)



#%% Function to plot 2x3 grid for bounding boxes and segmentation masks
def plot_category_grids(image_dict, category, size=1200):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"wspace": 0.01, "hspace": 0.01})
    images = list(image_dict[category].keys())[:3]
    
    # Plot OD images in the first row
    for col in range(3):
        ax = axes[0, col]
        if col < len(images):
            image_path = image_dict[category][images[col]]["ODPRED"]
            image = Image.open(image_path).resize((size, size))
            plot_with_rounded_edges(ax, image)
        else:
            ax.axis("off")

    # Plot SEG images in the second row
    for col in range(3):
        ax = axes[1, col]
        if col < len(images):
            image_path = image_dict[category][images[col]]["SEGPRED"]
            image = Image.open(image_path).resize((size, size))
            plot_with_rounded_edges(ax, image)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

    fig.savefig(results_folder / f"{category}.png", dpi=300)

    
#%% Cell 4: Plot 2x3 grid for "early"
plot_category_grids(image_dict, "early", 1200)

#%% Cell 5: Plot 2x3 grid for "normal"
plot_category_grids(image_dict, "normal", 1200)

#%% Cell 6: Plot 2x3 grid for "dense"
plot_category_grids(image_dict, "dense", 1200)

#%%
def plot_segmentation_comparison(image_dict, category, id=0, size=1200, mode="OD"):
    image_id = list(image_dict[category].keys())[id]  # Select the first image in the category
    fig, axes = plt.subplots(1, 2, figsize=(14, 12))
    
    # Plot ground truth segmentation
    ax = axes[0]
    image_path = image_dict[category][image_id][f"{mode}GT"]
    image = Image.open(image_path)
    plot_with_rounded_edges(ax, image, label="GT", size=size)
    
    # Plot predicted segmentation
    ax = axes[1]
    image_path = image_dict[category][image_id][f"{mode}PRED"]
    image = Image.open(image_path)
    plot_with_rounded_edges(ax, image, label="Pred", size=size)
    
    plt.tight_layout()
    plt.show()

    fig.savefig(results_folder / f"{mode.lower()}_comp.png", dpi=300)

#%%
plot_segmentation_comparison(image_dict, "normal", 1, 1200, mode="OD")

# %%
def plot_custom(image_dict, size=1200):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"wspace": 0.01, "hspace": 0.01})
    
    img1 = list(image_dict["early"].keys())[1]
    img2 = list(image_dict["normal"].keys())[0]
    img3 = list(image_dict["dense"].keys())[1]

    images = [
        image_dict["early"][img1],
        image_dict["normal"][img2],
        image_dict["dense"][img3]
    ]
    # Plot OD images in the first row
    for col in range(3):
        ax = axes[0, col]
        if col < len(images):
            image_path = images[col]["ODPRED"]
            image = Image.open(image_path).resize((size, size))
            plot_with_rounded_edges(ax, image)
        else:
            ax.axis("off")

    # Plot SEG images in the second row
    for col in range(3):
        ax = axes[1, col]
        if col < len(images):
            image_path = images[col]["SEGPRED"]
            image = Image.open(image_path).resize((size, size))
            plot_with_rounded_edges(ax, image)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

    fig.savefig(results_folder / "seg&box.png", dpi=300)

#%%
plot_custom(image_dict, 1600)
# %%
