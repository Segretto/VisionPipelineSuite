from PIL import Image, ImageDraw
import matplotlib.patches as patches


def plot_with_rounded_edges(ax, image, label=None, size=1200):
    """
    Helper function to add rounded edges and labels to a matplotlib axis.
    """
    image = image.convert("RGBA")
    image = image.resize((size, size))
    width, height = image.size

    # Create a mask for slightly rounded edges
    mask = Image.new("L", (width, height), 0)
    corner_radius = min(width, height) // 100
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width, height), radius=corner_radius, fill=255)

    # Apply mask
    image.putalpha(mask)
    ax.imshow(image)
    ax.axis("off")

    # Add label if provided
    box_size = 0.05
    if label:
        label_box = patches.FancyBboxPatch(
            (0.98 - box_size * 1.02, 0.98 - box_size * 1.02),
            box_size,
            box_size,
            boxstyle="round,pad=0.02",
            linewidth=1,
            edgecolor="black",
            facecolor="white",
            transform=ax.transAxes,
        )
        ax.add_patch(label_box)
        ax.text(
            0.98 - box_size / 2,
            0.98 - box_size / 2,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            fontweight="bold",
        )
