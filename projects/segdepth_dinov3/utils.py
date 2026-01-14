import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --------- Depth Utils ---------
def depth_to_colormap(
    depth_m: np.ndarray,
    dmin: float | None = None,
    dmax: float | None = None,
    colormap: int = cv2.COLORMAP_INFERNO,
    invert: bool = True,
    bgr: bool = True
) -> np.ndarray:
    """
    Convert an HxW depth map in meters to a color (BGR) visualization.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    assert depth.ndim == 2, "depth_m must be HxW"

    valid = np.isfinite(depth) & (depth > 0)

    # Choose visualization range
    if dmin is None or dmax is None:
        if np.any(valid):
            vals = depth[valid]
            if dmin is None:
                dmin = float(np.percentile(vals, 5))
            if dmax is None:
                dmax = float(np.percentile(vals, 95))
        else:
            dmin, dmax = 0.1, 1.0  # fallback if nothing valid
    if dmax <= dmin:
        dmax = dmin + 1e-6

    # Normalize to 0..255 for applyColorMap
    norm = (depth - dmin) / (dmax - dmin)
    norm = np.clip(norm, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    gray8 = (norm * 255.0).astype(np.uint8)

    # Colorize (BGR)
    color_bgr = cv2.applyColorMap(gray8, colormap)

    # Paint invalid pixels black
    if not np.all(valid):
        color_bgr[~valid] = (0, 0, 0)

    # Return in requested channel order
    if bgr:
        return color_bgr
    else:
        # Convert to RGB by swapping channels
        return color_bgr[:, :, ::-1].copy()

# --------- Segmentation Utils ---------
def outputs_to_maps(semantic_logits, image_size):
    """
    Convert model outputs at full resolution to semantic + instance maps.
    """

    # add batch dim if missing
    if semantic_logits.dim() == 3:
        semantic_logits = semantic_logits.unsqueeze(0)  # 1,C,H,W

    H_img, W_img = image_size

    # Upsample logits to image resolution
    semantic_logits_up = F.interpolate(semantic_logits, size=(H_img, W_img), mode='bilinear', align_corners=False)[0]  # C,H,W

    # semantic prediction
    semantic_prob = F.softmax(semantic_logits_up.unsqueeze(0), dim=1)[0]  # (C,H,W) tensor
    semantic_pred = torch.argmax(semantic_prob, dim=0).cpu().numpy().astype(np.int32)  # H,W

    return semantic_pred


def generate_segmentation_overlay(
    image,
    semantic_map,
    class_names=None,
    alpha=0.6,
    background_index=0,
    seed=42,
    draw_semantic_labels=True,
    semantic_label_fontsize=10
):
    """
    Generate a segmentation overlay image with optional labels drawn.
    """

    # Normalize image to uint8
    img = np.array(image)
    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            img_u8 = (img * 255).astype(np.uint8)
        else:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img.copy()

    H, W = semantic_map.shape[:2]

    # Palette for colors
    rng = np.random.RandomState(seed)
    n_colors = max(100, int(np.max(semantic_map) + 1))
    palette = (rng.randint(0, 256, size=(n_colors, 3))).astype(np.uint8)

    # Helper: overlay one mask
    def overlay(base, mask, color, alpha):
        out = base.astype(np.float32).copy()
        color_f = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        m3 = np.stack([mask]*3, axis=-1).astype(np.float32)
        out = out * (1 - alpha * m3) + color_f * (alpha * m3)
        return out.clip(0, 255).astype(np.uint8)

    # Apply overlays for all classes
    out_sem = img_u8.copy()
    unique_c = np.unique(semantic_map)
    label_info = []  # store (cls, y0, x0) for labels

    for cls in unique_c:
        if cls < 0 or cls == background_index:
            continue
        mask = (semantic_map == int(cls))
        if mask.sum() == 0:
            continue

        color = palette[int(cls) % len(palette)]
        out_sem = overlay(out_sem, mask, color, alpha)

        # Save centroid info for later labeling
        ys, xs = np.nonzero(mask)
        if ys.size > 0:
            y0 = int(np.mean(ys))
            x0 = int(np.mean(xs))
            label_info.append((cls, y0, x0))

    # Draw labels at the end, after all overlays
    if draw_semantic_labels:
        for cls, y0, x0 in label_info:
            label = class_names[cls] if (class_names is not None and cls < len(class_names)) else str(cls)
            font_scale = semantic_label_fontsize / 10
            thickness = max(1, semantic_label_fontsize // 4)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # Draw black background rectangle
            cv2.rectangle(out_sem, (x0 - text_w//2 - 2, y0 - text_h//2 - 2),
                          (x0 + text_w//2 + 2, y0 + text_h//2 + 2), (0, 0, 0), -1)
            # Draw white text
            cv2.putText(out_sem, label, (x0 - text_w//2, y0 + text_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out_sem

def visualize_combined(image, depth_map, semantic_map, class_names=None, figsize=(18, 6)):
    """
    Visualize original image, depth map, and semantic map side-by-side.
    """
    depth_color = depth_to_colormap(depth_map, bgr=False)
    
    seg_overlay = generate_segmentation_overlay(
        image, semantic_map, class_names=class_names, alpha=0.6,
        draw_semantic_labels=True, semantic_label_fontsize=10, background_index=0
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(depth_color)
    axes[1].set_title("Depth Estimation")
    axes[1].axis('off')

    axes[2].imshow(seg_overlay)
    axes[2].set_title("Semantic Segmentation")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show() # Note: In this environment, plt.show() might not work interactively, output to file often better.

def save_combined_visualization(image, depth_map, semantic_map, output_path, class_names=None, figsize=(18, 6)):
    """
    Visualize original image, depth map, and semantic map side-by-side and save to file.
    """
    depth_color = depth_to_colormap(depth_map, bgr=False)
    
    seg_overlay = generate_segmentation_overlay(
        image, semantic_map, class_names=class_names, alpha=0.6,
        draw_semantic_labels=True, semantic_label_fontsize=10, background_index=0
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(depth_color)
    axes[1].set_title("Depth Estimation")
    axes[1].axis('off')

    axes[2].imshow(seg_overlay)
    axes[2].set_title("Semantic Segmentation")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path

