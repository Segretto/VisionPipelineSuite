#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# ------------------------
# Utilities
# ------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

def setup_logger(verbosity: int):
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

def collect_images(root: Path) -> list[Path]:
    """Recursively collect image files from subfolders."""
    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {root}")
    files = [p for p in sorted(root.rglob("*")) if p.suffix in IMG_EXTS and p.is_file()]
    return files

def save_color_with_axes_cm(depth_m: np.ndarray, out_path: Path, ticks: int = 9, cmap_name: str = "turbo"):
    """
    Save a colorized depth with axes and a colorbar labeled in centimeters.
    We display the *metric* depth (meters) with a robust window [p1, p99],
    but the colorbar ticks/labels are in centimeters.
    """
    d_m = depth_m.copy()
    # Robust range in meters
    p1, p99 = np.nanpercentile(d_m, 1), np.nanpercentile(d_m, 99)
    # Fallback in degenerate cases
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = float(np.nanmin(d_m)), float(np.nanmax(d_m))
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        # Bail out to something viewable
        p1, p99 = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    im = ax.imshow(d_m, cmap=cmap_name, vmin=p1, vmax=p99, interpolation="nearest")
    ax.set_title("Predicted Depth")
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Build nice, dense ticks in centimeters
    ticks_m = np.linspace(p1, p99, num=max(3, ticks))  # meters
    ticks_cm = ticks_m * 100.0
    cbar.set_ticks(ticks_m)
    cbar.set_ticklabels([f"{t:.0f}" for t in ticks_cm])
    cbar.set_label("Depth (cm)", rotation=90)

    fig.tight_layout()
    fig.savefig(out_path.as_posix())
    plt.close(fig)

def save_mm_png(depth_m: np.ndarray, out_path: Path, scale_m: float):
    """Save 16-bit PNG in millimeters. (May look dark in viewers.)"""
    mm = np.clip(depth_m * scale_m * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(mm, mode="I;16").save(out_path.as_posix())

# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="VGGT depth with dataset scanning, batching, and cm colorbar")
    ap.add_argument("--input", type=Path, required=True,
                    help="Input folder containing subfolders with images (recursively scanned).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output folder to save results.")
    ap.add_argument("--model-id", type=str, default="facebook/VGGT-1B",
                    help='Model identifier or local path. Default: "facebook/VGGT-1B".')
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                    help="Inference device.")
    ap.add_argument("--scale-m", type=float, default=1.0,
                    help="Metric scale factor for depth (meters). Keep 1.0 if unknown.")
    ap.add_argument("--save-mm-png", action="store_true",
                    help="Also save 16-bit grayscale PNG (millimeters). Default: off.")
    ap.add_argument("--max-views", type=int, default=None,
                    help="Limit number of images passed to VGGT overall (useful for testing).")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Number of views to process per forward pass (chunk size).")
    ap.add_argument("--colormap", type=str, default="turbo",
                    help='Matplotlib colormap for previews (e.g., "turbo", "viridis").')
    ap.add_argument("--cb-ticks", type=int, default=9,
                    help="Number of ticks on the colorbar (in cm).")
    ap.add_argument("-v", "--verbose", action="count", default=1,
                    help="Increase logging verbosity (-v, -vv).")
    args = ap.parse_args()

    setup_logger(args.verbose)
    args.output.mkdir(parents=True, exist_ok=True)

    # Device / dtype
    device = torch.device(args.device)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    logging.info(f"Device: {device} | AMP dtype: {amp_dtype}")

    # Collect images
    image_paths = collect_images(args.input)
    if args.max_views is not None:
        image_paths = image_paths[: args.max_views]
    if len(image_paths) == 0:
        logging.error(f"No images found under {args.input}")
        return

    logging.info(f"Found {len(image_paths)} images.")
    for p in image_paths[:5]:
        logging.debug(f"  {p}")
    if len(image_paths) > 5:
        logging.debug("  ...")

    # Load model
    logging.info(f"Loading model: {args.model_id}")
    model = VGGT.from_pretrained(args.model_id).to(device).eval()

    # Preprocess ALL images into a single tensor (N,3,H,W)
    image_strs = [p.as_posix() for p in image_paths]
    logging.info("Preprocessing images...")
    images = load_and_preprocess_images(image_strs).to(device)  # (N,3,H,W)
    logging.info(f"Images tensor shape: {tuple(images.shape)}")

    N_total = images.shape[0]
    bs = max(1, args.batch_size)  # chunk size in number of views
    logging.info(f"Processing in chunks of {bs} views to avoid OOM.")

    # Iterate over chunks of views
    save_idx = 0
    t_all0 = time.perf_counter()
    for start in range(0, N_total, bs):
        end = min(start + bs, N_total)
        chunk = images[start:end]  # (n,3,H,W)
        n = chunk.shape[0]
        logging.info(f"Chunk {start:04d}:{end:04d} ({n} views)")

        with torch.no_grad():
            images_batched = chunk.unsqueeze(0)  # (B=1,n,3,H,W)
            logging.debug(f"Batched chunk shape: {tuple(images_batched.shape)}")

            # Aggregator
            t0 = time.perf_counter()
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(images_batched)
            t1 = time.perf_counter()
            logging.info(f"  Aggregator time: {t1 - t0:.3f}s")

            # Depth head
            t2 = time.perf_counter()
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batched, ps_idx)
            t3 = time.perf_counter()
            logging.info(f"  Depth head time: {t3 - t2:.3f}s | Chunk total: {t3 - t0:.3f}s")

        # Handle shape (B,n,H,W) or (B,n,H,W,1)
        if depth_map.dim() == 5 and depth_map.size(-1) == 1:
            depth_map = depth_map.squeeze(-1)
        if depth_map.dim() != 4 or depth_map.shape[0] != 1 or depth_map.shape[1] != n:
            raise RuntimeError(f"Unexpected depth_map shape for chunk: {tuple(depth_map.shape)}")

        # Save per-view outputs for this chunk
        for i in range(n):
            d_raw = depth_map[0, i].float().cpu().numpy()  # meters (up-to-scale)
            d_metric = d_raw * args.scale_m                 # meters after scaling

            d_min = float(np.nanmin(d_metric))
            d_mean = float(np.nanmean(d_metric))
            d_max = float(np.nanmax(d_metric))
            logging.info(f"  View {save_idx:03d} | depth min/mean/max (m): {d_min:.4f}/{d_mean:.4f}/{d_max:.4f}")

            base = f"depth_{save_idx:03d}"
            # Save raw metric depth as .npy
            np.save(args.output / f"{base}.npy", d_metric.astype(np.float32))
            # Save color preview with axes + colorbar in cm
            save_color_with_axes_cm(d_metric, args.output / f"{base}_color.png", ticks=args.cb_ticks, cmap_name=args.colormap)
            # Optional 16-bit grayscale in millimeters
            if args.save_mm_png:
                save_mm_png(d_raw, args.output / f"{base}_mm.png", scale_m=args.scale_m)

            save_idx += 1

    t_all1 = time.perf_counter()
    logging.info(f"Processed {save_idx} views in {t_all1 - t_all0:.3f}s")
    logging.info(f"Saved results to: {args.output.as_posix()}")

if __name__ == "__main__":
    main()
