#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 Frame Annotator (PyQt6 + Matplotlib) — extended UX
- Load folder of frames (sorted)
- Choose SAM2 model config + checkpoint
- Click LMB = positive, RMB = negative (click near an existing point (<5 px) to delete it)
- Side panel: manage labels/classes (persistent via config JSON)
- New column: lists masks present on the current frame; select & Delete
- Propagate N frames; Play preview
- Save to COCO (writes images_dir/coco.json + <script_dir>/annotations/<images_folder>.json)
- Prompt to save on exit

Requirements:
  pip install pyqt6 matplotlib pillow numpy torch pycocotools
  git clone https://github.com/facebookresearch/sam2 && pip install -e ./sam2

Note:
  Expects a JPEG sequence "video" with names 00000.jpg, 00001.jpg, ...
  If you pass a folder of arbitrary images, add a shim to convert/rename (as discussed earlier).
"""
import argparse
import sys
import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import tempfile
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import torch
from pycocotools import mask as maskUtils

from sam2.build_sam import build_sam2_video_predictor  # SAM2 video (images-as-video) API


# ----------------------------- helpers --------------------------------
def script_dir() -> Path:
    return Path(__file__).resolve().parent

def device_select():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        # perf knobs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return dev
    return torch.device("cpu")

def load_frames(folder: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    frames = [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]
    frames.sort(key=lambda p: int(p.stem))
    if not frames:
        raise FileNotFoundError(f"No JPEG sequence found in {folder} (00000.jpg …)")
    return frames

def prepare_jpeg_sequence(src_dir: Path) -> Path:
    """Create a temp dir with files named 00000.jpg ... for SAM2."""
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG",".JPEG",".PNG"}
    imgs = sorted([p for p in src_dir.iterdir() if p.suffix in exts])
    if not imgs:
        raise RuntimeError(f"No images found in {src_dir}")
    tmp = Path(tempfile.mkdtemp(prefix="sam2_seq_"))
    for i, p in enumerate(imgs):
        out = tmp / f"{i:05d}.jpg"
        if p.suffix.lower() in [".jpg",".jpeg"]:
            # fast path: hardlink or copy
            try:
                os.link(p, out)
            except OSError:
                shutil.copy2(p, out)
        else:
            Image.open(p).convert("RGB").save(out, "JPEG", quality=92)
    return tmp

# simple color palette
def color_rgba(idx: int, alpha: float = 0.5) -> Tuple[float,float,float,float]:
    base = [
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
        (0.89, 0.47, 0.76),
        (0.50, 0.50, 0.50),
        (0.74, 0.74, 0.13),
        (0.09, 0.75, 0.81),
    ]
    r,g,b = base[idx % len(base)]
    return (r,g,b,alpha)


@dataclass
class LabelEntry:
    name: str
    obj_id: int  # SAM2 object id

# ----------------------- Matplotlib canvas -----------------------------
class ImageCanvas(FigureCanvasQTAgg):
    clicked = QtCore.pyqtSignal(float, float, int)  # x, y, button (1=LMB, 2=RMB)

    def __init__(self):
        self.fig = Figure(figsize=(7, 5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.img_artist = None
        self.masks_artists = []  # list of (artist, obj_id)
        self.points_artists = [] # list of artists for clicks
        self._image_w = None
        self._image_h = None
        self.mpl_connect("button_press_event", self._on_click)

    def show_image(self, pil_img: Image.Image):
        self.ax.clear()
        self.ax.set_axis_off()
        # reset artist lists to avoid stale .remove() calls
        self.masks_artists = []
        self.points_artists = []

        self.img_artist = self.ax.imshow(pil_img)
        self._image_w, self._image_h = pil_img.size
        self.fig.tight_layout()
        self.draw_idle()

    def draw_mask_overlay(self, masks: Dict[int, np.ndarray]):
        # masks: obj_id -> HxW (bool/0-1)
        for a, _ in self.masks_artists:
            try:
                if getattr(a, "axes", None) is not None:
                    a.remove()
            except Exception:
                pass
        self.masks_artists.clear()
        for i, (obj_id, m) in enumerate(masks.items()):
            h, w = m.shape[-2:]
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            r,g,b,a = color_rgba(i)
            rgba[..., 0] = r
            rgba[..., 1] = g
            rgba[..., 2] = b
            rgba[..., 3] = (m.astype(np.float32) * a)
            artist = self.ax.imshow(rgba)
            self.masks_artists.append((artist, obj_id))
        self.draw_idle()

    def draw_points(self, pos_pts: List[Tuple[float,float]], neg_pts: List[Tuple[float,float]]):
        # safely remove old point artists
        for a in self.points_artists:
            try:
                if getattr(a, "axes", None) is not None:
                    a.remove()
            except Exception:
                pass
        self.points_artists.clear()

        if pos_pts:
            xs, ys = zip(*pos_pts)
            a1 = self.ax.scatter(xs, ys, c="lime", marker="*", s=200, edgecolor="white", linewidths=1.25)
            self.points_artists.append(a1)
        if neg_pts:
            xs, ys = zip(*neg_pts)
            a2 = self.ax.scatter(xs, ys, c="red", marker="*", s=200, edgecolor="white", linewidths=1.25)
            self.points_artists.append(a2)
        self.draw_idle()

    def _on_click(self, ev):
        if ev.xdata is None or ev.ydata is None:
            return
        btn = 1 if ev.button == 1 else (2 if ev.button == 3 else 0)
        if btn:
            self.clicked.emit(float(ev.xdata), float(ev.ydata), btn)

# --------------------------- Main Window -------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("SAM2 Annotator (Frames)")
        self.resize(1400, 860)

        # state
        self.args = args
        self.images_dir = prepare_jpeg_sequence(Path(args.images)).resolve()
        self.frames = load_frames(self.images_dir)
        self.frame_idx = 0
        self.pos_points: List[Tuple[float,float]] = []
        self.neg_points: List[Tuple[float,float]] = []
        self.video_segments: Dict[int, Dict[int, np.ndarray]] = {}  # frame_idx -> {obj_id: mask}
        self.labels: List[LabelEntry] = []  # label set (global)
        self.next_obj_id = 1
        self.current_label_obj: Optional[int] = None
        self._dirty = False  # whether there are unsaved changes

        # dirs/files for configs/annotations
        self.config_dir = (Path(args.config_dir) if args.config_dir else script_dir()/ "config").resolve()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.labels_cfg_path = self.config_dir / f"labels_{self.images_dir.name}.json"
        self.annotations_dir = script_dir() / "annotations"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        self._load_labels_config()

        # predictor
        self.device = device_select()
        self.predictor = build_sam2_video_predictor(args.model_config, args.checkpoint, device=self.device)
        # initialize with the frames directory as a "video"
        self.infer_state = self.predictor.init_state(video_path=str(self.images_dir))
        self.predictor.reset_state(self.infer_state)

        # --- classes (labels) ---
        self.labels: List[dict] = []      # [{id, name, color:(r,g,b)}]
        self.next_label_id = 1
        self.selected_label_id: Optional[int] = None  # class you’ll create with

        # --- object instances ---
        self.instances: Dict[int, dict] = {}  # instance_id -> {label_id, color}
        self.next_instance_id = 1
        self.active_instance_id: Optional[int] = None  # which instance to refine

        # per-frame masks: frame_idx -> {instance_id: mask}
        self.video_segments: Dict[int, Dict[int, np.ndarray]] = {}

        self._dirty = False


        # ---------------- UI ----------------
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        # left: image canvas
        self.canvas = ImageCanvas()
        self.canvas.clicked.connect(self.on_canvas_click)
        layout.addWidget(self.canvas, stretch=10)

        # middle: frame annotation list (masks on current frame)
        mid_panel = QtWidgets.QFrame()
        mid_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        mv = QtWidgets.QVBoxLayout(mid_panel)
        mv.addWidget(QtWidgets.QLabel("Frame Annotations"))
        self.frame_masks_list = QtWidgets.QListWidget()
        mv.addWidget(self.frame_masks_list, stretch=1)
        self.del_mask_btn = QtWidgets.QPushButton("Delete selected mask from this frame")
        mv.addWidget(self.del_mask_btn)
        layout.addWidget(mid_panel, stretch=3)

        # right: controls panel
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        v = QtWidgets.QVBoxLayout(panel)

        # model info
        self.model_lbl = QtWidgets.QLabel(
            f"Model: {Path(args.model_config).name}\nCheckpoint: {Path(args.checkpoint).name}"
        )
        self.model_lbl.setWordWrap(True)
        v.addWidget(self.model_lbl)

        # frame nav
        nav_box = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("⟵ Prev")
        self.next_btn = QtWidgets.QPushButton("Next ⟶")
        self.goto_edit = QtWidgets.QLineEdit("0")
        self.goto_btn = QtWidgets.QPushButton("Go")
        nav_box.addWidget(self.prev_btn)
        nav_box.addWidget(self.next_btn)
        nav_box.addWidget(QtWidgets.QLabel("Frame:"))
        nav_box.addWidget(self.goto_edit)
        nav_box.addWidget(self.goto_btn)
        v.addLayout(nav_box)

        # points/mask actions
        act_box = QtWidgets.QHBoxLayout()
        self.clear_pts_btn = QtWidgets.QPushButton("Clear points")
        self.apply_pts_btn = QtWidgets.QPushButton("Apply points → mask")
        act_box.addWidget(self.clear_pts_btn)
        act_box.addWidget(self.apply_pts_btn)
        v.addLayout(act_box)

        # labels list
        v.addWidget(QtWidgets.QLabel("Labels / Classes (persistent)"))
        self.labels_list = QtWidgets.QListWidget()
        self.labels_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        v.addWidget(self.labels_list, stretch=1)

        add_lbl_box = QtWidgets.QHBoxLayout()
        self.new_label_edit = QtWidgets.QLineEdit()
        self.add_label_btn = QtWidgets.QPushButton("Add label")
        add_lbl_box.addWidget(self.new_label_edit)
        add_lbl_box.addWidget(self.add_label_btn)
        v.addLayout(add_lbl_box)

        self.assign_label_btn = QtWidgets.QPushButton("Use selected label for next clicks (obj_id)")
        v.addWidget(self.assign_label_btn)

        # propagation
        v.addWidget(QtWidgets.QLabel("Propagate to next N frames"))
        prop_box = QtWidgets.QHBoxLayout()
        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setRange(1, 9999)
        self.n_spin.setValue(10)
        self.prop_btn = QtWidgets.QPushButton("Propagate")
        self.play_btn = QtWidgets.QPushButton("Play")
        prop_box.addWidget(self.n_spin)
        prop_box.addWidget(self.prop_btn)
        prop_box.addWidget(self.play_btn)
        v.addLayout(prop_box)

        # save to COCO
        self.save_coco_btn = QtWidgets.QPushButton("💾 Save annotations to COCO")
        v.addWidget(self.save_coco_btn)

        v.addStretch(1)
        layout.addWidget(panel, stretch=4)
        self.setCentralWidget(central)

        # wire handlers
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)
        self.goto_btn.clicked.connect(self.on_goto)
        self.clear_pts_btn.clicked.connect(self.on_clear_points)
        self.apply_pts_btn.clicked.connect(self.on_apply_points)
        self.add_label_btn.clicked.connect(self.on_add_label)
        self.assign_label_btn.clicked.connect(self.on_assign_label)
        self.play_btn.clicked.connect(self.on_play)
        self.prop_btn.clicked.connect(self.on_propagate)
        self.del_mask_btn.clicked.connect(self.on_delete_selected_mask)
        self.save_coco_btn.clicked.connect(self.on_save_coco)

        self._refresh_labels_ui()
        self.redraw()

    # ----------------------- config I/O -----------------------
    def _load_labels_config(self):
        if self.labels_cfg_path.exists():
            try:
                data = json.loads(self.labels_cfg_path.read_text())
                self.next_obj_id = int(data.get("next_obj_id", 1))
                self.labels = [LabelEntry(name=e["name"], obj_id=int(e["obj_id"])) for e in data.get("labels", [])]
            except Exception:
                # start clean if malformed
                self.next_obj_id = 1
                self.labels = []

    def _save_labels_config(self):
        data = {
            "next_obj_id": self.next_obj_id,
            "labels": [{"name": e.name, "obj_id": e.obj_id} for e in self.labels],
        }
        self.labels_cfg_path.write_text(json.dumps(data, indent=2))

    # ----------------------- UI actions -----------------------
    def on_prev(self):
        self.frame_idx = max(0, self.frame_idx - 1)
        self.redraw()

    def on_next(self):
        self.frame_idx = min(len(self.frames) - 1, self.frame_idx + 1)
        self.redraw()

    def on_goto(self):
        try:
            idx = int(self.goto_edit.text())
        except ValueError:
            return
        self.frame_idx = int(np.clip(idx, 0, len(self.frames) - 1))
        self.redraw()

    def on_clear_points(self):
        self.pos_points.clear()
        self.neg_points.clear()
        self.redraw_points()

    def on_add_label(self):
        name = self.new_label_edit.text().strip()
        if not name:
            return
        entry = LabelEntry(name=name, obj_id=self.next_obj_id)
        self.next_obj_id += 1
        self.labels.append(entry)
        self.new_label_edit.clear()
        self._save_labels_config()
        self._refresh_labels_ui()

    def on_assign_label(self):
        row = self.labels_list.currentRow()
        if row < 0:
            return
        self.current_label_obj = self.labels[row].obj_id
        QtWidgets.QMessageBox.information(
            self, "Label selected",
            f"Using object id [{self.current_label_obj}] for next clicks."
        )

    def on_delete_selected_mask(self):
        row = self.frame_masks_list.currentRow()
        if row < 0:
            return
        # item text format: "[obj_id] label_name" or fallback
        item = self.frame_masks_list.item(row).text()
        try:
            obj_id = int(item.split("]")[0].strip("["))
        except Exception:
            return
        # delete only from current frame
        if self.frame_idx in self.video_segments and obj_id in self.video_segments[self.frame_idx]:
            del self.video_segments[self.frame_idx][obj_id]
            if not self.video_segments[self.frame_idx]:
                del self.video_segments[self.frame_idx]
            self._dirty = True
            self.redraw()

    def on_canvas_click(self, x: float, y: float, button: int):
        # smart delete if near an existing point (within 5 px)
        def nearest_idx(pts: List[Tuple[float,float]], x, y, tol=5.0):
            if not pts:
                return -1
            arr = np.asarray(pts, dtype=np.float32)
            d = np.sqrt((arr[:,0]-x)**2 + (arr[:,1]-y)**2)
            j = int(np.argmin(d))
            return j if d[j] <= tol else -1

        if button == 1:   # LMB => positive
            j = nearest_idx(self.pos_points, x, y)
            if j >= 0:
                self.pos_points.pop(j)
            else:
                self.pos_points.append((x, y))
        elif button == 2: # RMB => negative
            j = nearest_idx(self.neg_points, x, y)
            if j >= 0:
                self.neg_points.pop(j)
            else:
                self.neg_points.append((x, y))
        self.redraw_points()

    def on_apply_points(self):
        if self.current_label_obj is None:
            QtWidgets.QMessageBox.warning(self, "No label",
                "Select/assign a label (obj_id) first.")
            return
        if not (self.pos_points or self.neg_points):
            return

        pts = np.array(self.pos_points + self.neg_points, dtype=np.float32)
        labs = np.array([1]*len(self.pos_points) + [0]*len(self.neg_points), dtype=np.int32)
        try:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=self.infer_state,
                frame_idx=int(self.frame_idx),
                obj_id=int(self.current_label_obj),
                points=pts,
                labels=labs,
            )
        except TypeError:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                self.infer_state, int(self.frame_idx), int(self.current_label_obj), pts, labs
            )

        mask_bin = (out_mask_logits[0] > 0.0).detach().cpu().numpy()
        self.video_segments.setdefault(self.frame_idx, {})[int(out_obj_ids[0])] = mask_bin
        self.pos_points.clear()
        self.neg_points.clear()
        self._dirty = True
        self.redraw()

    def on_propagate(self):
        limit = self.n_spin.value()
        start = self.frame_idx
        end = min(len(self.frames)-1, start + limit)

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.infer_state):
            if out_frame_idx > end:
                break
            masks = {
                int(out_obj_ids[i]): (out_mask_logits[i] > 0.0).detach().cpu().numpy()
                for i in range(len(out_obj_ids))
            }
            self.video_segments[out_frame_idx] = masks
        self._dirty = True
        self.redraw()

    def on_play(self):
        limit = self.n_spin.value()
        start = self.frame_idx
        end = min(len(self.frames)-1, start + limit)
        for f in range(start, end + 1):
            self.frame_idx = f
            self.redraw()
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(60)

    def on_save_coco(self):
        coco = self._build_coco()
        # 1) images_dir/coco.json
        out_main = self.images_dir / "coco.json"
        out_main.write_text(json.dumps(coco, indent=2))
        # 2) <script_dir>/annotations/<images_dir_stem>.json
        out_copy = self.annotations_dir / f"{self.images_dir.name}.json"
        shutil.copy2(out_main, out_copy)
        self._dirty = False
        QtWidgets.QMessageBox.information(self, "Saved", f"COCO saved:\n- {out_main}\n- {out_copy}")

    # ------------------------ drawing -------------------------
    def redraw(self):
        img = Image.open(self.frames[self.frame_idx]).convert("RGB")
        self.canvas.show_image(img)

        masks = self.video_segments.get(self.frame_idx, {})
        self.canvas.draw_mask_overlay(masks)
        self.redraw_points()

        self.goto_edit.setText(str(self.frame_idx))
        self._refresh_frame_masks_ui()

    def redraw_points(self):
        self.canvas.draw_points(self.pos_points, self.neg_points)

    # -------------------- UI helpers --------------------------
    def _refresh_labels_ui(self):
        self.labels_list.clear()
        for e in self.labels:
            self.labels_list.addItem(f"[{e.obj_id}] {e.name}")

    def _refresh_frame_masks_ui(self):
        self.frame_masks_list.clear()
        masks = self.video_segments.get(self.frame_idx, {})
        # build a map obj_id->name
        name_for = {e.obj_id: e.name for e in self.labels}
        for obj_id in sorted(masks.keys()):
            name = name_for.get(obj_id, f"obj_{obj_id}")
            self.frame_masks_list.addItem(f"[{obj_id}] {name}")

    # -------------------- COCO builder ------------------------
    def _build_coco(self) -> dict:
        """
        Build a minimal COCO (instances) dict from current masks.
        One image per frame; categories from global labels; annotations per frame.
        """
        # images
        images = []
        id_from_frame = {}
        for i, p in enumerate(self.frames, start=1):
            images.append({
                "id": i,
                "file_name": p.name,
                "width": Image.open(p).width,
                "height": Image.open(p).height,
            })
            id_from_frame[int(p.stem)] = i  # stem is zero-padded index

        # categories from labels list; map obj_id->category_id (stable by index)
        categories = []
        obj_to_cat = {}
        for cat_id, e in enumerate(self.labels, start=1):
            categories.append({
                "id": cat_id,
                "name": e.name,
                "supercategory": "object",
            })
            obj_to_cat[e.obj_id] = cat_id

        # annotations (RLE)
        annotations = []
        ann_id = 1
        for frame_idx, masks in self.video_segments.items():
            image_id = id_from_frame.get(frame_idx)
            if image_id is None:
                continue
            for obj_id, m in masks.items():
                if m.dtype != np.uint8:
                    m = (m > 0).astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(m))
                # pycocotools wants ascii for json
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("ascii")
                area = float(maskUtils.area(rle))
                bbox = [float(x) for x in maskUtils.toBbox(rle).tolist()]
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": obj_to_cat.get(obj_id, 1),
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                ann_id += 1

        return {
            "info": {"description": "SAM2 Annotator export"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    # -------------------- window close ------------------------
    def closeEvent(self, event: QtGui.QCloseEvent):
        # persist label set
        self._save_labels_config()
        # prompt to save coco if dirty
        if self._dirty:
            res = QtWidgets.QMessageBox.question(
                self, "Save COCO?",
                "You have unsaved annotation changes.\nSave COCO before exiting?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No | QtWidgets.QMessageBox.StandardButton.Cancel
            )
            if res == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if res == QtWidgets.QMessageBox.StandardButton.Yes:
                try:
                    self.on_save_coco()
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
        event.accept()

# ----------------------------- main -----------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="SAM2 Frame Annotator")
    ap.add_argument("--images", required=True, help="Folder with JPEG sequence (00000.jpg...)")
    ap.add_argument("--model-config", required=True, help="Path to SAM2 model yaml (e.g., sam2.1_hiera_l.yaml)")
    ap.add_argument("--checkpoint", required=True, help="Path to SAM2 checkpoint (e.g., sam2.1_hiera_large.pt)")
    ap.add_argument("--config-dir", default=str(script_dir()/ "config"),
                    help="Folder to store persistent configs (default: ./config near this script)")
    return ap.parse_args()

def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
