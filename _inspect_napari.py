"""
Quick segmentation + napari inspection (one file at a time).

Runs cpsam segmentation + QC locally and opens the result in napari.
For full tracks + divisions, use the Streamlit app (SAM2 in pipeline) or `_inspect_sam2_tracking.py`.

Usage:  .venv-cyto2\Scripts\python.exe -u _inspect_napari.py
"""
from pathlib import Path
import numpy as np
import tifffile as tf

from app_core.pipeline import (
    _run_cellpose_on_stack,
    _qc_centroids_from_masks,
)
from app_core.types import (
    SegmentationParams,
    QcParams,
)

# ── Which file to inspect ───────────────────────────────────────
MOVIE = Path(r"c:\Users\fgkb\Desktop\bioelectricity project\EF_200mV_5min_interval-2.tif")
LABEL = "Corneal epithelial (cpsam)"
START_FRAME: int = 0
TEST_FRAMES: int | None = 1

# ── Crop: take left or right half of image ──────────────────────
HALF: str | None = "left"  # "left", "right", or None for full

# ── Parameters (cpsam with auto diameter, GPU) ──────────────────
seg = SegmentationParams(diameter_px=None, cellprob_threshold=0.0, flow_threshold=0.4, use_gpu=True)
qc  = QcParams(min_area_px=400, max_area_px=8000, border_px=8, min_solidity=0.80, min_eccentricity=0.0, max_circularity=0.99)

# ── Run pipeline ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {LABEL}: {MOVIE.name}")
print(f"  Frames: {START_FRAME} to {START_FRAME + TEST_FRAMES - 1}" if TEST_FRAMES else "  Frames: all")
if HALF:
    print(f"  Half: {HALF}")
print(f"{'='*60}")

# Load and optionally crop
raw_full = tf.imread(str(MOVIE))
if TEST_FRAMES:
    raw_full = raw_full[START_FRAME:START_FRAME + TEST_FRAMES]
if HALF:
    _, H, W = raw_full.shape
    mid = W // 2
    if HALF == "left":
        raw_full = raw_full[:, :, :mid]
    else:
        raw_full = raw_full[:, :, mid:]
    print(f"  Cropped to {raw_full.shape}")

# Save cropped to temp file for pipeline
import tempfile, os
tmp = os.path.join(tempfile.gettempdir(), "_crop_test.tif")
tf.imwrite(tmp, raw_full)

print("  Segmenting ...")
masks = _run_cellpose_on_stack(Path(tmp), seg=seg, max_frames=None)
print(f"    masks shape: {masks.shape}")

print("  QC filtering ...")
pts, masks_filt = _qc_centroids_from_masks(masks, qc=qc)
print(f"    {len(pts)} detections kept")

# ── Open napari ─────────────────────────────────────────────────
print("\nOpening napari ...")
import napari

raw = raw_full[:masks_filt.shape[0]]
viewer = napari.Viewer(title=LABEL)

viewer.add_image(raw, name="raw", colormap="gray", blending="additive")
viewer.add_labels(masks_filt, name="masks")

if not pts.empty:
    viewer.add_points(
        pts[["frame", "y", "x"]].to_numpy(),
        size=6, face_color="yellow", name="centroids",
    )

viewer.reset_view()
napari.run()
