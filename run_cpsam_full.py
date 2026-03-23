"""
Full pipeline run with Cellpose-SAM (cpsam) on GPU.

Designed for Google Colab (T4 GPU, 15 GB VRAM).
Processes a control + EF file pair, exports all results.

Usage (Colab):
    !python run_cpsam_full.py

Usage (local):
    .venv-cyto2\Scripts\python.exe run_cpsam_full.py
"""
import sys, subprocess

def _colab_install():
    """Auto-install dependencies when running on Google Colab."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return  # not on Colab, skip
    pkgs = [
        "cellpose>=4.0",
        "segment-anything",
        "btrack==0.7.0",
        "tifffile",
        "scikit-image",
        "pandas",
        "opencv-python-headless",
        "imageio",
        "imageio-ffmpeg",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    )
    print("Colab dependencies installed.\n")

_colab_install()

from pathlib import Path
from app_core.pipeline import run_and_export
from app_core.types import (
    SegmentationParams,
    QcParams,
    TrackingParams,
    MetadataParams,
    OutputOptions,
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIGURE THESE FOR YOUR RUN                               ║
# ╚══════════════════════════════════════════════════════════════╝

# -- File paths (update to match your Colab/local paths) --------
CTRL_PATH = Path("Ctrl_0mV_5min_interval-2.tif")
EF_PATH   = Path("EF_200mV_5min_interval-2.tif")

# -- Output directory -------------------------------------------
OUT_DIR = Path("results_cpsam")

# -- Segmentation -----------------------------------------------
seg = SegmentationParams(
    model_type="cpsam",
    diameter_px=None,       # auto-detect cell size
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    use_gpu=True,           # True for Colab GPU
    denoise=False,
)

# -- Quality control --------------------------------------------
qc = QcParams(
    min_area_px=400,
    max_area_px=8000,
    border_px=8,
    min_solidity=0.80,
    min_eccentricity=0.0,
    max_circularity=0.99,
)

# -- Tracking ---------------------------------------------------
tr = TrackingParams(
    search_range_px=20.0,
    memory=2,
    min_track_len=3,
    apply_drift_correction=True,
)

# -- Metadata ---------------------------------------------------
meta = MetadataParams(
    dt_min=5.0,
    pixels_per_um=1.135,
    ef_on_frame=1,
    ef_axis="x",
    ef_sign=-1,
)

# -- Exports ----------------------------------------------------
out_opts = OutputOptions(
    export_tracks_csv=True,
    export_per_cell_csv=True,
    export_per_frame_csv=True,
    export_lineage_csv=True,
    export_masks_tiff=True,
    export_segmentation_overlay_mp4=True,
    export_tracking_overlay_mp4=True,
)

# -- Max frames (None = all frames) -----------------------------
MAX_FRAMES = None

# ╔══════════════════════════════════════════════════════════════╗
# ║  RUN                                                         ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Cellpose-SAM full pipeline run")
    print(f"  CTRL: {CTRL_PATH.name}")
    print(f"  EF:   {EF_PATH.name}")
    print(f"  Frames: {MAX_FRAMES or 'all'}")
    print(f"  GPU: {seg.use_gpu}")
    print(f"{'='*60}\n")

    result = run_and_export(
        mode="pair",
        run_dir=OUT_DIR,
        meta=meta,
        seg=seg,
        qc=qc,
        tr=tr,
        out_opts=out_opts,
        max_frames=MAX_FRAMES,
        ctrl_path=CTRL_PATH,
        ef_path=EF_PATH,
    )

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")

    if result.summary_ctrl:
        s = result.summary_ctrl
        print(f"\n  CTRL:")
        print(f"    Frames:     {s.n_frames}")
        print(f"    Cells:      {s.n_cells}")
        print(f"    Tracks:     {s.n_tracks}")
        print(f"    Avg track:  {s.mean_track_len:.1f} frames" if s.mean_track_len else "")
        print(f"    Divisions:  {s.n_divisions}" if s.n_divisions is not None else "")

    if result.summary_ef:
        s = result.summary_ef
        print(f"\n  EF:")
        print(f"    Frames:     {s.n_frames}")
        print(f"    Cells:      {s.n_cells}")
        print(f"    Tracks:     {s.n_tracks}")
        print(f"    Avg track:  {s.mean_track_len:.1f} frames" if s.mean_track_len else "")
        print(f"    Divisions:  {s.n_divisions}" if s.n_divisions is not None else "")
        if hasattr(s, "mean_directed_velocity_um_per_min") and s.mean_directed_velocity_um_per_min is not None:
            print(f"    Mean vel:   {s.mean_directed_velocity_um_per_min:.3f} um/min")
        if hasattr(s, "mean_directedness") and s.mean_directedness is not None:
            print(f"    Mean dir:   {s.mean_directedness:.3f}")

    print(f"\n  Output saved to: {OUT_DIR.resolve()}")
    print(f"{'='*60}\n")
