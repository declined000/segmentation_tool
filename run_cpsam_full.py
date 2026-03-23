"""
Batch Cellpose-SAM (cpsam) segmentation + tracking for all .tif files.

Designed for Google Colab (T4 GPU, 15 GB VRAM).
Automatically processes every .tif file in the working directory.

Usage (Colab):  !python run_cpsam_full.py
Usage (local):  .venv-cyto2/Scripts/python.exe run_cpsam_full.py
"""
import sys, subprocess, glob, time
from pathlib import Path

def _colab_install():
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "cellpose>=4.0", "segment-anything", "btrack==0.7.0",
         "tifffile", "scikit-image", "pandas",
         "opencv-python-headless", "imageio", "imageio-ffmpeg"]
    )
    print("Colab dependencies installed.\n")

_colab_install()

import numpy as np
import pandas as pd
import tifffile as tf
from app_core.pipeline import (
    _run_cellpose_on_stack,
    _qc_centroids_from_masks,
    _track_centroids,
    _build_lineage_df,
    _per_cell_metrics,
)
from app_core.exports import (
    export_segmentation_overlay_mp4,
    export_tracking_overlay_mp4,
    export_masks_tiff,
)
from app_core.types import (
    SegmentationParams,
    QcParams,
    TrackingParams,
    MetadataParams,
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  SETTINGS — permissive defaults, no tuning needed           ║
# ╚══════════════════════════════════════════════════════════════╝

seg = SegmentationParams(
    model_type="cpsam",
    diameter_px=None,        # auto-detect for any cell type
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    use_gpu=True,
)

qc = QcParams(
    min_area_px=50,          # small enough for neutrophils
    max_area_px=15000,       # large enough for epithelial
    border_px=8,
    min_solidity=0.50,       # permissive for irregular microglia
    min_eccentricity=0.0,
    max_circularity=0.99,
)

tr = TrackingParams(
    search_range_px=25.0,
    memory=2,
    min_track_len=3,
    apply_drift_correction=False,
)

meta = MetadataParams(
    dt_min=5.0,              # overridden per-file below if "1min" in filename
    pixels_per_um=1.135,
    ef_on_frame=1,
    ef_axis="x",
    ef_sign=-1,
)

OUT_DIR = Path("results_cpsam")

# ╔══════════════════════════════════════════════════════════════╗
# ║  RUN — processes every .tif in the directory                 ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    tif_files = sorted(Path(".").glob("*.tif"))
    if not tif_files:
        print("No .tif files found in current directory!")
        sys.exit(1)

    print(f"\nFound {len(tif_files)} .tif files:")
    for f in tif_files:
        print(f"  - {f.name}")
    print()

    for tif_path in tif_files:
        name = tif_path.stem
        out_dir = OUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect time interval from filename
        file_meta = meta
        if "1min" in name.lower() or "1_min" in name.lower():
            file_meta = MetadataParams(
                dt_min=1.0,
                pixels_per_um=meta.pixels_per_um,
                ef_on_frame=meta.ef_on_frame,
                ef_axis=meta.ef_axis,
                ef_sign=meta.ef_sign,
            )

        print(f"{'='*60}")
        print(f"  Processing: {tif_path.name}")
        print(f"  dt_min: {file_meta.dt_min}")
        print(f"{'='*60}")

        t0 = time.time()

        print("  Segmenting (cpsam on GPU) ...")
        masks = _run_cellpose_on_stack(tif_path, seg=seg, max_frames=None)
        print(f"    {masks.shape[0]} frames, shape {masks.shape[1]}x{masks.shape[2]}")
        seg_time = time.time() - t0
        print(f"    Segmentation: {seg_time:.0f}s ({seg_time/masks.shape[0]:.1f}s/frame)")

        print("  QC filtering ...")
        pts, masks_filt = _qc_centroids_from_masks(masks, qc=qc)
        print(f"    {len(pts)} detections kept")

        print("  Tracking ...")
        tracks = pd.DataFrame()
        lineage = pd.DataFrame()
        try:
            tracks = _track_centroids(masks_filt, tr=tr)
            n_tracks = tracks["particle"].nunique() if not tracks.empty else 0
            print(f"    {n_tracks} tracks")

            if not tracks.empty:
                tracks = tracks.sort_values(["particle", "frame"]).copy()
                tracks[["dx", "dy"]] = tracks.groupby("particle")[["x", "y"]].diff()

            lineage = _build_lineage_df(tracks, file_meta)
            n_div = int((lineage["n_children"] >= 2).sum()) if not lineage.empty else 0
            print(f"    {n_div} division events")
        except Exception as e:
            print(f"    WARNING: Tracking failed ({e})")
            print(f"    Segmentation + masks still saved. Run tracking locally.")

        per_cell = _per_cell_metrics(tracks, file_meta)

        # Export CSVs
        if not tracks.empty:
            tracks.to_csv(out_dir / "tracks.csv", index=False)
        if not per_cell.empty:
            per_cell.to_csv(out_dir / "per_cell.csv", index=True)
        if not lineage.empty:
            lineage.to_csv(out_dir / "lineage.csv", index=True)
        if not pts.empty:
            pts.to_csv(out_dir / "detections.csv", index=False)

        # Export masks
        export_masks_tiff(out_dir, masks_filt)

        # Export overlay videos
        print("  Exporting segmentation overlay video ...")
        export_segmentation_overlay_mp4(out_dir, tif_path, masks_filt, ef_on_frame=None)

        if not tracks.empty:
            print("  Exporting tracking overlay video ...")
            export_tracking_overlay_mp4(out_dir, tif_path, masks_filt, pts, tracks, ef_on_frame=None)
        else:
            print("  Skipping tracking overlay (no tracks).")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s\n")

    print(f"\n{'='*60}")
    print(f"  ALL DONE — results in {OUT_DIR.resolve()}")
    print(f"{'='*60}")
    for d in sorted(OUT_DIR.iterdir()):
        if d.is_dir():
            csvs = list(d.glob("*.csv"))
            mp4s = list(d.glob("*.mp4"))
            print(f"  {d.name}/  ({len(csvs)} CSVs, {len(mp4s)} videos)")
    print()
