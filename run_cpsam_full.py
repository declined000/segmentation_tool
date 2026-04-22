"""
Batch Cellpose-SAM (cpsam) segmentation + QC for all .tif in the current directory.

Writes per-movie folders under results_cpsam/<stem>/ (masks, overlay MP4, detections CSV).
For full segmentation + SAM2 tracking + metrics, use the Streamlit app instead.

Usage (local):
  .venv-cyto2\\Scripts\\python.exe run_cpsam_full.py
"""
import sys
import time
from pathlib import Path

from app_core.pipeline import (
    _run_cellpose_on_stack,
    _qc_centroids_from_masks,
)
from app_core.exports import (
    export_segmentation_overlay_mp4,
    export_masks_tiff,
)
from app_core.types import (
    SegmentationParams,
    QcParams,
)

seg = SegmentationParams(
    diameter_px=None,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    use_gpu=True,
)

qc = QcParams(
    min_area_px=50,
    max_area_px=15000,
    border_px=8,
    min_solidity=0.50,
    min_eccentricity=0.0,
    max_circularity=0.99,
)

OUT_DIR = Path("results_cpsam")

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

        print(f"{'='*60}")
        print(f"  Processing: {tif_path.name}")
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

        if not pts.empty:
            pts.to_csv(out_dir / "detections.csv", index=False)

        export_masks_tiff(out_dir, masks_filt)

        print("  Exporting segmentation overlay video ...")
        export_segmentation_overlay_mp4(out_dir, tif_path, masks_filt, ef_on_frame=None)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s\n")

    print(f"\n{'='*60}")
    print(f"  ALL DONE — results in {OUT_DIR.resolve()}")
    print("  Next: run the Streamlit app for SAM2 tracking + electrotaxis exports.")
    print(f"{'='*60}")
    for d in sorted(OUT_DIR.iterdir()):
        if d.is_dir():
            csvs = list(d.glob("*.csv"))
            tifs = list(d.glob("*.tif"))
            mp4s = list(d.glob("*.mp4"))
            print(f"  {d.name}/  ({len(csvs)} CSVs, {len(tifs)} TIFs, {len(mp4s)} videos)")
    print()
