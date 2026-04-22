"""
Run the full pipeline (cpsam segmentation + SAM2 tracking) on a single video.

Designed for Google Cloud VMs with GPU. Outputs go to results_cloud/<stem>/.

Usage:
    python run_cloud_test.py path/to/video.tif [--max-frames N] [--no-gpu]
"""
from __future__ import annotations

import argparse
import os
import site
import sys
import time
from pathlib import Path


def _prepend_torch_cuda_libs() -> None:
    """Linux: load PyTorch's bundled cuBLAS before other packages (e.g. OpenCV).

    Fixes ``Invalid handle. Cannot load symbol cublasLtCreate`` when
    ``LD_LIBRARY_PATH`` would otherwise pick a mismatched system libcublas.
    """
    if "--no-gpu" in sys.argv or not sys.platform.startswith("linux"):
        return
    bases: list[str] = []
    try:
        u = site.getusersitepackages()
        if u:
            bases.append(u)
    except Exception:
        pass
    try:
        bases.extend(site.getsitepackages())
    except Exception:
        pass
    for base in bases:
        lib = os.path.join(base, "torch", "lib")
        if os.path.isdir(lib):
            prev = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = lib + (os.pathsep + prev if prev else "")
            return


def _prime_cuda_before_pipeline() -> None:
    """Initialize CUDA via PyTorch before importing app_core (skimage/cellpose chain)."""
    if "--no-gpu" in sys.argv:
        return
    _prepend_torch_cuda_libs()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
            torch.zeros(1, device="cuda")
    except Exception:
        pass


_prime_cuda_before_pipeline()

from app_core.pipeline import run_single_movie
from app_core.types import (
    MetadataParams,
    OutputOptions,
    QcParams,
    SegmentationParams,
    TrackingParams,
)
from app_core.exports import (
    export_csvs,
    export_masks_tiff,
    export_segmentation_overlay_mp4,
    export_tracking_overlay_mp4,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tif", type=Path, help="Path to a stacked .tif video")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Process only the first N frames (default: all)")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Force CPU-only (slow, for debugging)")
    ap.add_argument("--sam4ct-path", type=str, default="sam4celltracking",
                    help="Path to cloned sam4celltracking repo")
    args = ap.parse_args()

    if not args.tif.exists():
        print(f"File not found: {args.tif}")
        sys.exit(1)

    use_gpu = not args.no_gpu
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("WARNING: --no-gpu not set but CUDA unavailable. Falling back to CPU.")
                use_gpu = False
            else:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            use_gpu = False

    meta = MetadataParams(dt_min=5.0, pixels_per_um=None)
    seg = SegmentationParams(diameter_px=None, cellprob_threshold=0.0,
                             flow_threshold=0.4, use_gpu=use_gpu)
    qc = QcParams(min_area_px=50, max_area_px=15000, border_px=8,
                  min_solidity=0.50, min_eccentricity=0.0, max_circularity=0.99)
    tr = TrackingParams(sam4ct_path=args.sam4ct_path)
    out_opts = OutputOptions(export_masks_tiff=True)

    out_dir = Path("results_cloud") / args.tif.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Pipeline: cpsam segmentation + SAM2 tracking")
    print(f"  Input:    {args.tif.name}")
    print(f"  GPU:      {use_gpu}")
    print(f"  Output:   {out_dir.resolve()}")
    print(f"{'='*60}\n")

    t0 = time.time()
    res = run_single_movie(
        args.tif, meta=meta, seg=seg, qc=qc, tr=tr,
        max_frames=args.max_frames,
    )
    elapsed = time.time() - t0

    print(f"\nPipeline finished in {elapsed:.0f}s")
    print(f"  Frames:      {res.masks_filt.shape[0]}")
    print(f"  Detections:  {len(res.pts)}")
    print(f"  Tracks:      {res.tracks['particle'].nunique() if not res.tracks.empty else 0}")

    if not res.lineage.empty and "n_children" in res.lineage.columns:
        n_div = int((res.lineage["n_children"] >= 2).sum())
        print(f"  Divisions:   {n_div}")

    print(f"\nExporting to {out_dir} ...")
    exported = export_csvs(out_dir, meta=meta, single=res, out_opts=out_opts)
    export_masks_tiff(out_dir, res.masks_filt)
    export_segmentation_overlay_mp4(out_dir, args.tif, res.masks_filt, ef_on_frame=None)

    if not res.tracks.empty:
        export_tracking_overlay_mp4(
            out_dir, args.tif, res.masks_filt, res.pts, res.tracks,
            ef_on_frame=None,
        )

    print(f"\nDone. Results in: {out_dir.resolve()}")
    for f in sorted(out_dir.iterdir()):
        sz = f.stat().st_size / 1024
        unit = "KB"
        if sz > 1024:
            sz /= 1024
            unit = "MB"
        print(f"  {f.name:40s} {sz:6.1f} {unit}")


if __name__ == "__main__":
    main()
