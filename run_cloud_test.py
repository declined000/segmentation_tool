"""
Run the full pipeline (cpsam segmentation + SAM2 tracking) on a single video.

Designed for Google Cloud VMs with GPU. Outputs go to results_cloud/<stem>/.

Usage:
    python3 run_cloud_test.py path/to/video.tif [--max-frames N] [--no-gpu]

Requires LD_LIBRARY_PATH set before launch (see scripts/gcloud_setup.sh).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict

# On Linux VMs with mixed CUDA trees, preload pip-packaged cuBLAS/cuDNN first.
if "--no-gpu" not in sys.argv:
    try:
        from app_core.cuda_preload import preload_cuda_user_libs
        preload_cuda_user_libs(verbose=True)

        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.init()
            _x = _torch.randn(2, 2, device="cuda")
            _y = _torch.matmul(_x, _x)          # force cuBLAS load
            del _x, _y
            print(f"GPU: {_torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA not available, falling back to CPU")
    except Exception as e:
        print(f"WARNING: CUDA init failed ({e}), falling back to CPU")

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
    ap.add_argument(
        "--adjudication-provider",
        type=str,
        choices=["gemini", "heuristic"],
        default="heuristic",
        help="Phase-1 ambiguity adjudication backend.",
    )
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
    tr = TrackingParams(
        sam4ct_path=args.sam4ct_path,
        adjudication_provider=args.adjudication_provider,
    )
    out_opts = OutputOptions(export_masks_tiff=True)

    out_dir = Path("results_cloud") / args.tif.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    params_json = out_dir / "params.json"
    params_json.write_text(
        json.dumps(
            {
                "runner": "run_cloud_test.py",
                "input": str(args.tif.resolve()),
                "max_frames": args.max_frames,
                "meta": asdict(meta),
                "seg": asdict(seg),
                "qc": asdict(qc),
                "tracking": asdict(tr),
                "output": asdict(out_opts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

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
    if res.adjudication_audit is not None and not res.adjudication_audit.empty:
        n_events = int(len(res.adjudication_audit))
        n_applied = int(res.adjudication_audit["applied"].sum()) if "applied" in res.adjudication_audit.columns else 0
        print(f"  Adjudicated events: {n_events} (applied: {n_applied})")
        if "final_applied_action" in res.adjudication_audit.columns:
            top_actions = (
                res.adjudication_audit["final_applied_action"]
                .value_counts()
                .head(5)
                .to_dict()
            )
            print(f"  Adjudication actions (top): {top_actions}")
        if "reason" in res.adjudication_audit.columns:
            n_unavail = int(
                res.adjudication_audit["reason"]
                .astype(str)
                .str.startswith("gemini_unavailable_")
                .sum()
            )
            print(f"  Gemini unavailable events: {n_unavail}")

    print(f"\nExporting to {out_dir} ...")
    exported = export_csvs(out_dir, meta=meta, single=res, out_opts=out_opts)
    export_masks_tiff(out_dir, res.masks_filt)
    export_segmentation_overlay_mp4(out_dir, args.tif, res.masks_filt, ef_on_frame=None)

    if not res.tracks.empty:
        export_tracking_overlay_mp4(
            out_dir, args.tif, res.masks_filt, res.pts, res.tracks,
            ef_on_frame=None,
        )

    quality_summary = {
        "frames": int(res.masks_filt.shape[0]),
        "detections": int(len(res.pts)),
        "tracks": int(res.tracks["particle"].nunique()) if not res.tracks.empty else 0,
        "divisions_ge2_children": int((res.lineage["n_children"] >= 2).sum())
        if (not res.lineage.empty and "n_children" in res.lineage.columns) else 0,
    }
    if res.adjudication_audit is not None and not res.adjudication_audit.empty:
        aa = res.adjudication_audit
        quality_summary["adjudication_events"] = int(len(aa))
        quality_summary["adjudication_applied"] = int(aa["applied"].sum()) if "applied" in aa.columns else 0
        quality_summary["adjudication_apply_rate"] = (
            float(aa["applied"].mean()) if "applied" in aa.columns else 0.0
        )
        if "provider" in aa.columns:
            quality_summary["providers"] = aa["provider"].value_counts().to_dict()
        if "vlm_decision" in aa.columns:
            quality_summary["vlm_decisions"] = aa["vlm_decision"].value_counts().to_dict()
        if "final_applied_action" in aa.columns:
            quality_summary["final_actions"] = aa["final_applied_action"].value_counts().to_dict()
        if "reason" in aa.columns:
            rs = aa["reason"].astype(str)
            quality_summary["gemini_unavailable_events"] = int(rs.str.startswith("gemini_unavailable_").sum())
    (out_dir / "quality_summary.json").write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")

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
