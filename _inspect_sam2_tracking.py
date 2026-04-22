"""Inspect SAM2 tracking results in napari (local; e.g. outputs from GUI or batch scripts)."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import tifffile as tf
from pathlib import Path

from app_core.pipeline import (
    _build_lineage_df,
    _per_cell_metrics,
    _drift_correct,
)
from app_core.types import MetadataParams

# ── Config ───────────────────────────────────────────────────
RAW_PATH = Path("Ctrl_0mV_5min_interval-2.tif")
RESULTS_DIR = Path("results_sam2_tracking/Ctrl_0mV_5min_interval-2")

META = MetadataParams(
    dt_min=5.0,
    pixels_per_um=1.135,
    ef_on_frame=1,
    ef_axis="x",
    ef_sign=-1,
)
MIN_TRACK_LEN = 3
APPLY_DRIFT_CORRECTION = True

# ── Load ─────────────────────────────────────────────────────
print("Loading raw images...")
raw = np.array(tf.memmap(str(RAW_PATH)))

print("Loading tracked masks...")
masks_path = RESULTS_DIR / "masks_tracked.tif"
masks_tracked = tf.imread(str(masks_path))
print(f"  {masks_tracked.shape}, {masks_tracked.max()} unique labels")

# Trim raw to match mask frames
raw = raw[:masks_tracked.shape[0]]

# Load tracks CSV if available, otherwise build from masks
tracks_csv = RESULTS_DIR / "tracks.csv"
if tracks_csv.exists():
    print("Loading tracks from CSV...")
    tracks = pd.read_csv(str(tracks_csv))
else:
    print("No tracks.csv found. Building from masks + res_track.txt...")
    from app_core.pipeline import _parse_res_track, _build_tracks_from_masks
    track_info = _parse_res_track(RESULTS_DIR / "res_track.txt")
    tracks = _build_tracks_from_masks(masks_tracked, track_info)
    if tracks.empty:
        print("  WARNING: Could not build tracks. Check res_track.txt.")

if not tracks.empty:
    print(f"  {tracks['particle'].nunique()} tracks, {len(tracks)} rows")
else:
    print("  No tracks loaded.")

# Filter short tracks
if not tracks.empty:
    counts = tracks.groupby("particle").size()
    keep = counts[counts >= MIN_TRACK_LEN].index
    tracks = tracks[tracks["particle"].isin(keep)].copy()
    print(f"  {tracks['particle'].nunique()} tracks after min_track_len={MIN_TRACK_LEN}")

# Drift correction
if APPLY_DRIFT_CORRECTION and not tracks.empty:
    print("Applying drift correction...")
    tracks = _drift_correct(tracks)
elif not tracks.empty:
    tracks = tracks.sort_values(["particle", "frame"]).copy()
    tracks[["dx", "dy"]] = tracks.groupby("particle")[["x", "y"]].diff()

# Lineage
print("Building lineage...")
lineage = _build_lineage_df(tracks, meta=META)
if not lineage.empty:
    div_rows = lineage[lineage["n_children"] >= 2]
    print(f"  {len(div_rows)} division events")
    if not div_rows.empty:
        print(div_rows[["parent_id", "children", "n_children", "division_angle_deg"]].to_string())
else:
    print("  No lineage data")

# Per-cell metrics
print("Computing per-cell metrics...")
per_cell = _per_cell_metrics(tracks, meta=META)
if not per_cell.empty:
    print(f"  {len(per_cell)} cells with metrics")
    if "avg_step_speed_um_per_min" in per_cell.columns:
        print(f"  Mean speed: {per_cell['avg_step_speed_um_per_min'].mean():.3f} um/min")
    if "directedness_cos" in per_cell.columns:
        print(f"  Mean directedness: {per_cell['directedness_cos'].mean():.3f}")

# ── Napari ───────────────────────────────────────────────────
print("\nOpening napari...")
import napari

viewer = napari.Viewer(title="SAM2 tracking - Ctrl_0mV_5min_interval-2")
viewer.add_image(raw, name="raw", colormap="gray")
viewer.add_labels(masks_tracked, name="masks (SAM2 tracked)")

if not tracks.empty:
    track_data = tracks[["particle", "frame", "y", "x"]].to_numpy()
    viewer.add_tracks(track_data, name="tracks")

viewer.reset_view()
print("Napari open. Use the slider to scrub through frames.")
napari.run()
