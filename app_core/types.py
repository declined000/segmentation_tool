from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Mode = Literal["single", "pair"]


@dataclass(frozen=True)
class MetadataParams:
    dt_min: float
    pixels_per_um: float | None  # if None: pixel units only

    # Pair-only
    ef_on_frame: int = 1
    ef_axis: Literal["x", "y"] = "x"
    ef_sign: int = -1  # cathode LEFT => negative x is positive directed motion


@dataclass(frozen=True)
class SegmentationParams:
    """Cellpose-SAM (cpsam) only — generalist segmentation, optional manual diameter."""

    diameter_px: float | None = None  # None => native cpsam sizing (recommended)
    cellprob_threshold: float = 0.5
    flow_threshold: float = 0.4
    use_gpu: bool = False


@dataclass(frozen=True)
class QcParams:
    min_area_px: int = 400
    max_area_px: int = 8000
    border_px: int = 8

    min_solidity: float = 0.80
    min_eccentricity: float = 0.0
    max_circularity: float = 0.99



@dataclass(frozen=True)
class TrackingParams:
    min_track_len: int = 10
    apply_drift_correction: bool = True

    # SAM2 (sam4celltracking) parameters
    sam2_window_size: int = 128     # patch size for SAM2 linking
    sam2_dis_threshold: int = 50    # min mask pixels to continue tracking
    sam2_neighbor_dist: int = 30    # max pixels to link mask to prediction
    sam4ct_path: str = "sam4celltracking"  # path to cloned repo


@dataclass(frozen=True)
class OutputOptions:
    export_tracks_csv: bool = True
    export_per_cell_csv: bool = True
    export_per_frame_csv: bool = True
    export_lineage_csv: bool = True
    export_masks_tiff: bool = False
    export_segmentation_overlay_mp4: bool = True
    export_tracking_overlay_mp4: bool = True
    export_per_step_velocity_csv: bool = False


@dataclass(frozen=True)
class RunPaths:
    root_dir: Path
    run_dir: Path
    input_dir: Path
    out_dir: Path


@dataclass(frozen=True)
class SingleRunResult:
    movie_path: Path
    masks_filt: np.ndarray  # (T,Y,X) int32
    pts: pd.DataFrame
    tracks: pd.DataFrame
    per_frame: pd.DataFrame
    per_cell: pd.DataFrame
    lineage: pd.DataFrame


@dataclass(frozen=True)
class PairRunResult:
    ctrl: SingleRunResult
    ef: SingleRunResult


@dataclass(frozen=True)
class Summary:
    n_frames: int
    n_cells: int
    n_tracks: int
    mean_track_len: float | None

    n_divisions: int | None = None

    # Pair-only summary fields can be computed from EF tracks/corrected steps
    mean_directed_velocity_um_per_min: float | None = None
    mean_directedness: float | None = None


@dataclass(frozen=True)
class ExportedPaths:
    params_json: Path
    tracks_csv: Path | None
    per_cell_csv: Path | None
    per_frame_csv: Path | None
    per_step_csv: Path | None
    lineage_csv: Path | None
    masks_tiff: Path | None
    segmentation_overlay_mp4: Path | None
    tracking_overlay_mp4: Path | None


def exported_with(paths: ExportedPaths, **updates) -> ExportedPaths:
    d = {
        "params_json": paths.params_json,
        "tracks_csv": paths.tracks_csv,
        "per_cell_csv": paths.per_cell_csv,
        "per_frame_csv": paths.per_frame_csv,
        "per_step_csv": paths.per_step_csv,
        "lineage_csv": paths.lineage_csv,
        "masks_tiff": paths.masks_tiff,
        "segmentation_overlay_mp4": paths.segmentation_overlay_mp4,
        "tracking_overlay_mp4": paths.tracking_overlay_mp4,
    }
    d.update(updates)
    return ExportedPaths(**d)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RunOutput:
    mode: Mode
    run_dir: Path

    # For QC preview in GUI (uint8 RGB)
    preview_rgb_single: np.ndarray | None = None
    preview_rgb_ctrl: np.ndarray | None = None
    preview_rgb_ef: np.ndarray | None = None

    summary_ctrl: Summary | None = None
    summary_ef: Summary | None = None
    summary_single: Summary | None = None

    exported_single: ExportedPaths | None = None
    exported_ctrl: ExportedPaths | None = None
    exported_ef: ExportedPaths | None = None

