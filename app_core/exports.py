from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tf

import cv2
import imageio.v2 as imageio
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

from .types import ExportedPaths, MetadataParams, OutputOptions, SingleRunResult


def export_csvs(
    out_dir: Path,
    *,
    meta: MetadataParams,
    single: SingleRunResult,
    out_opts: OutputOptions,
) -> ExportedPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_csv = None
    per_cell_csv = None
    per_frame_csv = None
    per_step_csv = None
    lineage_csv = None

    if out_opts.export_tracks_csv:
        tracks_csv = out_dir / "tracks.csv"
        single.tracks.to_csv(tracks_csv, index=False)

    if out_opts.export_per_cell_csv:
        per_cell_csv = out_dir / "per_cell.csv"
        single.per_cell.to_csv(per_cell_csv, index=True)  # particle is index

    if out_opts.export_per_frame_csv:
        per_frame_csv = out_dir / "per_frame.csv"
        single.per_frame.to_csv(per_frame_csv, index=False)

    if out_opts.export_per_step_velocity_csv and (meta.pixels_per_um is not None) and (not single.tracks.empty):
        step_csv = out_dir / "per_step_velocity.csv"
        step_start_frame = max(int(meta.ef_on_frame), 1)
        steps = single.tracks[single.tracks["frame"] >= step_start_frame].copy()
        um_per_px = 1.0 / float(meta.pixels_per_um)
        dx_col = "dx_corr" if "dx_corr" in steps.columns else "dx"
        dy_col = "dy_corr" if "dy_corr" in steps.columns else "dy"
        comp = steps[dx_col] if meta.ef_axis == "x" else steps[dy_col]
        steps["directed_velocity_um_per_min"] = (float(meta.ef_sign) * comp) * um_per_px / float(meta.dt_min)
        steps.to_csv(step_csv, index=False)
        per_step_csv = step_csv

    if out_opts.export_lineage_csv and not single.lineage.empty:
        lineage_csv = out_dir / "lineage.csv"
        single.lineage.to_csv(lineage_csv, index=True)

    return ExportedPaths(
        params_json=out_dir / "params.json",
        tracks_csv=tracks_csv,
        per_cell_csv=per_cell_csv,
        per_frame_csv=per_frame_csv,
        per_step_csv=per_step_csv,
        lineage_csv=lineage_csv,
        masks_tiff=None,
        segmentation_overlay_mp4=None,
        tracking_overlay_mp4=None,
    )


def export_masks_tiff(out_dir: Path, masks_filt: np.ndarray) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "masks_filt.tif"
    # int32 label images are fine for TIFF; keep as-is.
    tf.imwrite(str(path), masks_filt.astype(np.int32, copy=False))
    return path


def _compute_vmin_vmax(mm: np.ndarray, step: int = 5) -> tuple[float, float]:
    sample = np.asarray(mm[::step])
    vmin, vmax = np.percentile(sample, (1, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _to_uint8(img16: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = img16.astype(np.float32)
    x = (x - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def make_preview_rgb(movie_path: Path, masks_filt: np.ndarray, frame: int = 0) -> np.ndarray:
    mm = tf.memmap(str(movie_path))
    t = int(np.clip(frame, 0, int(masks_filt.shape[0]) - 1))
    vmin, vmax = _compute_vmin_vmax(mm)
    g = _to_uint8(mm[t], vmin, vmax)
    rgb = np.stack([g, g, g], axis=-1)
    b = find_boundaries(masks_filt[t] > 0, mode="outer")
    rgb[b, :] = (0, 255, 0)
    return rgb


def export_segmentation_overlay_mp4(
    out_dir: Path,
    movie_path: Path,
    masks_filt: np.ndarray,
    *,
    ef_on_frame: int | None,
    fps: int = 5,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "segmentation_overlay.mp4"

    mm = tf.memmap(str(movie_path))
    T = int(min(mm.shape[0], masks_filt.shape[0]))
    H, W = int(mm.shape[1]), int(mm.shape[2])
    vmin, vmax = _compute_vmin_vmax(mm)

    with imageio.get_writer(
        str(out_path),
        fps=fps,
        macro_block_size=1,
        pixelformat="yuv420p",
    ) as w:
        for t in range(T):
            g = _to_uint8(mm[t], vmin, vmax)
            rgb = np.stack([g, g, g], axis=-1)
            b = find_boundaries(masks_filt[t] > 0, mode="outer")
            rgb[b, :] = (0, 255, 0)

            if ef_on_frame is not None and t >= int(ef_on_frame):
                rgb[5:25, 5:25, :] = (255, 0, 0)

            w.append_data(rgb)

    return out_path


def _mask_overlay(gray01: np.ndarray, labels: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    return label2rgb(labels, image=gray01, bg_label=0, alpha=alpha, image_alpha=1.0)


def _to_float01(img16: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = img16.astype(np.float32)
    x = (x - vmin) / (vmax - vmin)
    return np.clip(x, 0.0, 1.0)


def _draw_centroids(rgb_u8: np.ndarray, pts_df: pd.DataFrame, t: int, *, color=(255, 255, 0)):
    if pts_df.empty:
        return
    this = pts_df[pts_df["frame"] == t]
    for _, r in this.iterrows():
        x = int(round(float(r["x"])))
        y = int(round(float(r["y"])))
        cv2.circle(rgb_u8, (x, y), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(rgb_u8, (x, y), 4, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def _draw_track_tails(rgb_u8: np.ndarray, tracks_df: pd.DataFrame, t: int, tail_frames: int, *, show_ids: bool):
    if tracks_df is None or tracks_df.empty:
        return
    t0 = max(0, int(t) - int(tail_frames))
    sub = tracks_df[(tracks_df["frame"] >= t0) & (tracks_df["frame"] <= t)].copy()
    if sub.empty:
        return

    for pid, g in sub.groupby("particle"):
        g = g.sort_values("frame")
        pts = [(int(round(x)), int(round(y))) for x, y in zip(g["x"], g["y"]) if np.isfinite(x) and np.isfinite(y)]
        if len(pts) < 2:
            continue

        h = int(pid) % 180
        col = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2RGB)[0, 0].tolist()
        col = (int(col[0]), int(col[1]), int(col[2]))

        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            cv2.line(rgb_u8, (x1, y1), (x2, y2), col, thickness=2, lineType=cv2.LINE_AA)

        if show_ids:
            xh, yh = pts[-1]
            cv2.putText(
                rgb_u8,
                str(int(pid)),
                (xh + 6, yh - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                rgb_u8,
                str(int(pid)),
                (xh + 6, yh - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                thickness=1,
                lineType=cv2.LINE_AA,
            )


def export_tracking_overlay_mp4(
    out_dir: Path,
    movie_path: Path,
    masks_filt: np.ndarray,
    pts: pd.DataFrame,
    tracks: pd.DataFrame,
    *,
    ef_on_frame: int | None,
    fps: int = 5,
    tail_frames: int = 10,
    show_ids: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tracking_overlay.mp4"

    mm = tf.memmap(str(movie_path))
    T = int(min(mm.shape[0], masks_filt.shape[0]))
    vmin, vmax = _compute_vmin_vmax(mm)

    with imageio.get_writer(
        str(out_path),
        fps=fps,
        macro_block_size=1,
        pixelformat="yuv420p",
    ) as w:
        for t in range(T):
            g = _to_float01(mm[t], vmin, vmax)
            rgb01 = _mask_overlay(g, masks_filt[t], alpha=0.35)
            frame = (rgb01 * 255).astype(np.uint8)

            _draw_track_tails(frame, tracks, t, tail_frames, show_ids=show_ids)
            _draw_centroids(frame, pts, t)

            if ef_on_frame is not None and t >= int(ef_on_frame):
                frame[5:25, 5:25, :] = (255, 0, 0)

            w.append_data(frame)

    return out_path

