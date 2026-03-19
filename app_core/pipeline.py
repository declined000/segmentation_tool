from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tf

from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential

import trackpy as tp

from .types import (
    MetadataParams,
    OutputOptions,
    PairRunResult,
    QcParams,
    RunOutput,
    SegmentationParams,
    SingleRunResult,
    Summary,
    TrackingParams,
)
from .exports import (
    export_csvs,
    export_masks_tiff,
    export_segmentation_overlay_mp4,
    export_tracking_overlay_mp4,
    make_preview_rgb,
)
from .types import exported_with


def _run_cellpose_on_stack(
    path: Path,
    seg: SegmentationParams,
    max_frames: int | None,
) -> np.ndarray:
    from cellpose import models  # imported here for faster env-check UI

    mm = tf.memmap(str(path))  # (T,Y,X)
    nT = int(mm.shape[0])
    T = nT if max_frames is None else min(nT, int(max_frames))

    if seg.denoise:
        from cellpose import denoise
        restore_type = f"denoise_{seg.model_type}"
        model = denoise.CellposeDenoiseModel(
            gpu=bool(seg.use_gpu),
            model_type=str(seg.model_type),
            restore_type=restore_type,
        )
    else:
        model = models.Cellpose(gpu=bool(seg.use_gpu), model_type=str(seg.model_type))

    masks: list[np.ndarray] = []
    for t in range(T):
        img = mm[t]
        result = model.eval(
            img,
            channels=[0, 0],
            diameter=None if seg.diameter_px is None else float(seg.diameter_px),
            flow_threshold=float(seg.flow_threshold),
            cellprob_threshold=float(seg.cellprob_threshold),
            normalize=True,
        )
        # CellposeDenoiseModel returns (masks, flows, styles, imgs_dn);
        # Cellpose returns (masks, flows, styles, diams)
        m = result[0]
        masks.append(m.astype(np.int32, copy=False))

    return np.stack(masks, axis=0)


def _qc_centroids_from_masks(
    masks: np.ndarray,
    qc: QcParams,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []
    masks_filt = np.zeros_like(masks, dtype=np.int32)

    for t in range(int(masks.shape[0])):
        m = masks[t]
        if int(m.max()) == 0:
            continue

        props = regionprops_table(
            m,
            properties=(
                "label",
                "centroid",
                "area",
                "eccentricity",
                "solidity",
                "perimeter",
            ),
        )
        df = pd.DataFrame(props)
        df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
        df["frame"] = t

        per = df["perimeter"].to_numpy(dtype=float)
        area = df["area"].to_numpy(dtype=float)
        circ = np.zeros_like(area)
        ok = per > 0
        circ[ok] = (4.0 * np.pi * area[ok]) / (per[ok] ** 2)
        df["circularity"] = circ

        H, W = m.shape
        keep = (
            (df["area"] >= int(qc.min_area_px))
            & (df["area"] <= int(qc.max_area_px))
            & (df["solidity"] >= float(qc.min_solidity))
            & (df["eccentricity"] >= float(qc.min_eccentricity))
            & (df["circularity"] <= float(qc.max_circularity))
            & (df["x"] >= int(qc.border_px))
            & (df["x"] <= (W - 1 - int(qc.border_px)))
            & (df["y"] >= int(qc.border_px))
            & (df["y"] <= (H - 1 - int(qc.border_px)))
        )
        df_keep = df[keep].copy()

        kept_labels = df_keep["label"].to_numpy(dtype=int)
        mf = np.where(np.isin(m, kept_labels), m, 0).astype(np.int32, copy=False)
        mf, fw, _ = relabel_sequential(mf)
        masks_filt[t] = mf

        for _, r in df_keep.iterrows():
            old_label = int(r["label"])
            new_label = int(fw[old_label])
            rows.append(
                {
                    "frame": int(r["frame"]),
                    "label": new_label,
                    "label_old": old_label,
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "area": float(r["area"]),
                    "eccentricity": float(r["eccentricity"]),
                    "solidity": float(r["solidity"]),
                    "circularity": float(r["circularity"]),
                }
            )

    pts = pd.DataFrame(rows)
    return pts, masks_filt


def _track_centroids(
    pts: pd.DataFrame,
    tr: TrackingParams,
) -> pd.DataFrame:
    if pts.empty:
        return pd.DataFrame()

    df_track = pts[["frame", "x", "y"]].copy()
    linked = tp.link_df(
        df_track,
        search_range=float(tr.search_range_px),
        memory=int(tr.memory),
    )

    linked = linked.sort_values(["particle", "frame"]).copy()
    linked["step"] = linked.groupby("particle")["frame"].diff()
    linked["dx_raw"] = linked.groupby("particle")["x"].diff()
    linked["dy_raw"] = linked.groupby("particle")["y"].diff()
    linked["jump_px"] = np.hypot(linked["dx_raw"], linked["dy_raw"])

    jump_max = float(tr.jump_max_factor) * float(tr.search_range_px)
    linked["break"] = (linked["step"] != 1) | (linked["jump_px"] > jump_max)
    linked["segment"] = linked.groupby("particle")["break"].cumsum().fillna(0).astype(int)

    linked["particle"] = (linked["particle"].astype(int) * 10000 + linked["segment"]).astype(int)
    linked.drop(columns=["segment"], inplace=True)

    counts = linked.groupby("particle").size()
    keep = counts[counts >= int(tr.min_track_len)].index
    linked = linked[linked["particle"].isin(keep)].copy()

    return linked


def _drift_correct(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return tracks

    tracks = tracks.sort_values(["particle", "frame"]).copy()
    tracks[["dx", "dy"]] = tracks.groupby("particle")[["x", "y"]].diff()

    step_rows = tracks.dropna(subset=["dx", "dy"]).copy()
    drift = (
        step_rows.groupby("frame")[["dx", "dy"]]
        .median()
        .rename(columns={"dx": "drift_dx", "dy": "drift_dy"})
    )
    tracks = tracks.join(drift, on="frame")

    tracks["dx_corr"] = tracks["dx"] - tracks["drift_dx"]
    tracks["dy_corr"] = tracks["dy"] - tracks["drift_dy"]
    return tracks


def _directed_component_um_per_min(
    tracks: pd.DataFrame,
    meta: MetadataParams,
) -> pd.Series:
    if tracks.empty:
        return pd.Series(dtype=float)

    dx_col = "dx_corr" if "dx_corr" in tracks.columns else "dx"
    dy_col = "dy_corr" if "dy_corr" in tracks.columns else "dy"
    comp = tracks[dx_col] if meta.ef_axis == "x" else tracks[dy_col]
    if meta.pixels_per_um is None:
        return pd.Series(dtype=float)
    um_per_px = 1.0 / float(meta.pixels_per_um)
    return (float(meta.ef_sign) * comp) * um_per_px / float(meta.dt_min)


def _per_cell_metrics(tracks: pd.DataFrame, meta: MetadataParams) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame()

    step_start_frame = max(int(meta.ef_on_frame), 1)
    dc = "dx_corr" if "dx_corr" in tracks.columns else "dx"
    dy_col = "dy_corr" if "dy_corr" in tracks.columns else "dy"
    steps = tracks[(tracks["frame"] >= step_start_frame)].dropna(subset=[dc, dy_col]).copy()
    if steps.empty:
        return pd.DataFrame()

    g = steps.groupby("particle")

    ef_comp = steps[dc] if meta.ef_axis == "x" else steps[dy_col]
    sum_ef = ef_comp.groupby(steps["particle"]).sum()

    sum_dx = g[dc].sum()
    sum_dy = g[dy_col].sum()
    step_dist = np.sqrt(steps[dc] ** 2 + steps[dy_col] ** 2)
    path_len = step_dist.groupby(steps["particle"]).sum()
    net_disp = np.sqrt(sum_dx ** 2 + sum_dy ** 2)

    out = pd.DataFrame({"n_steps": g.size()})

    # Old path-based directedness (kept for comparison)
    out["directedness_path"] = (float(meta.ef_sign) * sum_ef) / path_len.replace(0, np.nan)

    # Paper directedness: cos(theta) = net EF-axis displacement / net straight-line displacement
    # (Zhang et al., iScience 2022)
    out["directedness_cos"] = (float(meta.ef_sign) * sum_ef) / net_disp.replace(0, np.nan)

    if meta.pixels_per_um is not None:
        um_per_px = 1.0 / float(meta.pixels_per_um)
        dt = float(meta.dt_min)

        out["net_toward_cathode_um"] = (float(meta.ef_sign) * sum_ef) * um_per_px
        out["net_displacement_um"] = net_disp * um_per_px

        ef_step_vel = (float(meta.ef_sign) * ef_comp) * um_per_px / dt
        step_speed = step_dist * um_per_px / dt

        out["avg_step_speed_um_per_min"] = step_speed.groupby(steps["particle"]).mean()
        out["avg_speed_toward_cathode_um_per_min"] = ef_step_vel.groupby(steps["particle"]).mean()

    out = out.dropna(subset=["directedness_cos"])
    return out


def _per_frame_metrics(
    pts: pd.DataFrame,
    masks_shape: tuple[int, int, int],
    meta: MetadataParams,
) -> pd.DataFrame:
    T, H, W = (int(masks_shape[0]), int(masks_shape[1]), int(masks_shape[2]))
    base = pd.DataFrame({"frame": np.arange(T, dtype=int)})

    if pts.empty:
        base["n_cells"] = 0
        base["mean_area_px"] = np.nan
    else:
        g = pts.groupby("frame")
        base = base.merge(g.size().rename("n_cells"), on="frame", how="left")
        base = base.merge(g["area"].mean().rename("mean_area_px"), on="frame", how="left")
        base["n_cells"] = base["n_cells"].fillna(0).astype(int)

    if meta.pixels_per_um is not None:
        um_per_px = 1.0 / float(meta.pixels_per_um)
        area_um2 = (H * W) * (um_per_px**2)
        area_mm2 = area_um2 / 1e6
        if area_mm2 > 0:
            base["cells_per_mm2"] = base["n_cells"] / area_mm2

    return base


def _summary(tracks: pd.DataFrame, pts: pd.DataFrame, masks: np.ndarray) -> Summary:
    n_frames = int(masks.shape[0])
    # total detections kept across frames (after QC)
    n_cells = int(pts.shape[0]) if pts is not None and not pts.empty else 0
    n_tracks = int(tracks["particle"].nunique()) if (tracks is not None and not tracks.empty) else 0

    if tracks is not None and not tracks.empty:
        counts = tracks.groupby("particle").size()
        mean_track_len = float(counts.mean()) if len(counts) else None
    else:
        mean_track_len = None

    return Summary(
        n_frames=n_frames,
        n_cells=n_cells,
        n_tracks=n_tracks,
        mean_track_len=mean_track_len,
    )


def run_single_movie(
    movie_path: Path,
    meta: MetadataParams,
    seg: SegmentationParams,
    qc: QcParams,
    tr: TrackingParams,
    max_frames: int | None,
) -> SingleRunResult:
    masks = _run_cellpose_on_stack(movie_path, seg=seg, max_frames=max_frames)
    pts, masks_filt = _qc_centroids_from_masks(masks, qc=qc)
    tracks = _track_centroids(pts, tr=tr)

    if not tracks.empty and tr.apply_drift_correction:
        tracks = _drift_correct(tracks)
    elif not tracks.empty:
        tracks = tracks.sort_values(["particle", "frame"]).copy()
        tracks[["dx", "dy"]] = tracks.groupby("particle")[["x", "y"]].diff()

    per_frame = _per_frame_metrics(pts, masks_filt.shape, meta=meta)
    per_cell = _per_cell_metrics(tracks, meta=meta)

    return SingleRunResult(
        movie_path=movie_path,
        masks_filt=masks_filt,
        pts=pts,
        tracks=tracks,
        per_frame=per_frame,
        per_cell=per_cell,
    )


def run_pair(
    ctrl_path: Path,
    ef_path: Path,
    meta: MetadataParams,
    seg: SegmentationParams,
    qc: QcParams,
    tr: TrackingParams,
    max_frames: int | None,
) -> PairRunResult:
    ctrl = run_single_movie(ctrl_path, meta=meta, seg=seg, qc=qc, tr=tr, max_frames=max_frames)
    ef = run_single_movie(ef_path, meta=meta, seg=seg, qc=qc, tr=tr, max_frames=max_frames)
    return PairRunResult(ctrl=ctrl, ef=ef)


def run_and_export(
    *,
    mode: str,
    run_dir: Path,
    meta: MetadataParams,
    seg: SegmentationParams,
    qc: QcParams,
    tr: TrackingParams,
    out_opts: OutputOptions,
    max_frames: int | None,
    single_path: Path | None = None,
    ctrl_path: Path | None = None,
    ef_path: Path | None = None,
    qc_frame: int = 0,
) -> RunOutput:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "mode": mode,
        "meta": asdict(meta),
        "seg": asdict(seg),
        "qc": asdict(qc),
        "tracking": asdict(tr),
        "output": asdict(out_opts),
        "max_frames": max_frames,
        "single_path": str(single_path) if single_path else None,
        "ctrl_path": str(ctrl_path) if ctrl_path else None,
        "ef_path": str(ef_path) if ef_path else None,
    }
    params_json = run_dir / "params.json"
    params_json.write_text(json.dumps(params, indent=2), encoding="utf-8")

    if mode == "single":
        if single_path is None:
            raise ValueError("single_path is required for mode='single'")
        out_dir = run_dir / "single"
        out_dir.mkdir(parents=True, exist_ok=True)

        res = run_single_movie(single_path, meta=meta, seg=seg, qc=qc, tr=tr, max_frames=max_frames)
        preview_rgb = make_preview_rgb(single_path, res.masks_filt, frame=qc_frame)

        exported = export_csvs(out_dir, meta=meta, single=res, out_opts=out_opts)
        masks_path = export_masks_tiff(out_dir, res.masks_filt) if out_opts.export_masks_tiff else None
        seg_mp4 = (
            export_segmentation_overlay_mp4(out_dir, single_path, res.masks_filt, ef_on_frame=None)
            if out_opts.export_segmentation_overlay_mp4
            else None
        )
        track_mp4 = (
            export_tracking_overlay_mp4(out_dir, single_path, res.masks_filt, res.pts, res.tracks, ef_on_frame=None)
            if out_opts.export_tracking_overlay_mp4
            else None
        )

        exported = exported_with(
            exported,
            params_json=params_json,
            masks_tiff=masks_path,
            segmentation_overlay_mp4=seg_mp4,
            tracking_overlay_mp4=track_mp4,
        )
        summ = _summary(res.tracks, res.pts, res.masks_filt)

        return RunOutput(
            mode="single",
            run_dir=run_dir,
            preview_rgb_single=preview_rgb,
            summary_single=summ,
            exported_single=exported,
        )

    if mode == "pair":
        if ctrl_path is None or ef_path is None:
            raise ValueError("ctrl_path and ef_path are required for mode='pair'")

        ctrl_dir = run_dir / "ctrl"
        ef_dir = run_dir / "ef"
        ctrl_dir.mkdir(parents=True, exist_ok=True)
        ef_dir.mkdir(parents=True, exist_ok=True)

        pair = run_pair(ctrl_path, ef_path, meta=meta, seg=seg, qc=qc, tr=tr, max_frames=max_frames)

        preview_ctrl = make_preview_rgb(ctrl_path, pair.ctrl.masks_filt, frame=qc_frame)
        preview_ef = make_preview_rgb(ef_path, pair.ef.masks_filt, frame=qc_frame)

        exp_ctrl = export_csvs(ctrl_dir, meta=meta, single=pair.ctrl, out_opts=out_opts)
        exp_ef = export_csvs(ef_dir, meta=meta, single=pair.ef, out_opts=out_opts)

        if out_opts.export_masks_tiff:
            exp_ctrl = exported_with(exp_ctrl, masks_tiff=export_masks_tiff(ctrl_dir, pair.ctrl.masks_filt))
            exp_ef = exported_with(exp_ef, masks_tiff=export_masks_tiff(ef_dir, pair.ef.masks_filt))

        if out_opts.export_segmentation_overlay_mp4:
            exp_ctrl = exported_with(
                exp_ctrl,
                segmentation_overlay_mp4=export_segmentation_overlay_mp4(
                    ctrl_dir, ctrl_path, pair.ctrl.masks_filt, ef_on_frame=None
                ),
            )
            exp_ef = exported_with(
                exp_ef,
                segmentation_overlay_mp4=export_segmentation_overlay_mp4(
                    ef_dir, ef_path, pair.ef.masks_filt, ef_on_frame=int(meta.ef_on_frame)
                ),
            )

        if out_opts.export_tracking_overlay_mp4:
            exp_ctrl = exported_with(
                exp_ctrl,
                tracking_overlay_mp4=export_tracking_overlay_mp4(
                    ctrl_dir,
                    ctrl_path,
                    pair.ctrl.masks_filt,
                    pair.ctrl.pts,
                    pair.ctrl.tracks,
                    ef_on_frame=None,
                ),
            )
            exp_ef = exported_with(
                exp_ef,
                tracking_overlay_mp4=export_tracking_overlay_mp4(
                    ef_dir,
                    ef_path,
                    pair.ef.masks_filt,
                    pair.ef.pts,
                    pair.ef.tracks,
                    ef_on_frame=int(meta.ef_on_frame),
                ),
            )

        exp_ctrl = exported_with(exp_ctrl, params_json=params_json)
        exp_ef = exported_with(exp_ef, params_json=params_json)

        summ_ctrl = _summary(pair.ctrl.tracks, pair.ctrl.pts, pair.ctrl.masks_filt)
        summ_ef = _summary(pair.ef.tracks, pair.ef.pts, pair.ef.masks_filt)

        # Pair-level metrics reported from EF only (if µm conversion available)
        if meta.pixels_per_um is not None and not pair.ef.tracks.empty:
            step_start_frame = max(int(meta.ef_on_frame), 1)
            ef_steps = pair.ef.tracks[pair.ef.tracks["frame"] >= step_start_frame]
            dv = _directed_component_um_per_min(ef_steps, meta=meta).dropna()
            mean_dv = float(dv.mean()) if len(dv) else None
            mean_dir = float(pair.ef.per_cell["directedness_cos"].mean()) if (not pair.ef.per_cell.empty and "directedness_cos" in pair.ef.per_cell) else None
            summ_ef = Summary(**{**asdict(summ_ef), "mean_directed_velocity_um_per_min": mean_dv, "mean_directedness": mean_dir})  # type: ignore[arg-type]

        return RunOutput(
            mode="pair",
            run_dir=run_dir,
            preview_rgb_ctrl=preview_ctrl,
            preview_rgb_ef=preview_ef,
            summary_ctrl=summ_ctrl,
            summary_ef=summ_ef,
            exported_ctrl=exp_ctrl,
            exported_ef=exp_ef,
        )

    raise ValueError(f"Unknown mode: {mode}")

