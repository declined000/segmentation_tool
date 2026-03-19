from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import streamlit as st

from app_core.env_check import check_cellpose_cyto2, check_napari_qt, check_torch
from app_core.napari_launch import launch_napari
from app_core.pipeline import run_and_export
from app_core.types import (
    MetadataParams,
    OutputOptions,
    QcParams,
    SegmentationParams,
    TrackingParams,
)


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = APP_ROOT / "results"

HELP = {
    # Input / run control
    "mode": (
        "Choose what you want to analyze.\n\n"
        "- **Single**: one movie (no EF comparison)\n"
        "- **Pair**: one CTRL movie + one EF movie\n"
    ),
    "input_method": (
        "How to provide files.\n\n"
        "- **Path**: fastest, recommended (files stay where they are).\n"
        "- **Upload**: copies the file into this run folder."
    ),
    "movie_path": "Path to a .tif/.tiff movie on your computer.",
    "results_root": (
        "Where to save outputs.\n\n"
        "Outputs go to: `results/<run_name>/single/*` or `results/<run_name>/ctrl/*` + `/ef/*`."
    ),
    "run_name": (
        "A name for this run (used as the output folder).\n\n"
        "Tip: use something like `EF_200mV_rep1` so you can find it later."
    ),
    "preview_frames": (
        "How many frames to process in Preview mode.\n\n"
        "Preview is for tuning parameters before a full run so you can see a preview of your settings before you spend a lot of time analyzing the image.\n\n"
        "Typical: max 5-6\n\n"
        "If segmentation looks wrong, tweak diameter/thresholds and preview again."
    ),
    "run_mode": (
        "Preview: fast test on first N frames.\n"
        "Full: process the entire movie.\n"
        "Safe workflow: ALWAYS Preview first → Full once it looks good."
    ),
    "qc_frame": (
        "Which frame number to show in the **quick preview image**.\n\n"
        "**Does NOT affect processing.** It only changes what you see.\n\n"
        "Example for a 37-frame movie:\n"
        "- frame 0 = first image\n"
        "- frame 10 = middle\n"
        "- frame 36 = last\n"
    ),

    # Metadata
    "dt_min": (
        "Time between frames (minutes).\n\n"
        "Used to compute speeds (µm/min). If this is wrong, speeds will be wrong. Default for corneal data: 5.0."
    ),
    "pixels_per_um": (
        "Spatial calibration.\n\n"
        "Used to convert pixels → µm and compute density.\n"
        "- If unknown: leave blank (outputs stay in pixels).\n"
        "- If wrong: all physical units will be wrong.\n\n"
        "Example: `1.135` means 1.135 pixels per µm."
    ),
    "ef_on_frame": (
        "Frame index when the electric field starts (pair mode only).\n\n"
        "Example: if EF starts after the first image, use **1** (frame 0 is baseline)."
    ),
    "ef_axis": (
        "Which image axis corresponds to the EF direction.\n\n"
        "- `x`: left↔right\n"
        "- `y`: top↔bottom"
    ),
    "ef_sign": (
        "Which direction should count as **positive toward the cathode**.\n\n"
        "For your setup (cathode on the left): choose **left positive (-1)**."
    ),

    # Segmentation
    "model_type": (
        "Cellpose model.\n\n"
        "- **cyto2**: whole cells (phase contrast / brightfield)\n"
        "- **nuclei**: fluorescent nuclei\n\n"
        "For your movies: **cyto2**."
    ),
    "diameter_auto": (
        "If enabled, Cellpose estimates cell size automatically.\n\n"
        "Use Auto when you don't know the size or it varies a lot.\n"
        "Turn off Auto if cells are over-split or merged and you want control."
    ),
    "diameter_px": (
        "Expected cell diameter in pixels.\n\n"
        "If cells are **split into fragments** → increase diameter.\n"
        "If cells are **merged together** → decrease diameter.\n\n"
        "Safe starting point: **30 px** for medium adherent cells."
    ),
    "cellprob_thr": (
        "How strict detection is.\n\n"
        "Higher = fewer detections (more strict).\n"
        "Lower = more detections (can include noise).\n\n"
        "Start around **0.0–0.2**."
    ),
    "flow_thr": (
        "How strict mask shape consistency is.\n\n"
        "Higher = stricter masks (fewer weird shapes).\n"
        "Lower = more sensitive but can be noisier.\n\n"
        "Start around **0.4**."
    ),
    "use_gpu": (
        "Use GPU acceleration (if CUDA is correctly installed).\n\n"
        "If you're unsure, leave it off — CPU works fine for 37 frames."
    ),
    "denoise": (
        "Use Cellpose3 built-in image restoration before segmentation.\n\n"
        "Recommended for **noisy or phase-contrast images** with halo/ring artifacts.\n"
        "The model denoises each frame before detecting cells, reducing false detections.\n\n"
        "Adds ~1-2 seconds per frame. Use **Preview** to check if it helps."
    ),

    # QC
    "min_area": (
        "Remove small objects (debris) before tracking.\n\n"
        "If tiny specks remain → increase.\n"
        "If real small cells disappear → decrease."
    ),
    "max_area": (
        "Remove very large objects (merged blobs) before tracking.\n\n"
        "If merged blobs remain → decrease.\n"
        "If large real cells disappear → increase."
    ),
    "border_px": (
        "Ignore detections near the image edge.\n\n"
        "Edge cells are often cut off and cause tracking errors.\n"
        "If you lose too many valid edge cells → decrease.\n"
        "If you see half-cells/edge artifacts → increase."
    ),
    "min_solidity": (
        "Keeps objects that are more \"filled-in\" (drops ring/bubble-like artifacts).\n\n"
        "Higher = stricter (fewer kept).\n"
        "Lower = more permissive.\n\n"
        "Notebook default: **0.80**."
    ),
    "min_eccentricity": (
        "Keeps objects that are at least a bit elongated (drops very round artifacts).\n\n"
        "Higher = stricter (keeps more elongated shapes).\n"
        "Lower = more permissive.\n\n"
        "Notebook default: **0.15**."
    ),
    "max_circularity": (
        "Drops objects that are *too circular* (often bubbles/debris).\n\n"
        "Lower = stricter (removes more round objects).\n"
        "Higher = more permissive.\n\n"
        "Notebook default: **0.90**."
    ),

    # Tracking
    "apply_drift_correction": (
        "Subtract median frame-to-frame displacement from all tracks to remove global drift.\n\n"
        "Turn **off** if your experiment has no stage drift or if you want raw uncorrected tracks.\n\n"
        "Turn **on** (default) if the field of view shifts over time."
    ),
    "search_range": (
        "Max distance (pixels) a cell can move between frames.\n\n"
        "If tracks break often → increase.\n"
        "If tracks jump between nearby cells → decrease.\n\n"
        "Rule of thumb: slightly larger than the biggest expected per-frame movement."
    ),
    "memory": (
        "How many frames a cell may be missing and still keep the same track.\n\n"
        "If cells flicker/disappear briefly → increase.\n"
        "If identity swaps happen → decrease.\n\n"
        "Safe default: **1**."
    ),
    "min_track_len": (
        "Minimum number of frames required for a track to be kept.\n\n"
        "If too many short noisy tracks → increase.\n"
        "If you lose real but short-lived cells → decrease.\n\n"
        "Safe default for ~37 frames: **10**."
    ),
    "jump_factor": (
        "Cuts tracks when they make an unrealistic jump.\n\n"
        "Higher = allow bigger jumps.\n"
        "Lower = stricter (can reduce identity swaps).\n\n"
        "Safe default: **2.5**."
    ),

    # Outputs
    "export_tracks": "Export `tracks.csv` (per-frame tracking table). Usually useful.",
    "export_per_cell": "Export `per_cell.csv` (one row per tracked cell). Great for electrotaxis summaries.",
    "export_per_frame": (
        "Export `per_frame.csv` (frame-level: n_cells, mean area, and density if calibrated)."
    ),
    "export_masks": (
        "Export `masks_filt.tif` (label image stack).\n\n"
        "Useful for opening later in Napari. Optional."
    ),
    "export_seg_mp4": "Export `segmentation_overlay.mp4` (image + mask boundaries). Good for sharing/QC.",
    "export_track_mp4": "Export `tracking_overlay.mp4` (filled masks + centroids + track tails/IDs).",
    "export_step_csv": (
        "Export `per_step_velocity.csv` (advanced).\n\n"
        "Only needed if someone wants per-step directed velocity analysis."
    ),
    "open_napari_qc": (
        "Enable the Napari QC tools in this app.\n\n"
        "After a run, use **Launch Napari Now** to open the interactive viewer."
    ),
    "cell_preset": (
        "Pre-fill segmentation, QC, and tracking settings for common cell types.\n\n"
        "- **Elongated (corneal epithelial)**: default settings, tuned for medium elongated cells.\n"
        "- **Round / unpolarized**: looser shape filters for round cells that haven't polarized yet.\n"
        "- **Large (e.g. microglia)**: bigger diameter + area range + search range.\n"
        "- **Custom**: start from current values and tune manually.\n\n"
        "After selecting a preset, use **Preview** to verify the settings look right."
    ),
}

CELL_PRESETS = {
    "Elongated (corneal epithelial)": {
        "p_diameter_px": 30.0,
        "p_min_area": 200,
        "p_max_area": 6000,
        "p_border_px": 8,
        "p_min_solidity": 0.80,
        "p_min_eccentricity": 0.15,
        "p_max_circularity": 0.90,
        "p_search_range": 20.0,
        "p_memory": 1,
    },
    "Round / unpolarized": {
        "p_diameter_px": 30.0,
        "p_min_area": 400,
        "p_max_area": 6000,
        "p_border_px": 8,
        "p_min_solidity": 0.80,
        "p_min_eccentricity": 0.0,
        "p_max_circularity": 0.98,
        "p_search_range": 20.0,
        "p_memory": 2,
    },
    "Large (e.g. microglia)": {
        "p_diameter_px": 60.0,
        "p_min_area": 500,
        "p_max_area": 15000,
        "p_border_px": 15,
        "p_min_solidity": 0.70,
        "p_min_eccentricity": 0.0,
        "p_max_circularity": 0.98,
        "p_search_range": 30.0,
        "p_memory": 2,
    },
    "Custom": None,
}


def _save_upload(upload, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(upload.getbuffer())
    return dest


def _env_panel(wants_napari: bool):
    st.markdown("#### Environment check")
    statuses = [check_cellpose_cyto2(), check_torch()]
    if wants_napari:
        statuses.append(check_napari_qt())

    ok_all = True
    for s in statuses:
        if s.ok:
            st.success(f"{s.title}: OK")
        else:
            ok_all = False
            st.error(f"{s.title}: {s.details}")
    return ok_all


def _summary_box(title: str, summary):
    st.markdown(f"**{title}**")
    if summary is None:
        st.write("No summary.")
        return
    st.write(
        {
            "#frames": summary.n_frames,
            "#detections_kept": summary.n_cells,
            "#tracks": summary.n_tracks,
            "mean_track_len": summary.mean_track_len,
            "mean_directed_velocity_um_per_min": summary.mean_directed_velocity_um_per_min,
            "mean_directedness": summary.mean_directedness,
        }
    )


def _download_chip(*, label: str, path: Path | None, mime: str, key: str):
    if path is None:
        return
    p = Path(path)
    if not p.exists() or not p.is_file():
        return
    try:
        data = p.read_bytes()
    except OSError:
        return
    st.download_button(
        label=label,
        data=data,
        file_name=p.name,
        mime=mime,
        key=key,
    )


st.set_page_config(page_title="Electrotaxis Segmentation + Tracking", layout="wide")

st.markdown(
    """
<style>
/* Overall app */
.stApp { background: #f3f4f6; }
div.block-container { padding-top: 1.75rem; max-width: 1200px; }

/* Top bar: subtle (keeps menu but less visual weight) */
header[data-testid="stHeader"] { background: rgba(243, 244, 246, 0.6); backdrop-filter: blur(6px); }
div[data-testid="stDecoration"] { display: none; }

/* Pull sidebar content up (reduce the top gap) */
/* Streamlit offsets the sidebar below the top header bar; override so the sidebar starts higher. */
section[data-testid="stSidebar"] {
  top: 0rem !important;
  height: 100vh !important;
  padding-top: 0rem !important;
  background: linear-gradient(180deg, #334155 0%, #1e293b 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
  padding-top: 0rem !important;
}
/* Sidebar typography (don't force on *; it breaks light widgets) */
section[data-testid="stSidebar"] :is(p, span, small, label, li, a, div, h1, h2, h3, h4) { color: #f1f5f9; }
section[data-testid="stSidebar"] a { text-decoration: none; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.10) !important; }

/* Sidebar brand + section headers */
.sb-brand { display:flex; gap:0.65rem; align-items:center; margin: 0.25rem 0 0.85rem 0; }
.sb-logo {
  width: 36px; height: 36px; border-radius: 10px;
  display:flex; align-items:center; justify-content:center;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.10);
  font-size: 18px;
}
.sb-brand-title { font-size: 1.05rem; font-weight: 750; line-height: 1.05; margin: 0; padding: 0; }
.sb-brand-sub { font-size: 0.90rem; opacity: 0.85; margin: 0.10rem 0 0 0; padding: 0; }
.sb-section { font-size: 0.95rem; font-weight: 700; margin: 0.75rem 0 0.25rem 0; opacity: 0.95; }
.sb-section span { opacity: 0.85; margin-right: 0.35rem; }

/* Sidebar inputs: use light fields with dark text (readable) */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  background: rgba(248,250,252,0.98) !important;
  color: #111827 !important;
  caret-color: #111827 !important;
  border: 1px solid rgba(148,163,184,0.9) !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
  color: #64748b !important;
}

/* BaseWeb input wrapper (number_input etc.) */
section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
  background: rgba(248,250,252,0.98) !important;
  border-color: rgba(148,163,184,0.9) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] input {
  color: #111827 !important;
  caret-color: #111827 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder {
  color: #64748b !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] button {
  color: #334155 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] svg {
  color: #334155 !important;
}

/* Widget labels + help icons (keep readable on dark sidebar) */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
  color: rgba(241,245,249,0.98) !important;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] svg {
  color: #111827 !important;
  fill: #111827 !important;
  stroke: #111827 !important;
}

/* Help tooltip icon: force a real ?-in-circle icon (white circle, black outline + ?) */
:root{
  --help-qmark-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='10' fill='white' stroke='%23111827' stroke-width='2'/%3E%3Cpath fill='%23111827' d='M12 7a3.2 3.2 0 0 0-3.2 3.2h2A1.2 1.2 0 1 1 12 11.4c-.7.5-2 1.3-2 3.1V15h2v-.5c0-1 .5-1.4 1.3-2 1-.7 1.9-1.6 1.9-3.3A3.2 3.2 0 0 0 12 7z'/%3E%3Cpath fill='%23111827' d='M11 18h2v2h-2z'/%3E%3C/svg%3E");
}

section[data-testid="stSidebar"] [data-testid="stTooltipIcon"]{
  width: 20px !important;
  height: 20px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  opacity: 1 !important;
  cursor: help !important;
}
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg{
  opacity: 0 !important; /* keep hover target for tooltip */
}
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"]::before{
  content: "" !important;
  position: absolute !important;
  inset: 0 !important;
  background: var(--help-qmark-icon) center/20px 20px no-repeat !important;
  pointer-events: none !important; /* let hover reach underlying tooltip target */
}

/* Some Streamlit versions render the help icon as a button (not stTooltipIcon). Force it visible. */
section[data-testid="stSidebar"] button[aria-label="Help"],
section[data-testid="stSidebar"] button[title="Help"],
section[data-testid="stSidebar"] button[aria-label="help"],
section[data-testid="stSidebar"] button[title="help"] {
  opacity: 1 !important;
  visibility: visible !important;
  border-radius: 999px !important;
  width: 20px !important;
  height: 20px !important;
  padding: 0 !important;
  position: relative !important;
  background: transparent !important;
  cursor: help !important;
}
section[data-testid="stSidebar"] button[aria-label="Help"] svg,
section[data-testid="stSidebar"] button[title="Help"] svg,
section[data-testid="stSidebar"] button[aria-label="help"] svg,
section[data-testid="stSidebar"] button[title="help"] svg {
  opacity: 0 !important; /* keep hover target for tooltip */
}
section[data-testid="stSidebar"] button[aria-label="Help"]:hover svg,
section[data-testid="stSidebar"] button[title="Help"]:hover svg,
section[data-testid="stSidebar"] button[aria-label="help"]:hover svg,
section[data-testid="stSidebar"] button[title="help"]:hover svg {
  opacity: 0 !important;
}
section[data-testid="stSidebar"] button[aria-label="Help"]::before,
section[data-testid="stSidebar"] button[title="Help"]::before,
section[data-testid="stSidebar"] button[aria-label="help"]::before,
section[data-testid="stSidebar"] button[title="help"]::before {
  content: "" !important;
  position: absolute !important;
  inset: 0 !important;
  background: var(--help-qmark-icon) center/20px 20px no-repeat !important;
  pointer-events: none !important;
}

/* Selects */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(248,250,252,0.98) !important;
  border-color: rgba(148,163,184,0.9) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * {
  color: #111827 !important;
}

/* Expanders in sidebar: keep dark so text is readable */
section[data-testid="stSidebar"] details {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
  overflow: hidden;
}
section[data-testid="stSidebar"] details > summary {
  background: rgba(255,255,255,0.06) !important;
  color: #e5e7eb !important;
}
section[data-testid="stSidebar"] details > summary * { color: #e5e7eb !important; }

/* File uploader (dropzone) readability in dark sidebar */
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: rgba(248,250,252,0.98) !important; /* light box */
  border: 1px dashed rgba(148,163,184,0.9) !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
  color: #111827 !important; /* dark text on light box */
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
  color: #334155 !important;
  opacity: 0.95 !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
  background: #ffffff !important;
  border: 1px solid rgba(148,163,184,0.9) !important;
  color: #111827 !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] button:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
  background: #f1f5f9 !important;
}

/* Tidy download buttons so they look like small chips */
div[data-testid="stDownloadButton"] button {
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  font-size: 0.9rem;
}

/* Primary action button */
button[data-testid="stBaseButton-primary"] {
  background: #2563eb !important;
  border: 1px solid rgba(37,99,235,0.65) !important;
}
button[data-testid="stBaseButton-primary"]:hover {
  background: #1d4ed8 !important;
}

/* Alerts look more like cards */
div[data-testid="stAlert"] { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='font-size:2.1rem; font-weight:800; letter-spacing:-0.02em; margin: 0 0 0.75rem 0;'>"
    "Electrotaxis Segmentation + Tracking</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
<div class="sb-brand">
  <div class="sb-logo">⚡</div>
  <div>
    <div class="sb-brand-title">Electrotaxis</div>
    <div class="sb-brand-sub">Segmentation + Tracking</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sb-section'><span>☰</span>Input</div>", unsafe_allow_html=True)
    mode = st.radio("Mode", ["single", "pair"], horizontal=True, help=HELP["mode"])

    input_method = st.radio(
        "File input method",
        ["Path (recommended)", "Upload"],
        horizontal=False,
        help=HELP["input_method"],
    )

    if input_method.startswith("Path"):
        if mode == "single":
            single_path = st.text_input(
                "Movie path (.tif/.tiff)",
                value=str(APP_ROOT / "EF_200mV_5min_interval-2.tif"),
                help=HELP["movie_path"],
            )
            ctrl_path = ""
            ef_path = ""
        else:
            ctrl_path = st.text_input(
                "CTRL movie path (.tif/.tiff)",
                value=str(APP_ROOT / "Ctrl_0mV_5min_interval-2.tif"),
                help="Path to the CTRL movie TIFF.",
            )
            ef_path = st.text_input(
                "EF movie path (.tif/.tiff)",
                value=str(APP_ROOT / "EF_200mV_5min_interval-2.tif"),
                help="Path to the EF movie TIFF.",
            )
            single_path = ""
        upload_single = upload_ctrl = upload_ef = None
    else:
        if mode == "single":
            upload_single = st.file_uploader(
                "Movie (.tif/.tiff)",
                type=["tif", "tiff"],
                help="Uploads will be copied into `results/<run_name>/inputs/`.",
            )
            upload_ctrl = upload_ef = None
        else:
            upload_ctrl = st.file_uploader(
                "CTRL (.tif/.tiff)",
                type=["tif", "tiff"],
                help="Uploads will be copied into `results/<run_name>/inputs/`.",
            )
            upload_ef = st.file_uploader(
                "EF (.tif/.tiff)",
                type=["tif", "tiff"],
                help="Uploads will be copied into `results/<run_name>/inputs/`.",
            )
            upload_single = None
        single_path = ctrl_path = ef_path = ""

    st.divider()
    st.markdown("<div class='sb-section'><span>⤓</span>Output</div>", unsafe_allow_html=True)
    results_root = Path(st.text_input("Results folder", value=str(DEFAULT_RESULTS), help=HELP["results_root"]))
    run_name_in = st.text_input("Run name (optional)", value="", help=HELP["run_name"])
    run_name = (run_name_in or "").strip() or time.strftime("%Y%m%d_%H%M%S")

    st.divider()
    st.markdown("<div class='sb-section'><span>▶</span>Preview / full run</div>", unsafe_allow_html=True)
    preview_frames = st.slider(
        "Preview frames",
        min_value=2,
        max_value=20,
        value=5,
        step=1,
        help=HELP["preview_frames"],
    )
    run_mode = st.radio("Run mode", ["Preview", "Full"], horizontal=True, help=HELP["run_mode"])
    qc_frame = st.number_input(
        "QC frame index to display",
        min_value=0,
        value=0,
        step=1,
        help=HELP["qc_frame"],
    )

    st.divider()
    st.markdown("<div class='sb-section'><span>⚙</span>Imaging metadata</div>", unsafe_allow_html=True)
    dt_min = st.number_input("Δt (minutes)", min_value=0.01, value=5.0, step=0.5, help=HELP["dt_min"])
    pixels_per_um_val = st.text_input(
        "Pixels per µm (leave blank if unknown)",
        value="1.135",
        help=HELP["pixels_per_um"],
    )
    pixels_per_um = float(pixels_per_um_val) if pixels_per_um_val.strip() else None

    if mode == "pair":
        ef_on_frame = st.number_input("EF ON frame", min_value=0, value=1, step=1, help=HELP["ef_on_frame"])
        ef_axis = st.selectbox("EF axis", ["x", "y"], help=HELP["ef_axis"])
        ef_sign = st.selectbox(
            "EF sign (toward cathode)",
            ["left positive (-1)", "right positive (+1)"],
            help=HELP["ef_sign"],
        )
        ef_sign_val = -1 if ef_sign.startswith("left") else 1
    else:
        ef_on_frame = 1
        ef_axis = "x"
        ef_sign_val = -1

    st.divider()
    st.markdown("<div class='sb-section'><span>🔬</span>Cell type preset</div>", unsafe_allow_html=True)
    _preset_name = st.selectbox(
        "Cell type",
        list(CELL_PRESETS.keys()),
        index=0,
        help=HELP["cell_preset"],
    )
    if _preset_name != st.session_state.get("_last_preset"):
        st.session_state["_last_preset"] = _preset_name
        _pvals = CELL_PRESETS[_preset_name]
        if _pvals is not None:
            for _k, _v in _pvals.items():
                st.session_state[_k] = _v
            st.rerun()

    with st.expander("Segmentation", expanded=True):
        st.caption(
            "Defaults are tuned for medium adherent cells (~30 px diameter). "
            "For larger cells (e.g. microglia), increase **Diameter** and use **Preview** to check."
        )
        model_type = st.selectbox("Model", ["cyto2", "nuclei"], index=0, help=HELP["model_type"])
        diameter_auto = st.checkbox("Auto diameter", value=False, help=HELP["diameter_auto"])
        diameter_px = None if diameter_auto else float(
            st.number_input("Diameter (px)", min_value=5.0, value=30.0, step=1.0, help=HELP["diameter_px"], key="p_diameter_px")
        )
        cellprob_thr = float(
            st.slider(
                "Cellprob threshold",
                min_value=-6.0,
                max_value=6.0,
                value=0.0,
                step=0.1,
                help=HELP["cellprob_thr"],
            )
        )
        flow_thr = float(
            st.slider(
                "Flow threshold",
                min_value=0.0,
                max_value=2.0,
                value=0.4,
                step=0.05,
                help=HELP["flow_thr"],
            )
        )
        use_gpu = st.checkbox("Use GPU", value=False, help=HELP["use_gpu"])
        use_denoise = st.checkbox("Denoise before segmentation (Cellpose3)", value=False, help=HELP["denoise"])

    with st.expander("QC filter", expanded=False):
        st.caption(
            "These filters remove debris and artifacts before tracking. "
            "For larger cells, raise **Max area**. "
            "For rounder cells (e.g. unpolarized), raise **Max circularity** to ~0.98 "
            "and lower **Min eccentricity** to 0. "
            "If many real cells are missing, try raising **Min area** to filter debris instead."
        )
        min_area = int(st.number_input("Min area (px²)", min_value=0, value=200, step=50, help=HELP["min_area"], key="p_min_area"))
        max_area = int(st.number_input("Max area (px²)", min_value=0, value=6000, step=200, help=HELP["max_area"], key="p_max_area"))
        border_px = int(
            st.number_input("Border exclusion (px)", min_value=0, value=8, step=1, help=HELP["border_px"], key="p_border_px")
        )
        min_solidity = float(
            st.slider("Min solidity", min_value=0.0, max_value=1.0, value=0.80, step=0.01, help=HELP["min_solidity"], key="p_min_solidity")
        )
        min_eccentricity = float(
            st.slider(
                "Min eccentricity",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                help=HELP["min_eccentricity"],
                key="p_min_eccentricity",
            )
        )
        max_circularity = float(
            st.slider(
                "Max circularity",
                min_value=0.0,
                max_value=1.0,
                value=0.90,
                step=0.01,
                help=HELP["max_circularity"],
                key="p_max_circularity",
            )
        )

    with st.expander("Tracking", expanded=False):
        st.caption(
            "Controls how cells are linked across frames. "
            "If larger/faster cells lose their tracks, increase **Search range**. "
            "If cells flicker in and out of detection, increase **Memory** to 2-3."
        )
        search_range = float(
            st.number_input("Search range (px)", min_value=1.0, value=20.0, step=1.0, help=HELP["search_range"], key="p_search_range")
        )
        memory = int(st.number_input("Memory (frames)", min_value=0, value=1, step=1, help=HELP["memory"], key="p_memory"))
        min_track_len = int(
            st.number_input("Min track length", min_value=1, value=10, step=1, help=HELP["min_track_len"])
        )
        jump_factor = float(
            st.number_input("Jump max factor", min_value=1.0, value=2.5, step=0.1, help=HELP["jump_factor"])
        )
        apply_drift = st.checkbox("Apply drift correction", value=True, help=HELP["apply_drift_correction"])

    with st.expander("Outputs", expanded=False):
        export_tracks = st.checkbox("Export tracks.csv", value=True, help=HELP["export_tracks"])
        export_per_cell = st.checkbox("Export per_cell.csv", value=True, help=HELP["export_per_cell"])
        export_per_frame = st.checkbox("Export per_frame.csv", value=True, help=HELP["export_per_frame"])
        export_masks = st.checkbox("Export masks TIFF (masks_filt.tif)", value=False, help=HELP["export_masks"])
        export_seg_mp4 = st.checkbox("Export segmentation overlay video", value=True, help=HELP["export_seg_mp4"])
        export_track_mp4 = st.checkbox("Export tracking overlay video", value=True, help=HELP["export_track_mp4"])
        export_step_csv = st.checkbox("Export per-step directed velocity CSV", value=False, help=HELP["export_step_csv"])

    open_napari_qc = st.checkbox("Open Napari QC", value=False, help=HELP["open_napari_qc"])


with st.container(border=True):
    env_ok = _env_panel(wants_napari=open_napari_qc)

if not env_ok:
    st.stop()


def _resolve_inputs() -> tuple[str | None, str | None, str | None]:
    run_dir = results_root / run_name
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    if input_method.startswith("Path"):
        if mode == "single":
            return single_path.strip() or None, None, None
        return None, ctrl_path.strip() or None, ef_path.strip() or None

    # Upload mode: save to disk.
    if mode == "single":
        if upload_single is None:
            return None, None, None
        p = _save_upload(upload_single, input_dir / upload_single.name)
        return str(p), None, None
    else:
        if upload_ctrl is None or upload_ef is None:
            return None, None, None
        pc = _save_upload(upload_ctrl, input_dir / upload_ctrl.name)
        pe = _save_upload(upload_ef, input_dir / upload_ef.name)
        return None, str(pc), str(pe)


col_run, col_napari = st.columns([2, 1])

with col_run:
    with st.container(border=True):
        st.markdown("#### Run")
        can_run = True
        sp, cp, ep = _resolve_inputs()
        if mode == "single" and not sp:
            can_run = False
            st.info("Select a movie file to run.")
        if mode == "pair" and (not cp or not ep):
            can_run = False
            st.info("Select both CTRL and EF movie files to run.")

        if st.button("Run now", type="primary", disabled=not can_run):
            run_dir = results_root / run_name
            max_frames = int(preview_frames) if run_mode == "Preview" else None

            meta = MetadataParams(
                dt_min=float(dt_min),
                pixels_per_um=pixels_per_um,
                ef_on_frame=int(ef_on_frame),
                ef_axis=str(ef_axis),
                ef_sign=int(ef_sign_val),
            )
            seg = SegmentationParams(
                model_type=str(model_type),
                diameter_px=diameter_px,
                cellprob_threshold=float(cellprob_thr),
                flow_threshold=float(flow_thr),
                use_gpu=bool(use_gpu),
                denoise=bool(use_denoise),
            )
            qc = QcParams(
                min_area_px=int(min_area),
                max_area_px=int(max_area),
                border_px=int(border_px),
                min_solidity=float(min_solidity),
                min_eccentricity=float(min_eccentricity),
                max_circularity=float(max_circularity),
            )
            tr = TrackingParams(
                search_range_px=float(search_range),
                memory=int(memory),
                min_track_len=int(min_track_len if run_mode == "Full" else max(2, int(preview_frames) - 1)),
                jump_max_factor=float(jump_factor),
                apply_drift_correction=bool(apply_drift),
            )

            # If user wants napari QC, ensure masks are saved.
            out_opts = OutputOptions(
                export_tracks_csv=bool(export_tracks),
                export_per_cell_csv=bool(export_per_cell),
                export_per_frame_csv=bool(export_per_frame),
                export_masks_tiff=bool(export_masks or open_napari_qc),
                export_segmentation_overlay_mp4=bool(export_seg_mp4),
                export_tracking_overlay_mp4=bool(export_track_mp4),
                export_per_step_velocity_csv=bool(export_step_csv),
            )

            st.markdown(
                f"Processing `{Path(sp).name if sp else (Path(cp).name if cp else '')}`"
                + (" (Preview)" if max_frames is not None else " (Full)")
                + "…"
            )
            st.caption("Steps: Cellpose → QC filter → Tracking → Export")
            prog = st.progress(0, text="Starting…")
            prog.progress(10, text="Running segmentation + tracking… (this can take a bit)")

            with st.spinner("Running segmentation + tracking..."):
                out = run_and_export(
                    mode=mode,
                    run_dir=run_dir,
                    meta=meta,
                    seg=seg,
                    qc=qc,
                    tr=tr,
                    out_opts=out_opts,
                    max_frames=max_frames,
                    single_path=Path(sp) if sp else None,
                    ctrl_path=Path(cp) if cp else None,
                    ef_path=Path(ep) if ep else None,
                    qc_frame=int(qc_frame),
                )
            prog.progress(100, text="Done.")

            st.session_state["last_run"] = out
            st.success(f"Done. Outputs in: {out.run_dir}")


with col_napari:
    with st.container(border=True):
        st.markdown("#### Napari QC")
        st.caption("Launch Napari to do deep quality-control (opens a desktop window).")
        if st.button("Launch Napari Now", disabled=not open_napari_qc):
            last = st.session_state.get("last_run")
            if last is None:
                st.warning("Run the pipeline first.")
            else:
                # Launch napari using the same python environment
                launch_napari(last.run_dir, mode=last.mode, pixels_per_um=pixels_per_um)
                st.success("Launched napari.")


last = st.session_state.get("last_run")
if last is not None:
    with st.container(border=True):
        st.markdown("#### Quick QC (mask boundaries)")
        if last.mode == "single" and last.preview_rgb_single is not None:
            st.image(last.preview_rgb_single, caption="Preview frame with mask boundaries", clamp=True)
        elif last.mode == "pair":
            c1, c2 = st.columns(2)
            if last.preview_rgb_ctrl is not None:
                c1.image(last.preview_rgb_ctrl, caption="CTRL preview (boundaries)", clamp=True)
            if last.preview_rgb_ef is not None:
                c2.image(last.preview_rgb_ef, caption="EF preview (boundaries)", clamp=True)

    with st.container(border=True):
        st.markdown("#### Run summary")
        if last.mode == "single":
            _summary_box("Single", last.summary_single)
        else:
            c1, c2 = st.columns(2)
            with c1:
                _summary_box("CTRL", last.summary_ctrl)
            with c2:
                _summary_box("EF", last.summary_ef)

    with st.container(border=True):
        st.markdown("#### Downloads")
        st.caption("Tip: these are the main files people usually share/analyze.")
        st.write({"run_dir": str(last.run_dir)})

    def _downloads_for(prefix: str, exported):
        cols = st.columns(6)
        with cols[0]:
            _download_chip(
                label="params.json",
                path=getattr(exported, "params_json", None),
                mime="application/json",
                key=f"dl_{prefix}_params",
            )
        with cols[1]:
            _download_chip(
                label="tracks.csv",
                path=getattr(exported, "tracks_csv", None),
                mime="text/csv",
                key=f"dl_{prefix}_tracks",
            )
        with cols[2]:
            _download_chip(
                label="per_cell.csv",
                path=getattr(exported, "per_cell_csv", None),
                mime="text/csv",
                key=f"dl_{prefix}_per_cell",
            )
        with cols[3]:
            _download_chip(
                label="per_frame.csv",
                path=getattr(exported, "per_frame_csv", None),
                mime="text/csv",
                key=f"dl_{prefix}_per_frame",
            )
        with cols[4]:
            _download_chip(
                label="per_step_velocity.csv",
                path=getattr(exported, "per_step_csv", None),
                mime="text/csv",
                key=f"dl_{prefix}_per_step",
            )
        with cols[5]:
            # Videos can be large; show as optional downloads when present.
            _download_chip(
                label="tracking_overlay.mp4",
                path=getattr(exported, "tracking_overlay_mp4", None),
                mime="video/mp4",
                key=f"dl_{prefix}_track_mp4",
            )

    if last.mode == "single" and last.exported_single is not None:
        _downloads_for("single", last.exported_single)
        with st.expander("All exported paths", expanded=False):
            st.write({k: (str(v) if v else None) for k, v in last.exported_single.__dict__.items()})

    if last.mode == "pair":
        if last.exported_ctrl is not None:
            st.markdown("**CTRL**")
            _downloads_for("ctrl", last.exported_ctrl)
            with st.expander("CTRL: all exported paths", expanded=False):
                st.write({k: (str(v) if v else None) for k, v in last.exported_ctrl.__dict__.items()})
        if last.exported_ef is not None:
            st.markdown("**EF**")
            _downloads_for("ef", last.exported_ef)
            with st.expander("EF: all exported paths", expanded=False):
                st.write({k: (str(v) if v else None) for k, v in last.exported_ef.__dict__.items()})

