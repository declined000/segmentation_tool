from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def write_launch_script(
    run_dir: Path,
    *,
    mode: str,
    pixels_per_um: float | None,
) -> Path:
    run_dir = Path(run_dir)
    script_path = run_dir / "launch_napari_qc.py"

    # The script reads params.json to locate inputs, then loads masks/tracks from outputs.
    script = f"""\
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tf
import napari

RUN_DIR = Path(r\"{str(run_dir)}\")
MODE = {mode!r}
PIXELS_PER_UM = {pixels_per_um!r}

params = json.loads((RUN_DIR / 'params.json').read_text(encoding='utf-8'))

def _load_tracks_csv(p: Path):
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    # napari tracks: (n, 4) => [track_id, frame, y, x]
    if not set(['particle','frame','y','x']).issubset(df.columns):
        return None
    return df[['particle','frame','y','x']].to_numpy(dtype=float)

def _viewer_for_one(panel_dir: Path, title: str, movie_path: Path):
    masks = panel_dir / \"masks_filt.tif\"
    tracks_csv = panel_dir / \"tracks.csv\"

    vol = tf.memmap(str(movie_path))
    lab = tf.memmap(str(masks)) if masks.exists() else None
    tracks = _load_tracks_csv(tracks_csv)

    viewer = napari.Viewer(title=title)
    scale = None
    if PIXELS_PER_UM is not None:
        um_per_px = 1.0 / float(PIXELS_PER_UM)
        scale = (1.0, um_per_px, um_per_px)

    viewer.add_image(vol, name='image', scale=scale)
    if lab is not None:
        viewer.add_labels(lab, name='masks', scale=scale)
    if tracks is not None and tracks.shape[0] > 0:
        viewer.add_tracks(tracks, name='tracks', scale=scale)

    return viewer

if MODE == 'single':
    single_dir = RUN_DIR / 'single'
    movie = Path(params['single_path'])
    _viewer_for_one(single_dir, 'QC (single)', movie)
elif MODE == 'pair':
    ctrl_dir = RUN_DIR / 'ctrl'
    ef_dir = RUN_DIR / 'ef'
    ctrl = Path(params['ctrl_path'])
    ef = Path(params['ef_path'])
    _viewer_for_one(ctrl_dir, 'QC (CTRL)', ctrl)
    _viewer_for_one(ef_dir, 'QC (EF)', ef)
else:
    raise RuntimeError(f'Unknown mode: {{MODE}}')

napari.run()
"""

    script_path.write_text(script, encoding="utf-8")
    return script_path


def launch_napari(run_dir: Path, *, mode: str, pixels_per_um: float | None) -> subprocess.Popen:
    script = write_launch_script(run_dir, mode=mode, pixels_per_um=pixels_per_um)
    # Run in a separate process so Streamlit stays responsive.
    return subprocess.Popen([sys.executable, str(script)], cwd=str(run_dir))

