# Electrotaxis Segmentation + Tracking

End-to-end pipeline for cell electrotaxis analysis: **cpsam segmentation** + **SAM2 tracking** + automated electrotaxis metrics.

## Pipeline

1. **Segmentation** -- Cellpose-SAM (`cpsam`) segments cells per-frame with automatic diameter detection
2. **QC filtering** -- removes debris, edge artifacts, and halo rings based on area, solidity, eccentricity, and circularity
3. **Tracking** -- SAM2 (`sam4celltracking`) links cells across frames via backward propagation, recovers missing masks, and detects cell divisions (mitosis)
4. **Metrics** -- drift correction, per-cell speed/directedness, lineage trees, division angles

## Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main GUI (Streamlit web app) |
| `app_core/pipeline.py` | Core processing: segmentation, QC, SAM2 tracking, metrics |
| `app_core/types.py` | Dataclasses for all parameters and result types |
| `app_core/exports.py` | CSV/TIFF/MP4 export logic |
| `run_cpsam_full.py` | Optional local batch: cpsam + QC → `results_cpsam/` (masks + overlay only) |
| `docs/ROADMAP.md` | What is implemented vs planned next |
| `_inspect_napari.py` | Quick local segmentation + napari viewer |
| `_inspect_sam2_tracking.py` | Load SAM2 tracking results into napari |

Jupyter notebooks (`inspect_data*.ipynb`) may still mention legacy `cyto2` kernels; the supported workflow is the Streamlit app above.

## Setup

### 1. Python environment

```bash
python -m venv .venv-cyto2
.venv-cyto2\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. SAM2 tracking setup

The tracking backend requires the `sam4celltracking` repository and the SAM2.1 model.

```bash
# Clone the repo (into the project directory)
git clone --depth=1 https://github.com/zhuchen96/sam4celltracking.git

# Download the SAM2.1 model (~900 MB)
cd sam4celltracking/src/trained_models
# On Linux/Mac:
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# On Windows (PowerShell):
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -OutFile "sam2.1_hiera_large.pt"
cd ../../..
```

**GPU requirement**: SAM2 tracking needs a CUDA-capable GPU with at least 8 GB VRAM. A Tesla T4 (15 GB) or better is recommended.

### Windows + SAM2 (recommended: WSL2)

The upstream [sam4celltracking](https://github.com/zhuchen96/sam4celltracking) project documents **Linux** + CUDA. Native Windows may fail (import errors, missing `.pyd` equivalents for vendored extensions, or CUDA DLL issues) even after you clone the repo and download weights.

**Supported fix:** run the **same Streamlit GUI** inside **WSL2 (Ubuntu 22.04)** with **GPU in WSL** enabled:

1. Install WSL2 + Ubuntu (`wsl --install` or Microsoft Store).
2. Install the **latest NVIDIA driver on Windows** (WSL GPU uses the host driver).
3. In Ubuntu: install Python 3.10+, create a venv, `pip install -r requirements.txt`, clone `sam4celltracking`, download `sam2.1_hiera_large.pt` as above.
4. Put the project in the Linux filesystem for best I/O (e.g. `~/bioelectricity-project`) or run from `/mnt/c/...` (slower but works).
5. From Ubuntu: `python -m streamlit run streamlit_app.py` and open `http://localhost:8501` in the Windows browser.

Official overview: [GPU in WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute).

**Segmentation** (Cellpose / cpsam) can still be used on native Windows; only the **SAM2 linking** step needs the Linux-compatible stack.

## Run the GUI

```bash
python -m streamlit run streamlit_app.py
```

### No-terminal option (Windows)

Double-click `Run_Electrotaxis_App.bat`. On first run it creates `.venv-cyto2/` and installs dependencies automatically.

### macOS / Linux

```bash
chmod +x *.sh
./Setup_Electrotaxis.sh   # first time
./Run_Electrotaxis_App.sh
```

## Outputs

The app writes to `results/<run_name>/`:

- `tracks.csv` -- per-frame tracking table (frame, x, y, particle, parent, fate, generation)
- `per_cell.csv` -- per-cell metrics (speed, directedness, displacement)
- `per_frame.csv` -- frame-level cell counts and density
- `lineage.csv` -- division events, parent/child IDs, division angles
- `masks_filt.tif` -- QC-filtered label masks (optional)
- `segmentation_overlay.mp4` -- image with mask boundaries
- `tracking_overlay.mp4` -- filled masks with centroids and track tails
- `params.json` -- all parameters used for the run

## Notes

- Large files (TIFFs, MP4s, `results/`, `.venv-cyto2/`, `sam4celltracking/`) are git-ignored.
- Segmentation works on CPU; tracking requires GPU.
