# Bioelectricity project (electrotaxis)

Reproducible notebook to:

- Inspect CTRL/EF TIFF stacks (lazy loading + napari viewing)
- Run Cellpose segmentation + centroid tracking (trackpy)
- Export shareable MP4s:
  - Segmentation overlay (boundaries)
  - Napari-style overlay (filled masks + centroids + track tails/IDs)

## Files

- `inspect_data.ipynb`: main workflow notebook
- `inspect_tiffs.py`: quick TIFF metadata inspector
- `streamlit_app.py`: local GUI for non-coders
- `app_core/`: processing backend used by the GUI

## Setup (recommended)

Use a dedicated virtual environment (example):

```bash
python -m venv .venv-cyto2
.\.venv-cyto2\Scripts\python.exe -m pip install -r requirements.txt
```

Then run the notebook in that environment/kernel.

## Run the GUI (local web app)

From the project folder (in the same environment where dependencies are installed):

```bash
python -m streamlit run streamlit_app.py
```

### No-terminal option (recommended for teammates)

On Windows, you can **double-click**:

- `Run_Electrotaxis_App.bat`

On **first run**, it will automatically create the local environment (`.venv-cyto2/`) and install everything in `requirements.txt` (includes napari), then open the app in your browser (usually `http://localhost:8501`).

If you prefer to install first (or need to re-install), double-click:

- `Setup_Electrotaxis.bat`

If it doesn't open, check `launcher.log` in the project folder for the error.

The app writes outputs into `results/<run_name>/...`.

### macOS / Linux launcher scripts

In Terminal (first time you may need `chmod +x *.sh`):

- Install deps: `./Setup_Electrotaxis.sh`
- Run app: `./Run_Electrotaxis_App.sh`

These scripts create `.venv-cyto2/`, install requirements (macOS uses `requirements_mac.txt` if present), then run Streamlit on `http://localhost:8501`.

## Notes

- Large inputs/outputs (TIFFs, PDFs, MP4s, `results/`, `.venv-cyto2/`) are intentionally ignored by git via `.gitignore`.

