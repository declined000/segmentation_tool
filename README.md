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

## Setup (recommended)

Use a dedicated virtual environment (example):

```bash
python -m venv .venv-cyto2
.\.venv-cyto2\Scripts\python.exe -m pip install -r requirements.txt
```

Then run the notebook in that environment/kernel.

## Notes

- Large inputs/outputs (TIFFs, PDFs, MP4s, `results/`, `.venv-cyto2/`) are intentionally ignored by git via `.gitignore`.

