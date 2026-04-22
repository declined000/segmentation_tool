# Roadmap and status

## Implemented (current app)

- **Segmentation:** Cellpose-SAM (`cpsam`) only — no `cyto2` / `nuclei` / denoise branches in code or GUI.
- **QC:** Area, solidity, eccentricity, circularity, border exclusion.
- **Tracking:** SAM2 via cloned `sam4celltracking` + `sam2.1_hiera_large.pt` (Linux or WSL2 + CUDA recommended).
- **Metrics:** Drift correction (optional), directedness, speeds, displacement, lineage CSV where SAM2 provides parent IDs.
- **Delivery:** Streamlit GUI + CSV/MP4 exports; optional `run_cpsam_full.py` for batch masks only.

## Next — verify cpsam is “enough” (your item 4)

Run a **small, documented comparison** on the same movies (Preview then Full):

1. **Segmentation QC:** counts of cells/frame, obvious over-merge / over-split in overlay MP4, halo debris rate (qualitative scorecard).
2. **End-to-end:** same QC + SAM2 settings; compare `#tracks`, mean track length, `#division-like` lineage rows, directedness summaries vs any saved baseline (if you still have old exports).

Record outcomes in `results/` or a short `docs/EVAL_NOTES.md` so “cpsam after fixes” is evidence-based.

## How to *prove* segmentation vs SAM2 (tracking / lineage)

Goal: a defensible sentence like *“masks match expectation at division frames, but IDs/parents are wrong → bottleneck is linking.”*

### 1. Freeze what you compare

- Same raw TIFF, same **QC** sliders, same **`sam4celltracking`** checkout and **weights**.
- If you change segmentation or QC, **re-run the full pipeline** so `masks_filt` and SAM2 always stay in sync.

### 2. Decide if the bug is in **masks** (segmentation / QC)

On **10–20 frames** around a division or tight contact, look only at **`segmentation_overlay.mp4`** or **`masks_filt.tif`** in Napari (ignore track IDs first):

| What you see | Likely cause |
|--------------|----------------|
| One cell **splits into two masks too early**, or two cells **share one mask** when they should not | Segmentation / QC (or diameter if you forced manual diameter) |
| **Halo / bubble** labeled as cell, or real cell **removed** by QC | Segmentation / QC |
| Boundaries look **right every frame**, but you only hate the **colours / IDs jumping** | Not primarily segmentation — go to step 3 |

**Sanity check:** on the frame before vs after division, does the **number and placement** of mask instances match what you expect? If the mask stack never shows two daughters, SAM2 cannot fix that with linking alone.

### 3. Decide if the bug is in **linking** (SAM2 + params)

After step 2 looks good, inspect **`tracks.csv`** / **`lineage.csv`** and the **tracking overlay**:

| What you see | Likely cause |
|--------------|----------------|
| **ID swaps**, one-frame **gaps**, **wrong `parent`**, lineage division count ≠ your manual count on the same short clip | SAM2 linking, `min_track_len`, or SAM2 hyperparameters (`window`, `neighbor_dist`, `dis_threshold`) |
| Problem **moves** when you change SAM2 params **without** changing masks | Strong evidence for **tracking** bottleneck |

**Ablation:** export and **keep one** `masks_filt.tif` run; re-run linking-only if your workflow allows, or re-run full pipeline with **only** SAM2 parameters changed. If lineage errors change a lot while masks are identical, the linker dominates.

### 4. Light numbers (optional, no new models)

- **Short-lived tracks:** fraction of `particle` IDs with ≤2 frames — spikes with good masks suggest broken links, not bad blobs.
- **Per-frame counts:** compare `per_frame.csv` `n_cells` to connected components in `masks_filt` per frame; persistent mismatch suggests drops/merges in the tracked output vs raw masks.

### 5. When a VLM is justified

Only after you can cite step **2** (“masks OK on event frames”) **and** step **3** (“graph still wrong” or “fixed only by SAM2 params, not by mask edits”). Then VLM is a **narrow** tool (ambiguous crop QC, rare division review), not a replacement for this diagnosis.

---

## VLM (deferred — your item 5)

**Testing cpsam alone does *not* fully answer division/merge tracking.**

- **cpsam** mainly improves **per-frame masks** (boundaries, halos, touching cells). Better masks *reduce* linker errors but do not replace **temporal** reasoning.
- **Divisions and merges over time** still depend heavily on **SAM2 linking** and its parameters once masks are reasonable.

Use the **“How to *prove*…”** section above before investing in VLM: it separates **mask errors** from **identity / lineage errors** with a small amount of manual review plus an optional SAM2-only parameter ablation.

## Optional later

- VLM-assisted QC on ambiguous masks (see Cursor plan `full_pipeline_architecture`).
- Public benchmarks (e.g. CTC) vs other tools.
