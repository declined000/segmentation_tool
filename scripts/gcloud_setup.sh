#!/usr/bin/env bash
# Google Cloud VM one-time setup for the bioelectricity pipeline.
# Run this once after SSH-ing into a fresh Deep Learning VM (PyTorch + CUDA).
#
# Usage:
#   chmod +x scripts/gcloud_setup.sh && ./scripts/gcloud_setup.sh
set -euo pipefail

echo "=== 1/4  System packages ==="
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg git

echo "=== 2/4  Python dependencies ==="
pip install --quiet --upgrade pip
pip install --quiet \
    numpy==1.23.5 \
    "pandas>=2.0.3" \
    matplotlib==3.10.8 \
    tifffile==2025.5.10 \
    "dask[array]==2024.8.1" \
    scikit-image==0.24.0 \
    opencv-python-headless==4.11.0.86 \
    cellpose==3.1.1.3 \
    imageio==2.37.2 \
    imageio-ffmpeg==0.6.0

# PyTorch with CUDA should already be on the Deep Learning VM.
# Verify:
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found'; print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')"

echo "=== 3/4  Clone sam4celltracking ==="
if [ ! -d "sam4celltracking" ]; then
    git clone --depth=1 https://github.com/zhuchen96/sam4celltracking.git
else
    echo "  sam4celltracking/ already exists, skipping clone."
fi

echo "=== 4/4  Download SAM2.1 weights ==="
MODEL_DIR="sam4celltracking/src/trained_models"
MODEL_FILE="$MODEL_DIR/sam2.1_hiera_large.pt"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_FILE" ]; then
    wget -q --show-progress -O "$MODEL_FILE" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
else
    echo "  Weights already downloaded."
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "  VRAM:   $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB\")')"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Upload a .tif video:  gcloud compute scp LOCAL_FILE VM_NAME:~/bioelectricity-project/"
echo "  2. Run the pipeline:     python run_cloud_test.py YOUR_FILE.tif"
echo ""
