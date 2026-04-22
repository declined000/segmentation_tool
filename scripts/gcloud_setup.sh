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
export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet \
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

# Cellpose's pip metadata can pull a PyTorch wheel that does not match the
# VM's NVIDIA driver / libcublas. Re-pin CUDA PyTorch from the official index
# so cuBLAS loads (fixes "Invalid handle ... cublasLtCreate").
echo "  Re-aligning PyTorch + CUDA (cu124 wheels work on GCP CUDA 12.x drivers) ..."
python3 -m pip install --quiet --upgrade torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

_TORCH_LIB="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
# Persist for new SSH sessions (idempotent marker)
if ! grep -q "bioelectricity-torch-lib" ~/.bashrc 2>/dev/null; then
    echo "export LD_LIBRARY_PATH=\"${_TORCH_LIB}:\${LD_LIBRARY_PATH:-}\"  # bioelectricity-torch-lib" >> ~/.bashrc
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not found'; x=torch.zeros(1, device='cuda'); print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}, cuda tensor OK')"

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
echo "  GPU:    $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "  VRAM:   $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB\")')"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Upload a .tif video:  gcloud compute scp LOCAL_FILE VM_NAME:~/bioelectricity-project/"
echo "  2. Run the pipeline:     python3 run_cloud_test.py YOUR_FILE.tif"
echo ""
