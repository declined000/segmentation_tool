#!/usr/bin/env bash
# Launch pipeline with deterministic CUDA library ordering on Linux.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  ./scripts/run_cloud_gpu.sh path/to/video.tif [extra run_cloud_test args]"
  echo ""
  echo "Examples:"
  echo "  ./scripts/run_cloud_gpu.sh movie.tif --max-frames 50"
  echo "  ./scripts/run_cloud_gpu.sh movie.tif --sam4ct-path sam4celltracking"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

TORCH_LIB="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
CBLAS_LIB="$(python3 -c 'import nvidia.cublas.lib, os; print(os.path.dirname(nvidia.cublas.lib.__file__))')"
CUDNN_LIB="$(python3 -c 'import nvidia.cudnn.lib, os; print(os.path.dirname(nvidia.cudnn.lib.__file__))')"

# Avoid inheriting stale or stub-prone library paths from VM image defaults.
export LD_LIBRARY_PATH="${TORCH_LIB}:${CBLAS_LIB}:${CUDNN_LIB}"

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
python3 scripts/inspect_cuda_loader.py
python3 scripts/pinpoint_crash.py
python3 run_cloud_test.py "$@"
