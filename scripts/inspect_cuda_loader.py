#!/usr/bin/env python3
"""Inspect CUDA loader paths and detect stub-library collisions.

Run:
    python3 scripts/inspect_cuda_loader.py
"""
from __future__ import annotations

import ctypes
import os
import sys


def _print(msg: str) -> None:
    print(msg, flush=True)


def _is_stub_path(p: str) -> bool:
    q = p.lower().replace("\\", "/")
    return "/stubs/" in q or q.endswith("/stubs")


def main() -> int:
    _print("=== CUDA Loader Inspection ===")
    _print(f"python: {sys.executable}")
    _print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '<unset>')}")

    try:
        from app_core.cuda_preload import preload_cuda_user_libs

        loaded = preload_cuda_user_libs(verbose=True)
        if not loaded:
            _print("preload: no wheel CUDA libs found")
    except Exception as e:
        _print(f"preload failed: {e}")

    try:
        import torch

        _print(f"torch: {torch.__version__}")
        _print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            _print(f"gpu: {torch.cuda.get_device_name(0)}")
            x = torch.randn(8, 8, device="cuda")
            y = torch.matmul(x, x)
            del x, y
            _print("torch matmul: OK")
    except Exception as e:
        _print(f"torch failed: {type(e).__name__}: {e}")
        return 1

    # Try explicit libcublasLt open and print resolved path
    try:
        h = ctypes.CDLL("libcublasLt.so.12", mode=ctypes.RTLD_GLOBAL)
        _print(f"ctypes libcublasLt.so.12 loaded: {h._name}")  # type: ignore[attr-defined]
        if _is_stub_path(str(h._name)):  # type: ignore[attr-defined]
            _print("WARNING: libcublasLt resolved to stubs path")
            return 2
    except Exception as e:
        _print(f"ctypes libcublasLt load failed: {type(e).__name__}: {e}")
        return 1

    _print("inspection: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
