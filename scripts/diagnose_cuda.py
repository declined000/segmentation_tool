"""Diagnose CUDA / cuBLAS issues on the cloud VM.

Run:  python3 scripts/diagnose_cuda.py
"""
from __future__ import annotations
import os, sys, subprocess, importlib

def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def main() -> None:
    section("1. Environment")
    print(f"  Python:          {sys.executable}")
    print(f"  Version:         {sys.version}")
    print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(not set)')}")
    print(f"  sys.path (first 5):")
    for p in sys.path[:5]:
        print(f"    {p}")

    section("2. PyTorch")
    try:
        import torch
        print(f"  torch.__file__:  {torch.__file__}")
        print(f"  torch.__version__: {torch.__version__}")
        print(f"  CUDA available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version:    {torch.version.cuda}")
            print(f"  GPU:             {torch.cuda.get_device_name(0)}")
            print(f"  Capability:      {torch.cuda.get_device_capability(0)}")
    except Exception as e:
        print(f"  IMPORT FAILED: {e}")

    section("3. CUDA tensor test (before any other imports)")
    try:
        import torch
        x = torch.zeros(1, device="cuda")
        y = x + 1
        print(f"  torch.zeros + add on CUDA: OK  (result={y.item()})")
        del x, y
    except Exception as e:
        print(f"  FAILED: {e}")

    section("4. cuBLAS test (matmul triggers cuBLAS)")
    try:
        import torch
        a = torch.randn(4, 4, device="cuda")
        b = torch.randn(4, 4, device="cuda")
        c = torch.matmul(a, b)
        print(f"  matmul on CUDA: OK  (shape={c.shape})")
        del a, b, c
    except Exception as e:
        print(f"  FAILED: {e}")

    section("5. Import chain (step-by-step)")
    chain = [
        "numpy",
        "pandas",
        "tifffile",
        "skimage",
        "skimage.measure",
        "skimage.segmentation",
        "imageio",
        "cv2",
        "cellpose",
        "cellpose.models",
    ]
    for mod in chain:
        try:
            importlib.import_module(mod)
            print(f"  import {mod:30s} OK")
        except Exception as e:
            print(f"  import {mod:30s} FAILED: {e}")

        # After each import, test cuBLAS still works
        try:
            import torch
            if torch.cuda.is_available():
                a = torch.randn(2, 2, device="cuda")
                b = torch.matmul(a, a)
                del a, b
        except Exception as e:
            print(f"    ^^ cuBLAS BROKE after importing {mod}: {e}")
            break

    section("6. libcublas files on disk")
    try:
        result = subprocess.run(
            ["find", "/usr", "-name", "libcublasLt*", "-type", "f"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split("\n")[:10]:
            if line:
                print(f"  {line}")
    except Exception:
        pass

    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        result = subprocess.run(
            ["ls", "-la", torch_lib],
            capture_output=True, text=True, timeout=10
        )
        cublas = [l for l in result.stdout.split("\n") if "cublas" in l.lower()]
        for line in cublas[:5]:
            print(f"  torch/lib: {line.strip()}")
    except Exception:
        pass

    section("7. Full pipeline import")
    try:
        from app_core.pipeline import run_single_movie
        print("  app_core.pipeline imported OK")
    except Exception as e:
        print(f"  FAILED: {e}")

    section("8. Cellpose model creation")
    try:
        from cellpose import models
        import inspect
        params = inspect.signature(models.CellposeModel).parameters
        kwargs = {"gpu": True, "pretrained_model": "cpsam"}
        if "use_bfloat16" in params:
            kwargs["use_bfloat16"] = False
        model = models.CellposeModel(**kwargs)
        print(f"  CellposeModel created OK (params: {list(kwargs.keys())})")
    except Exception as e:
        print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print("  Diagnostics complete.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
