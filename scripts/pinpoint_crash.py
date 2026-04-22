#!/usr/bin/env python3
"""Pin down EXACTLY where cublasLtCreate fails.

Run: python3 scripts/pinpoint_crash.py
"""
from __future__ import annotations
import sys, os

print("=" * 60)
print("  PINPOINT: where does cuBLAS actually crash?")
print("=" * 60)

# ── Step A: torch matmul (known to work) ─────────────────────
print("\n[A] torch.matmul ... ", end="", flush=True)
import torch
a = torch.randn(64, 64, device="cuda")
b = torch.matmul(a, a)
print(f"OK  (torch {torch.__version__})")
del a, b

# ── Step B: import cellpose ──────────────────────────────────
print("[B] import cellpose.models ... ", end="", flush=True)
from cellpose import models
import inspect
print("OK")

# ── Step C: create model ─────────────────────────────────────
print("[C] CellposeModel(gpu=True, pretrained_model='cpsam') ... ", end="", flush=True)
params = inspect.signature(models.CellposeModel).parameters
kwargs = {"gpu": True, "pretrained_model": "cpsam"}
if "use_bfloat16" in params:
    kwargs["use_bfloat16"] = False
model = models.CellposeModel(**kwargs)
print("OK")

# ── Step D: torch matmul again (still ok after model load?) ──
print("[D] torch.matmul after model load ... ", end="", flush=True)
a = torch.randn(64, 64, device="cuda")
b = torch.matmul(a, a)
print(f"OK")
del a, b

# ── Step E: tiny inference on GPU ────────────────────────────
print("[E] model.eval on 64x64 dummy image (GPU) ... ", end="", flush=True)
import numpy as np
dummy = np.random.randint(0, 255, (64, 64), dtype=np.uint16)
try:
    masks, flows, styles = model.eval(dummy, diameter=30, flow_threshold=0.4,
                                       cellprob_threshold=0.0, normalize=True)
    print(f"OK  (masks shape: {masks.shape}, max label: {masks.max()})")
except Exception as e:
    print(f"\n  *** CRASHED at model.eval: {type(e).__name__}: {e}")
    # Retry on CPU to confirm it's GPU-specific
    print("\n[E2] Retrying model.eval on CPU ... ", end="", flush=True)
    model_cpu = models.CellposeModel(**{**kwargs, "gpu": False})
    masks, flows, styles = model_cpu.eval(dummy, diameter=30, flow_threshold=0.4,
                                           cellprob_threshold=0.0, normalize=True)
    print(f"OK on CPU  (masks: {masks.shape})")
    sys.exit(1)

# ── Step F: bigger image (closer to real data) ───────────────
print("[F] model.eval on 512x512 dummy image (GPU) ... ", end="", flush=True)
dummy_big = np.random.randint(0, 255, (512, 512), dtype=np.uint16)
try:
    masks2, _, _ = model.eval(dummy_big, diameter=30, flow_threshold=0.4,
                               cellprob_threshold=0.0, normalize=True)
    print(f"OK  (masks shape: {masks2.shape})")
except Exception as e:
    print(f"\n  *** CRASHED: {type(e).__name__}: {e}")
    sys.exit(1)

# ── Step G: full pipeline import + single frame ──────────────
print("[G] import app_core.pipeline ... ", end="", flush=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from app_core.pipeline import _run_cellpose_on_stack, _qc_centroids_from_masks
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  ALL STEPS PASSED — the crash is somewhere else.")
print("=" * 60)
