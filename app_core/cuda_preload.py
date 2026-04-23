from __future__ import annotations

import ctypes
import glob
import os
import platform


def preload_cuda_user_libs(verbose: bool = False) -> list[str]:
    """Preload cuBLAS/cuDNN shared libs from pip nvidia packages.

    Why:
    - Some Linux images expose multiple CUDA library trees (system CUDA, stubs,
      pip nvidia wheels). Dynamic loader order can pick a mismatched libcublasLt
      and crash later at runtime with:
      "Invalid handle. Cannot load symbol cublasLtCreate".
    - Preloading explicit wheel paths with RTLD_GLOBAL stabilizes resolution.

    Returns loaded library paths in preload order.
    """
    if platform.system() != "Linux":
        return []

    # Keep imports local so environments without nvidia wheels still work.
    try:
        import nvidia.cublas.lib as cublas_lib  # type: ignore
        import nvidia.cudnn.lib as cudnn_lib  # type: ignore
    except Exception:
        return []

    cublas_dir = os.path.dirname(cublas_lib.__file__)
    cudnn_dir = os.path.dirname(cudnn_lib.__file__)

    def _pick(pattern: str) -> str | None:
        hits = sorted(glob.glob(pattern))
        return hits[-1] if hits else None

    lib_cublaslt = _pick(os.path.join(cublas_dir, "libcublasLt.so*"))
    lib_cublas = _pick(os.path.join(cublas_dir, "libcublas.so*"))
    lib_cudnn = _pick(os.path.join(cudnn_dir, "libcudnn.so*"))

    loaded: list[str] = []
    for p in (lib_cublaslt, lib_cublas, lib_cudnn):
        if not p:
            continue
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        loaded.append(p)

    if loaded:
        # Make subprocesses and any later dlopen favor these folders.
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        pref = f"{cublas_dir}:{cudnn_dir}"
        os.environ["LD_LIBRARY_PATH"] = f"{pref}:{ld}" if ld else pref
        if verbose:
            print("Preloaded CUDA libs:")
            for p in loaded:
                print(f"  - {p}")

    return loaded
