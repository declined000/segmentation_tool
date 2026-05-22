from __future__ import annotations

import ctypes
import glob
import os
import platform
import site


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

    def _lib_dir_from_module_or_site(
        module_name: str,
        rel_lib_path: tuple[str, ...],
    ) -> str | None:
        # Keep imports local so environments without nvidia wheels still work.
        try:
            mod = __import__(module_name, fromlist=["dummy"])
            mod_file = getattr(mod, "__file__", None)
            if isinstance(mod_file, str) and mod_file:
                d = os.path.dirname(mod_file)
                if os.path.isdir(d):
                    return d
        except Exception:
            pass

        # Fallback for namespace-package layouts where __file__ may be None.
        for root in site.getsitepackages() + [site.getusersitepackages()]:
            cand = os.path.join(root, *rel_lib_path)
            if os.path.isdir(cand):
                return cand
        return None

    cublas_dir = _lib_dir_from_module_or_site("nvidia.cublas.lib", ("nvidia", "cublas", "lib"))
    cudnn_dir = _lib_dir_from_module_or_site("nvidia.cudnn.lib", ("nvidia", "cudnn", "lib"))
    if not cublas_dir and not cudnn_dir:
        return []

    def _pick(pattern: str) -> str | None:
        hits = sorted(glob.glob(pattern))
        return hits[-1] if hits else None

    lib_cublaslt = _pick(os.path.join(cublas_dir, "libcublasLt.so*")) if cublas_dir else None
    lib_cublas = _pick(os.path.join(cublas_dir, "libcublas.so*")) if cublas_dir else None
    lib_cudnn = _pick(os.path.join(cudnn_dir, "libcudnn.so*")) if cudnn_dir else None

    loaded: list[str] = []
    for p in (lib_cublaslt, lib_cublas, lib_cudnn):
        if not p:
            continue
        try:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            loaded.append(p)
        except Exception as e:
            if verbose:
                print(f"Failed to preload {p}: {e}")

    if loaded:
        # Make subprocesses and any later dlopen favor these folders.
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        pref_parts = [x for x in (cublas_dir, cudnn_dir) if x]
        pref = ":".join(pref_parts)
        os.environ["LD_LIBRARY_PATH"] = f"{pref}:{ld}" if ld else pref
        if verbose:
            print("Preloaded CUDA libs:")
            for p in loaded:
                print(f"  - {p}")

    return loaded
