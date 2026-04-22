"""SAM2 / sam4celltracking platform notes and path checks.

Upstream sam4celltracking documents **Linux** + CUDA. Native Windows often fails
on compiled extensions or torch ops; **WSL2 (Ubuntu) + NVIDIA CUDA in WSL** is
the supported way to run the same GUI stack on a Windows PC.
"""
from __future__ import annotations

import sys
from pathlib import Path


def is_native_windows() -> bool:
    return sys.platform == "win32"


WSL2_SAM2_GUIDANCE = """
Native Windows is often incompatible with sam4celltracking’s vendored SAM2
build (upstream lists **Linux**). Use **WSL2** so tracking runs in Linux while
you still use the same browser GUI.

1. Install **WSL2** + **Ubuntu 22.04** (Microsoft Store or `wsl --install`).
2. Install the **latest NVIDIA Windows driver** (supports WSL GPU).
3. Inside Ubuntu, install Python 3.10+, CUDA toolkit **or** use PyTorch’s
   CUDA wheels only (see PyTorch “Start Locally” for Linux + CUDA).
4. Clone this project under Linux (e.g. `~/bioelectricity-project`) or access it
   via `/mnt/c/Users/.../bioelectricity project` (slower I/O on `/mnt/c`).
5. Create a venv, `pip install -r requirements.txt`, follow README to clone
   `sam4celltracking` and download `sam2.1_hiera_large.pt`.
6. From that Ubuntu shell: `python -m streamlit run streamlit_app.py`
   then open http://localhost:8501 in Windows.

Docs: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
""".strip()


def sam2_layout_messages(sam4ct_path: Path | str) -> list[str]:
    """Human-readable checks for repo + weights (no imports)."""
    p = Path(sam4ct_path).resolve()
    src = p / "src"
    model = src / "trained_models" / "sam2.1_hiera_large.pt"
    msgs: list[str] = []
    if not src.is_dir():
        msgs.append(f"Missing sam4celltracking src: {src}")
    elif not (src / "linking_2d_general.py").is_file():
        msgs.append(f"Repo incomplete (no linking_2d_general.py): {src}")
    if not model.is_file():
        msgs.append(f"Missing SAM2.1 weights: {model}")
    return msgs


def format_tracking_failure(exc: BaseException) -> str:
    """User-facing message when SAM2 linking fails."""
    head = f"SAM2 tracking failed ({type(exc).__name__}): {exc}"
    if is_native_windows():
        return f"{head}\n\n{WSL2_SAM2_GUIDANCE}"
    return (
        f"{head}\n\n"
        "If this is an import or CUDA error, confirm Linux + NVIDIA driver, "
        "PyTorch with CUDA, and that sam4celltracking’s conda/setup script was "
        "followed where their README requires it."
    )
