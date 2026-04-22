from __future__ import annotations

from dataclasses import dataclass

from .sam2_support import is_native_windows, sam2_layout_messages


@dataclass(frozen=True)
class EnvStatus:
    ok: bool
    title: str
    details: str


def check_cellpose() -> EnvStatus:
    try:
        from cellpose import models  # type: ignore
    except Exception as e:  # noqa: BLE001
        return EnvStatus(
            ok=False,
            title="Cellpose import",
            details=f"Failed to import `cellpose`: {e}",
        )

    if not hasattr(models, "CellposeModel"):
        return EnvStatus(
            ok=False,
            title="Cellpose API",
            details="This app expects `cellpose.models.CellposeModel` (Cellpose 3.x). Install `cellpose==3.1.1.3`.",
        )

    return EnvStatus(
        ok=True,
        title="Cellpose (Cellpose-SAM)",
        details="CellposeModel available — pipeline uses pretrained `cpsam`.",
    )


def check_torch() -> EnvStatus:
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return EnvStatus(ok=False, title="PyTorch import", details=f"Failed to import `torch`: {e}")
    return EnvStatus(ok=True, title="PyTorch import", details="torch import OK.")


def check_napari_qt() -> EnvStatus:
    try:
        import napari  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return EnvStatus(ok=False, title="Napari import", details=f"Failed to import `napari`: {e}")

    # Qt bindings may be missing; importing PyQt5 gives clearer signal.
    try:
        import PyQt5  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return EnvStatus(
            ok=False,
            title="Qt bindings (PyQt5)",
            details=f"Napari needs Qt bindings. Install `pyqt5`. Import error: {e}",
        )

    return EnvStatus(ok=True, title="Napari + Qt", details="napari + PyQt5 import OK.")


def check_sam2_layout(sam4ct_path: str) -> EnvStatus:
    """Verify cloned repo + weights exist; warn on native Windows."""
    msgs = sam2_layout_messages(sam4ct_path)
    if msgs:
        return EnvStatus(
            ok=False,
            title="SAM2 (sam4celltracking) files",
            details="\n".join(msgs),
        )
    details = "sam4celltracking repo + sam2.1_hiera_large.pt found."
    if is_native_windows():
        details += (
            "\n\nNative Windows may still fail at import/runtime (upstream targets Linux). "
            "Use WSL2 + GPU if tracking errors — see README “Windows + SAM2”."
        )
    return EnvStatus(ok=True, title="SAM2 (sam4celltracking) files", details=details)

