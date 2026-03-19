from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvStatus:
    ok: bool
    title: str
    details: str


def check_cellpose_cyto2() -> EnvStatus:
    try:
        from cellpose import models  # type: ignore
    except Exception as e:  # noqa: BLE001
        return EnvStatus(
            ok=False,
            title="Cellpose import",
            details=f"Failed to import `cellpose`: {e}",
        )

    if not hasattr(models, "Cellpose"):
        return EnvStatus(
            ok=False,
            title="Cellpose API",
            details="This app expects cellpose<4 (classic `models.Cellpose`). Install `cellpose==3.1.1.3`.",
        )

    return EnvStatus(ok=True, title="Cellpose API", details="cellpose classic API is available.")


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

