from __future__ import annotations

from pathlib import Path
from typing import Any

import tifffile as tf


ROOT = Path(__file__).resolve().parent
FILES = [
    ROOT / "Ctrl_0mV_5min_interval-2.tif",
    ROOT / "EF_200mV_5min_interval-2.tif",
]


def _tag_value(tags: Any, name: str):
    t = tags.get(name)
    return None if t is None else t.value


def _as_float_resolution(value):
    # value can be (num, den) or already float-like
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            return None if den in (0, 0.0) else (num / den)
        return float(value)
    except Exception:
        return None


def _infer_pixel_size_um_per_px(res, resolution_unit):
    """
    TIFF uses pixels per unit:
      ResolutionUnit: 1=None, 2=inches, 3=centimeters
    Convert to micrometers/pixel when possible.
    """
    res_val = _as_float_resolution(res)
    if res_val in (None, 0):
        return None
    if resolution_unit == 2:  # inch
        return 25_400.0 / res_val  # um per inch / px_per_inch
    if resolution_unit == 3:  # cm
        return 10_000.0 / res_val  # um per cm / px_per_cm
    return None


def inspect_tif(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return out

    out["size_bytes"] = path.stat().st_size

    with tf.TiffFile(path) as tif:
        out["pages"] = len(tif.pages)
        out["is_imagej"] = bool(tif.is_imagej)
        out["has_ome_xml"] = bool(tif.ome_metadata)

        # Series-level (axes/shape/dtype)
        out["series"] = []
        for s in tif.series:
            out["series"].append(
                {
                    "shape": tuple(int(x) for x in s.shape),
                    "dtype": str(s.dtype),
                    "axes": getattr(s, "axes", None),
                }
            )

        ij = tif.imagej_metadata or {}
        out["imagej_spacing_z"] = ij.get("spacing")
        out["imagej_unit"] = ij.get("unit")
        out["imagej_metadata_keys"] = sorted(list(ij.keys()))

        # TIFF tags (often used for XY calibration)
        page0 = tif.pages[0]
        tags = page0.tags
        xres = _tag_value(tags, "XResolution")
        yres = _tag_value(tags, "YResolution")
        runit = _tag_value(tags, "ResolutionUnit")

        out["tiff_XResolution"] = xres
        out["tiff_YResolution"] = yres
        out["tiff_ResolutionUnit"] = runit

        out["inferred_pixel_size_x_um_per_px"] = _infer_pixel_size_um_per_px(xres, runit)
        out["inferred_pixel_size_y_um_per_px"] = _infer_pixel_size_um_per_px(yres, runit)

    return out


def _pretty(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d, indent=2, default=str)


def main() -> None:
    for p in FILES:
        print("=" * 100)
        print(p.name)
        print(_pretty(inspect_tif(p)))


if __name__ == "__main__":
    main()

