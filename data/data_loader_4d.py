"""Memory-aware TIFF loading for paired 3-D/4-D tomography datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tifffile

from .dataset_4d import Dataset4D


class TIFF4DLoader:
    DEFAULT_MEMMAP_THRESHOLD_GB = 8.0

    @staticmethod
    def estimate_memory(neutron_path: str, xray_path: str) -> Dict:
        neutron_path = Path(neutron_path)
        xray_path = Path(xray_path)
        if not neutron_path.exists():
            raise FileNotFoundError(f"Neutron file not found: {neutron_path}")
        if not xray_path.exists():
            raise FileNotFoundError(f"X-ray file not found: {xray_path}")
        with tifffile.TiffFile(neutron_path) as tif:
            neutron_shape = tif.series[0].shape
            neutron_dtype = tif.series[0].dtype
        with tifffile.TiffFile(xray_path) as tif:
            xray_shape = tif.series[0].shape
            xray_dtype = tif.series[0].dtype
        neutron_bytes = int(np.prod(neutron_shape, dtype=np.int64)) * np.dtype(neutron_dtype).itemsize
        xray_bytes = int(np.prod(xray_shape, dtype=np.int64)) * np.dtype(xray_dtype).itemsize
        total_bytes = neutron_bytes + xray_bytes
        return {
            "neutron_mb": neutron_bytes / 1024**2,
            "xray_mb": xray_bytes / 1024**2,
            "total_mb": total_bytes / 1024**2,
            "total_gb": total_bytes / 1024**3,
            "neutron_shape": neutron_shape,
            "xray_shape": xray_shape,
            "neutron_dtype": str(neutron_dtype),
            "xray_dtype": str(xray_dtype),
        }

    @staticmethod
    def _to_native_byte_order(array: np.ndarray) -> np.ndarray:
        """Return the array in native byte order.

        Big-endian TIFFs yield arrays whose dtype byte order differs from the
        machine's; downstream consumers (PyTorch tensors, some scikit-learn
        versions) reject such arrays. In-RAM arrays are byteswapped in place
        (no extra memory); read-only memory maps are returned unchanged and
        converted chunk-wise by the consumers instead.
        """
        if array.dtype.isnative:
            return array
        if isinstance(array, np.memmap) or not array.flags.writeable:
            return array
        array.byteswap(inplace=True)
        return array.view(array.dtype.newbyteorder("="))

    @staticmethod
    def _read(path: Path, use_memmap: bool):
        if not use_memmap:
            return TIFF4DLoader._to_native_byte_order(tifffile.imread(path))
        try:
            return tifffile.memmap(path, mode="r")
        except (ValueError, TypeError):
            # Compressed TIFFs cannot always be mapped directly.  tifffile's
            # out='memmap' path creates a temporary memory-mapped backing file.
            return tifffile.imread(path, out="memmap")

    @staticmethod
    def load(
        neutron_path: str,
        xray_path: str,
        use_memmap: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
        memmap_threshold_gb: float = DEFAULT_MEMMAP_THRESHOLD_GB,
    ) -> Dataset4D:
        neutron_path = Path(neutron_path)
        xray_path = Path(xray_path)
        valid, message = TIFF4DLoader.validate_files(neutron_path, xray_path)
        if not valid:
            raise ValueError(message)

        estimate = TIFF4DLoader.estimate_memory(neutron_path, xray_path)
        # Safety takes precedence over the legacy GUI's hard-coded False.  Large
        # datasets are mapped unless the threshold is explicitly set to infinity.
        effective_memmap = bool(use_memmap) or estimate["total_gb"] >= float(
            memmap_threshold_gb
        )

        def report(value, text):
            if cancel_check:
                cancel_check()
            if progress_callback:
                progress_callback(value, text)

        report(5, "Loading neutron data...")
        neutron_data = TIFF4DLoader._read(neutron_path, effective_memmap)
        report(45, "Loading X-ray data...")
        xray_data = TIFF4DLoader._read(xray_path, effective_memmap)
        report(75, "Validating data dimensions...")

        if neutron_data.ndim == 3:
            neutron_data = neutron_data[np.newaxis, ...]
            xray_data = xray_data[np.newaxis, ...]
        if neutron_data.shape != xray_data.shape:
            raise ValueError(
                f"Shape mismatch: neutron {neutron_data.shape} vs X-ray {xray_data.shape}"
            )

        metadata = TIFF4DLoader._extract_metadata(neutron_path, xray_path)
        metadata["memory_mapped"] = bool(effective_memmap)
        metadata["estimated_total_gb"] = float(estimate["total_gb"])
        report(95, "Creating dataset object...")
        dataset = Dataset4D(neutron_data, xray_data, metadata)
        report(100, f"Dataset loaded: {dataset.shape}")
        return dataset

    @staticmethod
    def _extract_metadata(neutron_path: Path, xray_path: Path) -> Dict:
        metadata = {
            "neutron_file": str(neutron_path),
            "xray_file": str(xray_path),
        }
        for modality, path in (("neutron", neutron_path), ("xray", xray_path)):
            try:
                with tifffile.TiffFile(path) as tif:
                    if tif.imagej_metadata:
                        metadata[f"{modality}_imagej"] = tif.imagej_metadata
                    page = tif.pages[0]
                    x_res = page.tags.get("XResolution")
                    y_res = page.tags.get("YResolution")
                    if x_res and y_res:
                        metadata[f"{modality}_resolution"] = {
                            "x": x_res.value,
                            "y": y_res.value,
                        }
            except Exception:
                # Metadata is optional and must never prevent loading pixel data.
                continue
        return metadata

    @staticmethod
    def validate_files(neutron_path: str | Path, xray_path: str | Path) -> Tuple[bool, str]:
        neutron_path = Path(neutron_path)
        xray_path = Path(xray_path)
        for label, path in (("Neutron", neutron_path), ("X-ray", xray_path)):
            if not path.exists():
                return False, f"{label} file not found: {path}"
            if path.suffix.lower() not in {".tif", ".tiff"}:
                return False, f"{label} file is not a TIFF: {path.suffix}"
        try:
            with tifffile.TiffFile(neutron_path) as tif:
                neutron_shape = tif.series[0].shape
            with tifffile.TiffFile(xray_path) as tif:
                xray_shape = tif.series[0].shape
            if len(neutron_shape) not in {3, 4}:
                return False, f"Neutron data must be 3-D or 4-D, got {len(neutron_shape)}D"
            if len(xray_shape) not in {3, 4}:
                return False, f"X-ray data must be 3-D or 4-D, got {len(xray_shape)}D"
            if neutron_shape != xray_shape:
                return False, f"Shape mismatch: neutron {neutron_shape} vs X-ray {xray_shape}"
            return True, "Files are valid"
        except Exception as exc:
            return False, f"Error validating files: {exc}"
