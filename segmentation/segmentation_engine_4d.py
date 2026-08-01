"""ROI-based segmentation for paired 3-D/4-D tomography datasets."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from utils.roi_manager import ROIManager

ProgressCallback = Optional[Callable[[int, str], None]]


class SegmentationEngine4D:
    """Apply histogram-space ROIs to paired neutron and X-ray volumes."""

    def __init__(self) -> None:
        self.segmentation_masks: dict[int, np.ndarray] = {}

    @staticmethod
    def _validate_pair(
        neutron_data: np.ndarray,
        xray_data: np.ndarray,
        expected_ndim: int,
    ) -> None:
        if neutron_data.shape != xray_data.shape:
            raise ValueError(
                f"Shape mismatch: neutron {neutron_data.shape} vs "
                f"X-ray {xray_data.shape}"
            )
        if neutron_data.ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim}D arrays, got {neutron_data.ndim}D"
            )
        if neutron_data.size == 0:
            raise ValueError("Cannot segment an empty volume")

    @staticmethod
    def _report(callback: ProgressCallback, value: int, message: str) -> None:
        if callback is not None:
            callback(int(value), message)

    def segment_volume(
        self,
        neutron_3d: np.ndarray,
        xray_3d: np.ndarray,
        roi_manager: ROIManager,
        progress_callback: ProgressCallback = None,
    ) -> np.ndarray:
        """Return a boolean mask for one paired 3-D volume."""
        self._validate_pair(neutron_3d, xray_3d, expected_ndim=3)
        if not roi_manager.has_roi():
            raise ValueError("No ROI defined in ROI manager")

        self._report(progress_callback, 10, "Checking voxels against ROI...")
        mask = np.asarray(
            roi_manager.is_inside_roi(neutron_3d, xray_3d), dtype=bool
        )
        if mask.shape != neutron_3d.shape:
            raise ValueError(
                f"ROI manager returned shape {mask.shape}; expected {neutron_3d.shape}"
            )

        num_segmented = int(np.count_nonzero(mask))
        percentage = 100.0 * num_segmented / mask.size
        self._report(
            progress_callback,
            100,
            f"Segmented {num_segmented:,} voxels ({percentage:.1f}%)",
        )
        return mask

    def segment_all_volumes(
        self,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
        roi_manager: ROIManager,
        progress_callback: ProgressCallback = None,
    ) -> np.ndarray:
        """Return a boolean 4-D mask for all timepoints."""
        self._validate_pair(neutron_4d, xray_4d, expected_ndim=4)
        if not roi_manager.has_roi():
            raise ValueError("No ROI defined in ROI manager")

        mask_4d = np.empty(neutron_4d.shape, dtype=bool)
        total = neutron_4d.shape[0]
        for timepoint in range(total):
            self._report(
                progress_callback,
                int(100 * timepoint / max(total, 1)),
                f"Segmenting timepoint {timepoint + 1}/{total}",
            )
            mask_4d[timepoint] = roi_manager.is_inside_roi(
                neutron_4d[timepoint], xray_4d[timepoint]
            )

        num_segmented = int(np.count_nonzero(mask_4d))
        percentage = 100.0 * num_segmented / mask_4d.size
        self._report(
            progress_callback,
            100,
            f"Complete: {num_segmented:,} voxels ({percentage:.1f}%)",
        )
        return mask_4d

    @staticmethod
    def get_segmentation_statistics(
        mask: np.ndarray,
        neutron_data: np.ndarray,
        xray_data: np.ndarray,
    ) -> dict:
        if mask.shape != neutron_data.shape or mask.shape != xray_data.shape:
            raise ValueError("Mask and modality arrays must have identical shapes")

        mask_bool = np.asarray(mask, dtype=bool)
        num_segmented = int(np.count_nonzero(mask_bool))
        total_voxels = int(mask_bool.size)
        percentage = 100.0 * num_segmented / total_voxels if total_voxels else 0.0

        stats = {
            "num_voxels": num_segmented,
            "total_voxels": total_voxels,
            "percentage": float(percentage),
        }
        if num_segmented == 0:
            return stats

        neutron_segmented = np.asarray(neutron_data)[mask_bool]
        xray_segmented = np.asarray(xray_data)[mask_bool]
        stats.update(
            {
                "neutron_mean": float(np.mean(neutron_segmented)),
                "neutron_std": float(np.std(neutron_segmented)),
                "neutron_min": float(np.min(neutron_segmented)),
                "neutron_max": float(np.max(neutron_segmented)),
                "xray_mean": float(np.mean(xray_segmented)),
                "xray_std": float(np.std(xray_segmented)),
                "xray_min": float(np.min(xray_segmented)),
                "xray_max": float(np.max(xray_segmented)),
            }
        )
        return stats

    @staticmethod
    def get_temporal_statistics(
        mask_4d: np.ndarray,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
    ) -> dict:
        if mask_4d.shape != neutron_4d.shape or mask_4d.shape != xray_4d.shape:
            raise ValueError("Mask and modality arrays must have identical shapes")
        if mask_4d.ndim != 4:
            raise ValueError("Temporal statistics require 4-D arrays")

        mask_bool = np.asarray(mask_4d, dtype=bool)
        num_timepoints = mask_bool.shape[0]
        volumes_per_time = np.count_nonzero(mask_bool, axis=(1, 2, 3)).astype(float)
        neutron_mean_per_time = np.full(num_timepoints, np.nan, dtype=float)
        xray_mean_per_time = np.full(num_timepoints, np.nan, dtype=float)

        for timepoint in range(num_timepoints):
            if volumes_per_time[timepoint] > 0:
                current_mask = mask_bool[timepoint]
                neutron_mean_per_time[timepoint] = float(
                    np.mean(neutron_4d[timepoint][current_mask])
                )
                xray_mean_per_time[timepoint] = float(
                    np.mean(xray_4d[timepoint][current_mask])
                )

        return {
            "num_timepoints": num_timepoints,
            "volumes_per_time": volumes_per_time,
            "neutron_mean_per_time": neutron_mean_per_time,
            "xray_mean_per_time": xray_mean_per_time,
            "total_segmented_voxels": int(np.count_nonzero(mask_bool)),
            "mean_volume": float(np.mean(volumes_per_time)),
            "std_volume": float(np.std(volumes_per_time)),
        }

    @staticmethod
    def apply_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.3,
    ) -> np.ndarray:
        """Return an RGB uint8 image with a vectorized coloured mask overlay."""
        image_array = np.asarray(image)
        mask_bool = np.asarray(mask, dtype=bool)
        if image_array.ndim != 2:
            raise ValueError("Overlay image must be 2-D")
        if mask_bool.shape != image_array.shape:
            raise ValueError(
                f"Mask shape {mask_bool.shape} does not match image {image_array.shape}"
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        finite = np.isfinite(image_array)
        if not np.any(finite):
            normalized = np.zeros(image_array.shape, dtype=np.float32)
        else:
            finite_values = image_array[finite].astype(np.float64, copy=False)
            minimum = float(np.min(finite_values))
            maximum = float(np.max(finite_values))
            if maximum <= minimum:
                normalized = np.zeros(image_array.shape, dtype=np.float32)
            else:
                normalized = (
                    (image_array.astype(np.float64, copy=False) - minimum)
                    / (maximum - minimum)
                    * 255.0
                ).astype(np.float32)
                normalized[~finite] = 0.0

        rgb = np.repeat(normalized[..., None], 3, axis=-1)
        color_array = np.asarray(color, dtype=np.float32)
        if color_array.shape != (3,):
            raise ValueError("color must be an RGB triplet")
        color_array = np.clip(color_array, 0.0, 255.0)
        rgb[mask_bool] = rgb[mask_bool] * (1.0 - alpha) + color_array * alpha
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)
