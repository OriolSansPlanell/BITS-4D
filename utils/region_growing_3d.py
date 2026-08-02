"""Fast and dtype-safe 3-D connected region growing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage


class RegionGrowing3D:
    """Connected-component region growing for one or two modalities."""

    @staticmethod
    def _validate_volume(volume: np.ndarray) -> np.ndarray:
        array = np.asarray(volume)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3-D volume, got {array.ndim}D")
        if array.size == 0:
            raise ValueError("Volume must not be empty")
        return array

    @staticmethod
    def _validate_seed(shape: tuple[int, int, int], seed: Tuple[int, int, int]) -> tuple[int, int, int]:
        if len(seed) != 3:
            raise ValueError("Seed must be a (z, y, x) tuple")
        z, y, x = (int(value) for value in seed)
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            raise IndexError(f"Seed {(z, y, x)} is outside volume shape {shape}")
        return z, y, x

    @staticmethod
    def _structure(connectivity: int) -> np.ndarray:
        if connectivity not in (1, 2, 3):
            raise ValueError("3-D connectivity must be 1 (6), 2 (18), or 3 (26)")
        return ndimage.generate_binary_structure(3, connectivity)

    @staticmethod
    def _component_from_candidate(
        candidate: np.ndarray,
        seed: tuple[int, int, int],
        connectivity: int,
    ) -> np.ndarray:
        if not candidate[seed]:
            return np.zeros(candidate.shape, dtype=bool)
        labels, _ = ndimage.label(candidate, structure=RegionGrowing3D._structure(connectivity))
        seed_label = int(labels[seed])
        if seed_label == 0:
            return np.zeros(candidate.shape, dtype=bool)
        return labels == seed_label

    @staticmethod
    def grow_region_univariate_3d(
        volume: np.ndarray,
        seed_z: int,
        seed_y: int,
        seed_x: int,
        tolerance: float,
        connectivity: int = 1,
    ) -> np.ndarray:
        array = RegionGrowing3D._validate_volume(volume)
        seed = RegionGrowing3D._validate_seed(array.shape, (seed_z, seed_y, seed_x))
        tolerance = float(tolerance)
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError("Tolerance must be finite and non-negative")

        seed_value = float(array[seed])
        if not np.isfinite(seed_value):
            return np.zeros(array.shape, dtype=bool)

        lower = seed_value - tolerance
        upper = seed_value + tolerance
        candidate = np.isfinite(array) & (array >= lower) & (array <= upper)
        return RegionGrowing3D._component_from_candidate(candidate, seed, connectivity)

    @staticmethod
    def grow_region_bivariate_3d(
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
        seed_z: int,
        seed_y: int,
        seed_x: int,
        neutron_tol: float,
        xray_tol: float,
        connectivity: int = 1,
    ) -> np.ndarray:
        neutron = RegionGrowing3D._validate_volume(neutron_vol)
        xray = RegionGrowing3D._validate_volume(xray_vol)
        if neutron.shape != xray.shape:
            raise ValueError(
                f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}"
            )
        seed = RegionGrowing3D._validate_seed(neutron.shape, (seed_z, seed_y, seed_x))

        neutron_tol = float(neutron_tol)
        xray_tol = float(xray_tol)
        if neutron_tol < 0 or xray_tol < 0:
            raise ValueError("Tolerances must be non-negative")
        if not np.isfinite(neutron_tol) or not np.isfinite(xray_tol):
            raise ValueError("Tolerances must be finite")

        neutron_seed = float(neutron[seed])
        xray_seed = float(xray[seed])
        if not np.isfinite(neutron_seed) or not np.isfinite(xray_seed):
            return np.zeros(neutron.shape, dtype=bool)

        candidate = (
            np.isfinite(neutron)
            & np.isfinite(xray)
            & (neutron >= neutron_seed - neutron_tol)
            & (neutron <= neutron_seed + neutron_tol)
            & (xray >= xray_seed - xray_tol)
            & (xray <= xray_seed + xray_tol)
        )
        return RegionGrowing3D._component_from_candidate(candidate, seed, connectivity)

    @staticmethod
    def bivariate_region_growing_3d(
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
        seed_3d: Tuple[int, int, int],
        neutron_tol: float,
        xray_tol: float,
        connectivity: int = 1,
    ) -> np.ndarray:
        z, y, x = seed_3d
        return RegionGrowing3D.grow_region_bivariate_3d(
            neutron_vol,
            xray_vol,
            z,
            y,
            x,
            neutron_tol,
            xray_tol,
            connectivity,
        )

    @staticmethod
    def univariate_region_growing_3d(
        volume: np.ndarray,
        seed_3d: Tuple[int, int, int],
        tolerance: float,
        connectivity: int = 1,
    ) -> np.ndarray:
        z, y, x = seed_3d
        return RegionGrowing3D.grow_region_univariate_3d(
            volume, z, y, x, tolerance, connectivity
        )

    @staticmethod
    def extract_slice_from_3d_mask(
        mask_3d: np.ndarray, axis: str, slice_index: int
    ) -> np.ndarray:
        mask = RegionGrowing3D._validate_volume(mask_3d)
        axis = axis.lower()
        if axis == "z":
            return mask[slice_index, :, :]
        if axis == "y":
            return mask[:, slice_index, :]
        if axis == "x":
            return mask[:, :, slice_index]
        raise ValueError("axis must be 'z', 'y', or 'x'")

    @staticmethod
    def get_3d_statistics(
        mask_3d: np.ndarray,
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
    ) -> dict | None:
        mask = np.asarray(mask_3d, dtype=bool)
        neutron = RegionGrowing3D._validate_volume(neutron_vol)
        xray = RegionGrowing3D._validate_volume(xray_vol)
        if mask.shape != neutron.shape or mask.shape != xray.shape:
            raise ValueError("Mask and modality volumes must have identical shapes")

        count = int(np.count_nonzero(mask))
        if count == 0:
            return None
        neutron_values = neutron[mask]
        xray_values = xray[mask]
        return {
            "voxels": count,
            "volume_fraction": float(count / mask.size),
            "neutron_mean": float(np.mean(neutron_values)),
            "neutron_std": float(np.std(neutron_values)),
            "neutron_min": float(np.min(neutron_values)),
            "neutron_max": float(np.max(neutron_values)),
            "xray_mean": float(np.mean(xray_values)),
            "xray_std": float(np.std(xray_values)),
            "xray_min": float(np.min(xray_values)),
            "xray_max": float(np.max(xray_values)),
        }
