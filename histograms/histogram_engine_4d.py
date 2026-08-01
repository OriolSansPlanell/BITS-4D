"""Robust 2-D histogram computation for paired 4-D datasets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

Range2D = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class HistogramData:
    histogram: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    data_range: Range2D
    num_voxels: int
    ignored_voxels: int = 0

    def to_log_scale(self) -> np.ndarray:
        return np.log10(self.histogram.astype(np.float64, copy=False) + 1.0)


class HistogramEngine4D:
    def __init__(self, bins: int = 256, cache_size: int = 10, use_gpu: bool = True):
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.bins = int(bins)
        self.cache_size = int(cache_size)
        self.use_gpu = bool(
            use_gpu and torch is not None and torch.cuda.is_available()
        )
        self._local_cache: OrderedDict[int, HistogramData] = OrderedDict()
        self._global_histogram: Optional[HistogramData] = None
        self._data_range: Optional[Range2D] = None

    @staticmethod
    def _validate_pair(neutron: np.ndarray, xray: np.ndarray) -> None:
        if neutron.shape != xray.shape:
            raise ValueError(
                f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}"
            )
        if neutron.size == 0:
            raise ValueError("Cannot compute a histogram from empty arrays")

    @staticmethod
    def _finite_pair(neutron: np.ndarray, xray: np.ndarray):
        neutron_flat = np.asarray(neutron).ravel()
        xray_flat = np.asarray(xray).ravel()
        finite = np.isfinite(neutron_flat) & np.isfinite(xray_flat)
        valid = int(np.count_nonzero(finite))
        if valid == 0:
            raise ValueError("No finite neutron/X-ray voxel pairs are available")
        return neutron_flat[finite], xray_flat[finite], neutron_flat.size - valid

    @staticmethod
    def _safe_range(values: np.ndarray) -> tuple[float, float]:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum > minimum:
            return minimum, maximum
        padding = max(abs(minimum) * 1e-6, 0.5)
        return minimum - padding, maximum + padding

    def compute_global_histogram(
        self,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
        subsample: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> HistogramData:
        self._validate_pair(neutron_4d, xray_4d)
        if neutron_4d.ndim != 4:
            raise ValueError("Global histogram inputs must be 4-D")
        factor = 1 if subsample is None else int(subsample)
        if factor < 1:
            raise ValueError("subsample must be a positive integer")
        if factor > 1:
            slicing = (slice(None, None, factor),) * 4
            neutron_4d = neutron_4d[slicing]
            xray_4d = xray_4d[slicing]

        if progress_callback:
            progress_callback(25, "Filtering invalid voxel pairs...")
        neutron, xray, ignored = self._finite_pair(neutron_4d, xray_4d)
        data_range = (self._safe_range(neutron), self._safe_range(xray))

        # A new global range invalidates every local histogram.
        self._data_range = data_range
        self._local_cache.clear()
        if progress_callback:
            progress_callback(60, "Computing global histogram...")
        result = self._compute(neutron, xray, data_range, ignored)
        self._global_histogram = result
        if progress_callback:
            progress_callback(100, "Global histogram complete")
        return result

    def compute_local_histogram(
        self,
        neutron_3d: np.ndarray,
        xray_3d: np.ndarray,
        timepoint: int,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> HistogramData:
        self._validate_pair(neutron_3d, xray_3d)
        if neutron_3d.ndim != 3:
            raise ValueError("Local histogram inputs must be 3-D")
        if self._data_range is None:
            raise RuntimeError("Must compute global histogram first")
        if timepoint in self._local_cache:
            self._local_cache.move_to_end(timepoint)
            if progress_callback:
                progress_callback(100, "Retrieved local histogram from cache")
            return self._local_cache[timepoint]

        neutron, xray, ignored = self._finite_pair(neutron_3d, xray_3d)
        result = self._compute(neutron, xray, self._data_range, ignored)
        if self.cache_size > 0:
            self._local_cache[int(timepoint)] = result
            self._local_cache.move_to_end(int(timepoint))
            while len(self._local_cache) > self.cache_size:
                self._local_cache.popitem(last=False)
        if progress_callback:
            progress_callback(100, "Local histogram complete")
        return result

    def _compute(
        self,
        neutron: np.ndarray,
        xray: np.ndarray,
        data_range: Range2D,
        ignored: int,
    ) -> HistogramData:
        if self.use_gpu:
            try:
                return self._compute_gpu(neutron, xray, data_range, ignored)
            except (RuntimeError, MemoryError):
                # A histogram should remain available even when CUDA memory is
                # insufficient.  The backend is disabled for subsequent calls.
                self.use_gpu = False
        return self._compute_cpu(neutron, xray, data_range, ignored)

    def _compute_cpu(
        self, neutron: np.ndarray, xray: np.ndarray, data_range: Range2D, ignored: int
    ) -> HistogramData:
        histogram, x_edges, y_edges = np.histogram2d(
            neutron,
            xray,
            bins=self.bins,
            range=[data_range[0], data_range[1]],
        )
        histogram = histogram.T.astype(np.uint64, copy=False)
        return self._make_data(histogram, x_edges, y_edges, data_range, len(neutron), ignored)

    def _compute_gpu(
        self, neutron: np.ndarray, xray: np.ndarray, data_range: Range2D, ignored: int
    ) -> HistogramData:
        device = torch.device("cuda")
        neutron_tensor = torch.as_tensor(neutron, device=device, dtype=torch.float64)
        xray_tensor = torch.as_tensor(xray, device=device, dtype=torch.float64)
        n_min, n_max = data_range[0]
        x_min, x_max = data_range[1]
        n_bins = torch.clamp(
            ((neutron_tensor - n_min) / (n_max - n_min) * self.bins).long(),
            0,
            self.bins - 1,
        )
        x_bins = torch.clamp(
            ((xray_tensor - x_min) / (x_max - x_min) * self.bins).long(),
            0,
            self.bins - 1,
        )
        indices = x_bins * self.bins + n_bins
        counts = torch.bincount(indices, minlength=self.bins * self.bins)
        histogram = counts.reshape(self.bins, self.bins).cpu().numpy().astype(np.uint64)
        x_edges = np.linspace(n_min, n_max, self.bins + 1)
        y_edges = np.linspace(x_min, x_max, self.bins + 1)
        return self._make_data(histogram, x_edges, y_edges, data_range, len(neutron), ignored)

    @staticmethod
    def _make_data(histogram, x_edges, y_edges, data_range, valid, ignored):
        return HistogramData(
            histogram=histogram,
            x_edges=x_edges,
            y_edges=y_edges,
            x_centers=(x_edges[:-1] + x_edges[1:]) / 2.0,
            y_centers=(y_edges[:-1] + y_edges[1:]) / 2.0,
            data_range=data_range,
            num_voxels=int(valid),
            ignored_voxels=int(ignored),
        )

    def precompute_all_local_histograms(
        self,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        self._validate_pair(neutron_4d, xray_4d)
        for timepoint in range(neutron_4d.shape[0]):
            self.compute_local_histogram(
                neutron_4d[timepoint], xray_4d[timepoint], timepoint
            )
            if progress_callback:
                progress_callback(
                    int(100 * (timepoint + 1) / neutron_4d.shape[0]),
                    f"Computed histogram {timepoint + 1}/{neutron_4d.shape[0]}",
                )

    def get_global_histogram(self) -> Optional[HistogramData]:
        return self._global_histogram

    def get_cached_local_histogram(self, timepoint: int) -> Optional[HistogramData]:
        return self._local_cache.get(timepoint)

    def clear_cache(self, include_global: bool = False) -> None:
        self._local_cache.clear()
        if include_global:
            self._global_histogram = None
            self._data_range = None

    def get_cache_stats(self) -> Dict:
        return {
            "cache_size": len(self._local_cache),
            "cache_limit": self.cache_size,
            "cached_timepoints": list(self._local_cache.keys()),
            "global_cached": self._global_histogram is not None,
        }
