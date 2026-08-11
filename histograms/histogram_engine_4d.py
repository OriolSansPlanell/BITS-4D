"""Chunked CPU/GPU 2-D histograms for large paired tomography datasets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

Range2D = Tuple[Tuple[float, float], Tuple[float, float]]
Progress = Optional[Callable[[int, str], None]]
CancelCheck = Optional[Callable[[], None]]


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
    """Compute histograms without materializing complete flattened datasets."""

    def __init__(
        self,
        bins: int = 256,
        cache_size: int = 10,
        use_gpu: bool = True,
        chunk_voxels: int = 4_000_000,
    ) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if chunk_voxels < 1:
            raise ValueError("chunk_voxels must be positive")
        self.bins = int(bins)
        self.cache_size = int(cache_size)
        self.chunk_voxels = int(chunk_voxels)
        self.use_gpu = bool(
            use_gpu and torch is not None and torch.cuda.is_available()
        )
        self._local_cache: OrderedDict[tuple, HistogramData] = OrderedDict()
        self._global_histogram: Optional[HistogramData] = None
        self._data_range: Optional[Range2D] = None

    @staticmethod
    def _validate_pair(neutron: np.ndarray, xray: np.ndarray, ndim: int) -> None:
        if neutron.shape != xray.shape:
            raise ValueError(
                f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}"
            )
        if neutron.ndim != ndim:
            raise ValueError(f"Expected {ndim}D arrays, got {neutron.ndim}D")
        if neutron.size == 0:
            raise ValueError("Cannot compute a histogram from empty arrays")

    @staticmethod
    def _report(callback: Progress, value: int, message: str) -> None:
        if callback is not None:
            callback(int(value), message)

    @staticmethod
    def _cancel(cancel_check: CancelCheck) -> None:
        if cancel_check is not None:
            cancel_check()

    def _iter_chunks(
        self,
        neutron: np.ndarray,
        xray: np.ndarray,
        subsample: int = 1,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield paired 1-D chunks, keeping at most one volume slice in memory."""
        if neutron.ndim == 4:
            time_indices = range(0, neutron.shape[0], subsample)
            for timepoint in time_indices:
                n_volume = np.asarray(neutron[timepoint])
                x_volume = np.asarray(xray[timepoint])
                if subsample > 1:
                    n_volume = n_volume[::subsample, ::subsample, ::subsample]
                    x_volume = x_volume[::subsample, ::subsample, ::subsample]
                n_flat = n_volume.reshape(-1)
                x_flat = x_volume.reshape(-1)
                for start in range(0, n_flat.size, self.chunk_voxels):
                    stop = min(start + self.chunk_voxels, n_flat.size)
                    yield n_flat[start:stop], x_flat[start:stop]
        else:
            n_volume = np.asarray(neutron)
            x_volume = np.asarray(xray)
            if subsample > 1:
                n_volume = n_volume[::subsample, ::subsample, ::subsample]
                x_volume = x_volume[::subsample, ::subsample, ::subsample]
            n_flat = n_volume.reshape(-1)
            x_flat = x_volume.reshape(-1)
            for start in range(0, n_flat.size, self.chunk_voxels):
                stop = min(start + self.chunk_voxels, n_flat.size)
                yield n_flat[start:stop], x_flat[start:stop]

    def _scan_range(
        self,
        neutron: np.ndarray,
        xray: np.ndarray,
        subsample: int,
        progress_callback: Progress,
        cancel_check: CancelCheck,
    ) -> tuple[Range2D, int, int]:
        n_min = np.inf
        n_max = -np.inf
        x_min = np.inf
        x_max = -np.inf
        valid_total = 0
        ignored_total = 0
        processed = 0
        expected = max(1, int(np.ceil(neutron.size / max(subsample**neutron.ndim, 1))))

        for n_chunk, x_chunk in self._iter_chunks(neutron, xray, subsample):
            self._cancel(cancel_check)
            finite = np.isfinite(n_chunk) & np.isfinite(x_chunk)
            valid = int(np.count_nonzero(finite))
            valid_total += valid
            ignored_total += int(finite.size - valid)
            if valid:
                n_values = n_chunk[finite]
                x_values = x_chunk[finite]
                n_min = min(n_min, float(np.min(n_values)))
                n_max = max(n_max, float(np.max(n_values)))
                x_min = min(x_min, float(np.min(x_values)))
                x_max = max(x_max, float(np.max(x_values)))
            processed += finite.size
            self._report(
                progress_callback,
                min(40, int(40 * processed / expected)),
                "Scanning finite data range...",
            )

        if valid_total == 0:
            raise ValueError("No finite neutron/X-ray voxel pairs are available")
        return (
            (self._expand_constant(n_min, n_max), self._expand_constant(x_min, x_max)),
            valid_total,
            ignored_total,
        )

    @staticmethod
    def _expand_constant(minimum: float, maximum: float) -> tuple[float, float]:
        if maximum > minimum:
            return float(minimum), float(maximum)
        padding = max(abs(float(minimum)) * 1e-6, 0.5)
        return float(minimum - padding), float(maximum + padding)

    def compute_global_histogram(
        self,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
        subsample: Optional[int] = None,
        progress_callback: Progress = None,
        cancel_check: CancelCheck = None,
    ) -> HistogramData:
        self._validate_pair(neutron_4d, xray_4d, ndim=4)
        factor = 1 if subsample is None else int(subsample)
        if factor < 1:
            raise ValueError("subsample must be a positive integer")

        data_range, valid, ignored = self._scan_range(
            neutron_4d,
            xray_4d,
            factor,
            progress_callback,
            cancel_check,
        )
        self._data_range = data_range
        self._local_cache.clear()
        result = self._accumulate(
            neutron_4d,
            xray_4d,
            data_range,
            factor,
            valid,
            ignored,
            progress_callback,
            cancel_check,
            progress_start=40,
        )
        self._global_histogram = result
        self._report(progress_callback, 100, "Global histogram complete")
        return result

    def compute_local_histogram(
        self,
        neutron_3d: np.ndarray,
        xray_3d: np.ndarray,
        timepoint: int,
        progress_callback: Progress = None,
        cancel_check: CancelCheck = None,
    ) -> HistogramData:
        self._validate_pair(neutron_3d, xray_3d, ndim=3)
        if self._data_range is None:
            raise RuntimeError("Must compute global histogram first")
        key = (int(timepoint), neutron_3d.shape, str(neutron_3d.dtype), str(xray_3d.dtype))
        if key in self._local_cache:
            self._local_cache.move_to_end(key)
            self._report(progress_callback, 100, "Retrieved local histogram from cache")
            return self._local_cache[key]

        valid = 0
        ignored = 0
        for n_chunk, x_chunk in self._iter_chunks(neutron_3d, xray_3d):
            finite = np.isfinite(n_chunk) & np.isfinite(x_chunk)
            valid += int(np.count_nonzero(finite))
            ignored += int(finite.size - np.count_nonzero(finite))
        if valid == 0:
            raise ValueError("No finite neutron/X-ray voxel pairs are available")

        result = self._accumulate(
            neutron_3d,
            xray_3d,
            self._data_range,
            1,
            valid,
            ignored,
            progress_callback,
            cancel_check,
            progress_start=0,
        )
        if self.cache_size > 0:
            self._local_cache[key] = result
            self._local_cache.move_to_end(key)
            while len(self._local_cache) > self.cache_size:
                self._local_cache.popitem(last=False)
        self._report(progress_callback, 100, "Local histogram complete")
        return result

    def _accumulate(
        self,
        neutron: np.ndarray,
        xray: np.ndarray,
        data_range: Range2D,
        subsample: int,
        valid_total: int,
        ignored_total: int,
        progress_callback: Progress,
        cancel_check: CancelCheck,
        progress_start: int,
    ) -> HistogramData:
        try:
            if self.use_gpu:
                histogram = self._accumulate_gpu(
                    neutron,
                    xray,
                    data_range,
                    subsample,
                    valid_total,
                    progress_callback,
                    cancel_check,
                    progress_start,
                )
            else:
                histogram = self._accumulate_cpu(
                    neutron,
                    xray,
                    data_range,
                    subsample,
                    valid_total,
                    progress_callback,
                    cancel_check,
                    progress_start,
                )
        except (RuntimeError, MemoryError):
            # CUDA OOM or backend failure: discard partial counts and recompute
            # deterministically on CPU.
            self.use_gpu = False
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            histogram = self._accumulate_cpu(
                neutron,
                xray,
                data_range,
                subsample,
                valid_total,
                progress_callback,
                cancel_check,
                progress_start,
            )

        x_edges = np.linspace(*data_range[0], self.bins + 1, dtype=np.float64)
        y_edges = np.linspace(*data_range[1], self.bins + 1, dtype=np.float64)
        return HistogramData(
            histogram=histogram,
            x_edges=x_edges,
            y_edges=y_edges,
            x_centers=(x_edges[:-1] + x_edges[1:]) / 2.0,
            y_centers=(y_edges[:-1] + y_edges[1:]) / 2.0,
            data_range=data_range,
            num_voxels=int(valid_total),
            ignored_voxels=int(ignored_total),
        )

    def _accumulate_cpu(
        self,
        neutron,
        xray,
        data_range,
        subsample,
        valid_total,
        progress_callback,
        cancel_check,
        progress_start,
    ) -> np.ndarray:
        histogram = np.zeros((self.bins, self.bins), dtype=np.uint64)
        processed_valid = 0
        for n_chunk, x_chunk in self._iter_chunks(neutron, xray, subsample):
            self._cancel(cancel_check)
            finite = np.isfinite(n_chunk) & np.isfinite(x_chunk)
            if np.any(finite):
                chunk_hist, _, _ = np.histogram2d(
                    n_chunk[finite],
                    x_chunk[finite],
                    bins=self.bins,
                    range=[data_range[0], data_range[1]],
                )
                histogram += chunk_hist.T.astype(np.uint64, copy=False)
                processed_valid += int(np.count_nonzero(finite))
            self._report(
                progress_callback,
                progress_start
                + int((100 - progress_start) * processed_valid / max(valid_total, 1)),
                "Accumulating histogram chunks on CPU...",
            )
        return histogram

    def _accumulate_gpu(
        self,
        neutron,
        xray,
        data_range,
        subsample,
        valid_total,
        progress_callback,
        cancel_check,
        progress_start,
    ) -> np.ndarray:
        histogram = np.zeros((self.bins, self.bins), dtype=np.uint64)
        processed_valid = 0
        device = torch.device("cuda")
        n_min, n_max = data_range[0]
        x_min, x_max = data_range[1]

        for n_chunk, x_chunk in self._iter_chunks(neutron, xray, subsample):
            self._cancel(cancel_check)
            finite = np.isfinite(n_chunk) & np.isfinite(x_chunk)
            if not np.any(finite):
                continue
            # dtype=np.float64 also normalizes byte order: torch rejects
            # non-native arrays, which big-endian TIFF memory maps produce.
            n_tensor = torch.as_tensor(
                np.asarray(n_chunk[finite], dtype=np.float64),
                device=device, dtype=torch.float64
            )
            x_tensor = torch.as_tensor(
                np.asarray(x_chunk[finite], dtype=np.float64),
                device=device, dtype=torch.float64
            )
            n_bins = torch.clamp(
                ((n_tensor - n_min) / (n_max - n_min) * self.bins).long(),
                0,
                self.bins - 1,
            )
            x_bins = torch.clamp(
                ((x_tensor - x_min) / (x_max - x_min) * self.bins).long(),
                0,
                self.bins - 1,
            )
            counts = torch.bincount(
                x_bins * self.bins + n_bins,
                minlength=self.bins * self.bins,
            )
            histogram += counts.reshape(self.bins, self.bins).cpu().numpy().astype(np.uint64)
            processed_valid += int(np.count_nonzero(finite))
            self._report(
                progress_callback,
                progress_start
                + int((100 - progress_start) * processed_valid / max(valid_total, 1)),
                "Accumulating histogram chunks on GPU...",
            )
            del n_tensor, x_tensor, n_bins, x_bins, counts
        return histogram

    def precompute_all_local_histograms(
        self,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
        progress_callback: Progress = None,
        cancel_check: CancelCheck = None,
    ) -> None:
        self._validate_pair(neutron_4d, xray_4d, ndim=4)
        for timepoint in range(neutron_4d.shape[0]):
            self._cancel(cancel_check)
            self.compute_local_histogram(
                neutron_4d[timepoint],
                xray_4d[timepoint],
                timepoint,
                cancel_check=cancel_check,
            )
            self._report(
                progress_callback,
                int(100 * (timepoint + 1) / neutron_4d.shape[0]),
                f"Computed histogram {timepoint + 1}/{neutron_4d.shape[0]}",
            )

    def get_global_histogram(self) -> Optional[HistogramData]:
        return self._global_histogram

    def get_cached_local_histogram(self, timepoint: int) -> Optional[HistogramData]:
        for key, value in reversed(self._local_cache.items()):
            if key[0] == int(timepoint):
                return value
        return None

    def clear_cache(self, include_global: bool = False) -> None:
        self._local_cache.clear()
        if include_global:
            self._global_histogram = None
            self._data_range = None

    def get_cache_stats(self) -> Dict:
        return {
            "cache_size": len(self._local_cache),
            "cache_limit": self.cache_size,
            "cached_timepoints": [key[0] for key in self._local_cache],
            "global_cached": self._global_histogram is not None,
            "backend": "gpu" if self.use_gpu else "cpu",
            "chunk_voxels": self.chunk_voxels,
        }
