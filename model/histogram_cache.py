"""Binned sufficient statistics for fitting a mixture without touching voxels.

A Gaussian mixture in the (neutron, X-ray) plane depends on the data only
through per-component sums of ``1``, ``v`` and ``v vᵀ``. Those sums are
additive, so they can be accumulated **per histogram bin** once and reused
for every EM iteration: a 512-bin grid turns 38 million voxels into at most
262 144 bins, and in practice far fewer because most of the plane is empty.

What is and is not approximated
───────────────────────────────
The usual shortcut — fit on bin *centres* weighted by counts — is lossy: it
inflates every covariance by the bin variance (h²/12 per axis, the Sheppard
bias) and it cannot represent where inside a bin the mass actually sits.

This cache instead stores, per bin, the count and the **first and second
moments of the voxels that fell in it**. Given per-bin responsibilities the
M-step is then algebraically identical to the one you would get from the
voxels themselves — no bin-centre approximation anywhere. The single
remaining approximation is that all voxels sharing a bin share a
responsibility, whose error shrinks with the bin width and vanishes wherever
one component dominates a bin.

Because the exact within-bin scatter is retained, the log-likelihood this
cache reports is a genuine per-voxel quantity, so BIC/ICL penalties must use
the **voxel** count (:attr:`HistogramCache.num_voxels`), not the number of
occupied bins.

Orientation
───────────
Bins are addressed as a flat id ``neutron_bin * bins + xray_bin``.
:meth:`HistogramCache.to_image` converts back to the ``[X-ray, neutron]``
layout that :class:`~histograms.histogram_engine_4d.HistogramData` uses for
display, so the two never have to be reasoned about at the same time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Sums of squares of large intensities overflow float32 quickly, so the
# accumulators are float64 throughout. The per-bin arrays are small.
_ACCUMULATOR_DTYPE = np.float64


@dataclass
class HistogramCache:
    """Per-bin sufficient statistics of one (neutron, X-ray) volume pair.

    Attributes
    ----------
    bin_ids
        Flat id of each occupied bin, ``neutron_bin * bins + xray_bin``.
    counts
        Voxels in each occupied bin.
    sums
        ``[M, 2]`` sum of ``(neutron, xray)`` over each bin.
    scatter
        ``[M, 2, 2]`` sum of ``v vᵀ`` over each bin (second moments about
        the origin, not about the mean).
    bin_index
        Per-voxel flat bin id in volume shape, or ``None`` when not kept.
        ``-1`` marks a voxel that is invalid or out of range.
    """

    bins: int
    neutron_edges: np.ndarray
    xray_edges: np.ndarray
    bin_ids: np.ndarray
    counts: np.ndarray
    sums: np.ndarray
    scatter: np.ndarray
    num_voxels: int
    out_of_range: int = 0
    invalid_voxels: int = 0
    shape: Optional[Tuple[int, ...]] = None
    bin_index: Optional[np.ndarray] = None

    # ── derived quantities ───────────────────────────────────────────────
    @property
    def num_bins(self) -> int:
        """Occupied bins — the size of the EM problem."""
        return int(self.bin_ids.size)

    @property
    def means(self) -> np.ndarray:
        """``[M, 2]`` mean position of the voxels in each bin.

        This is the value a Gaussian is evaluated at, and it is the true
        within-bin mean rather than the bin centre.
        """
        return self.sums / self.counts[:, None]

    @property
    def within_bin_scatter(self) -> np.ndarray:
        """``[M, 2, 2]`` scatter of each bin about its own mean."""
        means = self.means
        return self.scatter - self.counts[:, None, None] * (
            means[:, :, None] * means[:, None, :]
        )

    def totals(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Overall ``(count, sum, scatter)`` — the whole-volume moments."""
        return (
            float(self.counts.sum()),
            self.sums.sum(axis=0),
            self.scatter.sum(axis=0),
        )

    def support_area(self) -> float:
        """Area of the histogram's data range, for the uniform outlier density."""
        return float(
            (self.neutron_edges[-1] - self.neutron_edges[0])
            * (self.xray_edges[-1] - self.xray_edges[0])
        )

    def to_image(self) -> np.ndarray:
        """Dense ``[X-ray bin, neutron bin]`` counts, matching HistogramData."""
        image = np.zeros((self.bins, self.bins), dtype=np.float64)
        neutron_bin, xray_bin = np.divmod(self.bin_ids, self.bins)
        image[xray_bin, neutron_bin] = self.counts
        return image

    def row_index_volume(self) -> np.ndarray:
        """Per-voxel row into the compact per-bin arrays; ``-1`` if none.

        This is the lookup that turns any per-bin quantity — a likelihood, a
        responsibility, a fraction — into a per-voxel one without ever
        materialising it for every voxel.
        """
        if self.bin_index is None or self.shape is None:
            raise RuntimeError(
                "This needs the per-voxel bin index; build the cache with "
                "store_bin_index=True"
            )
        lookup = np.full(self.bins * self.bins, -1, dtype=np.int32)
        lookup[self.bin_ids] = np.arange(self.bin_ids.size, dtype=np.int32)

        flat_index = self.bin_index.reshape(-1)
        rows = np.full(flat_index.shape, -1, dtype=np.int32)
        known = flat_index >= 0
        rows[known] = lookup[flat_index[known]]
        return rows.reshape(self.shape)

    def expand_to_voxels(self, per_bin_values: np.ndarray,
                         fill=0.0, rows=None) -> np.ndarray:
        """Scatter a per-bin quantity back over the volume.

        *per_bin_values* is ``[M]`` or ``[M, K]``; the result has the volume
        shape (plus the trailing axis). Pass *rows* from
        :meth:`row_index_volume` to reuse the lookup across several calls.
        """
        values = np.asarray(per_bin_values)
        rows = self.row_index_volume() if rows is None else np.asarray(rows)
        flat_rows = rows.reshape(-1)
        known = flat_rows >= 0
        safe = np.where(known, flat_rows, 0)

        if values.ndim == 1:
            output = np.where(known, values[safe], fill)
            return output.reshape(self.shape)

        output = np.where(
            known[:, None], values[safe], fill
        ).astype(values.dtype, copy=False)
        return output.reshape(tuple(self.shape) + (values.shape[1],))


def build_histogram_cache(
    neutron_volume,
    xray_volume,
    neutron_edges,
    xray_edges,
    valid_mask=None,
    store_bin_index: bool = False,
    chunk_voxels: int = 8_000_000,
) -> HistogramCache:
    """Accumulate per-bin sufficient statistics over one volume pair.

    *neutron_edges* / *xray_edges* come from the global histogram, so every
    timepoint's cache lives on the same grid and its components are directly
    comparable across time.

    Voxels outside the edge range are **excluded**, not clipped into the edge
    bins — clipping would build a spike of foreign mass at the border that a
    mixture component would then try to explain.
    """
    neutron = np.asarray(neutron_volume)
    xray = np.asarray(xray_volume)
    if neutron.shape != xray.shape:
        raise ValueError(
            f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}"
        )
    neutron_edges = np.asarray(neutron_edges, dtype=np.float64)
    xray_edges = np.asarray(xray_edges, dtype=np.float64)
    bins = int(neutron_edges.size - 1)
    if bins < 1 or xray_edges.size - 1 != bins:
        raise ValueError("neutron_edges and xray_edges must define the same bin count")

    neutron_low, neutron_high = float(neutron_edges[0]), float(neutron_edges[-1])
    xray_low, xray_high = float(xray_edges[0]), float(xray_edges[-1])
    neutron_span = neutron_high - neutron_low
    xray_span = xray_high - xray_low
    if neutron_span <= 0 or xray_span <= 0:
        raise ValueError("Histogram edges must span a positive range")

    total_bins = bins * bins
    counts = np.zeros(total_bins, dtype=_ACCUMULATOR_DTYPE)
    sum_n = np.zeros(total_bins, dtype=_ACCUMULATOR_DTYPE)
    sum_x = np.zeros(total_bins, dtype=_ACCUMULATOR_DTYPE)
    sum_nn = np.zeros(total_bins, dtype=_ACCUMULATOR_DTYPE)
    sum_xx = np.zeros(total_bins, dtype=_ACCUMULATOR_DTYPE)
    sum_nx = np.zeros(total_bins, dtype=_ACCUMULATOR_DTYPE)

    bin_index = (
        np.full(neutron.shape, -1, dtype=np.int32) if store_bin_index else None
    )

    flat_neutron = neutron.reshape(-1)
    flat_xray = xray.reshape(-1)
    flat_valid = None if valid_mask is None else np.asarray(
        valid_mask, dtype=bool
    ).reshape(-1)
    if flat_valid is not None and flat_valid.size != flat_neutron.size:
        raise ValueError("valid_mask must match the volume shape")
    flat_bin_index = None if bin_index is None else bin_index.reshape(-1)

    num_voxels = 0
    out_of_range = 0
    invalid_voxels = 0
    chunk = max(int(chunk_voxels), 1)

    for start in range(0, flat_neutron.size, chunk):
        stop = min(start + chunk, flat_neutron.size)
        # float64 also normalises byte order, which big-endian TIFFs need
        block_n = np.asarray(flat_neutron[start:stop], dtype=np.float64)
        block_x = np.asarray(flat_xray[start:stop], dtype=np.float64)

        keep = np.isfinite(block_n) & np.isfinite(block_x)
        if flat_valid is not None:
            keep &= flat_valid[start:stop]
        invalid_voxels += int(keep.size - np.count_nonzero(keep))

        inside = keep & (
            (block_n >= neutron_low) & (block_n <= neutron_high)
            & (block_x >= xray_low) & (block_x <= xray_high)
        )
        out_of_range += int(np.count_nonzero(keep) - np.count_nonzero(inside))
        if not inside.any():
            continue

        values_n = block_n[inside]
        values_x = block_x[inside]
        neutron_bin = np.minimum(
            ((values_n - neutron_low) / neutron_span * bins).astype(np.int64),
            bins - 1,
        )
        xray_bin = np.minimum(
            ((values_x - xray_low) / xray_span * bins).astype(np.int64),
            bins - 1,
        )
        flat_bin = neutron_bin * bins + xray_bin

        np.add.at(counts, flat_bin, 1.0)
        np.add.at(sum_n, flat_bin, values_n)
        np.add.at(sum_x, flat_bin, values_x)
        np.add.at(sum_nn, flat_bin, values_n * values_n)
        np.add.at(sum_xx, flat_bin, values_x * values_x)
        np.add.at(sum_nx, flat_bin, values_n * values_x)

        num_voxels += int(values_n.size)
        if flat_bin_index is not None:
            positions = np.flatnonzero(inside) + start
            flat_bin_index[positions] = flat_bin.astype(np.int32)

    occupied = np.flatnonzero(counts > 0)
    compact_counts = counts[occupied]
    sums = np.stack([sum_n[occupied], sum_x[occupied]], axis=1)
    scatter = np.empty((occupied.size, 2, 2), dtype=_ACCUMULATOR_DTYPE)
    scatter[:, 0, 0] = sum_nn[occupied]
    scatter[:, 1, 1] = sum_xx[occupied]
    scatter[:, 0, 1] = sum_nx[occupied]
    scatter[:, 1, 0] = sum_nx[occupied]

    return HistogramCache(
        bins=bins,
        neutron_edges=neutron_edges,
        xray_edges=xray_edges,
        bin_ids=occupied.astype(np.int64),
        counts=compact_counts,
        sums=sums,
        scatter=scatter,
        num_voxels=num_voxels,
        out_of_range=out_of_range,
        invalid_voxels=invalid_voxels,
        shape=tuple(neutron.shape),
        bin_index=bin_index,
    )


def cache_from_histogram_data(neutron_volume, xray_volume, histogram_data,
                              **kwargs) -> HistogramCache:
    """Build a cache on the grid of an existing :class:`HistogramData`."""
    return build_histogram_cache(
        neutron_volume, xray_volume,
        histogram_data.x_edges, histogram_data.y_edges, **kwargs
    )


def moments_from_mask(neutron_volume, xray_volume, mask,
                      max_samples: int = 5_000_000) -> Optional[dict]:
    """Count, mean and covariance of the voxels a mask selects.

    Used to turn a manual ROI at T0 into the prior moments of a mixture
    component. Returns None when the mask selects nothing usable.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    neutron = np.asarray(neutron_volume)
    if mask_bool.shape != neutron.shape:
        raise ValueError("mask must match the volume shape")

    values_n = np.asarray(neutron[mask_bool], dtype=np.float64)
    values_x = np.asarray(np.asarray(xray_volume)[mask_bool], dtype=np.float64)
    finite = np.isfinite(values_n) & np.isfinite(values_x)
    values_n = values_n[finite]
    values_x = values_x[finite]
    count = int(values_n.size)
    if count == 0:
        return None

    if count > max_samples:
        chosen = np.random.default_rng(0).choice(count, max_samples, replace=False)
        sample_n, sample_x = values_n[chosen], values_x[chosen]
    else:
        sample_n, sample_x = values_n, values_x

    mean = np.array([sample_n.mean(), sample_x.mean()], dtype=np.float64)
    if sample_n.size > 1:
        covariance = np.cov(np.stack([sample_n, sample_x]), bias=False)
    else:
        covariance = np.zeros((2, 2), dtype=np.float64)
    return {
        "count": count,
        "mean": mean,
        "covariance": np.atleast_2d(covariance).astype(np.float64),
    }
