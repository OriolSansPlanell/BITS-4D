"""Which voxels are allowed to take part in a fit, a metric or an export.

Reconstructed tomograms carry voxels that are not measurements: zero padding
outside the reconstruction circle, NaN/Inf from a failed slice, detector
saturation — and, in a paired dataset, **regions one modality covers and the
other does not**. Nothing in the pipeline used to exclude any of them, so
they were binned, segmented and counted like data.

Both channels, not either
────────────────────────
A paired measurement is only meaningful where *both* instruments measured.
If the neutron and X-ray fields of view differ, the non-overlapping region
has a real value in one channel and nothing in the other; treated as data it
forms a large, static, perfectly-zero-in-one-axis blob that a class will
happily absorb — inflating that class's spread and pinning part of it to a
value no material has. Rejecting a voxel only when *both* channels are empty
misses this entirely, which is why the default requires both.

The default policy rejects what is provably not a measurement: non-finite
values, and the exact sentinel the padding was written with (usually 0), in
either channel. Intensity floors are available and auto-derivable but are
opt-in, because a floor tuned on one dataset will silently delete a
genuinely low-attenuation phase in the next one — so the software proposes
one and shows what it would remove rather than applying it unasked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class ValidityPolicy:
    """How to decide whether a voxel is a measurement.

    Parameters
    ----------
    reject_non_finite
        Drop NaN/Inf. Always sensible; on by default.
    sentinel_values
        Exact values written by the reconstruction for "no data". ``(0.0,)``
        catches the usual zero padding.
    require_both_channels
        Reject a voxel when *either* channel is missing (default). Set False
        only for a dataset where a sentinel value is genuinely meaningful in
        one modality — and check what that admits before you do.
    neutron_floor, xray_floor
        Optional hard lower bounds. ``None`` (the default) means no floor.
        Use :func:`auto_floor` or :meth:`ValidityPolicy.from_data` rather
        than a literal copied from elsewhere.
    neutron_ceiling, xray_ceiling
        Optional hard upper bounds, for detector saturation.
    """

    reject_non_finite: bool = True
    sentinel_values: Sequence[float] = (0.0,)
    require_both_channels: bool = True
    neutron_floor: Optional[float] = None
    xray_floor: Optional[float] = None
    neutron_ceiling: Optional[float] = None
    xray_ceiling: Optional[float] = None

    @classmethod
    def from_data(cls, neutron_volume, xray_volume, quantile: float = 0.001,
                  sigma: float = 3.0, **kwargs) -> "ValidityPolicy":
        """Policy with floors derived from the data's own lower tails.

        A suggestion to inspect, not a rule to trust: look at what it would
        remove (:func:`validity_report`) before adopting it.
        """
        base = cls(**kwargs)
        provisional = build_valid_mask(neutron_volume, xray_volume, base)
        base.neutron_floor = auto_floor(
            neutron_volume, provisional, quantile, sigma
        )
        base.xray_floor = auto_floor(xray_volume, provisional, quantile, sigma)
        return base

    def describe(self) -> str:
        parts = []
        if self.reject_non_finite:
            parts.append("non-finite")
        if self.sentinel_values:
            scope = "either channel" if self.require_both_channels else "both channels"
            parts.append(
                "sentinels " + ", ".join(f"{v:g}" for v in self.sentinel_values)
                + f" in {scope}"
            )
        for label, low, high in (
            ("neutron", self.neutron_floor, self.neutron_ceiling),
            ("X-ray", self.xray_floor, self.xray_ceiling),
        ):
            if low is not None or high is not None:
                bounds = f"{'-inf' if low is None else f'{low:g}'}" \
                         f" .. {'inf' if high is None else f'{high:g}'}"
                parts.append(f"{label} in [{bounds}]")
        return "; ".join(parts) if parts else "everything accepted"


DEFAULT_POLICY = ValidityPolicy()


def _channel_has_data(volume, sentinels, reject_non_finite: bool) -> np.ndarray:
    """Per-channel mask of voxels this instrument actually measured."""
    array = np.asarray(volume)
    present = np.ones(array.shape, dtype=bool)
    if reject_non_finite:
        present &= np.isfinite(array)
    for sentinel in sentinels:
        with np.errstate(invalid="ignore"):
            present &= array != sentinel
    return present


def build_valid_mask(neutron_volume, xray_volume,
                     policy: Optional[ValidityPolicy] = None) -> np.ndarray:
    """Boolean mask of the voxels that count as measurements."""
    policy = policy or DEFAULT_POLICY
    neutron = np.asarray(neutron_volume)
    xray = np.asarray(xray_volume)
    if neutron.shape != xray.shape:
        raise ValueError(
            f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}"
        )

    neutron_ok = _channel_has_data(
        neutron, policy.sentinel_values, policy.reject_non_finite
    )
    xray_ok = _channel_has_data(
        xray, policy.sentinel_values, policy.reject_non_finite
    )
    if policy.require_both_channels:
        valid = neutron_ok & xray_ok
    else:
        valid = neutron_ok | xray_ok
        if policy.reject_non_finite:
            valid &= np.isfinite(neutron) & np.isfinite(xray)

    for volume, floor, ceiling in (
        (neutron, policy.neutron_floor, policy.neutron_ceiling),
        (xray, policy.xray_floor, policy.xray_ceiling),
    ):
        with np.errstate(invalid="ignore"):
            if floor is not None:
                valid &= volume >= floor
            if ceiling is not None:
                valid &= volume <= ceiling
    return valid


def auto_floor(volume, valid_mask=None, quantile: float = 0.001,
               sigma: float = 3.0) -> float:
    """Suggest a floor from the lower tail of the data itself.

    Takes the *quantile* point of the valid data and steps back *sigma*
    robust standard deviations (from the median absolute deviation, which
    padding cannot skew as badly as the plain σ can).
    """
    array = np.asarray(volume, dtype=np.float64)
    if valid_mask is not None:
        array = array[np.asarray(valid_mask, dtype=bool)]
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    low = float(np.quantile(array, quantile))
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return low - sigma * 1.4826 * mad


# Retained name from the previous release
estimate_floor = auto_floor


def channel_coverage(neutron_volume, xray_volume,
                     policy: Optional[ValidityPolicy] = None) -> dict:
    """How far the two modalities' fields of view actually agree.

    ``neutron_only`` and ``xray_only`` are the voxels one instrument measured
    and the other did not. A large value there means the two scans do not
    cover the same region, and every paired quantity is only meaningful on
    the overlap.
    """
    policy = policy or DEFAULT_POLICY
    neutron = np.asarray(neutron_volume)
    xray = np.asarray(xray_volume)
    neutron_ok = _channel_has_data(
        neutron, policy.sentinel_values, policy.reject_non_finite
    )
    xray_ok = _channel_has_data(
        xray, policy.sentinel_values, policy.reject_non_finite
    )
    total = int(neutron.size)
    both = int(np.count_nonzero(neutron_ok & xray_ok))
    neutron_only = int(np.count_nonzero(neutron_ok & ~xray_ok))
    xray_only = int(np.count_nonzero(~neutron_ok & xray_ok))
    neither = total - both - neutron_only - xray_only
    return {
        "total_voxels": total,
        "both_channels": both,
        "neutron_only": neutron_only,
        "xray_only": xray_only,
        "neither": neither,
        "overlap_fraction": both / total if total else 0.0,
        "neutron_only_fraction": neutron_only / total if total else 0.0,
        "xray_only_fraction": xray_only / total if total else 0.0,
    }


def validity_report(neutron_volume, xray_volume,
                    policy: Optional[ValidityPolicy] = None) -> dict:
    """Counts behind a validity mask, for logging and for the alarm below."""
    policy = policy or DEFAULT_POLICY
    neutron = np.asarray(neutron_volume)
    xray = np.asarray(xray_volume)
    valid = build_valid_mask(neutron, xray, policy)
    coverage = channel_coverage(neutron, xray, policy)

    total = int(neutron.size)
    n_valid = int(np.count_nonzero(valid))
    non_finite = int(
        np.count_nonzero(~(np.isfinite(neutron) & np.isfinite(xray)))
    )
    sentinel = 0
    for value in policy.sentinel_values:
        with np.errstate(invalid="ignore"):
            sentinel += int(
                np.count_nonzero((neutron == value) | (xray == value))
            )
    report = {
        "total_voxels": total,
        "valid_voxels": n_valid,
        "rejected_voxels": total - n_valid,
        "rejected_fraction": (total - n_valid) / total if total else 0.0,
        "non_finite_voxels": non_finite,
        "sentinel_voxels": sentinel,
        "policy": policy.describe(),
    }
    report.update(coverage)
    return report


def find_acquisition_steps(reports: Sequence[dict],
                           tolerance: float = 0.02) -> list:
    """Timepoints where the fraction of usable data jumps.

    A step in how much of the volume is not a measurement is an acquisition
    change — a shifted field of view, a different reconstruction, a detector
    fault. Absolute volume comparisons across such a step are not valid, so
    it is worth knowing about before the numbers are read as physics.

    Returns ``(timepoint, previous_fraction, fraction)`` for every jump
    larger than *tolerance*.
    """
    steps = []
    for index in range(1, len(reports)):
        previous = reports[index - 1]["rejected_fraction"]
        current = reports[index]["rejected_fraction"]
        if abs(current - previous) > tolerance:
            steps.append((index, previous, current))
    return steps
