"""Which voxels are allowed to take part in a fit, a metric or an export.

Reconstructed tomograms carry voxels that are not measurements: zero padding
outside the reconstruction circle, NaN/Inf from a failed slice, and detector
saturation. Nothing in the pipeline used to exclude them, so they were
binned, fitted and segmented like data — a padding region large enough to
rival a real phase gets absorbed into whichever class happens to be nearest,
inflating that class's spread and dragging its centroid.

The default policy here rejects only what is provably not a measurement:
non-finite values, and the exact sentinel value the padding was written with
(usually 0). Everything else is opt-in, because a hard intensity floor is
dataset-specific and will silently delete a genuinely low-attenuation phase
in a dataset it was not tuned for.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        catches the usual zero padding. A voxel is rejected when *either*
        channel holds a sentinel, because a pair is only meaningful when both
        modalities measured it.
    neutron_floor, xray_floor
        Optional hard lower bounds. ``None`` (the default) means no floor.
        Use :func:`estimate_floor` rather than a literal — a floor copied
        from another dataset is how a real phase gets deleted.
    neutron_ceiling, xray_ceiling
        Optional hard upper bounds, for detector saturation.
    """

    reject_non_finite: bool = True
    sentinel_values: Sequence[float] = (0.0,)
    neutron_floor: Optional[float] = None
    xray_floor: Optional[float] = None
    neutron_ceiling: Optional[float] = None
    xray_ceiling: Optional[float] = None

    def describe(self) -> str:
        parts = []
        if self.reject_non_finite:
            parts.append("non-finite")
        if self.sentinel_values:
            parts.append(
                "sentinels " + ", ".join(f"{v:g}" for v in self.sentinel_values)
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


def build_valid_mask(neutron_volume, xray_volume,
                     policy: Optional[ValidityPolicy] = None) -> np.ndarray:
    """Boolean mask of the voxels that count as measurements.

    Both volumes must have the same shape; a voxel is valid only when both
    channels are.
    """
    policy = policy or DEFAULT_POLICY
    neutron = np.asarray(neutron_volume)
    xray = np.asarray(xray_volume)
    if neutron.shape != xray.shape:
        raise ValueError(
            f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}"
        )

    valid = np.ones(neutron.shape, dtype=bool)
    if policy.reject_non_finite:
        valid &= np.isfinite(neutron) & np.isfinite(xray)

    for sentinel in policy.sentinel_values:
        # Compare on the finite part only; NaN == x is False anyway, but this
        # keeps the comparison free of invalid-value warnings.
        with np.errstate(invalid="ignore"):
            valid &= ~((neutron == sentinel) & (xray == sentinel))

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


def estimate_floor(volume, valid_mask=None, quantile: float = 0.001,
                   sigma: float = 3.0) -> float:
    """Suggest a floor from the lower tail of the data itself.

    Takes the *quantile* point of the valid data and steps back *sigma*
    robust standard deviations (from the median absolute deviation, which the
    padding cannot skew as badly as the plain σ can). The result is a
    suggestion to inspect and record — not something to apply blindly.
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


def validity_report(neutron_volume, xray_volume,
                    policy: Optional[ValidityPolicy] = None) -> dict:
    """Counts behind a validity mask, for logging and for the alarm below."""
    policy = policy or DEFAULT_POLICY
    neutron = np.asarray(neutron_volume)
    xray = np.asarray(xray_volume)
    valid = build_valid_mask(neutron, xray, policy)
    total = int(neutron.size)
    n_valid = int(np.count_nonzero(valid))
    non_finite = int(
        np.count_nonzero(~(np.isfinite(neutron) & np.isfinite(xray)))
    )
    sentinel = 0
    for value in policy.sentinel_values:
        with np.errstate(invalid="ignore"):
            sentinel += int(np.count_nonzero((neutron == value) & (xray == value)))
    return {
        "total_voxels": total,
        "valid_voxels": n_valid,
        "rejected_voxels": total - n_valid,
        "rejected_fraction": (total - n_valid) / total if total else 0.0,
        "non_finite_voxels": non_finite,
        "sentinel_voxels": sentinel,
        "policy": policy.describe(),
    }


def find_acquisition_steps(reports: Sequence[dict],
                           tolerance: float = 0.02) -> list:
    """Timepoints where the rejected fraction jumps.

    A step in how much of the volume is not a measurement is an acquisition
    change — a shifted field of view, a different reconstruction, a detector
    fault. Absolute volume comparisons across such a step are not valid, so
    it is worth knowing about before the numbers are interpreted as physics.

    Returns a list of ``(timepoint, previous_fraction, fraction)`` for every
    jump larger than *tolerance*.
    """
    steps = []
    for index in range(1, len(reports)):
        previous = reports[index - 1]["rejected_fraction"]
        current = reports[index]["rejected_fraction"]
        if abs(current - previous) > tolerance:
            steps.append((index, previous, current))
    return steps
