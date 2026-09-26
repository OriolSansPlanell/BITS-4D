"""Checking and correcting the alignment of the two modalities.

The bivariate histogram pairs each neutron voxel with the X-ray voxel at the
same index. If the volumes are offset by even one voxel, every interface
contributes pairs from two different materials: the class clouds smear
towards each other, streaks appear between them, and a thin phase can
vanish into its neighbours. The streak score ``S_h`` notices this; this
module measures the offset and removes it.

Method
──────
The modalities have different contrast (a material bright in one can be
dark in the other), so intensities cannot be compared directly. What they
share is *dependence*: when aligned, the X-ray value is best predicted by
the neutron value. That is what mutual information (MI) of the joint
histogram measures, and correct alignment maximises it.

The offset is found by coordinate search over **whole-voxel** shifts
(no resampling: interpolation averages noise and raises MI by itself, so
comparisons involving resampled data are biased), then refined to a
sub-voxel value by a parabola through MI at the best shift and its two
neighbours. Along each axis the best shift must beat its neighbours clearly;
an axis along which the sample has no structure (an extruded geometry)
gives equal MI everywhere, and its component is left at zero rather than
guessed.

The correction moves the X-ray volume by the **nearest whole number of
voxels** and reports the sub-voxel remainder rather than interpolating it
away: linear (or cubic) resampling averages neighbouring voxels, which
narrows the noise of the X-ray channel, so the class clouds shrink and
partial-volume voxels fall outside them. On the synthetic validation a
half-voxel offset corrected by interpolation segmented worse than the same
offset left alone (``docs/validation.md``).

Only a rigid translation is estimated: the offsets that occur in practice
between two co-mounted instruments. Rotation, scaling or deformation need a
full registration package (elastix, ANTs) before loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class Alignment:
    """How far the X-ray volume sits from the neutron volume."""

    shift: Tuple[float, float, float]   # (z, y, x) to apply to the X-ray volume
    mi_before: float
    mi_after: float
    error: float                        # share of MI gained by the shift
    #: Per axis: did mutual information confirm the component?
    supported: Tuple[bool, bool, bool] = (True, True, True)

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.shift))

    @property
    def improves(self) -> bool:
        """Some component of the shift is confirmed by mutual information."""
        return any(self.supported) and self.magnitude > 0

    @property
    def correction(self) -> Tuple[int, int, int]:
        """The whole-voxel shift to apply (components under 0.75 → 0)."""
        return tuple(int(np.round(v)) if abs(v) >= 0.75 else 0
                     for v in self.shift)

    @property
    def worth_correcting(self) -> bool:
        """A whole voxel or more, confirmed by mutual information."""
        return any(self.correction) and self.improves

    def describe(self) -> str:
        z, y, x = self.shift
        return (f"X-ray offset (z, y, x) = ({z:+.2f}, {y:+.2f}, {x:+.2f}) "
                f"voxels; mutual information {self.mi_before:.3f} → "
                f"{self.mi_after:.3f}")


def mutual_information(first, second, bins: int = 64,
                       valid: Optional[np.ndarray] = None) -> float:
    """Mutual information (nats) of two volumes' joint histogram."""
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    keep = np.isfinite(first) & np.isfinite(second)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool).ravel()
    if keep.sum() < 2:
        return 0.0
    joint, _, _ = np.histogram2d(first[keep], second[keep], bins=bins)
    joint /= joint.sum()
    marginal_a = joint.sum(axis=1, keepdims=True)
    marginal_b = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log(
        joint[nonzero] / (marginal_a @ marginal_b)[nonzero])))


def apply_shift(volume, shift: Sequence[float], order: int = 1) -> np.ndarray:
    """Shift *volume* by *shift* voxels (z, y, x); vacated voxels become NaN,
    so the validity mask leaves them out instead of inventing values."""
    volume = np.asarray(volume, dtype=np.float32)
    if not any(abs(float(s)) > 1e-6 for s in shift):
        return volume.copy()
    if all(float(s).is_integer() for s in shift):
        return _integer_shift(volume, shift)
    return ndimage.shift(volume, shift, order=order, mode="constant",
                         cval=np.nan).astype(np.float32)


def estimate_alignment(neutron, xray, max_shift: int = 4,
                       tolerance: float = 0.01, bins: int = 32,
                       smoothing: float = 1.0,
                       max_samples: int = 400_000, seed: int = 0) -> Alignment:
    """Offset of *xray* relative to *neutron* (rigid translation).

    Both volumes are first smoothed identically (noise otherwise dominates
    the joint histogram and makes MI fluctuate from shift to shift). Then a
    coordinate search over whole-voxel shifts in ``[-max_shift, max_shift]``
    per axis maximises mutual information on a fixed sample of voxels, and a
    parabola through MI at the best shift and its two neighbours gives the
    sub-voxel part. Whole-voxel shifts need no resampling, so the comparison
    carries no interpolation bias.

    An axis counts as *supported* when MI two voxels either side of the peak
    is clearly lower (relative drop above *tolerance*): the sample has
    structure along it. Along an unsupported axis (an extruded geometry) the
    component is left at zero rather than guessed.
    """
    neutron = _prepared(neutron, smoothing)
    xray = _prepared(xray, smoothing)
    shape = neutron.shape
    margin = [min(max_shift + 2, n // 3) for n in shape]
    reach = [max(m - 2, 0) for m in margin]
    inner = tuple(slice(m, n - m) for m, n in zip(margin, shape))
    coordinates = np.stack(np.meshgrid(
        *[np.arange(s.start, s.stop) for s in inner], indexing="ij"
    ), axis=-1).reshape(-1, len(shape))
    rng = np.random.default_rng(seed)
    if len(coordinates) > max_samples:
        coordinates = coordinates[rng.choice(len(coordinates), max_samples,
                                             replace=False)]
    fixed = neutron[tuple(coordinates.T)]
    keep = np.isfinite(fixed)

    cache = {}

    def mi_at(offset):
        key = tuple(int(v) for v in offset)
        if key not in cache:
            source = coordinates - np.asarray(key)
            # The margin keeps every evaluated shift inside the array;
            # clipping is only a safety net
            for axis, size in enumerate(shape):
                np.clip(source[:, axis], 0, size - 1, out=source[:, axis])
            cache[key] = mutual_information(
                fixed, xray[tuple(source.T)], bins=bins, valid=keep)
        return cache[key]

    current = np.zeros(len(shape), dtype=int)
    for _sweep in range(3):
        moved = False
        for axis in range(len(shape)):
            scores = {}
            for value in range(-reach[axis], reach[axis] + 1):
                candidate = current.copy()
                candidate[axis] = value
                scores[value] = mi_at(candidate)
            best = max(scores, key=scores.get)
            if best != current[axis]:
                current[axis] = best
                moved = True
        if not moved:
            break

    final = np.zeros(len(shape))
    supported = []
    at_best = mi_at(current)
    for axis in range(len(shape)):
        if reach[axis] < 1:
            supported.append(False)
            continue
        steps = {}
        for step in (-2, -1, 1, 2):
            candidate = current.copy()
            candidate[axis] += step
            steps[step] = mi_at(candidate)
        far = max(steps[-2], steps[2])
        structured = at_best > 0 and (at_best - far) > tolerance * at_best
        supported.append(bool(structured))
        if not structured:
            continue
        below, above = steps[-1], steps[1]
        curvature = below - 2.0 * at_best + above
        delta = 0.5 * (below - above) / curvature if curvature < 0 else 0.0
        final[axis] = current[axis] + float(np.clip(delta, -0.5, 0.5))

    before = mi_at(np.zeros(len(shape), dtype=int))
    return Alignment(
        shift=tuple(float(v) for v in final),
        mi_before=before, mi_after=at_best,
        error=float(1.0 - before / at_best) if at_best > 0 else 0.0,
        supported=tuple(bool(v) for v in supported),
    )


def _prepared(volume, smoothing):
    """Float copy with NaN filled by the median, lightly smoothed."""
    volume = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(volume)
    if not finite.all():
        fill = float(np.nanmedian(volume)) if finite.any() else 0.0
        volume = np.where(finite, volume, fill)
    if smoothing > 0:
        volume = ndimage.gaussian_filter(volume, smoothing)
    return volume.astype(np.float64)


def check_series_alignment(dataset, timepoints: Optional[Sequence[int]] = None,
                           cancel_check=None):
    """Alignment at several timepoints: ``[(timepoint, Alignment)]``.

    A constant offset is a mounting offset (correct it once); an offset that
    changes over time means something moved during the experiment.
    """
    if timepoints is None:
        count = dataset.num_timepoints
        timepoints = sorted({0, count // 2, count - 1})
    results = []
    for timepoint in timepoints:
        if cancel_check:
            cancel_check()
        neutron, xray = dataset.get_volume_at_time(timepoint)
        results.append((int(timepoint), estimate_alignment(neutron, xray)))
    return results


def _integer_shift(volume, offset):
    """Shift by whole voxels (a copy with NaN in the vacated margin)."""
    volume = np.asarray(volume, dtype=np.float32)
    result = np.full(volume.shape, np.nan, dtype=np.float32)
    source, target = [], []
    for step, size in zip(offset, volume.shape):
        step = int(step)
        if step >= 0:
            source.append(slice(0, size - step))
            target.append(slice(step, size))
        else:
            source.append(slice(-step, size))
            target.append(slice(0, size + step))
    result[tuple(target)] = volume[tuple(source)]
    return result


def resample_to_shape(volume, shape: Sequence[int], order: int = 1) -> np.ndarray:
    """Resample *volume* to *shape* — for modalities scanned at different
    resolutions over the same field of view."""
    volume = np.asarray(volume, dtype=np.float32)
    factors = [target / current for target, current in zip(shape, volume.shape)]
    result = ndimage.zoom(volume, factors, order=order)
    # zoom can be one voxel off in each axis; crop or pad to the exact shape
    out = np.full(tuple(shape), np.nan, dtype=np.float32)
    region = tuple(slice(0, min(a, b)) for a, b in zip(result.shape, shape))
    out[region] = result[region]
    return out
