"""Instrumental drift of the histogram, measured on chemically inert classes.

Over a long series the whole (neutron, X-ray) cloud migrates: beam current,
detector gain, scatter build-up and reconstruction changes all move it. A
boundary drawn once at T0 — a polygon or a classifier trained there — cannot
follow that, so it slowly segments the wrong thing while reporting no error.

Anchor classes are the fix. A phase that is chemically inert in the
experiment cannot really change, so *any* movement of its histogram centroid
is instrumental by definition, and that movement estimates the drift for
everything else.

Estimated in parameter space, not on the data
─────────────────────────────────────────────
The obvious implementation rewrites every volume into a drift-corrected
frame. This module does not, for two reasons: it would cost a full pass over
every voxel at every timepoint, and — worse — it would silently change the
units of every histogram, statistic and export downstream, so numbers would
stop being comparable with anything produced before.

Instead the estimate is applied to the **model**: a component anchored at
``(μ₀, Σ₀)`` at T0 is anchored at ``(s⊙μ₀ + d, S Σ₀ Sᵀ)`` at time *t*. The
fit is identical, no voxel is touched, and every export stays in native
intensity units. :meth:`DriftTracker.to_reference_frame` is available for the
cases where normalised *values* really are wanted, but it is opt-in.

Chicken and egg
───────────────
Measuring an anchor's centroid at time *t* seems to need the segmentation
that the drift estimate is supposed to enable. It does not: the anchor is a
dense, isolated mode, so a mean-shift started from its T0 position and run on
the timepoint's own histogram finds it without any labels at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class DriftEstimate:
    """Drift of one timepoint relative to the reference."""

    timepoint: Optional[int] = None
    shift: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    scale: np.ndarray = field(
        default_factory=lambda: np.ones(2, dtype=np.float64)
    )
    per_anchor: Dict[str, np.ndarray] = field(default_factory=dict)
    rejected_anchors: List[str] = field(default_factory=list)
    residual: float = 0.0
    scale_estimated: bool = False

    @property
    def magnitude(self) -> float:
        """Length of the shift in intensity units."""
        return float(np.hypot(*self.shift))

    def transform_mean(self, mean) -> np.ndarray:
        """Map a reference-frame centroid into this timepoint's frame."""
        return self.scale * np.asarray(mean, dtype=np.float64) + self.shift

    def transform_covariance(self, covariance) -> np.ndarray:
        scaling = np.diag(self.scale)
        return scaling @ np.asarray(covariance, dtype=np.float64) @ scaling

    def describe(self) -> str:
        text = (
            f"shift (Δn, Δx) = ({self.shift[0]:+.4g}, {self.shift[1]:+.4g})"
        )
        if self.scale_estimated:
            text += f", scale = ({self.scale[0]:.5g}, {self.scale[1]:.5g})"
        if self.rejected_anchors:
            text += f", rejected: {', '.join(self.rejected_anchors)}"
        return text


class DriftTracker:
    """Estimate histogram drift from anchor classes.

    Parameters
    ----------
    anchor_classes
        Names of the classes to trust as inert. Choose them on physics and on
        evidence — a class whose segmented volume is flat across the series
        and that has no role in the reaction. Estimating from reactive
        classes regresses the physics away, so nothing else is ever used.
    estimate_scale
        Also fit a per-axis gain. Needs at least two anchors that sit at
        different positions on that axis; otherwise the gain is not
        identifiable and stays 1.0 with ``scale_estimated=False``.
    kernel_width
        Mean-shift kernel size as a multiple of each anchor's own σ.
    max_shift_sigma
        An anchor that moves further than this many of its own σ is assumed
        to have latched onto a different mode and is dropped from the fit.
    min_separation_sigma
        Two anchors that converge to within this many σ of each other have
        landed on the *same* mode, and cannot both be right. The one that
        moved further is dropped. Without this check a drift estimate can be
        built from one mode counted twice — which looks entirely healthy,
        because neither anchor moved implausibly far on its own.
    """

    def __init__(
        self,
        anchor_classes: Sequence[str],
        estimate_scale: bool = False,
        kernel_width: float = 1.5,
        max_shift_sigma: float = 4.0,
        min_separation_sigma: float = 1.0,
        max_iter: int = 40,
        tol: float = 1e-4,
    ) -> None:
        if not anchor_classes:
            raise ValueError("At least one anchor class is required")
        self.anchor_classes = list(anchor_classes)
        self.estimate_scale = bool(estimate_scale)
        self.kernel_width = float(kernel_width)
        self.max_shift_sigma = float(max_shift_sigma)
        self.min_separation_sigma = float(min_separation_sigma)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reference: Dict[str, dict] = {}

    # ── reference ────────────────────────────────────────────────────────
    def fit_reference(self, moments_by_class: Dict[str, dict]) -> Dict[str, dict]:
        """Record the anchors' T0 moments.

        *moments_by_class* maps a class name to ``{'mean', 'covariance',
        'count'}`` — exactly what
        :func:`model.histogram_cache.moments_from_mask` returns.
        """
        self.reference = {}
        for name in self.anchor_classes:
            moments = moments_by_class.get(name)
            if moments is None:
                continue
            covariance = np.asarray(moments["covariance"], dtype=np.float64)
            spread = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            if not np.all(np.isfinite(spread)) or np.all(spread <= 0):
                # A perfectly uniform anchor gives the kernel no width; fall
                # back to a small fraction of the anchor's own magnitude.
                magnitude = float(np.max(np.abs(moments["mean"]))) or 1.0
                spread = np.full(2, 0.01 * magnitude)
            self.reference[name] = {
                "mean": np.asarray(moments["mean"], dtype=np.float64),
                "covariance": covariance,
                "sigma": np.maximum(spread, 1e-9),
                "count": int(moments.get("count", 1)),
            }
        if not self.reference:
            raise ValueError(
                "None of the anchor classes "
                f"{self.anchor_classes} were found in the reference moments"
            )
        return self.reference

    @property
    def is_fitted(self) -> bool:
        return bool(self.reference)

    # ── estimation ───────────────────────────────────────────────────────
    def _mean_shift(self, cache, start: np.ndarray,
                    sigma: np.ndarray) -> np.ndarray:
        """Locate the mode of *cache* nearest *start*."""
        positions = cache.means            # [M, 2] true within-bin means
        weights = cache.counts             # [M]
        width = np.maximum(self.kernel_width * sigma, 1e-12)
        inverse_variance = 1.0 / (width ** 2)

        centre = np.asarray(start, dtype=np.float64).copy()
        for _ in range(self.max_iter):
            offset = positions - centre
            exponent = -0.5 * np.sum((offset ** 2) * inverse_variance, axis=1)
            # Guard against a kernel that has drifted off the data entirely
            peak = float(exponent.max()) if exponent.size else 0.0
            kernel = weights * np.exp(exponent - peak)
            mass = float(kernel.sum())
            if mass <= 0:
                return centre
            updated = (kernel[:, None] * positions).sum(axis=0) / mass
            movement = float(np.max(np.abs(updated - centre) / width))
            centre = updated
            if movement < self.tol:
                break
        return centre

    def _collisions(self, observed: Dict[str, np.ndarray]) -> List[tuple]:
        """Anchor pairs that converged onto one another."""
        names = list(observed)
        pairs = []
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                scale = np.maximum(
                    self.reference[first]["sigma"], self.reference[second]["sigma"]
                )
                separation = np.abs(observed[first] - observed[second]) / scale
                if np.all(separation < self.min_separation_sigma):
                    pairs.append((first, second))
        return pairs

    def estimate(self, cache, timepoint: Optional[int] = None,
                 previous: Optional[DriftEstimate] = None) -> DriftEstimate:
        """Drift of one timepoint's histogram cache against the reference.

        *previous* is the estimate for the timepoint before this one. Passing
        it makes the search **cumulative**: the mean-shift starts from where
        the anchor was last seen rather than from T0, and the plausibility
        check is applied to the step since then. Without it, a series whose
        total drift exceeds a few σ would have every anchor rejected as
        implausible precisely when the drift matters most — the estimator
        would give up exactly where it is needed.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_reference() before estimate()")

        estimate = DriftEstimate(timepoint=timepoint)
        observed: Dict[str, np.ndarray] = {}
        weights: Dict[str, float] = {}
        displacement: Dict[str, float] = {}

        for name, anchor in self.reference.items():
            expected = (
                anchor["mean"] if previous is None
                else previous.transform_mean(anchor["mean"])
            )
            found = self._mean_shift(cache, expected, anchor["sigma"])
            movement = np.abs(found - expected) / anchor["sigma"]
            if np.any(movement > self.max_shift_sigma):
                estimate.rejected_anchors.append(name)
                continue
            observed[name] = found
            # The reported drift always maps the T0 frame onto this timepoint
            estimate.per_anchor[name] = found - anchor["mean"]
            weights[name] = float(anchor["count"])

        # Anchors that landed on the same mode: reject every one involved.
        # Which of them is the impostor cannot be told apart from the anchors
        # alone — both moved a similar, individually plausible distance — so
        # picking one would be a guess dressed up as a measurement. Dropping
        # them leaves the estimate to the anchors that stayed distinct, or to
        # the previous timepoint. This is also why estimation should run
        # incrementally: a series stepped through one timepoint at a time
        # never asks an anchor to make the ambiguous jump in the first place.
        collided = set()
        for first, second in self._collisions(observed):
            collided.update((first, second))
        for name in collided:
            del observed[name]
            estimate.per_anchor.pop(name, None)
            weights.pop(name, None)
            estimate.rejected_anchors.append(name)

        if not observed:
            # Every anchor was rejected. Carry the previous estimate forward
            # rather than snapping back to zero: the drift did not vanish
            # just because this timepoint's anchors could not be located.
            estimate.residual = float("nan")
            if previous is not None:
                estimate.shift = previous.shift.copy()
                estimate.scale = previous.scale.copy()
                estimate.scale_estimated = previous.scale_estimated
            return estimate

        names = list(observed)
        total_weight = sum(weights[name] for name in names) or 1.0

        if self.estimate_scale and len(names) >= 2:
            reference_points = np.array(
                [self.reference[name]["mean"] for name in names]
            )
            observed_points = np.array([observed[name] for name in names])
            anchor_weights = np.array([weights[name] for name in names])
            scale = np.ones(2)
            shift = np.zeros(2)
            for axis in range(2):
                x_values = reference_points[:, axis]
                y_values = observed_points[:, axis]
                spread = np.average(
                    (x_values - np.average(x_values, weights=anchor_weights)) ** 2,
                    weights=anchor_weights,
                )
                if spread <= 0:
                    # Anchors coincide on this axis: gain is unidentifiable
                    shift[axis] = np.average(
                        y_values - x_values, weights=anchor_weights
                    )
                    continue
                x_mean = np.average(x_values, weights=anchor_weights)
                y_mean = np.average(y_values, weights=anchor_weights)
                covariance = np.average(
                    (x_values - x_mean) * (y_values - y_mean),
                    weights=anchor_weights,
                )
                scale[axis] = covariance / spread
                shift[axis] = y_mean - scale[axis] * x_mean
                estimate.scale_estimated = True
            estimate.scale = scale
            estimate.shift = shift
        else:
            estimate.shift = sum(
                weights[name] * estimate.per_anchor[name] for name in names
            ) / total_weight

        residuals = [
            float(np.hypot(*(observed[name] - estimate.transform_mean(
                self.reference[name]["mean"]))))
            for name in names
        ]
        estimate.residual = float(np.mean(residuals)) if residuals else 0.0
        return estimate

    # ── optional value-space normalisation ───────────────────────────────
    @staticmethod
    def to_reference_frame(neutron_values, xray_values,
                           estimate: DriftEstimate):
        """Map measured values back into the reference frame.

        Only for the cases that genuinely need normalised *values* — an
        export meant to be compared voxel-for-voxel across time, say. The
        segmenter does not use it: it transforms the model instead, which is
        equivalent for the fit and leaves the data in its own units.
        """
        neutron = (np.asarray(neutron_values, dtype=np.float64)
                   - estimate.shift[0]) / estimate.scale[0]
        xray = (np.asarray(xray_values, dtype=np.float64)
                - estimate.shift[1]) / estimate.scale[1]
        return neutron, xray


def estimate_process_noise(estimates: Sequence[DriftEstimate],
                           floor: float = 1e-6) -> np.ndarray:
    """Instrumental noise floor, from what the anchors still do after correction.

    The anchors cannot really move, so whatever movement survives drift
    correction is measurement noise. That is the *smallest* step a component
    should be allowed to take between timepoints; a reactive class is then
    permitted to move faster than it by an explicit factor rather than by
    accident.

    Returns the per-axis variance of the residual anchor movement.
    """
    residuals = []
    for estimate in estimates:
        for name, movement in estimate.per_anchor.items():
            corrected = np.asarray(movement, dtype=np.float64) - estimate.shift
            residuals.append(corrected)
    if len(residuals) < 2:
        return np.full(2, floor)
    stacked = np.asarray(residuals, dtype=np.float64)
    return np.maximum(stacked.var(axis=0), floor)
