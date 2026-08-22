"""ROI-anchored Gaussian mixture over the bimodal histogram.

The idea worth keeping from the v17 proposal: a manually drawn ROI and a
free Gaussian mixture are **the two limits of one model**, not two rival
methods. Put a Normal-Inverse-Wishart prior on each component, centred on
the moments of the voxels the ROI selects at T0, and let its strength κ₀
interpolate:

* ``κ₀ → ∞`` pins every component at its T0 position — the fixed-ROI
  behaviour, which cannot follow drift;
* ``κ₀ → 0`` is an unconstrained mixture, which follows drift and also
  follows noise, swaps components and collapses onto dense regions;
* finite ``κ₀`` follows the data as far as the prior allows and no further.

Anchor strength instead of a raw pseudo-count
─────────────────────────────────────────────
κ₀ is a pseudo-count, so ``κ₀ = 1000`` means something completely different
for a class of 4 000 voxels than for one of 4 000 000, and it is useless as a
user-facing control. :func:`anchor_strength_to_kappa` maps a dimensionless
``strength ∈ [0, 1]`` onto ``κ₀ = n·s/(1−s)``, where *n* is the class's own
T0 size. At ``s = 0.5`` the prior is worth exactly as much as the data; 0 is
a free mixture and 1 is frozen, at every class size and on every dataset.

The M-step is exact
───────────────────
Fitting runs on a :class:`~model.histogram_cache.HistogramCache`, which
carries per-bin first and second moments, so the parameter updates are
algebraically the voxel-level ones. Responsibilities are shared within a bin
— that is the only approximation — and even the responsibility uses the
within-bin scatter rather than the bin centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

DIMENSION = 2
_LOG_2PI = math.log(2.0 * math.pi)
_MAX_KAPPA = 1e12


def anchor_strength_to_kappa(strength: float, class_voxels: float) -> float:
    """Dimensionless anchor strength in [0, 1] → NIW pseudo-count κ₀.

    ``0`` is a free mixture, ``0.5`` weighs the T0 anchor exactly as much as
    the data, ``1`` freezes the component. Because the scale is the class's
    own size, the same slider position means the same thing for a 0.96 %
    phase and a 36 % one.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    count = max(float(class_voxels), 1.0)
    if strength >= 1.0:
        return _MAX_KAPPA
    return count * strength / (1.0 - strength)


@dataclass
class ComponentPrior:
    """The T0 anchor of one mixture component."""

    name: str
    mean: np.ndarray
    covariance: np.ndarray
    kappa: float
    nu: float
    weight: float = 1.0
    count: float = 1.0
    fixed: bool = False
    class_id: Optional[int] = None

    def scaled(self, drift_estimate) -> "ComponentPrior":
        """This anchor expressed in a drifted timepoint's frame."""
        if drift_estimate is None:
            return self
        return ComponentPrior(
            name=self.name,
            mean=drift_estimate.transform_mean(self.mean),
            covariance=drift_estimate.transform_covariance(self.covariance),
            kappa=self.kappa,
            nu=self.nu,
            weight=self.weight,
            count=self.count,
            fixed=self.fixed,
            class_id=self.class_id,
        )


@dataclass
class MixturePrior:
    """Priors for every component, plus the outlier weight."""

    components: List[ComponentPrior] = field(default_factory=list)
    dirichlet_strength: float = 0.0
    outlier_weight: float = 1e-3

    @property
    def names(self) -> List[str]:
        return [component.name for component in self.components]

    @property
    def n_components(self) -> int:
        return len(self.components)

    def scaled(self, drift_estimate) -> "MixturePrior":
        return MixturePrior(
            components=[c.scaled(drift_estimate) for c in self.components],
            dirichlet_strength=self.dirichlet_strength,
            outlier_weight=self.outlier_weight,
        )


@dataclass
class FitResult:
    """Fitted parameters and the diagnostics needed to judge them."""

    names: List[str]
    means: np.ndarray                 # [K, 2]
    covariances: np.ndarray           # [K, 2, 2]
    weights: np.ndarray               # [K] mixture weights of real components
    outlier_weight: float
    responsibilities: np.ndarray      # [M, K (+1 if outlier)]
    log_density: np.ndarray           # [M, K (+1)] per-voxel mean log density
    counts: np.ndarray                # [K] effective voxels per component
    log_likelihood: float
    n_iter: int
    converged: bool
    num_voxels: int
    has_outlier: bool = False
    timepoint: Optional[int] = None
    drift: Optional[object] = None
    prior: Optional[MixturePrior] = None

    @property
    def n_components(self) -> int:
        return len(self.names)

    @property
    def n_parameters(self) -> int:
        """Free parameters, for BIC/ICL."""
        per_component = DIMENSION + DIMENSION * (DIMENSION + 1) // 2
        total = self.n_components * per_component + (self.n_components - 1)
        return total + (1 if self.has_outlier else 0)

    def bic(self) -> float:
        """Lower is better. ``N`` is the **voxel** count, not the bin count.

        Fitting happens on bins, but the log-likelihood is a per-voxel
        quantity, so penalising by the number of occupied bins would make the
        criterion depend on the histogram resolution rather than on the data.
        """
        return -2.0 * self.log_likelihood + self.n_parameters * math.log(
            max(self.num_voxels, 2)
        )

    def entropy(self) -> float:
        """Voxel-weighted entropy of the responsibilities."""
        resp = np.clip(self.responsibilities, 1e-300, 1.0)
        return float(-np.sum(self.bin_counts[:, None] * resp * np.log(resp)))

    def icl(self) -> float:
        """BIC plus the assignment entropy — penalises overlapping components.

        Preferred over BIC when the question is "is this a genuinely separate
        phase", because a component that merely shadows another buys
        likelihood without buying a clean assignment.
        """
        return self.bic() + 2.0 * self.entropy()

    # populated by the fitter; kept out of the constructor signature
    bin_counts: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def hard_labels_per_bin(self) -> np.ndarray:
        """Most likely component for each bin; ``-1`` marks the outlier."""
        best = np.argmax(self.responsibilities, axis=1)
        if self.has_outlier:
            best = np.where(best >= self.n_components, -1, best)
        return best.astype(np.int32)

    def moved_sigma(self) -> Dict[str, float]:
        """How far each component moved from its anchor, in anchor σ.

        This is the quantity to look at when choosing the anchor strength: a
        component that has moved many σ is being driven by the data, one that
        has not moved at all is being held by the prior.
        """
        if self.prior is None:
            return {}
        moved = {}
        for index, component in enumerate(self.prior.components):
            sigma = np.sqrt(np.maximum(np.diag(component.covariance), 1e-30))
            offset = (self.means[index] - component.mean) / sigma
            moved[component.name] = float(np.hypot(*offset))
        return moved

    def overlap(self) -> Dict[tuple, float]:
        """Bhattacharyya overlap coefficient for every component pair."""
        result = {}
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                result[(self.names[i], self.names[j])] = _bhattacharyya(
                    self.means[i], self.covariances[i],
                    self.means[j], self.covariances[j],
                )
        return result


def _bhattacharyya(mean_a, covariance_a, mean_b, covariance_b) -> float:
    """Overlap coefficient in [0, 1]; 1 means the components coincide."""
    average = 0.5 * (covariance_a + covariance_b)
    determinant = np.linalg.det(average)
    det_a = np.linalg.det(covariance_a)
    det_b = np.linalg.det(covariance_b)
    if determinant <= 0 or det_a <= 0 or det_b <= 0:
        return float("nan")
    difference = np.asarray(mean_a) - np.asarray(mean_b)
    try:
        mahalanobis = float(difference @ np.linalg.solve(average, difference))
    except np.linalg.LinAlgError:
        return float("nan")
    distance = 0.125 * mahalanobis + 0.5 * math.log(
        determinant / math.sqrt(det_a * det_b)
    )
    return float(math.exp(-max(distance, 0.0)))


def _regularise(covariance: np.ndarray, ridge: float) -> np.ndarray:
    """Keep a covariance positive definite without distorting its shape."""
    matrix = 0.5 * (covariance + covariance.T)
    scale = max(float(np.trace(matrix)) / DIMENSION, 0.0)
    floor = max(ridge * scale, ridge)
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if determinant <= 0 or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        matrix = matrix + np.eye(DIMENSION) * floor
    else:
        matrix = matrix + np.eye(DIMENSION) * (floor * 1e-3)
    return matrix


def _inverse_and_logdet(covariance: np.ndarray):
    """Closed-form 2×2 inverse plus log-determinant."""
    a, b = covariance[0, 0], covariance[0, 1]
    c, d = covariance[1, 0], covariance[1, 1]
    determinant = a * d - b * c
    if determinant <= 0 or not np.isfinite(determinant):
        return None, None
    inverse = np.array([[d, -b], [-c, a]], dtype=np.float64) / determinant
    return inverse, math.log(determinant)


class ROIAnchoredMixture:
    """MAP-EM for a mixture anchored on manually drawn ROIs.

    Parameters
    ----------
    outlier_component
        Add a uniform density over the histogram support. It absorbs padding,
        unexpected materials and reconstruction artifacts instead of forcing
        them into whichever real class happens to be nearest — a model-level
        defence that works even when the validity mask misses something.
    reject_margin
        Voxels whose best responsibility falls below this are reported as
        unassigned rather than being given the argmax label. ``None``
        disables abstention.
    """

    def __init__(
        self,
        outlier_component: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        ridge: float = 1e-6,
        reject_margin: Optional[float] = None,
    ) -> None:
        self.outlier_component = bool(outlier_component)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.reject_margin = reject_margin

    # ── prior construction ───────────────────────────────────────────────
    @staticmethod
    def prior_from_moments(
        moments_by_class: Dict[str, dict],
        anchor_strength: float = 0.5,
        per_class_strength: Optional[Dict[str, float]] = None,
        dirichlet_strength: float = 0.0,
        outlier_weight: float = 1e-3,
        fixed_classes: Sequence[str] = (),
        class_ids: Optional[Dict[str, int]] = None,
    ) -> MixturePrior:
        """Build a prior from the T0 moments of each ROI.

        *moments_by_class* maps a class name to ``{'mean', 'covariance',
        'count'}``. ``ν₀`` is tied to ``κ₀`` so that position and shape are
        anchored equally firmly, and ``Ψ₀`` is set so the prior *mode* of the
        covariance is exactly the ROI's covariance.
        """
        per_class_strength = per_class_strength or {}
        class_ids = class_ids or {}
        total = sum(
            max(float(m.get("count", 0)), 0.0) for m in moments_by_class.values()
        ) or 1.0

        components = []
        for name, moments in moments_by_class.items():
            count = max(float(moments.get("count", 1)), 1.0)
            strength = float(per_class_strength.get(name, anchor_strength))
            kappa = anchor_strength_to_kappa(strength, count)
            covariance = _regularise(
                np.asarray(moments["covariance"], dtype=np.float64), 1e-6
            )
            components.append(
                ComponentPrior(
                    name=name,
                    mean=np.asarray(moments["mean"], dtype=np.float64),
                    covariance=covariance,
                    kappa=kappa,
                    nu=kappa,
                    weight=count / total,
                    count=count,
                    fixed=name in set(fixed_classes),
                    class_id=class_ids.get(name),
                )
            )
        if not components:
            raise ValueError("No usable ROI moments to build a prior from")
        return MixturePrior(
            components=components,
            dirichlet_strength=float(dirichlet_strength),
            outlier_weight=float(outlier_weight),
        )

    # ── EM ───────────────────────────────────────────────────────────────
    def fit(self, cache, prior: MixturePrior,
            initial_means: Optional[np.ndarray] = None,
            initial_covariances: Optional[np.ndarray] = None,
            initial_weights: Optional[np.ndarray] = None,
            cancel_check=None) -> FitResult:
        """Run MAP-EM on a histogram cache."""
        if cache.num_bins == 0:
            raise ValueError("The histogram cache is empty; nothing to fit")

        n_components = prior.n_components
        bin_counts = cache.counts
        bin_means = cache.means
        bin_scatter = cache.within_bin_scatter
        total_voxels = float(bin_counts.sum())

        means = (
            np.array([c.mean for c in prior.components], dtype=np.float64)
            if initial_means is None else np.array(initial_means, dtype=np.float64)
        )
        covariances = (
            np.array([c.covariance for c in prior.components], dtype=np.float64)
            if initial_covariances is None
            else np.array(initial_covariances, dtype=np.float64)
        )
        weights = (
            np.array([c.weight for c in prior.components], dtype=np.float64)
            if initial_weights is None else np.array(initial_weights, dtype=np.float64)
        )
        weights = np.maximum(weights, 1e-12)

        has_outlier = self.outlier_component
        outlier_weight = float(prior.outlier_weight) if has_outlier else 0.0
        if has_outlier:
            weights *= (1.0 - outlier_weight) / weights.sum()
        else:
            weights /= weights.sum()

        area = cache.support_area()
        log_uniform = -math.log(area) if area > 0 else 0.0

        # Dirichlet MAP: alpha_k = 1 + strength * prior weight
        alpha_minus_one = prior.dirichlet_strength * np.array(
            [c.weight for c in prior.components], dtype=np.float64
        )

        anchor_means = np.array([c.mean for c in prior.components])
        anchor_kappa = np.array([c.kappa for c in prior.components])
        anchor_nu = np.array([c.nu for c in prior.components])
        # Psi0 chosen so the prior mode of Sigma is exactly the ROI covariance
        anchor_psi = np.array([
            (c.nu + DIMENSION + 1.0) * c.covariance for c in prior.components
        ])
        is_fixed = np.array([c.fixed for c in prior.components], dtype=bool)

        n_columns = n_components + (1 if has_outlier else 0)
        log_density = np.zeros((cache.num_bins, n_columns), dtype=np.float64)
        responsibilities = np.zeros_like(log_density)

        previous_ll = -np.inf
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            if cancel_check:
                cancel_check()

            # ── E-step ───────────────────────────────────────────────────
            for k in range(n_components):
                inverse, log_det = _inverse_and_logdet(covariances[k])
                if inverse is None:
                    covariances[k] = _regularise(covariances[k], max(self.ridge, 1e-4))
                    inverse, log_det = _inverse_and_logdet(covariances[k])
                    if inverse is None:
                        log_density[:, k] = -np.inf
                        continue
                offset = bin_means - means[k]
                quadratic = np.einsum("mi,ij,mj->m", offset, inverse, offset)
                # Mean log-density over the voxels in the bin, not the
                # log-density at the bin's mean: the correction is the
                # within-bin scatter seen through this component. Cancellation
                # can push a single-voxel bin's scatter slightly negative.
                within = np.maximum(
                    np.einsum("ij,mji->m", inverse, bin_scatter), 0.0
                ) / bin_counts
                log_density[:, k] = -0.5 * (
                    DIMENSION * _LOG_2PI + log_det + quadratic + within
                )
            if has_outlier:
                log_density[:, -1] = log_uniform

            log_weights = np.log(
                np.concatenate([weights, [outlier_weight]]) if has_outlier
                else weights
            )
            joint = log_density + log_weights
            peak = joint.max(axis=1, keepdims=True)
            # A bin no component can explain at all would otherwise produce
            # -inf - -inf; give it the outlier/uniform assignment instead.
            unexplained = ~np.isfinite(peak[:, 0])
            if unexplained.any():
                peak[unexplained, 0] = 0.0
                joint[unexplained] = np.where(
                    np.isfinite(joint[unexplained]), joint[unexplained], -745.0
                )
            np.exp(joint - peak, out=responsibilities)
            normaliser = np.maximum(
                responsibilities.sum(axis=1, keepdims=True), 1e-300
            )
            log_likelihood = float(
                np.sum(bin_counts * (peak[:, 0] + np.log(normaliser[:, 0])))
            )
            responsibilities /= normaliser

            # ── M-step ───────────────────────────────────────────────────
            weighted = responsibilities * bin_counts[:, None]
            effective = weighted.sum(axis=0)

            for k in range(n_components):
                count = float(effective[k])
                if count <= 0:
                    means[k] = anchor_means[k]
                    covariances[k] = anchor_psi[k] / (anchor_nu[k] + DIMENSION + 1.0)
                    continue
                if is_fixed[k]:
                    continue

                # cache.sums and cache.scatter are already summed over the
                # voxels in each bin, so they take the bare responsibility —
                # weighting them by the bin count as well would count every
                # voxel c_m times and inflate the covariance by roughly the
                # mean bin occupancy.
                sum_vector = responsibilities[:, k] @ cache.sums
                data_mean = sum_vector / count
                second = np.einsum(
                    "m,mij->ij", responsibilities[:, k], cache.scatter
                )
                scatter = second - count * np.outer(data_mean, data_mean)

                kappa = anchor_kappa[k]
                means[k] = (kappa * anchor_means[k] + count * data_mean) / (
                    kappa + count
                )
                offset = data_mean - anchor_means[k]
                psi = (
                    anchor_psi[k]
                    + scatter
                    + (kappa * count / (kappa + count)) * np.outer(offset, offset)
                )
                covariances[k] = _regularise(
                    psi / (anchor_nu[k] + count + DIMENSION + 1.0), self.ridge
                )

            if has_outlier:
                outlier_weight = float(
                    max(effective[-1] / max(total_voxels, 1.0), 1e-8)
                )
            denominator = float(alpha_minus_one.sum() + total_voxels)
            weights = (alpha_minus_one + effective[:n_components]) / denominator
            weights = np.maximum(weights, 1e-12)
            available = 1.0 - outlier_weight if has_outlier else 1.0
            weights *= available / weights.sum()

            if abs(log_likelihood - previous_ll) <= self.tol * abs(
                max(log_likelihood, 1.0)
            ):
                converged = True
                previous_ll = log_likelihood
                break
            previous_ll = log_likelihood

        result = FitResult(
            names=list(prior.names),
            means=means,
            covariances=covariances,
            weights=weights,
            outlier_weight=outlier_weight,
            responsibilities=responsibilities.copy(),
            log_density=log_density.copy(),
            counts=(responsibilities * bin_counts[:, None]).sum(axis=0)[
                :n_components
            ],
            log_likelihood=float(previous_ll),
            n_iter=iteration,
            converged=converged,
            num_voxels=int(cache.num_voxels),
            has_outlier=has_outlier,
            prior=prior,
        )
        result.bin_counts = bin_counts
        return result

    # ── outputs ──────────────────────────────────────────────────────────
    def label_bins(self, result: FitResult) -> np.ndarray:
        """Per-bin labels with abstention.

        ``-1`` marks a bin the model declines to assign: either the outlier
        component won it, or no component reached ``reject_margin``. A model
        built to track drift has to be able to say it does not recognise
        something, otherwise the first unmodelled phase is silently absorbed.
        """
        labels = result.hard_labels_per_bin()
        if self.reject_margin is not None:
            best = result.responsibilities[:, : result.n_components].max(axis=1)
            labels = np.where(best < float(self.reject_margin), -1, labels)
        return labels.astype(np.int32)
