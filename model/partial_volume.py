"""Mixing lines modelled as fractions instead of forced into hard classes.

Some "classes" in a bimodal histogram are not compact clusters at all: they
are elongated ridges running between two pure phases, made of voxels that
contain part of each. A rim around a steel pin, or the boundary shell of a
lithium deposit, is a **partial-volume** effect of the reconstruction, and
its elongation is the giveaway — an anisotropy of 2.4 against a 1.1–1.2
baseline for genuine phases.

Assigning such a ridge a hard label is ill-posed: the answer for a voxel that
is 40 % lithium is not "lithium" or "indium", it is 0.4. A voxel that is a
fraction α of phase *a* has

    μ(α) = α μ_a + (1−α) μ_b
    Σ(α) = α² Σ_a + (1−α)² Σ_b + σ² I

so discretising α on J bins turns the ridge into a small finite mixture that
EM handles with no special cases, and the reported ``E[α]`` per voxel is a
continuous, physically meaningful quantity rather than a category.

This also explains, rather than merely tolerating, most of the discrepancy
between a fixed histogram polygon and a trained classifier: they disagree
about a boundary shell whose true membership is fractional, so neither of
them was right and the disagreement was never evidence about the interior.

Fitted after the parents, not inside them
─────────────────────────────────────────
The ladder is built from the *fitted* parent components rather than being
optimised jointly. That keeps the parents' EM exactly as derived, keeps the
ladder consistent with wherever drift has moved the parents to, and means a
mispaired mixel degrades a fractional map instead of destabilising the whole
mixture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class MixelComponent:
    """A phase pair whose boundary voxels are modelled as fractions."""

    name: str
    phase_a: str
    phase_b: str
    n_alpha: int = 10
    noise: Optional[float] = None

    def alphas(self) -> np.ndarray:
        """Fraction of *phase_a*, at the centre of each of the J bins."""
        return (np.arange(self.n_alpha, dtype=np.float64) + 0.5) / self.n_alpha


@dataclass
class MixelLadder:
    """The discretised mixing line between two fitted components."""

    component: MixelComponent
    alphas: np.ndarray            # [J]
    means: np.ndarray             # [J, 2]
    covariances: np.ndarray       # [J, 2, 2]
    mean_a: np.ndarray
    mean_b: np.ndarray

    @property
    def axis(self) -> np.ndarray:
        """Unit vector along the line joining the parent centroids."""
        direction = self.mean_a - self.mean_b
        norm = float(np.hypot(*direction))
        return direction / norm if norm > 0 else direction


def build_mixel_ladder(
    component: MixelComponent,
    mean_a, covariance_a,
    mean_b, covariance_b,
) -> MixelLadder:
    """Discretise the mixing line between two fitted phases."""
    mean_a = np.asarray(mean_a, dtype=np.float64)
    mean_b = np.asarray(mean_b, dtype=np.float64)
    covariance_a = np.asarray(covariance_a, dtype=np.float64)
    covariance_b = np.asarray(covariance_b, dtype=np.float64)

    alphas = component.alphas()
    means = alphas[:, None] * mean_a + (1.0 - alphas)[:, None] * mean_b
    covariances = (
        (alphas ** 2)[:, None, None] * covariance_a
        + ((1.0 - alphas) ** 2)[:, None, None] * covariance_b
    )
    if component.noise is None:
        # Default the extra reconstruction noise to the smaller parent's
        # scale, so the ladder is never sharper than the phases it joins.
        noise = 0.25 * min(
            float(np.trace(covariance_a)) / 2.0,
            float(np.trace(covariance_b)) / 2.0,
        )
    else:
        noise = float(component.noise) ** 2
    covariances = covariances + np.eye(2) * max(noise, 1e-12)

    return MixelLadder(
        component=component,
        alphas=alphas,
        means=means,
        covariances=covariances,
        mean_a=mean_a,
        mean_b=mean_b,
    )


def _log_gaussian(points: np.ndarray, mean: np.ndarray,
                  covariance: np.ndarray) -> np.ndarray:
    determinant = float(np.linalg.det(covariance))
    if determinant <= 0:
        return np.full(points.shape[0], -np.inf)
    inverse = np.linalg.inv(covariance)
    offset = points - mean
    quadratic = np.einsum("mi,ij,mj->m", offset, inverse, offset)
    return -0.5 * (2 * _LOG_2PI + math.log(determinant) + quadratic)


def fraction_per_bin(ladder: MixelLadder, points) -> Tuple[np.ndarray, np.ndarray]:
    """``(E[alpha], total log-likelihood)`` for each point on the mixing line.

    *points* is ``[M, 2]`` — the within-bin means of a histogram cache.
    ``E[alpha]`` is the posterior-weighted fraction of *phase_a*.
    """
    points = np.asarray(points, dtype=np.float64)
    log_density = np.stack(
        [
            _log_gaussian(points, ladder.means[j], ladder.covariances[j])
            for j in range(ladder.alphas.size)
        ],
        axis=1,
    )
    peak = log_density.max(axis=1, keepdims=True)
    finite = np.isfinite(peak[:, 0])
    peak[~finite, 0] = 0.0
    weights = np.exp(log_density - peak)
    total = weights.sum(axis=1)
    safe = np.maximum(total, 1e-300)
    expected = (weights @ ladder.alphas) / safe
    log_likelihood = peak[:, 0] + np.log(safe) - math.log(ladder.alphas.size)
    return expected, log_likelihood


def alignment_angle(covariance, mean_a, mean_b) -> float:
    """Angle in degrees between a component's long axis and its parents' line.

    The check that a declared mixel really is one: a genuine mixing line runs
    along the line joining the two pure phases. The source plan sets 15° as
    the acceptance threshold. Returns NaN when the component is isotropic
    enough that it has no meaningful long axis.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    if eigenvalues[0] <= 0:
        return float("nan")
    if eigenvalues[1] > 0 and math.sqrt(eigenvalues[0] / eigenvalues[1]) < 1.05:
        return float("nan")          # isotropic: no long axis to compare

    principal = eigenvectors[:, order[0]]
    direction = np.asarray(mean_a, dtype=np.float64) - np.asarray(
        mean_b, dtype=np.float64
    )
    norm = float(np.hypot(*direction))
    if norm <= 0:
        return float("nan")
    direction = direction / norm
    cosine = abs(float(np.dot(principal, direction)))
    return float(math.degrees(math.acos(min(max(cosine, 0.0), 1.0))))


def elongation(covariance) -> float:
    """Ratio of the long to the short axis; 1 is isotropic."""
    eigenvalues = np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if eigenvalues[0] <= 0:
        return float("inf") if eigenvalues[1] > 0 else 1.0
    return float(math.sqrt(eigenvalues[1] / eigenvalues[0]))


def detect_mixing_lines(
    result,
    elongation_threshold: float = 1.5,
    angle_tolerance: float = 15.0,
    max_pairs_per_class: int = 1,
) -> List[MixelComponent]:
    """Suggest which fitted components are mixing lines, and between what.

    A component qualifies when it is elongated beyond *elongation_threshold*
    **and** its long axis points along the line joining two other components
    to within *angle_tolerance*. Requiring both is what separates a genuine
    partial-volume ridge from a phase that simply happens to be anisotropic;
    the pair that gives the best alignment is the proposed parentage.

    The source plan declares these pairs by hand from the ROI names. Deriving
    them from the fit instead means the geometry has to agree before a class
    is reinterpreted, which is the same check the plan uses to *verify* a
    hand-declared pair — so doing it first costs nothing and catches a
    mis-declared pair before it produces a fractional map.
    """
    suggestions: List[MixelComponent] = []
    n_components = result.n_components

    for index in range(n_components):
        if elongation(result.covariances[index]) < elongation_threshold:
            continue
        candidates = []
        for a in range(n_components):
            for b in range(a + 1, n_components):
                if index in (a, b):
                    continue
                angle = alignment_angle(
                    result.covariances[index], result.means[a], result.means[b]
                )
                if not np.isfinite(angle) or angle > angle_tolerance:
                    continue
                # The mixel must sit between its parents, not beyond them
                span = result.means[a] - result.means[b]
                length = float(np.dot(span, span))
                if length <= 0:
                    continue
                position = float(
                    np.dot(result.means[index] - result.means[b], span)
                ) / length
                if not 0.0 <= position <= 1.0:
                    continue
                candidates.append((angle, a, b))

        candidates.sort()
        for angle, a, b in candidates[:max_pairs_per_class]:
            suggestions.append(
                MixelComponent(
                    name=result.names[index],
                    phase_a=result.names[a],
                    phase_b=result.names[b],
                )
            )
    return suggestions


def fractional_maps(
    result,
    cache,
    mixels: Sequence[MixelComponent],
) -> Dict[str, np.ndarray]:
    """Per-bin ``E[alpha]`` for every declared mixing line.

    Returns ``{mixel name: [M] expected fraction of phase_a}``. Use
    :meth:`model.histogram_cache.HistogramCache.expand_to_voxels` to turn one
    into a volume.
    """
    index_of = {name: index for index, name in enumerate(result.names)}
    points = cache.means
    maps: Dict[str, np.ndarray] = {}

    for mixel in mixels:
        if mixel.phase_a not in index_of or mixel.phase_b not in index_of:
            continue
        a = index_of[mixel.phase_a]
        b = index_of[mixel.phase_b]
        ladder = build_mixel_ladder(
            mixel,
            result.means[a], result.covariances[a],
            result.means[b], result.covariances[b],
        )
        expected, _ = fraction_per_bin(ladder, points)
        maps[mixel.name] = expected
    return maps


def verify_mixels(result, mixels: Sequence[MixelComponent],
                  angle_tolerance: float = 15.0) -> Dict[str, dict]:
    """Check each declared pair before its fractional map is trusted."""
    index_of = {name: index for index, name in enumerate(result.names)}
    report: Dict[str, dict] = {}
    for mixel in mixels:
        entry = {
            "phase_a": mixel.phase_a,
            "phase_b": mixel.phase_b,
            "angle_deg": float("nan"),
            "elongation": float("nan"),
            "accepted": False,
            "reason": "",
        }
        if mixel.name not in index_of:
            entry["reason"] = "component not in the fit"
        elif mixel.phase_a not in index_of or mixel.phase_b not in index_of:
            entry["reason"] = "declared parent is not a fitted component"
        else:
            index = index_of[mixel.name]
            entry["elongation"] = elongation(result.covariances[index])
            entry["angle_deg"] = alignment_angle(
                result.covariances[index],
                result.means[index_of[mixel.phase_a]],
                result.means[index_of[mixel.phase_b]],
            )
            if not np.isfinite(entry["angle_deg"]):
                entry["reason"] = "component is isotropic; no mixing axis"
            elif entry["angle_deg"] > angle_tolerance:
                entry["reason"] = (
                    f"long axis is {entry['angle_deg']:.1f}° off the line "
                    f"joining {mixel.phase_a} and {mixel.phase_b}"
                )
            else:
                entry["accepted"] = True
                entry["reason"] = "aligned with its parents"
        report[mixel.name] = entry
    return report
