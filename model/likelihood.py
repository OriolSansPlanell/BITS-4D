"""Fixed material classes and the match table that scores voxels against them.

The premise, and why it changes the design
──────────────────────────────────────────
Neutron and X-ray attenuation coefficients are **material constants**. The
map from a material to its position in the (neutron, X-ray) plane is fixed by
physics, not estimated from data. On the battery series this was developed
on (26 timepoints; not distributed with the code) the chemically inert
classes barely moved — Air 0.035σ, Aluminium 0.062σ — while the classes that
did move were the ones whose *composition* was changing, which is signal and
must not be tracked away. What can be reproduced from this repository is the
synthetic validation in ``validation/`` (see ``docs/validation.md``).

So letting every class mean float, as a fitted mixture does, is wrong by
construction. A class centroid that moves is a class absorbing material that
should have left it. **Voxels migrate between fixed classes; classes do not
chase voxels.**

In locked mode there is no parameter estimation at all. Each class is a fixed
locus with a fixed spread, taken from the voxels its ROI selects at T0 (or,
better, from tabulated cross-sections). What varies per timepoint is only
*which class each voxel belongs to*.

Three consequences worth stating, because they are what makes the mode
trustworthy:

* classes cannot merge or swap — they have no freedom to;
* class names are the ROI names in ROI order, because there is no matching
  step that could get them wrong;
* timepoints are fully independent, so results are order-independent,
  reproducible and trivially parallel.

Speed
─────
Each class density is evaluated once per histogram *bin*, not once per voxel:
262 144 evaluations instead of 38.5 million, and only once for the whole
series while the loci stay locked. Scoring a voxel is then a table lookup.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DIMENSION = 2
_LOG_2PI = math.log(2.0 * math.pi)

#: Label reserved for voxels that match nothing, or that are outside the mask.
UNCLASSIFIED = 0


@dataclass
class MaterialClass:
    """A fixed locus in the (neutron, X-ray) plane.

    Attributes
    ----------
    name
        The user's name for it. Carried through to every output unchanged.
    mu
        ``(neutron, xray)`` centre.
    sigma
        2×2 covariance.
    source
        ``'roi'`` when derived from the voxels a polygon selects at T0,
        ``'physics'`` when computed from tabulated cross-sections.
    inert
        Declared not to change during the experiment. Used as the null
        control in the health check: if an inert class's volume moves, the
        segmentation is wrong, not the sample.
    weight
        Relative prevalence, from the T0 fraction. Only breaks ties between
        otherwise equally good matches.
    """

    name: str
    mu: np.ndarray
    sigma: np.ndarray
    source: str = "roi"
    inert: bool = False
    weight: float = 1.0
    voxels_t0: int = 0
    #: Timepoint the class was defined at (its ROI's timepoint). A phase
    #: that only appears later is defined where it exists.
    defined_at: Optional[int] = None
    #: Optional shape beyond one Gaussian: ``[(weight, mu, sigma), ...]``
    #: whose weights sum to 1. ``mu``/``sigma`` above stay the overall
    #: moments, used wherever one centre is needed (drift, reports).
    components: Optional[List[Tuple[float, np.ndarray, np.ndarray]]] = None

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=np.float64).reshape(DIMENSION)
        self.sigma = np.asarray(self.sigma, dtype=np.float64).reshape(
            DIMENSION, DIMENSION
        )
        if self.components is not None:
            self.components = [
                (float(w), np.asarray(m, dtype=np.float64).reshape(DIMENSION),
                 _regularise(np.asarray(c, dtype=np.float64)))
                for w, m, c in self.components
            ]

    @property
    def n_components(self) -> int:
        return len(self.components) if self.components else 1

    @property
    def elongation(self) -> float:
        """Ratio of the long axis to the short one; 1 is round."""
        eigenvalues = np.linalg.eigvalsh(self.sigma)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if eigenvalues[0] <= 0:
            return float("inf") if eigenvalues[1] > 0 else 1.0
        return float(math.sqrt(eigenvalues[1] / eigenvalues[0]))

    def describe(self) -> str:
        spread = np.sqrt(np.maximum(np.diag(self.sigma), 0.0))
        return (
            f"{self.name}: centre ({self.mu[0]:.6g}, {self.mu[1]:.6g}), "
            f"spread ({spread[0]:.4g}, {spread[1]:.4g}), from {self.source}"
            + (f" at T{self.defined_at}" if self.defined_at is not None else "")
            + (f", {self.n_components} components" if self.components else "")
            + (", control material" if self.inert else "")
        )


def _regularise(covariance: np.ndarray, floor_fraction: float = 1e-6):
    """Keep a covariance usable without distorting its shape."""
    matrix = 0.5 * (
        np.asarray(covariance, dtype=np.float64)
        + np.asarray(covariance, dtype=np.float64).T
    )
    scale = max(float(np.trace(matrix)) / DIMENSION, 0.0)
    floor = max(floor_fraction * scale, 1e-12)
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if determinant <= 0 or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        matrix = matrix + np.eye(DIMENSION) * max(floor, 1e-9)
    return matrix


class ClassLibrary:
    """The set of material classes a segmentation is built from.

    Order is fixed and meaningful: class *k* is label ``k + 1`` in every
    output, and label 0 is always Unclassified. Nothing reorders it, so an
    exported label volume means the same thing as the panel the user drew.
    """

    def __init__(self, classes: Sequence[MaterialClass]) -> None:
        self.classes: List[MaterialClass] = list(classes)
        names = [material.name for material in self.classes]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate class names: {names}")
        #: ``[(name, reason)]`` for every requested class that could not be
        #: defined. Never silent: also raised as a ClassDefinitionWarning.
        self.dropped: List[Tuple[str, str]] = []
        #: ``[(name, note)]`` for classes that were defined but on thin
        #: evidence (very few voxels).
        self.notes: List[Tuple[str, str]] = []

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self):
        return iter(self.classes)

    def __getitem__(self, index):
        return self.classes[index]

    @property
    def names(self) -> List[str]:
        return [material.name for material in self.classes]

    @property
    def inert_names(self) -> List[str]:
        return [m.name for m in self.classes if m.inert]

    def label_values(self) -> Dict[str, int]:
        """``{name: label}``. Label 0 is Unclassified and is never a class."""
        return {m.name: index for index, m in enumerate(self.classes, start=1)}

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def mark_inert(self, names: Sequence[str]) -> None:
        wanted = set(names)
        for material in self.classes:
            material.inert = material.name in wanted

    def describe(self) -> str:
        return "\n".join(material.describe() for material in self.classes)

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def from_masks(
        cls,
        neutron_volume,
        xray_volume,
        class_masks: Dict[str, np.ndarray],
        valid_mask=None,
        inert: Sequence[str] = (),
        max_samples: int = 5_000_000,
        max_components: int = 1,
        min_voxels: int = 30,
        timepoint: Optional[int] = None,
    ) -> "ClassLibrary":
        """Moments of the voxels each ROI selects at the reference timepoint.

        A class whose mask selects no usable voxel cannot be defined here; it
        is recorded in ``library.dropped`` with the reason and reported as a
        :class:`ClassDefinitionWarning` — never dropped silently. A phase
        that only exists later should be defined where it exists: see
        :meth:`from_sources`. A class with fewer than *min_voxels* voxels is
        kept but noted in ``library.notes``, because its spread is then a
        guess.

        With ``max_components > 1`` each class may be a small Gaussian
        mixture (the number chosen by BIC), for clouds that are curved or
        skewed by beam hardening, cupping or scatter.

        This is the definition available today. The better one — loci
        computed from tabulated cross-sections and calibrated by two known
        reference materials in the scan — makes the ROIs a *validation* of
        the physics rather than the definition. :meth:`from_physics` is the
        entry point for it.
        """
        sources = {
            name: (neutron_volume, xray_volume, mask, valid_mask, timepoint)
            for name, mask in class_masks.items()
        }
        return cls.from_sources(
            sources, inert=inert, max_samples=max_samples,
            max_components=max_components, min_voxels=min_voxels,
        )

    @classmethod
    def from_sources(
        cls,
        sources: Dict[str, tuple],
        inert: Sequence[str] = (),
        max_samples: int = 5_000_000,
        max_components: int = 1,
        min_voxels: int = 30,
    ) -> "ClassLibrary":
        """Each class from its own volume pair — its own timepoint.

        *sources* maps a class name to ``(neutron, xray, mask, valid,
        timepoint)`` (``valid`` and ``timepoint`` may be None). This is how a
        phase that is absent at the reference timepoint — a reaction product,
        a deposit — is defined from a timepoint where it exists; the class is
        then locked like any other and tracked through the whole series,
        including the timepoints before it appeared, where it should come out
        empty.
        """
        inert_names = set(inert)
        rng = np.random.default_rng(0)
        collected = []
        dropped: List[Tuple[str, str]] = []
        for name, source in sources.items():
            neutron, xray, mask, valid, timepoint = (tuple(source) + (None,) * 5)[:5]
            neutron = np.asarray(neutron)
            xray = np.asarray(xray)
            selected = np.asarray(mask, dtype=bool)
            if valid is not None:
                selected = selected & np.asarray(valid, dtype=bool)
            count = int(np.count_nonzero(selected))
            where = f" at timepoint {timepoint}" if timepoint is not None else ""
            if count == 0:
                dropped.append((name, (
                    f"its region selects no measured voxel{where} — the "
                    f"phase may be absent or thinner than the region there; "
                    f"define it at a timepoint where it exists"
                )))
                continue
            values_n = np.asarray(neutron[selected], dtype=np.float64)
            values_x = np.asarray(xray[selected], dtype=np.float64)
            finite = np.isfinite(values_n) & np.isfinite(values_x)
            values_n, values_x = values_n[finite], values_x[finite]
            if values_n.size == 0:
                dropped.append((name, f"none of its voxels{where} has both "
                                      f"a neutron and an X-ray value"))
                continue
            if values_n.size > max_samples:
                chosen = rng.choice(values_n.size, max_samples, replace=False)
                values_n, values_x = values_n[chosen], values_x[chosen]
            collected.append((name, values_n, values_x, count, timepoint))

        if dropped:
            for name, reason in dropped:
                warnings.warn(
                    f"Class '{name}' could not be defined: {reason}.",
                    ClassDefinitionWarning, stacklevel=3,
                )
        if not collected:
            raise ValueError(
                "None of the class masks selected any valid voxels"
                + ("".join(f"; '{n}': {r}" for n, r in dropped))
            )

        total = sum(item[3] for item in collected) or 1
        materials = []
        notes: List[Tuple[str, str]] = []
        for name, values_n, values_x, count, timepoint in collected:
            points = np.stack([values_n, values_x])
            mean = points.mean(axis=1)
            covariance = (
                np.cov(points, bias=False) if values_n.size > 1
                else np.eye(DIMENSION)
            )
            if count < min_voxels:
                notes.append((name, (
                    f"defined from only {count} voxel(s), so its spread is "
                    f"poorly known; enlarge its region or define it at a "
                    f"timepoint where there is more of it"
                )))
            components = None
            if max_components > 1 and values_n.size >= 50 * max_components:
                components = _fit_components(points.T, max_components, rng)
            materials.append(
                MaterialClass(
                    name=name,
                    mu=mean,
                    sigma=_regularise(np.atleast_2d(covariance)),
                    source="roi",
                    inert=name in inert_names,
                    weight=count / total,
                    voxels_t0=count,
                    defined_at=timepoint,
                    components=components,
                )
            )
        library = cls(materials)
        library.dropped = dropped
        library.notes = notes
        return library

    @classmethod
    def from_physics(
        cls,
        loci: Dict[str, Sequence[float]],
        spreads: Dict[str, Sequence[float]],
        inert: Sequence[str] = (),
    ) -> "ClassLibrary":
        """Classes placed by tabulated cross-sections rather than by ROIs.

        *loci* maps a material name to its expected ``(neutron, xray)``
        position **already calibrated to the scan's intensity units** — the
        usual calibration is a linear fit through two materials whose
        coefficients are known and which are present in the scan (air and
        aluminium are ideal). *spreads* gives the per-axis σ to expect,
        which is an instrument property (noise, partial volume), not a
        material one.

        With this path the ROIs stop being the definition of the classes and
        become an independent check on it.
        """
        inert_names = set(inert)
        materials = []
        for name, centre in loci.items():
            spread = np.asarray(spreads[name], dtype=np.float64).reshape(DIMENSION)
            materials.append(
                MaterialClass(
                    name=name,
                    mu=np.asarray(centre, dtype=np.float64),
                    sigma=_regularise(np.diag(spread ** 2)),
                    source="physics",
                    inert=name in inert_names,
                )
            )
        return cls(materials)


class ClassDefinitionWarning(UserWarning):
    """A requested class could not be defined from its region."""


def _fit_components(points, max_components, rng, max_fit=50_000):
    """A Gaussian mixture of 1..*max_components* for one class, by BIC.

    Fitted on standardised values (so the covariance floor means the same
    thing whatever the intensity units) and mapped back. Returns None when
    one component is best — the plain Gaussian is then exact.
    """
    from sklearn.mixture import GaussianMixture

    points = np.asarray(points, dtype=np.float64)
    if len(points) > max_fit:
        points = points[rng.choice(len(points), max_fit, replace=False)]
    centre = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale <= 0] = 1.0
    standard = (points - centre) / scale
    best, best_bic = None, np.inf
    for n in range(1, max_components + 1):
        model = GaussianMixture(n_components=n, covariance_type="full",
                                random_state=0, n_init=2).fit(standard)
        bic = model.bic(standard)
        if bic < best_bic:
            best, best_bic = model, bic
    if best is None or best.n_components == 1:
        return None
    transform = np.diag(scale)
    return [
        (float(w), centre + scale * m, transform @ c @ transform)
        for w, m, c in zip(best.weights_, best.means_, best.covariances_)
    ]


def _gaussian_bin_scores(points, scatter, counts, mu, sigma):
    """Mean log-density of each bin's voxels under one Gaussian."""
    covariance = _regularise(sigma)
    a, b = covariance[0, 0], covariance[0, 1]
    c, d = covariance[1, 0], covariance[1, 1]
    determinant = a * d - b * c
    if determinant <= 0 or not np.isfinite(determinant):
        return np.full(points.shape[0], -np.inf)
    inverse = np.array([[d, -b], [-c, a]], dtype=np.float64) / determinant
    offset = points - mu
    quadratic = np.einsum("mi,ij,mj->m", offset, inverse, offset)
    # Mean log-density over the voxels in the bin rather than the density
    # at the bin's mean: the correction is the within-bin spread seen
    # through this class.
    within = np.maximum(np.einsum("ij,mji->m", inverse, scatter), 0.0) / counts
    return -0.5 * (DIMENSION * _LOG_2PI + math.log(determinant)
                   + quadratic + within)


@dataclass
class MatchTable:
    """Per-bin match scores of every class, plus the Unclassified floor.

    ``scores`` is ``[n_bins, K]`` in natural log units — higher is a better
    match. It is computed once and reused for every timepoint while the class
    loci are locked, which is what makes locked mode fast.
    """

    scores: np.ndarray
    names: List[str]
    unclassified_score: float
    bin_ids: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(self.scores.shape[1])

    def with_unclassified(self) -> np.ndarray:
        """``[n_bins, K + 1]`` with the Unclassified column last."""
        column = np.full(
            (self.scores.shape[0], 1), self.unclassified_score, dtype=np.float32
        )
        return np.concatenate([self.scores.astype(np.float32), column], axis=1)

    def best_class_per_bin(self) -> np.ndarray:
        """Index of the best-matching class per bin; ``-1`` for Unclassified."""
        stacked = self.with_unclassified()
        best = np.argmax(stacked, axis=1)
        return np.where(best >= self.n_classes, -1, best).astype(np.int32)


def match_table(library: ClassLibrary, cache,
                unclassified_floor: float = 1e-4) -> MatchTable:
    """Score every occupied histogram bin against every class.

    *unclassified_floor* is the weight of a uniform density spread over the
    histogram's support. A voxel whose best class match is worse than that
    uniform baseline is genuinely unexplained, and lands in Unclassified
    rather than in whichever class happens to be nearest — which is how a
    large region of padding ends up inside a real material otherwise.
    """
    points = cache.means                      # true within-bin means
    scatter = cache.within_bin_scatter
    counts = cache.counts
    scores = np.empty((cache.num_bins, len(library)), dtype=np.float64)

    for index, material in enumerate(library):
        if material.components:
            # log Σ_j w_j N_j, evaluated stably
            parts = np.stack([
                math.log(max(weight, 1e-12))
                + _gaussian_bin_scores(points, scatter, counts, mu, sigma)
                for weight, mu, sigma in material.components
            ])
            peak = np.max(parts, axis=0)
            safe = np.where(np.isfinite(peak), peak, 0.0)
            density = safe + np.log(np.sum(np.exp(parts - safe), axis=0))
            density = np.where(np.isfinite(peak), density, -np.inf)
        else:
            density = _gaussian_bin_scores(points, scatter, counts,
                                           material.mu, material.sigma)
        scores[:, index] = density + math.log(max(material.weight, 1e-12))

    area = cache.support_area()
    uniform = -math.log(area) if area > 0 else 0.0
    return MatchTable(
        scores=scores,
        names=library.names,
        unclassified_score=uniform + math.log(max(unclassified_floor, 1e-30)),
        bin_ids=cache.bin_ids,
    )
