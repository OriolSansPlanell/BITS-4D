"""Honest accuracy numbers, and error bars on the curves.

Two quantities in the existing pipeline do not mean what they appear to.

**Training accuracy.** Voxels are spatially autocorrelated: a voxel's
neighbour is very nearly a duplicate of it. Scoring on the training sample —
or on a random k-fold split of it, or on an out-of-bag sample drawn from the
same voxels — asks the model to recognise data it has effectively already
seen, which is why such numbers sit above 95 % regardless of whether the
segmentation is any good. :func:`block_cross_validation` holds out
*contiguous 3-D blocks* instead, so the held-out voxels have no near-copies
in training. It returns a substantially lower and honest number.

**Volume curves without error bars.** A class volume per timepoint is a point
estimate that depends on the training subsample and the forest's random
state. :func:`bootstrap_bands` resamples both and reports a band, which is
what turns "these two methods differ by 6 %" into a statement that can be
true or false rather than merely observed.

Plus one diagnostic worth running before anything else:
:func:`anchoring_index` reports how much of a classifier's decision function
is carried by features that are *identical at every timepoint*. A model whose
predictions rest on frozen T0 geometry will look excellent on the timepoint
it was trained on and degrade quietly everywhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── block splitting ──────────────────────────────────────────────────────────

def block_ids_for_volume(shape: Sequence[int],
                         grid: Sequence[int] = (5, 5, 5)) -> np.ndarray:
    """Partition a volume into a grid of contiguous blocks.

    Returns an int32 volume whose value is the block index of each voxel.
    """
    shape = tuple(int(size) for size in shape)
    grid = tuple(int(count) for count in grid)
    if len(shape) != 3 or len(grid) != 3:
        raise ValueError("Both shape and grid must be 3-D")

    axes = []
    for size, divisions in zip(shape, grid):
        divisions = max(min(divisions, size), 1)
        edges = np.linspace(0, size, divisions + 1).astype(np.int64)
        index = np.zeros(size, dtype=np.int64)
        for block in range(divisions):
            index[edges[block]:edges[block + 1]] = block
        axes.append((index, divisions))

    z_index, _ = axes[0]
    y_index, y_divisions = axes[1]
    x_index, x_divisions = axes[2]
    ids = (
        z_index[:, None, None] * (y_divisions * x_divisions)
        + y_index[None, :, None] * x_divisions
        + x_index[None, None, :]
    )
    return np.ascontiguousarray(ids.astype(np.int32))


# ── scoring ──────────────────────────────────────────────────────────────────

def confusion_matrix(truth, prediction, n_classes: Optional[int] = None) -> np.ndarray:
    truth = np.asarray(truth).reshape(-1).astype(np.int64)
    prediction = np.asarray(prediction).reshape(-1).astype(np.int64)
    if truth.size != prediction.size:
        raise ValueError("truth and prediction must be the same length")
    if n_classes is None:
        n_classes = int(max(truth.max(initial=0), prediction.max(initial=0)) + 1)
    counts = np.bincount(
        truth * n_classes + prediction, minlength=n_classes * n_classes
    )
    return counts.reshape(n_classes, n_classes).astype(np.float64)


def per_class_iou(matrix: np.ndarray) -> np.ndarray:
    """Intersection over union per class; NaN for a class that never occurs."""
    intersection = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, intersection / union, np.nan)
    return iou


def cohen_kappa(matrix: np.ndarray) -> float:
    """Agreement corrected for what chance alone would produce."""
    total = matrix.sum()
    if total <= 0:
        return float("nan")
    observed = np.trace(matrix) / total
    expected = float(
        (matrix.sum(axis=0) / total) @ (matrix.sum(axis=1) / total)
    )
    if expected >= 1.0:
        return float("nan")
    return float((observed - expected) / (1.0 - expected))


@dataclass
class ValidationResult:
    accuracy: float
    kappa: float
    iou: Dict[int, float] = field(default_factory=dict)
    mean_iou: float = float("nan")
    n_folds: int = 0
    fold_kappa: List[float] = field(default_factory=list)
    confusion: Optional[np.ndarray] = None

    def describe(self) -> str:
        return (
            f"block CV over {self.n_folds} folds: "
            f"accuracy {100 * self.accuracy:.2f}%, kappa {self.kappa:.3f}, "
            f"mean IoU {self.mean_iou:.3f}"
        )


def block_cross_validation(
    features,
    labels,
    block_ids,
    estimator_factory: Callable[[], object],
    n_folds: Optional[int] = None,
    random_state: int = 0,
    max_train_samples: Optional[int] = None,
    progress_callback=None,
) -> ValidationResult:
    """Leave-one-group-out cross-validation over contiguous spatial blocks.

    *features*, *labels* and *block_ids* are parallel 1-D-indexed arrays over
    the sampled voxels. *estimator_factory* returns a fresh scikit-learn-style
    estimator (``fit`` / ``predict``) for each fold.

    Held-out blocks share no boundary-adjacent voxels with the training set
    except along their faces, so a model cannot score well by memorising its
    neighbours — which is exactly what an in-sample or random-fold score
    lets it do.
    """
    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    if not (features.shape[0] == labels.size == blocks.size):
        raise ValueError("features, labels and block_ids must be parallel")

    unique_blocks = np.unique(blocks)
    if unique_blocks.size < 2:
        raise ValueError("Block cross-validation needs at least two blocks")
    rng = np.random.default_rng(random_state)
    if n_folds is not None and n_folds < unique_blocks.size:
        unique_blocks = rng.choice(unique_blocks, n_folds, replace=False)

    n_classes = int(labels.max() + 1)
    total_confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    fold_kappa: List[float] = []

    for position, held_out in enumerate(unique_blocks):
        test = blocks == held_out
        train = ~test
        if not test.any() or not train.any():
            continue
        if np.unique(labels[train]).size < 2:
            continue

        train_index = np.flatnonzero(train)
        if max_train_samples is not None and train_index.size > max_train_samples:
            train_index = rng.choice(train_index, max_train_samples, replace=False)

        estimator = estimator_factory()
        estimator.fit(features[train_index], labels[train_index])
        predicted = estimator.predict(features[test])

        matrix = confusion_matrix(labels[test], predicted, n_classes)
        total_confusion += matrix
        fold_kappa.append(cohen_kappa(matrix))
        if progress_callback:
            progress_callback(
                int(100 * (position + 1) / len(unique_blocks)),
                f"Block {position + 1}/{len(unique_blocks)}",
            )

    iou = per_class_iou(total_confusion)
    total = total_confusion.sum()
    return ValidationResult(
        accuracy=float(np.trace(total_confusion) / total) if total else float("nan"),
        kappa=cohen_kappa(total_confusion),
        iou={index: float(value) for index, value in enumerate(iou)},
        mean_iou=float(np.nanmean(iou)) if iou.size else float("nan"),
        n_folds=len(fold_kappa),
        fold_kappa=fold_kappa,
        confusion=total_confusion,
    )


# ── anchoring ────────────────────────────────────────────────────────────────

def anchoring_index(importances, feature_names: Sequence[str],
                    anchored_features: Sequence[str]) -> float:
    """Share of a model's decision function carried by time-invariant features.

    Coordinates normalised at T0 are identical at every timepoint, so any
    weight on them is memory of where things *were*, not evidence about what
    they are. ``A > 0.20`` means a fifth of the segmentation is T0 geometry.

    Note on what this number is: impurity-based ``feature_importances_`` is
    biased toward continuous, high-cardinality features, and normalised
    coordinates are precisely that. Read the value from those importances as
    an **upper bound**, and confirm with
    :func:`permutation_anchoring_index` on held-out blocks before concluding
    a model is or is not anchored.
    """
    importances = np.asarray(importances, dtype=np.float64)
    if importances.size != len(feature_names):
        raise ValueError("importances and feature_names must be the same length")
    total = float(importances.sum())
    if total <= 0:
        return float("nan")
    anchored = set(anchored_features)
    share = sum(
        float(value) for value, name in zip(importances, feature_names)
        if name in anchored
    )
    return share / total


def permutation_importance(
    estimator,
    features,
    labels,
    feature_names: Sequence[str],
    n_repeats: int = 5,
    random_state: int = 0,
    scorer: Optional[Callable] = None,
) -> Dict[str, float]:
    """Drop in score when each feature is shuffled, on held-out data.

    Unlike impurity importance this is measured on data the model did not
    train on, and it is not biased by feature cardinality. It is also the
    only importance that answers the per-class question: at under 1 % of the
    volume, a minority phase is invisible in a global impurity ranking.
    """
    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(random_state)

    def default_scorer(estimator_, x, y):
        return float(np.mean(estimator_.predict(x) == y))

    score = scorer or default_scorer
    baseline = score(estimator, features, labels)

    result: Dict[str, float] = {}
    for index, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            shuffled = features.copy()
            shuffled[:, index] = shuffled[rng.permutation(shuffled.shape[0]), index]
            drops.append(baseline - score(estimator, shuffled, labels))
        result[name] = float(np.mean(drops))
    return result


def permutation_anchoring_index(
    estimator, features, labels, feature_names: Sequence[str],
    anchored_features: Sequence[str], **kwargs
) -> float:
    """:func:`anchoring_index` computed from permutation importance."""
    importance = permutation_importance(
        estimator, features, labels, feature_names, **kwargs
    )
    positive = {name: max(value, 0.0) for name, value in importance.items()}
    total = sum(positive.values())
    if total <= 0:
        return float("nan")
    anchored = set(anchored_features)
    return sum(
        value for name, value in positive.items() if name in anchored
    ) / total


# ── temporal generalisation ──────────────────────────────────────────────────

def temporal_generalisation_matrix(
    train_timepoints: Sequence[int],
    predict_timepoints: Sequence[int],
    fit_at: Callable[[int], object],
    score_at: Callable[[object, int], float],
    progress_callback=None,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """``M[t_train, t_predict]`` of a per-class agreement score.

    A flat row means the model generalises across time. A row that decays
    with ``|t_predict − t_train|`` means it is extrapolating, and the decay
    slope is a direct measurement of how fast the model goes stale — its
    half-life is the interval at which the segmentation should be re-anchored.
    """
    train_timepoints = list(train_timepoints)
    predict_timepoints = list(predict_timepoints)
    matrix = np.full(
        (len(train_timepoints), len(predict_timepoints)), np.nan, dtype=np.float64
    )
    for row, train in enumerate(train_timepoints):
        model = fit_at(train)
        for column, predict in enumerate(predict_timepoints):
            matrix[row, column] = float(score_at(model, predict))
        if progress_callback:
            progress_callback(
                int(100 * (row + 1) / max(len(train_timepoints), 1)),
                f"Trained at T{train}",
            )
    return matrix, train_timepoints, predict_timepoints


def staleness_half_life(matrix: np.ndarray, train_timepoints: Sequence[int],
                        predict_timepoints: Sequence[int]) -> Dict[int, float]:
    """Timepoints until a model's score falls to half its on-diagonal value."""
    half_lives: Dict[int, float] = {}
    predict = np.asarray(predict_timepoints, dtype=np.float64)
    for row, train in enumerate(train_timepoints):
        scores = matrix[row]
        if train not in predict_timepoints:
            continue
        diagonal = scores[predict_timepoints.index(train)]
        if not np.isfinite(diagonal) or diagonal <= 0:
            continue
        distance = np.abs(predict - train)
        order = np.argsort(distance)
        target = 0.5 * diagonal
        crossing = float("inf")
        for index in order:
            if np.isfinite(scores[index]) and scores[index] <= target:
                crossing = float(distance[index])
                break
        half_lives[int(train)] = crossing
    return half_lives


# ── uncertainty ──────────────────────────────────────────────────────────────

def bootstrap_bands(
    estimate: Callable[[int], Dict[str, float]],
    n_resamples: int = 20,
    quantiles: Tuple[float, float] = (0.16, 0.84),
    progress_callback=None,
) -> Dict[str, Dict[str, float]]:
    """Median and band for a quantity computed under *n_resamples* seeds.

    *estimate* takes a seed and returns ``{name: value}`` — for a volume
    curve, one entry per class. Returns
    ``{name: {'median', 'low', 'high', 'width'}}``.

    This is what settles a comparison between two methods: if the difference
    between them lies inside the band, they never disagreed in the first
    place.
    """
    samples: Dict[str, List[float]] = {}
    for index in range(n_resamples):
        for name, value in estimate(index).items():
            samples.setdefault(name, []).append(float(value))
        if progress_callback:
            progress_callback(
                int(100 * (index + 1) / max(n_resamples, 1)),
                f"Resample {index + 1}/{n_resamples}",
            )

    bands: Dict[str, Dict[str, float]] = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        low, high = np.quantile(array, quantiles)
        bands[name] = {
            "median": float(np.median(array)),
            "low": float(low),
            "high": float(high),
            "width": float(high - low),
            "n": int(array.size),
        }
    return bands


def difference_within_band(difference: float, band: Dict[str, float]) -> bool:
    """Is an observed difference smaller than the uncertainty on it?"""
    return abs(float(difference)) <= abs(band["high"] - band["low"])
