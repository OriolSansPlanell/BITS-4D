"""Scoring a label series against the truth."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    """2|A∩B| / (|A| + |B|); 1.0 when both are empty (nothing to find)."""
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    total = int(prediction.sum()) + int(truth.sum())
    if total == 0:
        return 1.0
    return 2.0 * int(np.count_nonzero(prediction & truth)) / total


def oracle_match(prediction: np.ndarray, truth: np.ndarray,
                 n_classes: int) -> np.ndarray:
    """Relabel an unsupervised result to best match the truth.

    Hungarian assignment on the overlap matrix — the most favourable
    naming the method could possibly have. Clusters left unmatched become 0.
    This makes the unsupervised baselines an *upper bound* on what they
    would score in practice, where the matching has to be done by hand.
    """
    prediction = np.asarray(prediction)
    predicted_ids = [v for v in np.unique(prediction) if v >= 0]
    overlap = np.zeros((len(predicted_ids), n_classes))
    for row, value in enumerate(predicted_ids):
        selected = prediction == value
        for k in range(n_classes):
            overlap[row, k] = np.count_nonzero(selected & (truth == k + 1))
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {predicted_ids[r]: c + 1 for r, c in zip(rows, cols)}
    result = np.zeros(prediction.shape, dtype=np.int16)
    for value, label in mapping.items():
        result[prediction == value] = label
    return result


def score_series(labels: np.ndarray, truth: np.ndarray,
                 classes: Sequence[str],
                 region: np.ndarray = None) -> Dict[str, Dict[str, list]]:
    """Per class: Dice and volume-fraction error at every timepoint.

    Timepoints where a class is absent in both prediction and truth score
    Dice 1 (correctly empty); absent in the truth but predicted scores 0.
    *region* (``(T, Z, Y, X)`` bool) restricts scoring to the voxels that
    can be classified at all — e.g. those with both measurements after an
    offset was corrected.
    """
    result: Dict[str, Dict[str, list]] = {}
    for k, name in enumerate(classes, start=1):
        dices, errors = [], []
        for t in range(truth.shape[0]):
            keep = np.ones(truth[t].shape, bool) if region is None else region[t]
            predicted = (labels[t] == k)[keep]
            actual = (truth[t] == k)[keep]
            dices.append(dice(predicted, actual))
            errors.append(float(predicted.mean() - actual.mean()))
        result[name] = {"dice": dices, "volume_error": errors}
    return result


def summarise(scores: Dict[str, Dict[str, list]]) -> Dict[str, float]:
    """Mean Dice per class over the series, plus the overall mean."""
    summary = {name: float(np.mean(entry["dice"]))
               for name, entry in scores.items()}
    summary["mean"] = float(np.mean(list(summary.values())))
    return summary
