"""Spatial regularisation whose costs are learned from the manual T0 labels.

A mixture over the histogram has no spatial term at all: it classifies each
voxel from its two intensities and nothing else, so its raw output is
speckled and, on its own, worse than the classifier it replaces. The
likelihood supplies the chemistry; this module supplies the coherence.

Learned, not assumed
────────────────────
The generic Potts model charges the same price for every class boundary. But
the T0 label volume already says which boundaries are common — lithium meets
the separator constantly and steel almost never — so the pairwise cost is
counted from face adjacencies in that volume instead of assumed uniform.
That is the second place the manual work enters the model, after the
mixture prior.

Two solvers, chosen by memory
─────────────────────────────
Mean-field keeps a full ``[Z, Y, X, K]`` responsibility array, which is
K × 4 bytes per voxel — around 1.4 GB for a 38-million-voxel volume with 9
classes, before temporaries. ICM keeps hard labels instead and costs about
9 bytes per voxel *regardless of K*, at the price of a coarser, greedier
optimum. :meth:`ROIDerivedMRF.refine` picks between them from a memory
budget unless told which to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# 6-connectivity: one entry per positive axis direction
_AXES = (0, 1, 2)
FORBIDDEN_COST = 1e3


@dataclass
class MRFDiagnostics:
    method: str
    sweeps: int
    changed_fraction: float
    energy: float


class UnaryScores:
    """Per-voxel class scores held as a per-bin table plus a lookup.

    The unary term is a function of the two intensities alone, so it takes
    only ``K`` numbers *per occupied histogram bin* — a few hundred kilobytes
    — rather than ``K`` numbers per voxel. Keeping it in that form is what
    lets ICM run on a volume whose dense score array would not fit in memory:
    it only ever needs one class column at a time, which this materialises on
    demand.
    """

    def __init__(self, table, row_index, fill: float = -1e30) -> None:
        self.table = np.asarray(table, dtype=np.float32)
        self.row_index = np.asarray(row_index)
        if self.table.ndim != 2:
            raise ValueError("table must be [n_bins, n_classes]")
        if self.row_index.ndim != 3:
            raise ValueError("row_index must be a 3-D volume")
        self.fill = float(fill)
        self._known = self.row_index >= 0
        self._safe = np.where(self._known, self.row_index, 0)

    @property
    def shape(self):
        return tuple(self.row_index.shape) + (self.table.shape[1],)

    @property
    def n_classes(self) -> int:
        return int(self.table.shape[1])

    @property
    def volume_shape(self):
        return tuple(self.row_index.shape)

    def column(self, k: int) -> np.ndarray:
        """The score volume for one class."""
        return np.where(
            self._known, self.table[self._safe, k], np.float32(self.fill)
        ).astype(np.float32, copy=False)

    def dense(self) -> np.ndarray:
        """The full ``[Z, Y, X, K]`` array. Only for mean-field."""
        return np.where(
            self._known[..., None], self.table[self._safe],
            np.float32(self.fill),
        ).astype(np.float32, copy=False)

    def valid_mask(self) -> np.ndarray:
        return self._known


class ROIDerivedMRF:
    """Markov random field over the class labels of a volume.

    Parameters
    ----------
    beta
        Strength of the spatial term. 0 disables it and reproduces the raw
        per-voxel mixture labels.
    n_sweeps
        Mean-field / ICM iterations.
    contrast_sigma
        Width of the contrast-sensitive edge weight. ``None`` estimates it
        from the volume, ``0`` disables edge weighting (plain Potts
        geometry). Real interfaces are cheap to cross, noise is not.
    max_cost
        Cap on a learned pairwise cost, so a boundary that never occurred at
        T0 stays expensive rather than infinite.
    """

    def __init__(
        self,
        beta: float = 1.0,
        n_sweeps: int = 5,
        contrast_sigma: Optional[float] = None,
        max_cost: float = 5.0,
        memory_budget_gb: float = 2.0,
    ) -> None:
        self.beta = float(beta)
        self.n_sweeps = int(n_sweeps)
        self.contrast_sigma = contrast_sigma
        self.max_cost = float(max_cost)
        self.memory_budget_gb = float(memory_budget_gb)
        self.pairwise: Optional[np.ndarray] = None
        self.class_names: Optional[Sequence[str]] = None

    # ── learning the pairwise cost ───────────────────────────────────────
    def fit_pairwise_from_labels(
        self,
        labels,
        n_classes: Optional[int] = None,
        valid_mask=None,
        class_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Count face adjacencies in a label volume and turn them into costs.

        ``V(k, l) = max(0, -log( p(k,l) / sqrt(p(k,k) p(l,l)) ))`` — zero when
        two classes touch as readily as each touches itself, large when they
        avoid one another. ``V(k, k)`` is zero by construction, so the field
        only ever charges for *changing* label.
        """
        label_volume = np.asarray(labels)
        if label_volume.ndim != 3:
            raise ValueError("The T0 label volume must be 3-D")
        classes = int(n_classes if n_classes is not None
                      else label_volume.max() + 1)
        if classes < 1:
            raise ValueError("Need at least one class")

        valid = (
            np.ones(label_volume.shape, dtype=bool) if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        )
        counts = np.zeros((classes, classes), dtype=np.float64)
        for axis in _AXES:
            if label_volume.shape[axis] < 2:
                continue
            first = np.take(label_volume, np.arange(label_volume.shape[axis] - 1), axis)
            second = np.take(label_volume, np.arange(1, label_volume.shape[axis]), axis)
            first_valid = np.take(valid, np.arange(valid.shape[axis] - 1), axis)
            second_valid = np.take(valid, np.arange(1, valid.shape[axis]), axis)
            usable = first_valid & second_valid
            a = first[usable].astype(np.int64)
            b = second[usable].astype(np.int64)
            inside = (a >= 0) & (a < classes) & (b >= 0) & (b < classes)
            a, b = a[inside], b[inside]
            flat = np.bincount(a * classes + b, minlength=classes * classes)
            counts += flat.reshape(classes, classes)
        counts = counts + counts.T          # unordered pairs

        total = counts.sum()
        if total <= 0:
            self.pairwise = potts_cost(classes)
            self.class_names = class_names
            return self.pairwise

        probability = counts / total
        self_adjacency = np.diag(probability).copy()
        cost = np.full((classes, classes), self.max_cost, dtype=np.float64)
        for k in range(classes):
            for l in range(classes):
                if k == l:
                    cost[k, l] = 0.0
                    continue
                reference = np.sqrt(self_adjacency[k] * self_adjacency[l])
                if reference <= 0:
                    # A class with no interior (isolated voxels) gives the
                    # ratio no meaning; fall back to a plain Potts penalty.
                    cost[k, l] = 1.0
                    continue
                joint = probability[k, l]
                if joint <= 0:
                    cost[k, l] = self.max_cost
                    continue
                cost[k, l] = min(
                    max(-np.log(joint / reference), 0.0), self.max_cost
                )
        self.pairwise = 0.5 * (cost + cost.T)
        self.class_names = class_names
        return self.pairwise

    def forbid(self, first: int, second: int,
               cost: float = FORBIDDEN_COST) -> None:
        """Make a class pair effectively unable to share a face.

        Used for partial-volume components, which are physically only able to
        border their two parent phases.
        """
        if self.pairwise is None:
            raise RuntimeError("Fit or set the pairwise costs first")
        self.pairwise[first, second] = cost
        self.pairwise[second, first] = cost

    def allow_only(self, component: int, neighbours: Sequence[int],
                   cost: float = FORBIDDEN_COST) -> None:
        """Restrict *component* to bordering itself and *neighbours*."""
        if self.pairwise is None:
            raise RuntimeError("Fit or set the pairwise costs first")
        allowed = set(int(n) for n in neighbours) | {int(component)}
        for other in range(self.pairwise.shape[0]):
            if other not in allowed:
                self.forbid(int(component), other, cost)

    # ── edge weights ─────────────────────────────────────────────────────
    def _edge_weights(self, neutron, xray) -> Optional[list]:
        """Per-axis contrast-sensitive weights, or None for plain geometry."""
        if self.contrast_sigma is not None and self.contrast_sigma == 0:
            return None
        if neutron is None or xray is None:
            return None

        first = np.asarray(neutron, dtype=np.float32)
        second = np.asarray(xray, dtype=np.float32)
        # Put both modalities on a comparable scale so neither dominates
        weights = []
        scales = []
        for volume in (first, second):
            finite = volume[np.isfinite(volume)]
            spread = float(np.std(finite)) if finite.size else 1.0
            scales.append(spread if spread > 0 else 1.0)

        differences = []
        for axis in _AXES:
            if first.shape[axis] < 2:
                differences.append(None)
                continue
            delta = np.zeros(first.shape, dtype=np.float32)
            slicer_low = [slice(None)] * 3
            slicer_high = [slice(None)] * 3
            slicer_low[axis] = slice(0, -1)
            slicer_high[axis] = slice(1, None)
            gap_first = (first[tuple(slicer_high)] - first[tuple(slicer_low)]) / scales[0]
            gap_second = (second[tuple(slicer_high)] - second[tuple(slicer_low)]) / scales[1]
            delta[tuple(slicer_low)] = np.sqrt(gap_first ** 2 + gap_second ** 2)
            differences.append(delta)

        finite_gaps = np.concatenate([
            d[np.isfinite(d)].ravel() for d in differences if d is not None
        ]) if any(d is not None for d in differences) else np.zeros(1, np.float32)
        sigma = self.contrast_sigma
        if sigma is None:
            # Mean gap is a scale-free, outlier-tolerant default
            sigma = float(np.mean(finite_gaps)) if finite_gaps.size else 1.0
        sigma = max(float(sigma), 1e-6)

        for delta in differences:
            if delta is None:
                weights.append(None)
            else:
                weights.append(
                    np.exp(-0.5 * np.square(delta / sigma, dtype=np.float32))
                )
        return weights

    # ── inference ────────────────────────────────────────────────────────
    def estimate_memory_gb(self, n_voxels: int, n_classes: int) -> float:
        """Mean-field peak memory, in GiB (two float32 [V, K] arrays)."""
        return 2.0 * n_voxels * n_classes * 4 / (1024 ** 3)

    def refine(
        self,
        log_unary,
        neutron=None,
        xray=None,
        valid_mask=None,
        method: str = "auto",
        initial_labels=None,
        cancel_check=None,
    ) -> Tuple[np.ndarray, MRFDiagnostics]:
        """Regularise per-voxel unary scores into a coherent labelling.

        *log_unary* is ``[Z, Y, X, K]``: the log of the mixture posterior (or
        any per-voxel score) before spatial smoothing. Returns
        ``(labels, diagnostics)`` where labels are ``int32`` and ``-1`` marks
        an invalid voxel.
        """
        if isinstance(log_unary, UnaryScores):
            scores = log_unary
        else:
            scores = np.asarray(log_unary, dtype=np.float32)
            if scores.ndim != 4:
                raise ValueError("log_unary must be [Z, Y, X, K]")
        shape = scores.shape[:3]
        n_classes = scores.shape[3]
        if self.pairwise is None:
            self.pairwise = potts_cost(n_classes)
        if self.pairwise.shape[0] != n_classes:
            raise ValueError(
                f"Pairwise cost is {self.pairwise.shape[0]}×"
                f"{self.pairwise.shape[0]} but there are {n_classes} classes"
            )

        valid = (
            None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        )
        if method == "auto":
            budget = self.estimate_memory_gb(int(np.prod(shape)), n_classes)
            method = "mean_field" if budget <= self.memory_budget_gb else "icm"

        weights = self._edge_weights(neutron, xray)
        if self.beta <= 0 or self.n_sweeps <= 0:
            labels = _argmax_scores(scores).astype(np.int32)
            if valid is not None:
                labels[~valid] = -1
            return labels, MRFDiagnostics(method="none", sweeps=0,
                                          changed_fraction=0.0, energy=float("nan"))

        if method == "mean_field":
            dense = scores.dense() if isinstance(scores, UnaryScores) else scores
            labels, changed = self._mean_field(dense, weights, valid, cancel_check)
        elif method == "icm":
            labels, changed = self._icm(
                scores, weights, valid, initial_labels, cancel_check
            )
        else:
            raise ValueError(f"Unknown method {method!r}")

        energy = self._energy(scores, labels, weights, valid)
        return labels, MRFDiagnostics(
            method=method, sweeps=self.n_sweeps,
            changed_fraction=changed, energy=energy,
        )

    # ── solvers ──────────────────────────────────────────────────────────
    def _neighbour_message(self, responsibilities, weights):
        """Σ over neighbours of the responsibility mass, per class."""
        total = np.zeros_like(responsibilities)
        for axis in _AXES:
            if responsibilities.shape[axis] < 2:
                continue
            low = [slice(None)] * 4
            high = [slice(None)] * 4
            low[axis] = slice(0, -1)
            high[axis] = slice(1, None)
            low_key, high_key = tuple(low), tuple(high)

            forward = responsibilities[high_key]
            backward = responsibilities[low_key]
            if weights is not None and weights[axis] is not None:
                weight = weights[axis][..., None]
                total[low_key] += forward * weight[low_key[:3]]
                total[high_key] += backward * weight[low_key[:3]]
            else:
                total[low_key] += forward
                total[high_key] += backward
        return total

    def _mean_field(self, scores, weights, valid, cancel_check):
        pairwise = self.pairwise.astype(np.float32)
        responsibilities = _softmax(scores)
        if valid is not None:
            responsibilities[~valid] = 0.0

        previous = np.argmax(responsibilities, axis=3)
        changed = 0.0
        for _ in range(self.n_sweeps):
            if cancel_check:
                cancel_check()
            neighbour = self._neighbour_message(responsibilities, weights)
            # message[..., k] = Σ_l neighbour[..., l] · V(k, l)
            message = neighbour @ pairwise.T
            responsibilities = _softmax(scores - self.beta * message)
            if valid is not None:
                responsibilities[~valid] = 0.0
            current = np.argmax(responsibilities, axis=3)
            changed = float(np.mean(current != previous))
            previous = current

        labels = previous.astype(np.int32)
        if valid is not None:
            labels[~valid] = -1
        return labels, changed

    def _icm(self, scores, weights, valid, initial_labels, cancel_check):
        pairwise = self.pairwise.astype(np.float32)
        n_classes = scores.shape[3]
        column = (
            scores.column if isinstance(scores, UnaryScores)
            else (lambda k: scores[..., k])
        )
        if initial_labels is None:
            labels = _argmax_scores(scores).astype(np.int32)
        else:
            labels = np.asarray(initial_labels, dtype=np.int32).copy()
        if valid is not None:
            labels[~valid] = 0

        changed = 0.0
        for _ in range(self.n_sweeps):
            if cancel_check:
                cancel_check()
            best_score = np.full(labels.shape, -np.inf, dtype=np.float32)
            best_label = np.zeros(labels.shape, dtype=np.int32)
            for k in range(n_classes):
                # One volume-sized temporary, whatever K is
                candidate = np.array(column(k), dtype=np.float32, copy=True)
                for axis in _AXES:
                    if labels.shape[axis] < 2:
                        continue
                    low = [slice(None)] * 3
                    high = [slice(None)] * 3
                    low[axis] = slice(0, -1)
                    high[axis] = slice(1, None)
                    low_key, high_key = tuple(low), tuple(high)
                    weight = (
                        None if weights is None or weights[axis] is None
                        else weights[axis][low_key]
                    )
                    forward = pairwise[k][labels[high_key]]
                    backward = pairwise[k][labels[low_key]]
                    if weight is not None:
                        forward = forward * weight
                        backward = backward * weight
                    candidate[low_key] -= self.beta * forward
                    candidate[high_key] -= self.beta * backward
                improved = candidate > best_score
                best_score = np.where(improved, candidate, best_score)
                best_label = np.where(improved, k, best_label)
            changed = float(np.mean(best_label != labels))
            labels = best_label.astype(np.int32)

        if valid is not None:
            labels[~valid] = -1
        return labels, changed

    def _energy(self, scores, labels, weights, valid) -> float:
        """Total energy of a labelling — lower is better."""
        usable = labels >= 0
        if not usable.any():
            return float("nan")
        safe = np.where(usable, labels, 0)
        if isinstance(scores, UnaryScores):
            chosen = np.zeros(labels.shape, dtype=np.float32)
            for k in range(scores.n_classes):
                np.copyto(chosen, scores.column(k), where=safe == k)
            unary = -chosen
        else:
            unary = -np.take_along_axis(scores, safe[..., None], axis=3)[..., 0]
        energy = float(unary[usable].sum())

        pairwise = self.pairwise
        for axis in _AXES:
            if labels.shape[axis] < 2:
                continue
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis] = slice(0, -1)
            high[axis] = slice(1, None)
            low_key, high_key = tuple(low), tuple(high)
            pair_ok = usable[low_key] & usable[high_key]
            if not pair_ok.any():
                continue
            cost = pairwise[safe[low_key][pair_ok], safe[high_key][pair_ok]]
            if weights is not None and weights[axis] is not None:
                cost = cost * weights[axis][low_key][pair_ok]
            energy += self.beta * float(cost.sum())
        return energy


def potts_cost(n_classes: int, off_diagonal: float = 1.0) -> np.ndarray:
    """The uniform baseline: every class change costs the same."""
    cost = np.full((n_classes, n_classes), float(off_diagonal), dtype=np.float64)
    np.fill_diagonal(cost, 0.0)
    return cost


def _argmax_scores(scores) -> np.ndarray:
    """Per-voxel argmax over classes, for dense arrays and lazy tables alike."""
    if not isinstance(scores, UnaryScores):
        return np.argmax(scores, axis=3)
    best_value = np.full(scores.volume_shape, -np.inf, dtype=np.float32)
    best_label = np.zeros(scores.volume_shape, dtype=np.int32)
    for k in range(scores.n_classes):
        column = scores.column(k)
        improved = column > best_value
        best_value = np.where(improved, column, best_value)
        best_label = np.where(improved, k, best_label)
    return best_label


def _softmax(scores: np.ndarray) -> np.ndarray:
    peak = scores.max(axis=-1, keepdims=True)
    exponent = np.exp(scores - peak, dtype=np.float32)
    return exponent / np.maximum(
        exponent.sum(axis=-1, keepdims=True), np.float32(1e-30)
    )


def adjacency_summary(pairwise: np.ndarray,
                      class_names: Sequence[str]) -> Dict[tuple, float]:
    """Learned costs as a readable ``{(class_a, class_b): cost}`` mapping."""
    summary = {}
    for i, name_i in enumerate(class_names):
        for j, name_j in enumerate(class_names):
            if j <= i:
                continue
            summary[(name_i, name_j)] = float(pairwise[i, j])
    return summary
