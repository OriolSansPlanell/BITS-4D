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
Mean-field keeps a full ``[Z, Y, X, K]`` responsibility array and several
of its kind while it updates: measured peak ≈ ``32·K + 30`` bytes per voxel
(``python -m validation.scaling``) — about 11 GB for a 38-million-voxel
volume with 9 classes. ICM keeps hard labels instead: ≈ 75 bytes per voxel
*regardless of K*, at the price of a coarser, greedier optimum.
:meth:`ROIDerivedMRF.refine` picks between them from a memory budget unless
told which to use.

Chunking
────────
Both solvers update every voxel from its neighbours' previous state
(synchronous sweeps), so after ``n`` sweeps a voxel depends only on voxels
within ``n`` steps of it. A volume that does not fit is therefore refined
in slabs along z, each padded with ``n_sweeps + 1`` extra slices on either
side (the halo) that are computed and thrown away. Edge weights and their
scale are computed once for the whole volume, so the labels are *identical*
to an unchunked run — only the peak memory changes. With a lazy
:class:`UnaryScores` input, the volume-wide cost falls to the 4-byte label
output; the ``32·K + 30`` bytes apply to the slab (plus its halo) only, so
mean-field stays the default solver on volumes of any size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# 6-connectivity: one entry per positive axis direction
_AXES = (0, 1, 2)

#: Measured peak bytes per voxel (validation/scaling.py, float32 scores)
MEAN_FIELD_BYTES_PER_CLASS = 32
MEAN_FIELD_BYTES_FIXED = 30
ICM_BYTES = 75
FORBIDDEN_COST = 1e3


@dataclass
class MRFDiagnostics:
    method: str
    sweeps: int
    changed_fraction: float
    energy: float
    energy_trace: list = None
    monotone: bool = True
    chunks: int = 1

    def describe(self) -> str:
        text = (
            f"{self.method}, {self.sweeps} sweeps, "
            f"{100 * self.changed_fraction:.2f}% of voxels changed on the last"
        )
        if self.chunks > 1:
            text += f" (in {self.chunks} slabs)"
        if not self.monotone:
            text += " — WARNING: cost did not fall monotonically"
        return text


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

    def slab(self, start: int, stop: int) -> "UnaryScores":
        """The same scores for slices ``start:stop`` (the table is shared)."""
        return UnaryScores(self.table, self.row_index[start:stop], self.fill)


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
        damping: float = 0.5,
        strict: bool = False,
    ) -> None:
        self.beta = float(beta)
        self.n_sweeps = int(n_sweeps)
        self.contrast_sigma = contrast_sigma
        self.max_cost = float(max_cost)
        self.memory_budget_gb = float(memory_budget_gb)
        self.damping = float(damping)
        self.strict = bool(strict)
        self.pairwise: Optional[np.ndarray] = None
        self.class_names: Optional[Sequence[str]] = None

    # ── learning the pairwise cost ───────────────────────────────────────
    def adjacency_counts(self, labels, n_classes: Optional[int] = None,
                         valid_mask=None) -> np.ndarray:
        """Face-adjacency counts ``n[k, l]`` in a label volume."""
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
        return counts + counts.T          # unordered pairs

    def fit_pairwise_from_labels(
        self,
        labels,
        n_classes: Optional[int] = None,
        valid_mask=None,
        class_names: Optional[Sequence[str]] = None,
        zero_diagonal: bool = True,
        epsilon: float = 1.0,
    ) -> np.ndarray:
        """Count face adjacencies at T0 and turn them into boundary costs.

        ``V(k, l) = -log( (n_kl + ε) / Σ_m (n_km + ε) )``, symmetrised. A
        class pair that touches constantly in the user's own segmentation is
        cheap to place side by side; one that never touches is expensive.
        This is strictly better than charging the same for every boundary,
        and it costs nothing — the counts come from work already done.

        *zero_diagonal* subtracts each row's self-cost before symmetrising.
        Without it a class whose own voxels are less reliably adjacent —
        which is exactly what a small or thin class looks like — pays a
        standing penalty for existing, on top of the penalty for bordering
        anything. That is a quiet bias against the classes most at risk of
        being smoothed away, so it is removed by default; the raw form is
        available for comparison.
        """
        counts = self.adjacency_counts(labels, n_classes, valid_mask)
        classes = counts.shape[0]
        if counts.sum() <= 0:
            self.pairwise = potts_cost(classes)
            self.class_names = class_names
            return self.pairwise

        padded = counts + float(epsilon)
        conditional = padded / padded.sum(axis=1, keepdims=True)
        cost = -np.log(conditional)
        if zero_diagonal:
            cost = cost - np.diag(cost)[:, None]
        cost = 0.5 * (cost + cost.T)
        np.clip(cost, 0.0, self.max_cost, out=cost)
        if zero_diagonal:
            np.fill_diagonal(cost, 0.0)

        self.pairwise = cost
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
    def _edge_parameters(self, neutron, xray, slab: int = 32):
        """``(scales, sigma)`` of the contrast-sensitive weights, or None.

        *scales* put both modalities on a comparable footing (each one's
        standard deviation); *sigma* is the mean normalised gap between
        face neighbours unless ``contrast_sigma`` fixes it. Both are
        accumulated slab by slab, so no volume-sized temporary is made and a
        chunked refinement uses exactly the numbers a whole-volume one does.
        """
        if self.contrast_sigma is not None and self.contrast_sigma == 0:
            return None
        if neutron is None or xray is None:
            return None
        volumes = (neutron, xray)
        depth = np.shape(neutron)[0]
        scales = []
        for volume in volumes:
            total, count = 0.0, 0
            for start in range(0, depth, slab):
                part = np.asarray(volume[start:start + slab], dtype=np.float64)
                finite = part[np.isfinite(part)]
                total += float(finite.sum())
                count += finite.size
            mean = total / count if count else 0.0
            squares = 0.0
            for start in range(0, depth, slab):
                part = np.asarray(volume[start:start + slab], dtype=np.float64)
                finite = part[np.isfinite(part)]
                squares += float(np.square(finite - mean).sum())
            spread = float(np.sqrt(squares / count)) if count else 1.0
            scales.append(spread if spread > 0 else 1.0)

        sigma = self.contrast_sigma
        if sigma is None:
            shape = np.shape(neutron)
            total, count = 0.0, 0
            for axis in _AXES:
                if shape[axis] < 2:
                    continue
                # The last layer along each axis has no forward neighbour and
                # counts as a zero gap (as it always has)
                count += int(np.prod(shape)) // shape[axis]
                for start in range(0, depth, slab):
                    stop = min(start + slab + (1 if axis == 0 else 0), depth)
                    gaps = _normalised_gaps(neutron[start:stop], xray[start:stop],
                                            scales, axis)
                    if gaps is None:
                        continue
                    finite = gaps[np.isfinite(gaps)]
                    total += float(finite.sum(dtype=np.float64))
                    count += finite.size
            sigma = total / count if count else 1.0
        return scales, max(float(sigma), 1e-6)

    def _edge_weights(self, neutron, xray, parameters="auto") -> Optional[list]:
        """Per-axis contrast-sensitive weights, or None for plain geometry.

        ``weights[axis][i]`` weighs the face between voxel ``i`` and its
        forward neighbour along *axis* (0 on the last layer). A face touching
        an unmeasured voxel gets weight 0: it carries no evidence.
        """
        if isinstance(parameters, str):
            parameters = self._edge_parameters(neutron, xray)
        if parameters is None or neutron is None or xray is None:
            return None
        scales, sigma = parameters
        weights = []
        shape = np.shape(neutron)
        for axis in _AXES:
            if shape[axis] < 2:
                weights.append(None)
                continue
            weight = np.zeros(shape, dtype=np.float32)
            low = [slice(None)] * 3
            low[axis] = slice(0, -1)
            gaps = _normalised_gaps(neutron, xray, scales, axis)
            gaps /= np.float32(sigma)
            np.square(gaps, out=gaps)
            gaps *= np.float32(-0.5)
            np.exp(gaps, out=gaps)
            weight[tuple(low)] = np.nan_to_num(gaps, nan=0.0)
            weights.append(weight)
        return weights

    # ── inference ────────────────────────────────────────────────────────
    @staticmethod
    def mean_field_bytes_per_voxel(n_classes: int) -> int:
        return MEAN_FIELD_BYTES_PER_CLASS * n_classes + MEAN_FIELD_BYTES_FIXED

    def estimate_memory_gb(self, n_voxels: int, n_classes: int) -> float:
        """Mean-field peak memory on *n_voxels*, in GiB (measured model)."""
        return (n_voxels * self.mean_field_bytes_per_voxel(n_classes)
                / (1024 ** 3))

    @property
    def halo(self) -> int:
        """Slices of padding that make a slab's result exact."""
        return self.n_sweeps + 1

    def plan_slabs(self, shape, n_classes: int) -> Tuple[str, Optional[int]]:
        """``(method, slab_thickness)`` for a volume under the budget.

        Mean-field on the whole volume if it fits; otherwise mean-field on
        z-slabs as thick as the budget allows (at least as thick as the halo,
        or the padding would cost more than the slab); otherwise ICM, in
        slabs if even its ~75 bytes per voxel do not fit. ``None`` means no
        chunking.
        """
        depth = int(shape[0])
        per_slice = int(np.prod(shape[1:]))
        budget = self.memory_budget_gb * 1024 ** 3
        if self.estimate_memory_gb(depth * per_slice, n_classes) <= self.memory_budget_gb:
            return "mean_field", None
        # The label output (4 B per voxel) is volume-wide; the rest per slab
        budget_for_slabs = budget - 4 * depth * per_slice
        rows = int(budget_for_slabs // (self.mean_field_bytes_per_voxel(n_classes)
                                        * per_slice)) - 2 * self.halo
        if rows >= max(1, self.halo):
            return "mean_field", rows
        icm_bytes = ICM_BYTES * per_slice
        if icm_bytes * depth <= budget:
            return "icm", None
        rows = int(budget // icm_bytes) - 2 * self.halo
        return "icm", (rows if rows >= 1 else None)

    def refine(
        self,
        log_unary,
        neutron=None,
        xray=None,
        valid_mask=None,
        method: str = "auto",
        initial_labels=None,
        cancel_check=None,
        slab_thickness: Optional[int] = None,
    ) -> Tuple[np.ndarray, MRFDiagnostics]:
        """Regularise per-voxel unary scores into a coherent labelling.

        *log_unary* is ``[Z, Y, X, K]``: the log of the mixture posterior (or
        any per-voxel score) before spatial smoothing. Returns
        ``(labels, diagnostics)`` where labels are ``int32`` and ``-1`` marks
        an invalid voxel. *slab_thickness* forces chunking along z (see the
        module notes); ``"auto"`` also chooses it from the memory budget.
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
            method, planned = self.plan_slabs(shape, n_classes)
            if slab_thickness is None:
                slab_thickness = planned

        chunked = (slab_thickness is not None
                   and 0 < slab_thickness < shape[0])
        parameters = None
        weights = None
        if self.beta > 0 and self.n_sweeps > 0:
            parameters = self._edge_parameters(neutron, xray)
            if not chunked:
                weights = self._edge_weights(neutron, xray, parameters)
        if self.beta <= 0 or self.n_sweeps <= 0:
            labels = _argmax_scores(scores).astype(np.int32)
            if valid is not None:
                labels[~valid] = -1
            return labels, MRFDiagnostics(method="none", sweeps=0,
                                          changed_fraction=0.0, energy=float("nan"))

        if method not in ("mean_field", "icm"):
            raise ValueError(f"Unknown method {method!r}")
        chunks = 1
        if chunked:
            labels, changed, trace, chunks = self._refine_in_slabs(
                method, scores, neutron, xray, parameters, valid,
                initial_labels, cancel_check, int(slab_thickness))
            # The last sweep's energy is the energy of these labels
            energy = trace[-1] if trace else float("nan")
        else:
            labels, changed, trace = self._solve(
                method, scores, weights, valid, initial_labels, cancel_check)
            energy = self._energy(scores, labels, weights, valid)
        # The total cost should fall every sweep. If it rises, the refinement
        # is cycling rather than settling, and any result it produces is an
        # arbitrary point in that cycle rather than an answer.
        monotone = _is_monotone(trace)
        if not monotone and self.strict:
            raise RuntimeError(
                "Spatial refinement did not settle: total cost rose between "
                f"sweeps ({[round(v, 3) for v in trace]}). Reduce the "
                "smoothing strength or increase damping."
            )
        return labels, MRFDiagnostics(
            method=method, sweeps=self.n_sweeps,
            changed_fraction=changed, energy=energy,
            energy_trace=trace, monotone=monotone, chunks=chunks,
        )

    def _solve(self, method, scores, weights, valid, initial_labels,
               cancel_check, core=None):
        if method == "mean_field":
            dense = scores.dense() if isinstance(scores, UnaryScores) else scores
            return self._mean_field(dense, weights, valid, cancel_check, core)
        return self._icm(scores, weights, valid, initial_labels, cancel_check,
                         core)

    def _refine_in_slabs(self, method, scores, neutron, xray, parameters,
                         valid, initial_labels, cancel_check, thickness):
        """Run the solver slab by slab; each slab's core is exact.

        Returns the same ``(labels, changed, trace)`` as an unchunked run
        (up to floating-point summation order in the energies), plus the
        number of slabs.
        """
        depth = scores.shape[0]
        halo = self.halo
        labels = np.empty(scores.shape[:3], dtype=np.int32)
        changed_voxels = 0.0
        trace = None
        chunks = 0
        for start in range(0, depth, thickness):
            stop = min(start + thickness, depth)
            low, high = max(0, start - halo), min(depth, stop + halo)
            part = (scores.slab(low, high) if isinstance(scores, UnaryScores)
                    else scores[low:high])
            # Weights for the slab (one extra slice gives the faces of its
            # last layer), from the volume-wide scale
            part_weights = None
            if parameters is not None:
                extra = min(high + 1, depth)
                part_weights = [
                    None if w is None else w[:high - low]
                    for w in self._edge_weights(neutron[low:extra],
                                                xray[low:extra], parameters)
                ]
            part_valid = None if valid is None else valid[low:high]
            part_initial = (None if initial_labels is None
                            else np.asarray(initial_labels)[low:high])
            core = (start - low, stop - low)
            result, changed, part_trace = self._solve(
                method, part, part_weights, part_valid, part_initial,
                cancel_check, core)
            labels[start:stop] = result[core[0]:core[1]]
            changed_voxels += changed * (stop - start)
            trace = (list(part_trace) if trace is None else
                     [a + b for a, b in zip(trace, part_trace)])
            chunks += 1
        return labels, changed_voxels / depth, trace or [], chunks

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

    def _mean_field(self, scores, weights, valid, cancel_check, core=None):
        region = _core_slice(core)
        pairwise = self.pairwise.astype(np.float32)
        responsibilities = _softmax(scores)
        if valid is not None:
            responsibilities[~valid] = 0.0

        previous = np.argmax(responsibilities, axis=3)
        changed = 0.0
        trace = []
        damping = np.float32(np.clip(self.damping, 0.0, 1.0))
        for _ in range(self.n_sweeps):
            if cancel_check:
                cancel_check()
            neighbour = self._neighbour_message(responsibilities, weights)
            # message[..., k] = Σ_l neighbour[..., l] · V(k, l)
            message = neighbour @ pairwise.T
            updated = _softmax(scores - self.beta * message)
            # Damped update. An undamped synchronous sweep can settle into a
            # two-cycle that flips a whole region back and forth forever and
            # looks, from outside, like an unstable segmentation.
            responsibilities = (
                updated if damping >= 1.0
                else damping * updated + (1.0 - damping) * responsibilities
            )
            if valid is not None:
                responsibilities[~valid] = 0.0
            current = np.argmax(responsibilities, axis=3)
            changed = float(np.mean(current[region] != previous[region]))
            previous = current
            trace.append(self._energy(scores, _mask_labels(current, valid),
                                      weights, valid, core))

        labels = previous.astype(np.int32)
        if valid is not None:
            labels[~valid] = -1
        return labels, changed, trace

    def _icm(self, scores, weights, valid, initial_labels, cancel_check,
             core=None):
        region = _core_slice(core)
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
        trace = []
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
            changed = float(np.mean(best_label[region] != labels[region]))
            labels = best_label.astype(np.int32)
            trace.append(
                self._energy(scores, _mask_labels(labels, valid), weights,
                             valid, core)
            )

        if valid is not None:
            labels[~valid] = -1
        return labels, changed, trace

    def _energy(self, scores, labels, weights, valid, core=None) -> float:
        """Total energy of a labelling — lower is better.

        With *core* ``(start, stop)`` only the unary terms of slices
        ``start:stop`` count, and the pair terms whose lower voxel lies
        there: summed over the slabs of a chunked run, that is the energy
        of the whole volume, each term counted once.
        """
        usable = labels >= 0
        if not usable.any():
            return float("nan")
        if core is not None:
            owned = np.zeros(labels.shape[0], dtype=bool)
            owned[core[0]:core[1]] = True
            owned = np.broadcast_to(owned[:, None, None], labels.shape)
        safe = np.where(usable, labels, 0)
        if isinstance(scores, UnaryScores):
            chosen = np.zeros(labels.shape, dtype=np.float32)
            for k in range(scores.n_classes):
                np.copyto(chosen, scores.column(k), where=safe == k)
            unary = -chosen
        else:
            unary = -np.take_along_axis(scores, safe[..., None], axis=3)[..., 0]
        counted = usable if core is None else usable & owned
        energy = float(unary[counted].sum())

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
            if core is not None:
                pair_ok &= owned[low_key]
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


def _normalised_gaps(neutron, xray, scales, axis):
    """Joint intensity step to the forward neighbour along *axis*.

    ``sqrt((Δn / s_n)² + (Δx / s_x)²)`` as float32, one layer shorter than
    the input along *axis*; None if the input is a single layer there.
    """
    first = np.asarray(neutron, dtype=np.float32)
    second = np.asarray(xray, dtype=np.float32)
    if first.shape[axis] < 2:
        return None
    low = [slice(None)] * 3
    high = [slice(None)] * 3
    low[axis] = slice(0, -1)
    high[axis] = slice(1, None)
    low, high = tuple(low), tuple(high)
    gap = np.subtract(first[high], first[low])
    gap /= np.float32(scales[0])
    np.square(gap, out=gap)
    other = np.subtract(second[high], second[low])
    other /= np.float32(scales[1])
    np.square(other, out=other)
    gap += other
    del other
    np.sqrt(gap, out=gap)
    return gap


def _core_slice(core):
    """Index selecting a slab's core slices (everything when None)."""
    return slice(None) if core is None else slice(core[0], core[1])


def _mask_labels(labels, valid):
    """Labels with invalid voxels marked, without touching the original."""
    if valid is None:
        return labels
    return np.where(valid, labels, -1)


def _is_monotone(trace, tolerance: float = 1e-4) -> bool:
    """Did the cost fall (or hold) at every sweep?"""
    values = [value for value in (trace or []) if np.isfinite(value)]
    for previous, current in zip(values, values[1:]):
        allowance = tolerance * max(abs(previous), 1.0)
        if current > previous + allowance:
            return False
    return True


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
