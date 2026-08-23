"""Locked-mode segmentation: fixed material classes, per-timepoint refinement.

This is the default path. Class positions are fixed — attenuation
coefficients are material constants, so a class that moves is a class
absorbing material that should have left it. Per timepoint the work is:

1. build the validity mask (both channels);
2. accumulate the histogram cache on the shared bin grid;
3. look up each voxel's match to every class from a table computed once;
4. resolve the assignment spatially;
5. report.

No parameters are estimated, so timepoints are fully independent: the result
is the same run forwards, backwards or in parallel, and three of the failure
modes an adaptive fit can have — classes merging, class identities permuting,
and oscillation between timepoints — are structurally impossible rather than
guarded against.

Choosing the smoothing strength
───────────────────────────────
Spatial smoothing has a strength, and left at a default it is the single most
destructive parameter here: too high and a small class is simply erased,
silently, with everything downstream still looking healthy. At ~1 % of the
volume a minority phase is the first thing to go.

:func:`auto_smoothing` therefore chooses it, by measuring rather than
guessing: the strongest setting at which **no class loses more than a set
share of its unsmoothed volume**, and the classes declared unchanging stay
unchanged. The whole sweep is kept, because the answer is only trustworthy
if you can see the curve it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from model.histogram_cache import build_histogram_cache
from model.likelihood import UNCLASSIFIED, ClassLibrary, MatchTable, match_table
from model.spatial_prior import ROIDerivedMRF, UnaryScores
from model.validity import ValidityPolicy, build_valid_mask, validity_report

#: Default sweep for the automatic smoothing search.
SMOOTHING_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _with_unclassified_row(cost: np.ndarray) -> np.ndarray:
    """Extend a K×K boundary cost to (K+1)×(K+1) for the Unclassified label."""
    cost = np.asarray(cost, dtype=np.float64)
    size = cost.shape[0]
    off_diagonal = cost[~np.eye(size, dtype=bool)] if size > 1 else np.array([1.0])
    neutral = float(np.median(off_diagonal)) if off_diagonal.size else 1.0

    extended = np.full((size + 1, size + 1), neutral, dtype=np.float64)
    extended[:size, :size] = cost
    extended[size, size] = 0.0
    return extended


def _unmatched_advice(outcome) -> str:
    """Why voxels stopped matching, and what to do about it.

    A material missing from the definitions is unmatched from the very first
    timepoint. Unmatched voxels that *accumulate* over the series mean
    something else: the measurement itself has moved, so definitions taken at
    the first timepoint no longer describe the same materials. Those two need
    opposite responses, and telling them apart is a matter of looking at when
    the mismatch appeared.
    """
    entries = list(getattr(outcome, "timepoints", []))
    if len(entries) < 3:
        return "You may be missing a material."

    shares = [entry.unclassified_fraction for entry in entries]
    grows = shares[-1] > 3 * max(shares[0], 1e-6) and shares[0] < 0.02
    if grows:
        return (
            "It was fine at the start and got steadily worse, which means "
            "the measurement drifted rather than that a material is missing. "
            "Run Check Instrument Stability; if the histogram has moved, "
            "either re-draw the materials on a later timepoint or allow the "
            "definitions to move (Advanced)."
        )
    return (
        "You may be missing a material. Look at where those voxels fall on "
        "the histogram and draw a region for them."
    )


class SegmentationRefused(RuntimeError):
    """Raised when a result would be silently wrong, so it is not returned.

    Carries ``findings``: one plain sentence per problem, each naming the
    class involved and what to do about it.
    """

    def __init__(self, findings: Sequence[str]):
        self.findings = list(findings)
        super().__init__("; ".join(self.findings))


@dataclass
class TimepointSegmentation:
    """One timepoint's result."""

    timepoint: int
    labels: np.ndarray                       # int32, 0 = Unclassified
    class_names: List[str]
    voxel_counts: Dict[str, int] = field(default_factory=dict)
    #: Measured voxels that matched no class. Distinct from *excluded_voxels*
    #: — "we looked and found nothing" is a different statement from "there
    #: was nothing to look at", and only the first one means a class may be
    #: missing.
    unclassified_voxels: int = 0
    excluded_voxels: int = 0
    valid_voxels: int = 0
    total_voxels: int = 0
    validity: Dict[str, float] = field(default_factory=dict)
    refinement: Optional[object] = None
    unsmoothed_counts: Dict[str, int] = field(default_factory=dict)
    fractions: Dict[str, np.ndarray] = field(default_factory=dict)

    def mask_for(self, name: str) -> np.ndarray:
        if name not in self.class_names:
            raise KeyError(f"No class named {name!r} in this result")
        return self.labels == (self.class_names.index(name) + 1)

    @property
    def unclassified_fraction(self) -> float:
        """Share of the **measured** voxels that matched no class."""
        return self.unclassified_voxels / max(self.valid_voxels, 1)

    def smoothing_retention(self) -> Dict[str, float]:
        """Per class, the share of its unsmoothed volume that survived.

        Compared at the *same* timepoint, so it isolates what smoothing did
        from what the sample did. A class that genuinely shrinks between
        timepoints shrinks in both numbers and scores 1.0 here.
        """
        if not self.unsmoothed_counts:
            return {name: 1.0 for name in self.class_names}
        return {
            name: (
                self.voxel_counts.get(name, 0) / self.unsmoothed_counts[name]
                if self.unsmoothed_counts.get(name) else 1.0
            )
            for name in self.class_names
        }

    def budget_closes(self) -> bool:
        """Every voxel is counted exactly once — a bug alarm, not a metric."""
        assigned = sum(self.voxel_counts.values())
        return (
            assigned + self.unclassified_voxels + self.excluded_voxels
            == self.total_voxels
        )


@dataclass
class SeriesSegmentation:
    """The whole run."""

    timepoints: List[TimepointSegmentation] = field(default_factory=list)
    library: Optional[ClassLibrary] = None
    smoothing: float = 0.0
    smoothing_sweep: Optional[list] = None
    pairwise_cost: Optional[np.ndarray] = None
    mode: str = "locked"

    def __iter__(self):
        return iter(self.timepoints)

    def __len__(self) -> int:
        return len(self.timepoints)

    @property
    def class_names(self) -> List[str]:
        return list(self.library.names) if self.library else []

    def counts_table(self) -> Dict[int, Dict[str, int]]:
        return {
            entry.timepoint: dict(entry.voxel_counts)
            for entry in self.timepoints
        }

    def volume_curve(self, name: str) -> List[int]:
        return [entry.voxel_counts.get(name, 0) for entry in self.timepoints]

    def masks_by_timepoint(self) -> Dict[int, Dict[str, np.ndarray]]:
        return {
            entry.timepoint: {
                name: entry.mask_for(name) for name in entry.class_names
            }
            for entry in self.timepoints
        }


class LockedSegmenter:
    """Segment a series against fixed material classes.

    Parameters
    ----------
    library
        The fixed classes, in the order they will be labelled.
    prior
        Spatial refinement. ``None`` assigns each voxel to its best match
        with no spatial term at all, which is what the histogram alone can
        say and is speckled by construction.
    validity_policy
        Which voxels count as measurements.
    min_retention
        A class that keeps less than this share of its unsmoothed volume has
        been smoothed away. Used both by the automatic search and, at the
        harder threshold below, as a refusal condition.
    """

    def __init__(
        self,
        library: ClassLibrary,
        prior: Optional[ROIDerivedMRF] = None,
        validity_policy: Optional[ValidityPolicy] = None,
        bins: int = 256,
        min_retention: float = 0.80,
        abort_retention: float = 0.50,
        max_unclassified: float = 0.05,
        inert_tolerance: float = 0.05,
        unclassified_floor: float = 1e-4,
    ) -> None:
        self.library = library
        self.prior = prior
        self.validity_policy = validity_policy or ValidityPolicy()
        self.bins = int(bins)
        self.min_retention = float(min_retention)
        self.abort_retention = float(abort_retention)
        self.max_unclassified = float(max_unclassified)
        self.inert_tolerance = float(inert_tolerance)
        self.unclassified_floor = float(unclassified_floor)
        self.neutron_edges: Optional[np.ndarray] = None
        self.xray_edges: Optional[np.ndarray] = None

    # ── setup ────────────────────────────────────────────────────────────
    def set_grid(self, neutron_edges, xray_edges) -> None:
        """Fix the histogram grid every timepoint shares."""
        self.neutron_edges = np.asarray(neutron_edges, dtype=np.float64)
        self.xray_edges = np.asarray(xray_edges, dtype=np.float64)

    def learn_boundaries(self, label_volume, valid_mask=None) -> np.ndarray:
        """Derive the boundary costs from the reference label volume.

        The returned matrix has one extra row and column for Unclassified,
        priced at the typical cost of any other boundary: free would let
        smoothing flood unmatched voxels across the volume, and prohibitive
        would push them into a real class — which is the outcome Unclassified
        exists to prevent.
        """
        if self.prior is None:
            return None
        learned = self.prior.fit_pairwise_from_labels(
            np.maximum(np.asarray(label_volume), 0),
            n_classes=len(self.library),
            valid_mask=valid_mask,
            class_names=self.library.names,
        )
        self.prior.pairwise = _with_unclassified_row(learned)
        self.prior.class_names = list(self.library.names) + ["Unclassified"]
        return self.prior.pairwise

    # ── one timepoint ────────────────────────────────────────────────────
    def segment_timepoint(
        self,
        neutron_volume,
        xray_volume,
        timepoint: int = 0,
        beta: Optional[float] = None,
        table: Optional[MatchTable] = None,
        cancel_check=None,
    ) -> TimepointSegmentation:
        """Segment one volume pair against the fixed classes."""
        if self.neutron_edges is None:
            raise RuntimeError("Call set_grid() before segmenting")

        valid = build_valid_mask(
            neutron_volume, xray_volume, self.validity_policy
        )
        cache = build_histogram_cache(
            neutron_volume, xray_volume,
            self.neutron_edges, self.xray_edges,
            valid_mask=valid, store_bin_index=True,
        )
        if cache.num_bins == 0:
            raise SegmentationRefused([
                f"Timepoint {timepoint} has no usable voxels at all. Check "
                f"that both volumes contain data for this timepoint."
            ])

        scores = (table or match_table(
            self.library, cache, self.unclassified_floor
        )).with_unclassified()
        rows = cache.row_index_volume()
        n_classes = len(self.library)

        if self.prior is None or beta == 0.0 or (
            beta is None and self.prior.beta == 0.0
        ):
            best = np.argmax(scores, axis=1).astype(np.int32)
            per_voxel = cache.expand_to_voxels(
                best, fill=np.int32(n_classes), rows=rows
            ).astype(np.int32)
            refinement = None
        else:
            original_beta = self.prior.beta
            if beta is not None:
                self.prior.beta = float(beta)
            try:
                per_voxel, refinement = self.prior.refine(
                    UnaryScores(scores, rows),
                    neutron=neutron_volume, xray=xray_volume,
                    valid_mask=valid & (rows >= 0),
                    cancel_check=cancel_check,
                )
            finally:
                self.prior.beta = original_beta
            per_voxel = np.where(per_voxel < 0, n_classes, per_voxel)

        # Shift so that Unclassified is 0 and class k is k+1
        labels = np.where(
            per_voxel >= n_classes, np.int32(UNCLASSIFIED),
            (per_voxel + 1).astype(np.int32),
        )
        labels[~valid] = UNCLASSIFIED

        counts = {
            name: int(np.count_nonzero(labels == index))
            for index, name in enumerate(self.library.names, start=1)
        }
        return TimepointSegmentation(
            timepoint=timepoint,
            labels=labels,
            class_names=list(self.library.names),
            voxel_counts=counts,
            unclassified_voxels=int(
                np.count_nonzero((labels == UNCLASSIFIED) & valid)
            ),
            excluded_voxels=int(np.count_nonzero(~valid)),
            valid_voxels=int(np.count_nonzero(valid)),
            total_voxels=int(labels.size),
            validity=validity_report(
                neutron_volume, xray_volume, self.validity_policy
            ),
            refinement=refinement,
        )

    # ── automatic smoothing strength ─────────────────────────────────────
    def auto_smoothing(
        self,
        neutron_volume,
        xray_volume,
        grid: Sequence[float] = SMOOTHING_GRID,
        timepoint: int = 0,
        progress_callback=None,
        cancel_check=None,
    ):
        """Strongest smoothing that costs no class its volume.

        Returns ``(strength, sweep)``. *sweep* is one row per grid point:
        the per-class volumes, each class's retention against the unsmoothed
        result, the movement of the control materials, and whether that point
        was acceptable. Keep it — it is the evidence for the choice, and the
        curve is worth looking at directly.
        """
        if self.neutron_edges is None:
            raise RuntimeError("Call set_grid() before searching")
        grid = sorted(set(float(value) for value in grid))
        if 0.0 not in grid:
            grid = [0.0] + grid

        valid = build_valid_mask(
            neutron_volume, xray_volume, self.validity_policy
        )
        cache = build_histogram_cache(
            neutron_volume, xray_volume,
            self.neutron_edges, self.xray_edges,
            valid_mask=valid, store_bin_index=True,
        )
        table = match_table(self.library, cache, self.unclassified_floor)

        baseline: Optional[Dict[str, int]] = None
        sweep = []
        chosen = 0.0
        for position, value in enumerate(grid):
            if cancel_check:
                cancel_check()
            result = self.segment_timepoint(
                neutron_volume, xray_volume, timepoint=timepoint,
                beta=value, table=table, cancel_check=cancel_check,
            )
            if baseline is None:
                baseline = dict(result.voxel_counts)

            retention = {
                name: (
                    result.voxel_counts.get(name, 0) / baseline[name]
                    if baseline.get(name) else 1.0
                )
                for name in self.library.names
            }
            inert_shift = {
                name: abs(retention[name] - 1.0)
                for name in self.library.inert_names
            }
            unclassified = result.unclassified_fraction
            worst = min(retention.values()) if retention else 1.0
            acceptable = (
                worst >= self.min_retention
                and all(
                    shift <= self.inert_tolerance for shift in inert_shift.values()
                )
                and unclassified <= self.max_unclassified
            )
            sweep.append({
                "smoothing": value,
                "volumes": dict(result.voxel_counts),
                "retention": retention,
                "worst_retention": worst,
                "worst_class": (
                    min(retention, key=retention.get) if retention else None
                ),
                "control_shift": inert_shift,
                "unclassified_fraction": unclassified,
                "acceptable": acceptable,
            })
            if acceptable:
                chosen = value
            if progress_callback:
                progress_callback(
                    int(100 * (position + 1) / len(grid)),
                    f"Testing smoothing {value:g}",
                )
        return chosen, sweep

    # ── the series ───────────────────────────────────────────────────────
    def segment_series(
        self,
        dataset,
        timepoints: Optional[Sequence[int]] = None,
        beta: Optional[float] = None,
        progress_callback=None,
        cancel_check=None,
        enforce_guards: bool = True,
    ) -> SeriesSegmentation:
        """Segment every timepoint. Order does not affect the result."""
        indices = (
            list(range(dataset.num_timepoints)) if timepoints is None
            else list(timepoints)
        )
        strength = (
            beta if beta is not None
            else (self.prior.beta if self.prior is not None else 0.0)
        )
        outcome = SeriesSegmentation(
            library=self.library, smoothing=float(strength),
            pairwise_cost=None if self.prior is None else self.prior.pairwise,
        )

        for position, timepoint in enumerate(indices):
            if cancel_check:
                cancel_check()
            if progress_callback:
                progress_callback(
                    int(100 * position / max(len(indices), 1)),
                    f"Timepoint {timepoint + 1} of {len(indices)}",
                )
            neutron, xray = dataset.get_volume_at_time(timepoint)
            entry = self.segment_timepoint(
                neutron, xray, timepoint=timepoint, beta=strength,
                cancel_check=cancel_check,
            )
            if strength > 0:
                # The same timepoint without smoothing, so the retention
                # guard measures what smoothing did rather than what the
                # sample did. Unsmoothed is a table lookup, so it is cheap.
                unsmoothed = self.segment_timepoint(
                    neutron, xray, timepoint=timepoint, beta=0.0,
                    cancel_check=cancel_check,
                )
                entry.unsmoothed_counts = dict(unsmoothed.voxel_counts)
            outcome.timepoints.append(entry)

        if enforce_guards:
            findings = self.check_guards(outcome)
            if findings:
                raise SegmentationRefused(findings)
        if progress_callback:
            progress_callback(100, "Done")
        return outcome

    # ── refusal conditions ───────────────────────────────────────────────
    def check_guards(self, outcome: SeriesSegmentation) -> List[str]:
        """Conditions under which a result must not be presented as an answer.

        These are not warnings. Each one means a number somewhere downstream
        would be wrong in a way nothing later would reveal.
        """
        findings: List[str] = []
        if not outcome.timepoints:
            return ["Nothing was segmented."]

        # Smoothing must not be what removes a class. A class shrinking
        # between timepoints is the measurement and is never a refusal — that
        # is the whole point of running a series. What is checked here is the
        # same timepoint with and without smoothing.
        for entry in outcome.timepoints:
            for name, kept in entry.smoothing_retention().items():
                if kept < self.abort_retention:
                    findings.append(
                        f"Smoothing removed {100 * (1 - kept):.0f}% of the "
                        f"class '{name}' at timepoint {entry.timepoint}. "
                        f"Reduce the smoothing strength."
                    )
            if findings:
                break

        for entry in outcome.timepoints:
            share = entry.unclassified_fraction
            if share > self.max_unclassified:
                findings.append(
                    f"At timepoint {entry.timepoint}, {100 * share:.0f}% of "
                    f"the measured voxels did not match any material you "
                    f"defined. " + _unmatched_advice(outcome)
                )
                break

            if not entry.budget_closes():
                findings.append(
                    f"At timepoint {entry.timepoint} the voxel count does not "
                    f"add up — some voxels were counted more than once. This "
                    f"is a bug; please report it."
                )
                break

            if entry.refinement is not None and not entry.refinement.monotone:
                findings.append(
                    f"The spatial cleanup at timepoint {entry.timepoint} did "
                    f"not settle — it cycled between two answers. Reduce the "
                    f"smoothing strength."
                )
                break
        return findings
