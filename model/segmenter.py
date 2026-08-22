"""The HMRF-EM loop: one pass over the series, one model carried through it.

Per timepoint:

1. build the validity mask, so padding and failed voxels take no part;
2. accumulate the histogram cache on the global bin grid;
3. estimate instrumental drift from the anchor classes and move the *prior*
   into this timepoint's frame — the data is never rewritten;
4. run MAP-EM on the cache under that prior;
5. let the temporal model clip an implausible jump and freeze a collapsed
   component;
6. hand the per-bin posterior to the MRF, which turns it into a spatially
   coherent labelling;
7. emit labels, parameters, fractional maps and diagnostics.

Because step 4 emits a full parameter set every time, the component
trajectories *are* the evolution analysis — there is no separate pass needed
to find out how the histogram changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from model.drift_tracker import DriftTracker
from model.histogram_cache import build_histogram_cache, moments_from_mask
from model.mixture import MixturePrior, ROIAnchoredMixture
from model.partial_volume import MixelComponent, fractional_maps, verify_mixels
from model.spatial_prior import ROIDerivedMRF, UnaryScores
from model.temporal import DriftTransition, TemporalModel
from model.validity import ValidityPolicy, build_valid_mask, validity_report

UNASSIGNED = -1


@dataclass
class TimepointResult:
    """Everything one timepoint produced."""

    timepoint: int
    labels: np.ndarray                       # int32 volume, -1 = unassigned
    class_names: List[str]
    voxel_counts: Dict[str, int] = field(default_factory=dict)
    fit: Optional[object] = None
    drift: Optional[object] = None
    mrf: Optional[object] = None
    transition: Optional[object] = None
    validity: Dict[str, float] = field(default_factory=dict)
    fractions: Dict[str, np.ndarray] = field(default_factory=dict)
    unassigned_voxels: int = 0

    def mask_for(self, name: str) -> np.ndarray:
        """Boolean mask of one class."""
        if name not in self.class_names:
            raise KeyError(f"No class named {name!r} in this result")
        return self.labels == self.class_names.index(name)


@dataclass
class SegmentationResult:
    """The whole series."""

    timepoints: List[TimepointResult] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    prior: Optional[MixturePrior] = None
    mixel_report: Dict[str, dict] = field(default_factory=dict)
    pairwise_cost: Optional[np.ndarray] = None

    def __iter__(self):
        return iter(self.timepoints)

    def __len__(self) -> int:
        return len(self.timepoints)

    def counts_table(self) -> Dict[int, Dict[str, int]]:
        """``{timepoint: {class: voxels}}`` — the volume curves."""
        return {
            result.timepoint: dict(result.voxel_counts)
            for result in self.timepoints
        }

    def parameter_trajectories(self) -> Dict[str, Dict[str, list]]:
        """``{class: {'timepoint', 'centroid_n', 'centroid_x', ...}}``."""
        trajectories: Dict[str, Dict[str, list]] = {
            name: {
                "timepoint": [], "centroid_n": [], "centroid_x": [],
                "sigma_n": [], "sigma_x": [], "weight": [], "voxels": [],
            }
            for name in self.class_names
        }
        for result in self.timepoints:
            if result.fit is None:
                continue
            for index, name in enumerate(result.fit.names):
                if name not in trajectories:
                    continue
                entry = trajectories[name]
                entry["timepoint"].append(result.timepoint)
                entry["centroid_n"].append(float(result.fit.means[index][0]))
                entry["centroid_x"].append(float(result.fit.means[index][1]))
                covariance = result.fit.covariances[index]
                entry["sigma_n"].append(float(np.sqrt(max(covariance[0, 0], 0))))
                entry["sigma_x"].append(float(np.sqrt(max(covariance[1, 1], 0))))
                entry["weight"].append(float(result.fit.weights[index]))
                entry["voxels"].append(
                    int(result.voxel_counts.get(name, 0))
                )
        return trajectories


class SequentialSegmenter:
    """Run the anchored mixture across a time series.

    Parameters
    ----------
    mixture
        The :class:`~model.mixture.ROIAnchoredMixture` to fit.
    mrf
        Spatial regulariser. ``None`` skips it, which gives the raw per-voxel
        mixture labels — speckled, and only useful as a control.
    temporal
        How components may move between timepoints. Defaults to
        :class:`~model.temporal.DriftTransition`.
    drift_tracker
        Instrumental drift from anchor classes. ``None`` disables drift
        correction, which together with
        :class:`~model.temporal.StaticTransition` reproduces frozen-boundary
        behaviour for comparison.
    """

    def __init__(
        self,
        mixture: Optional[ROIAnchoredMixture] = None,
        mrf: Optional[ROIDerivedMRF] = None,
        temporal: Optional[TemporalModel] = None,
        drift_tracker: Optional[DriftTracker] = None,
        validity_policy: Optional[ValidityPolicy] = None,
        mixels: Sequence[MixelComponent] = (),
        bins: int = 256,
        store_bin_index: bool = True,
        mrf_method: str = "auto",
    ) -> None:
        self.mixture = mixture or ROIAnchoredMixture()
        self.mrf = mrf
        self.temporal = temporal or DriftTransition()
        self.drift_tracker = drift_tracker
        self.validity_policy = validity_policy or ValidityPolicy()
        self.mixels = list(mixels)
        self.bins = int(bins)
        self.store_bin_index = bool(store_bin_index)
        self.mrf_method = mrf_method

        self.prior: Optional[MixturePrior] = None
        self.class_names: List[str] = []
        self.neutron_edges: Optional[np.ndarray] = None
        self.xray_edges: Optional[np.ndarray] = None

    # ── setup ────────────────────────────────────────────────────────────
    def prepare(
        self,
        neutron_volume,
        xray_volume,
        class_masks: Dict[str, np.ndarray],
        neutron_edges,
        xray_edges,
        anchor_strength: float = 0.5,
        per_class_strength: Optional[Dict[str, float]] = None,
        dirichlet_strength: float = 0.0,
        outlier_weight: float = 1e-3,
        label_volume=None,
    ) -> MixturePrior:
        """Turn the manual T0 segmentation into the model's prior.

        *class_masks* are the reference timepoint's class masks — exactly the
        layers the histogram ROIs produced. *neutron_edges* / *xray_edges*
        come from the global histogram, so every timepoint shares one grid.
        """
        if not class_masks:
            raise ValueError("At least one class mask is needed to build a prior")
        self.neutron_edges = np.asarray(neutron_edges, dtype=np.float64)
        self.xray_edges = np.asarray(xray_edges, dtype=np.float64)

        valid = build_valid_mask(neutron_volume, xray_volume, self.validity_policy)
        moments = {}
        for name, mask in class_masks.items():
            usable = np.asarray(mask, dtype=bool) & valid
            entry = moments_from_mask(neutron_volume, xray_volume, usable)
            if entry is not None:
                moments[name] = entry
        if not moments:
            raise ValueError(
                "None of the class masks selected any valid voxels"
            )

        self.class_names = list(moments)
        self.prior = ROIAnchoredMixture.prior_from_moments(
            moments,
            anchor_strength=anchor_strength,
            per_class_strength=per_class_strength,
            dirichlet_strength=dirichlet_strength,
            outlier_weight=outlier_weight,
        )

        if self.drift_tracker is not None:
            self.drift_tracker.fit_reference(moments)

        if self.mrf is not None:
            if label_volume is None:
                label_volume = np.full(valid.shape, -1, dtype=np.int32)
                for index, name in enumerate(self.class_names):
                    label_volume[np.asarray(class_masks[name], dtype=bool)] = index
            usable = np.asarray(label_volume) >= 0
            self.mrf.fit_pairwise_from_labels(
                np.maximum(np.asarray(label_volume), 0),
                n_classes=len(self.class_names),
                valid_mask=usable & valid,
                class_names=self.class_names,
            )
            self._constrain_mixels()
        return self.prior

    def _constrain_mixels(self) -> None:
        """A mixing line may only touch the two phases it lies between."""
        if self.mrf is None or not self.mixels:
            return
        index_of = {name: i for i, name in enumerate(self.class_names)}
        for mixel in self.mixels:
            if mixel.name not in index_of:
                continue
            parents = [
                index_of[phase] for phase in (mixel.phase_a, mixel.phase_b)
                if phase in index_of
            ]
            if len(parents) == 2:
                self.mrf.allow_only(index_of[mixel.name], parents)

    # ── the loop ─────────────────────────────────────────────────────────
    def run(
        self,
        dataset,
        timepoints: Optional[Sequence[int]] = None,
        progress_callback=None,
        cancel_check=None,
    ) -> SegmentationResult:
        """Segment a series, carrying the model from one timepoint to the next."""
        if self.prior is None:
            raise RuntimeError("Call prepare() before run()")

        indices = (
            list(range(dataset.num_timepoints)) if timepoints is None
            else list(timepoints)
        )
        result = SegmentationResult(
            class_names=list(self.class_names), prior=self.prior
        )
        previous_fit = None
        previous_drift = None

        for position, timepoint in enumerate(indices):
            if cancel_check:
                cancel_check()
            if progress_callback:
                progress_callback(
                    int(100 * position / max(len(indices), 1)),
                    f"Timepoint {timepoint + 1}/{len(indices)}: building histogram",
                )

            neutron, xray = dataset.get_volume_at_time(timepoint)
            valid = build_valid_mask(neutron, xray, self.validity_policy)
            cache = build_histogram_cache(
                neutron, xray, self.neutron_edges, self.xray_edges,
                valid_mask=valid,
                store_bin_index=self.store_bin_index or self.mrf is not None,
            )

            drift = None
            if self.drift_tracker is not None and self.drift_tracker.is_fitted:
                drift = self.drift_tracker.estimate(
                    cache, timepoint, previous=previous_drift
                )
                previous_drift = drift

            prior = self.temporal.prior_for(
                timepoint, self.prior, previous_fit, drift
            )
            if progress_callback:
                progress_callback(
                    int(100 * (position + 0.4) / max(len(indices), 1)),
                    f"Timepoint {timepoint + 1}/{len(indices)}: fitting mixture",
                )
            fit = self.mixture.fit(
                cache, prior,
                initial_means=None if previous_fit is None else previous_fit.means,
                initial_covariances=(
                    None if previous_fit is None else previous_fit.covariances
                ),
                cancel_check=cancel_check,
            )
            fit.timepoint = timepoint
            fit.drift = drift
            transition = self.temporal.post_fit(timepoint, fit, previous_fit)

            if progress_callback:
                progress_callback(
                    int(100 * (position + 0.7) / max(len(indices), 1)),
                    f"Timepoint {timepoint + 1}/{len(indices)}: spatial refinement",
                )
            labels, mrf_diagnostics = self._label_volume(
                cache, fit, neutron, xray, valid, cancel_check
            )

            fractions = {}
            if self.mixels and cache.bin_index is not None:
                rows = cache.row_index_volume()
                for name, per_bin in fractional_maps(
                    fit, cache, self.mixels
                ).items():
                    fractions[name] = cache.expand_to_voxels(
                        per_bin.astype(np.float32), fill=np.float32("nan"),
                        rows=rows,
                    )

            counts = {
                name: int(np.count_nonzero(labels == index))
                for index, name in enumerate(fit.names)
            }
            result.timepoints.append(
                TimepointResult(
                    timepoint=timepoint,
                    labels=labels,
                    class_names=list(fit.names),
                    voxel_counts=counts,
                    fit=fit,
                    drift=drift,
                    mrf=mrf_diagnostics,
                    transition=transition,
                    validity=validity_report(neutron, xray, self.validity_policy),
                    fractions=fractions,
                    unassigned_voxels=int(np.count_nonzero(labels == UNASSIGNED)),
                )
            )
            previous_fit = fit

        if self.mixels and previous_fit is not None:
            result.mixel_report = verify_mixels(previous_fit, self.mixels)
        if self.mrf is not None:
            result.pairwise_cost = self.mrf.pairwise
        if progress_callback:
            progress_callback(100, "Segmentation complete")
        return result

    # ── labelling ────────────────────────────────────────────────────────
    def _label_volume(self, cache, fit, neutron, xray, valid, cancel_check):
        """Per-bin posterior → per-voxel labels, spatially regularised."""
        n_components = fit.n_components
        log_posterior = np.log(
            np.maximum(fit.responsibilities[:, :n_components], 1e-300)
        ).astype(np.float32)

        reject = self.mixture.reject_margin
        best = fit.responsibilities[:, :n_components].max(axis=1)
        rejected_bins = np.zeros(best.shape, dtype=bool)
        if fit.has_outlier:
            rejected_bins |= (
                fit.responsibilities[:, -1] > best
            )
        if reject is not None:
            rejected_bins |= best < float(reject)

        if cache.bin_index is None:
            # No spatial pass is possible without the per-voxel lookup
            raise RuntimeError(
                "Labelling needs the per-voxel bin index; build the cache "
                "with store_bin_index=True"
            )
        rows = cache.row_index_volume()

        if self.mrf is None:
            per_bin = np.argmax(log_posterior, axis=1).astype(np.int32)
            per_bin[rejected_bins] = UNASSIGNED
            labels = cache.expand_to_voxels(
                per_bin, fill=np.int32(UNASSIGNED), rows=rows
            ).astype(np.int32)
            labels[~valid] = UNASSIGNED
            return labels, None

        scores = UnaryScores(log_posterior, rows)
        labels, diagnostics = self.mrf.refine(
            scores, neutron=neutron, xray=xray,
            valid_mask=valid & (rows >= 0),
            method=self.mrf_method, cancel_check=cancel_check,
        )
        # A bin the model declined stays declined even if the MRF would have
        # smoothed it into a neighbour: abstention is information.
        if rejected_bins.any():
            declined = cache.expand_to_voxels(
                rejected_bins, fill=False, rows=rows
            )
            labels = np.where(declined, np.int32(UNASSIGNED), labels)
        labels[~valid] = UNASSIGNED
        return labels.astype(np.int32), diagnostics
