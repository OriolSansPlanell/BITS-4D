"""Model-based time-series segmentation for paired 3-D/4-D datasets.

A histogram ROI drawn once at T0 is a *frozen* partition of the
(neutron, X-ray) plane: it cannot follow the instrumental drift that every
long series carries, and it forces a hard label onto voxels that are
physically a mixture of two phases. This package replaces the frozen
partition with a model that keeps the manual work as a prior rather than as
a hard constraint:

    p(labels, fractions | data)  ∝  mixture likelihood
                                 ×  spatial prior
                                 ×  temporal transition

* :mod:`model.mixture` — the mixture, anchored on the ROI moments.
* :mod:`model.spatial_prior` — the MRF, whose costs are learned from the
  T0 label volume.
* :mod:`model.temporal` — how a component may move between timepoints.
* :mod:`model.partial_volume` — mixing lines modelled as fractions.
* :mod:`model.segmenter` — the HMRF-EM loop that runs them.

with three prerequisites the rest depends on:

* :mod:`model.validity` — which voxels are measurements at all.
* :mod:`model.histogram_cache` — per-bin sufficient statistics.
* :mod:`model.drift_tracker` — instrumental drift from inert anchor classes.

Everything here is GUI-independent and scriptable.
"""

from model.drift_tracker import DriftEstimate, DriftTracker, estimate_process_noise
from model.histogram_cache import (
    HistogramCache,
    build_histogram_cache,
    cache_from_histogram_data,
    moments_from_mask,
)
from model.mixture import (
    ComponentPrior,
    FitResult,
    MixturePrior,
    ROIAnchoredMixture,
    anchor_strength_to_kappa,
)
from model.health_check import Finding, HealthReport, Status, run_health_check
from model.likelihood import (
    UNCLASSIFIED,
    ClassLibrary,
    MatchTable,
    MaterialClass,
    match_table,
)
from model.locked import (
    SMOOTHING_GRID,
    LockedSegmenter,
    SegmentationRefused,
    SeriesSegmentation,
    TimepointSegmentation,
)
from model.partial_volume import (
    MixelComponent,
    build_mixel_ladder,
    detect_mixing_lines,
    fractional_maps,
    verify_mixels,
)
from model.segmenter import (
    SegmentationResult,
    SequentialSegmenter,
    TimepointResult,
)
from model.spatial_prior import ROIDerivedMRF, UnaryScores, potts_cost
from model.temporal import DriftTransition, StaticTransition, TemporalModel
from model.validity import (
    ValidityPolicy,
    auto_floor,
    build_valid_mask,
    channel_coverage,
    estimate_floor,
    find_acquisition_steps,
    validity_report,
)

__all__ = [
    "ValidityPolicy",
    "build_valid_mask",
    "auto_floor",
    "channel_coverage",
    "estimate_floor",
    "MaterialClass",
    "ClassLibrary",
    "MatchTable",
    "match_table",
    "UNCLASSIFIED",
    "LockedSegmenter",
    "SeriesSegmentation",
    "TimepointSegmentation",
    "SegmentationRefused",
    "SMOOTHING_GRID",
    "run_health_check",
    "HealthReport",
    "Finding",
    "Status",
    "validity_report",
    "find_acquisition_steps",
    "HistogramCache",
    "build_histogram_cache",
    "cache_from_histogram_data",
    "moments_from_mask",
    "DriftTracker",
    "DriftEstimate",
    "estimate_process_noise",
    "ROIAnchoredMixture",
    "MixturePrior",
    "ComponentPrior",
    "FitResult",
    "anchor_strength_to_kappa",
    "ROIDerivedMRF",
    "UnaryScores",
    "potts_cost",
    "TemporalModel",
    "DriftTransition",
    "StaticTransition",
    "MixelComponent",
    "build_mixel_ladder",
    "detect_mixing_lines",
    "fractional_maps",
    "verify_mixels",
    "SequentialSegmenter",
    "SegmentationResult",
    "TimepointResult",
]
