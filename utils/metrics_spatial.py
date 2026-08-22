"""Metrics computed in the volume rather than in the histogram plane.

Every quantity in :mod:`utils.histogram_metrics` lives in the
(neutron, X-ray) plane. That is a real gap, because a segmentation failure is
usually a *spatial* one: a class that is right in aggregate but scattered
into a thousand speckles, a deposit whose centre of mass has walked across
the sample, a rim that belongs to a boundary rather than to a phase. None of
those change the histogram much, so a histogram-only metric set cannot see
them at all.

The decisive diagnostic
───────────────────────
:func:`disagreement_topology` answers the question two segmentations of the
same data always raise: *is this a real disagreement, or a boundary?* Erode
the disagreeing voxels and watch what survives. A shell one or two voxels
thick vanishes immediately — the methods agree about where the material is
and differ only on partial-volume voxels, whose membership is fractional
anyway. Compact clumps that survive three or four erosions are a genuine
disagreement about the interior, and worth investigating.

Rows use the same long-format schema as the histogram metrics, so the two
CSVs concatenate. Pass :data:`SPATIAL_METRIC_INFO` (or the merged registry
from :func:`combined_registry`) to
:func:`utils.histogram_metrics.write_metrics_csv`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from utils.histogram_metrics import (
    METRIC_INFO,
    PER_CLASS_METRICS,
    SCALAR_METRICS,
    MetricsRow,
)

SPATIAL_METRIC_INFO: Dict[str, Dict[str, str]] = {
    "com_z_k": {
        "label": "Centre of mass, Z",
        "unit": "voxels",
        "meaning": "Where the class sits along Z",
        "better": "n/a",
    },
    "com_y_k": {
        "label": "Centre of mass, Y",
        "unit": "voxels",
        "meaning": "Where the class sits along Y",
        "better": "n/a",
    },
    "com_x_k": {
        "label": "Centre of mass, X",
        "unit": "voxels",
        "meaning": "Where the class sits along X",
        "better": "n/a",
    },
    "com_drift_k": {
        "label": "Centre-of-mass drift vs first timepoint",
        "unit": "voxels",
        "meaning": "How far the class has physically moved since T0",
        "better": "n/a",
    },
    "rg_k": {
        "label": "Radius of gyration",
        "unit": "voxels",
        "meaning": "How spread out the class is in space; grows when a "
                   "compact deposit becomes diffuse",
        "better": "n/a",
    },
    "n_components_k": {
        "label": "Connected components",
        "unit": "count",
        "meaning": "Speckle count; a coherent phase is a few components, a "
                   "noisy segmentation is thousands",
        "better": "lower",
    },
    "largest_frac_k": {
        "label": "Largest component fraction",
        "unit": "fraction in [0, 1]",
        "meaning": "Share of the class in its single biggest piece",
        "better": "higher",
    },
    "sa_vol_k": {
        "label": "Surface-to-volume ratio",
        "unit": "faces per voxel",
        "meaning": "Roughness of the class boundary; speckle drives it up",
        "better": "lower",
    },
    "interface_kl": {
        "label": "Interface area between two classes",
        "unit": "faces",
        "meaning": "Shared surface between a class pair — the quantity that "
                   "governs reaction kinetics at a boundary",
        "better": "n/a",
    },
    "f_rind": {
        "label": "Disagreement that is a boundary rind",
        "unit": "fraction in [0, 1]",
        "meaning": "Share of disagreeing voxels that erode away within two "
                   "voxels; near 1 means the methods agree on morphology",
        "better": "higher",
    },
    "n_interior_components": {
        "label": "Interior disagreement clumps",
        "unit": "count",
        "meaning": "Compact disagreements surviving erosion — genuine "
                   "conflict about where the material is",
        "better": "lower",
    },
    "disagreement_voxels": {
        "label": "Disagreeing voxels",
        "unit": "voxels",
        "meaning": "Total voxels the two segmentations label differently",
        "better": "lower",
    },
}

SPATIAL_PER_CLASS_METRICS = (
    "com_z_k", "com_y_k", "com_x_k", "com_drift_k", "rg_k",
    "n_components_k", "largest_frac_k", "sa_vol_k", "interface_kl",
)
SPATIAL_SCALAR_METRICS = (
    "f_rind", "n_interior_components", "disagreement_voxels",
)


def combined_registry():
    """``(metric_info, scalar_metrics, per_class_metrics)`` for both modules."""
    info = dict(METRIC_INFO)
    info.update(SPATIAL_METRIC_INFO)
    return (
        info,
        tuple(SCALAR_METRICS) + SPATIAL_SCALAR_METRICS,
        tuple(PER_CLASS_METRICS) + SPATIAL_PER_CLASS_METRICS,
    )


def _structure(connectivity: int = 1):
    from scipy.ndimage import generate_binary_structure

    return generate_binary_structure(3, connectivity)


# ── per-class spatial descriptors ────────────────────────────────────────────

def class_spatial_metrics(mask, connectivity: int = 1) -> Dict[str, float]:
    """Shape descriptors of one class mask."""
    mask_bool = np.asarray(mask, dtype=bool)
    total = int(np.count_nonzero(mask_bool))
    if total == 0:
        return {}

    coordinates = np.array(np.nonzero(mask_bool), dtype=np.float64)
    centre = coordinates.mean(axis=1)
    offsets = coordinates - centre[:, None]
    radius_of_gyration = float(np.sqrt(np.mean(np.sum(offsets ** 2, axis=0))))

    from scipy.ndimage import label as connected_label

    labelled, count = connected_label(mask_bool, structure=_structure(connectivity))
    if count > 0:
        sizes = np.bincount(labelled.reshape(-1))[1:]
        largest = float(sizes.max()) / total
    else:
        largest = 0.0

    return {
        "com_z_k": float(centre[0]),
        "com_y_k": float(centre[1]),
        "com_x_k": float(centre[2]),
        "rg_k": radius_of_gyration,
        "n_components_k": float(count),
        "largest_frac_k": largest,
        "sa_vol_k": surface_area(mask_bool) / total,
    }


def surface_area(mask) -> float:
    """Internal faces where the mask meets something else. 6-connectivity."""
    mask_bool = np.asarray(mask, dtype=bool)
    faces = 0
    for axis in range(mask_bool.ndim):
        if mask_bool.shape[axis] < 2:
            continue
        low = [slice(None)] * mask_bool.ndim
        high = [slice(None)] * mask_bool.ndim
        low[axis] = slice(0, -1)
        high[axis] = slice(1, None)
        faces += int(
            np.count_nonzero(mask_bool[tuple(low)] != mask_bool[tuple(high)])
        )
    return float(faces)


def interface_area(mask_a, mask_b) -> float:
    """Faces shared between two classes — the contact area between phases.

    Only voxels belonging to exactly one of the two masks count, so overlap
    is ignored and a mask compared with itself has no interface rather than
    reporting its own interior faces.
    """
    first = np.asarray(mask_a, dtype=bool)
    second = np.asarray(mask_b, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Both masks must have the same shape")
    shared = first & second
    first = first & ~shared
    second = second & ~shared
    faces = 0
    for axis in range(first.ndim):
        if first.shape[axis] < 2:
            continue
        low = [slice(None)] * first.ndim
        high = [slice(None)] * first.ndim
        low[axis] = slice(0, -1)
        high[axis] = slice(1, None)
        low_key, high_key = tuple(low), tuple(high)
        faces += int(np.count_nonzero(first[low_key] & second[high_key]))
        faces += int(np.count_nonzero(second[low_key] & first[high_key]))
    return float(faces)


# ── the rind-vs-blob diagnostic ──────────────────────────────────────────────

def disagreement_topology(
    mask_a,
    mask_b,
    max_erosion: int = 4,
    rind_erosions: int = 2,
    min_interior_size: int = 8,
    connectivity: int = 1,
) -> Dict[str, float]:
    """Is the disagreement between two masks a boundary rind or a real blob?

    Erodes the symmetric difference by one voxel at a time and records how
    much survives. ``f_rind`` is the share gone by *rind_erosions* — close to
    1 means the two methods place the material in the same place and differ
    only on its shell, where membership is fractional in any case.
    ``n_interior_components`` counts the compact survivors, which are the
    disagreements actually worth investigating.

    Read the two together. ``f_rind`` is scale-dependent: a genuinely
    displaced *small* object erodes away in two voxels just as a shell does,
    and will score as a rind. ``n_interior_components`` does not have that
    problem, so a high ``f_rind`` with a non-zero component count is still a
    real disagreement — about something thin.
    """
    from scipy.ndimage import binary_erosion, label as connected_label

    first = np.asarray(mask_a, dtype=bool)
    second = np.asarray(mask_b, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Both masks must have the same shape")

    difference = first ^ second
    total = int(np.count_nonzero(difference))
    result = {
        "disagreement_voxels": float(total),
        "f_rind": float("nan"),
        "n_interior_components": float("nan"),
    }
    if total == 0:
        result["f_rind"] = 1.0
        result["n_interior_components"] = 0.0
        return result

    structure = _structure(connectivity)
    survival = []
    eroded = difference
    # Both reported numbers describe the same erosion depth: the fraction
    # gone by then, and how many clumps are still standing. Counting the
    # clumps deeper than that would report zero for anything thin, exactly
    # where the fraction is least informative.
    interior = None
    for step in range(1, max_erosion + 1):
        eroded = binary_erosion(eroded, structure=structure)
        remaining = int(np.count_nonzero(eroded))
        survival.append(remaining / total)
        result[f"survival_{step}"] = remaining / total
        if step == rind_erosions:
            result["f_rind"] = 1.0 - remaining / total
            interior = eroded

    if interior is None:
        interior = eroded
        if np.isnan(result["f_rind"]) and survival:
            result["f_rind"] = 1.0 - survival[-1]

    labelled, count = connected_label(interior, structure=structure)
    if count > 0:
        sizes = np.bincount(labelled.reshape(-1))[1:]
        result["n_interior_components"] = float(
            int(np.count_nonzero(sizes >= min_interior_size))
        )
    else:
        result["n_interior_components"] = 0.0
    return result


# ── assembling rows ──────────────────────────────────────────────────────────

def spatial_metrics_rows(
    masks_by_timepoint: Dict[int, Dict[str, np.ndarray]],
    connectivity: int = 1,
    interfaces: bool = True,
    max_interface_pairs: int = 45,
) -> list:
    """One :class:`MetricsRow` per timepoint of spatial descriptors.

    *masks_by_timepoint* is ``{timepoint: {class name: boolean mask}}``.
    Centre-of-mass drift is measured against the first timepoint a class
    appears in.
    """
    rows = []
    reference_centres: Dict[str, np.ndarray] = {}

    for timepoint in sorted(masks_by_timepoint):
        masks = masks_by_timepoint[timepoint]
        row = MetricsRow(scope="timepoint", timepoint=timepoint)
        per_class: Dict[str, Dict[str, float]] = {}

        for name, mask in masks.items():
            values = class_spatial_metrics(mask, connectivity=connectivity)
            if not values:
                continue
            centre = np.array(
                [values["com_z_k"], values["com_y_k"], values["com_x_k"]]
            )
            if name not in reference_centres:
                reference_centres[name] = centre
            values["com_drift_k"] = float(
                np.linalg.norm(centre - reference_centres[name])
            )
            for metric, value in values.items():
                per_class.setdefault(metric, {})[name] = value

        if interfaces:
            names = sorted(masks)
            pairs = 0
            for index, first in enumerate(names):
                for second in names[index + 1:]:
                    if pairs >= max_interface_pairs:
                        break
                    area = interface_area(masks[first], masks[second])
                    if area > 0:
                        per_class.setdefault("interface_kl", {})[
                            f"{first}|{second}"
                        ] = area
                    pairs += 1

        row.per_class = per_class
        rows.append(row)
    return rows


def comparison_rows(
    masks_a: Dict[int, Dict[str, np.ndarray]],
    masks_b: Dict[int, Dict[str, np.ndarray]],
    max_erosion: int = 4,
) -> list:
    """Rind-vs-blob rows comparing two segmentations of the same series.

    Emitted as scalar metrics per timepoint (aggregated over classes) plus a
    per-class breakdown of the disagreeing voxel count.
    """
    rows = []
    for timepoint in sorted(set(masks_a) & set(masks_b)):
        row = MetricsRow(scope="timepoint", timepoint=timepoint)
        per_class: Dict[str, Dict[str, float]] = {}
        rind_values = []
        interior = 0.0
        disagreement = 0.0

        for name in sorted(set(masks_a[timepoint]) & set(masks_b[timepoint])):
            topology = disagreement_topology(
                masks_a[timepoint][name], masks_b[timepoint][name],
                max_erosion=max_erosion,
            )
            per_class.setdefault("disagreement_voxels", {})[name] = topology[
                "disagreement_voxels"
            ]
            if np.isfinite(topology["f_rind"]):
                rind_values.append(topology["f_rind"])
            if np.isfinite(topology["n_interior_components"]):
                interior += topology["n_interior_components"]
            disagreement += topology["disagreement_voxels"]

        row.scalars["f_rind"] = (
            float(np.mean(rind_values)) if rind_values else None
        )
        row.scalars["n_interior_components"] = interior
        row.scalars["disagreement_voxels"] = disagreement
        row.per_class = per_class
        rows.append(row)
    return rows
