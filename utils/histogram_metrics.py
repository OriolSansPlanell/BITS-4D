"""Ground-truth-free metrics for the bimodal histogram and its classes.

Which metrics are here, and why
───────────────────────────────
The reference implementation (``metrics_table.py`` /
``metrics_table_morphology.py``) splits its metrics by whether they need a
ground-truth label volume. Only the ones that do **not** are reproduced
here, because measured data has no phantom to anchor on:

* **Shape metrics** — ``S_h``, ``S_v``, ``S_d``, ``A_x`` come from the
  histogram alone. ``Delta_n`` needs a *reference* rather than ground
  truth; for a time series the natural reference is the first timepoint,
  so it reports how far the neutron marginal has shifted since then.
* **Class metrics** — ``sigma_x_k``, ``sigma_n_k``, ``E_k`` and the
  Davies–Bouldin index ``DB`` are computed over the *segmentation classes*
  the user created. In the reference these are flagged "ground truth
  (matching only)" because there they describe GMM components matched to
  known materials; measured against classes that already exist they are
  ordinary internal cluster indices and need no ground truth.

Deliberately **not** included: ``eps_k``, ``CE`` and ``O_ab``. All three
measure distance to *known* material positions or labels, which do not
exist for measured data — reporting them would mean inventing a reference.
``CD`` below is the honest time-series analogue of ``CE``: instead of the
error against a known position it reports how far each class centroid has
drifted from the first timepoint.

Orientation
───────────
``HistogramData.histogram`` is ``[X-ray bin, neutron bin]`` and is drawn
with neutron on the x-axis and X-ray on the y-axis. "Horizontal" and
"vertical" below refer to that display, so ``S_h`` is spread along neutron
at fixed X-ray. (The reference code puts X-ray on the horizontal axis, so
its S_h/S_v are the mirror of these.)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

EPSILON = 1e-12

# name -> (label, unit, what it means, direction that is "better")
METRIC_INFO: Dict[str, Dict[str, str]] = {
    "S_h": {
        "label": "Horizontal streak score",
        "unit": "dimensionless",
        "meaning": "Spread along neutron at a fixed X-ray value; flags "
                   "misalignment between the two modalities",
        "better": "lower",
    },
    "S_v": {
        "label": "Vertical streak score",
        "unit": "dimensionless",
        "meaning": "Spread along X-ray at a fixed neutron value; flags ring "
                   "artifacts",
        "better": "lower",
    },
    "S_d": {
        "label": "Diagonal smear score",
        "unit": "Pearson rho in [-1, 1]",
        "meaning": "Correlation between the modalities; flags beam hardening "
                   "or scatter",
        "better": "closer to zero",
    },
    "A_x": {
        "label": "X-ray marginal asymmetry",
        "unit": "(mean - median) / sigma",
        "meaning": "Skew of the X-ray marginal; flags cupping",
        "better": "closer to zero",
    },
    "Delta_n": {
        "label": "Neutron marginal shift vs first timepoint",
        "unit": "intensity",
        "meaning": "Drift of the mean neutron intensity since T0; flags "
                   "scatter build-up or sample change",
        "better": "closer to zero",
    },
    "DB": {
        "label": "Davies-Bouldin index (over segmentation classes)",
        "unit": "dimensionless",
        "meaning": "Class separability in the histogram plane",
        "better": "lower",
    },
    "CD": {
        "label": "Mean class centroid drift vs first timepoint",
        "unit": "intensity",
        "meaning": "How far the classes have moved in the histogram since T0",
        "better": "lower",
    },
    "n_classes": {
        "label": "Classes measured",
        "unit": "count",
        "meaning": "Number of segmentation classes contributing",
        "better": "n/a",
    },
    "sigma_n_k": {
        "label": "Class spread along neutron",
        "unit": "intensity",
        "meaning": "Per-class standard deviation of neutron intensity",
        "better": "lower",
    },
    "sigma_x_k": {
        "label": "Class spread along X-ray",
        "unit": "intensity",
        "meaning": "Per-class standard deviation of X-ray intensity",
        "better": "lower",
    },
    "E_k": {
        "label": "Class elongation",
        "unit": "dimensionless (>= 1)",
        "meaning": "Anisotropy of a class in the histogram plane",
        "better": "lower",
    },
    "centroid_n_k": {
        "label": "Class centroid, neutron",
        "unit": "intensity",
        "meaning": "Mean neutron intensity of the class",
        "better": "n/a",
    },
    "centroid_x_k": {
        "label": "Class centroid, X-ray",
        "unit": "intensity",
        "meaning": "Mean X-ray intensity of the class",
        "better": "n/a",
    },
    "drift_k": {
        "label": "Class centroid drift vs first timepoint",
        "unit": "intensity",
        "meaning": "Distance this class has moved in the histogram since T0",
        "better": "lower",
    },
    "voxels_k": {
        "label": "Class voxel count",
        "unit": "voxels",
        "meaning": "Segmented voxels in the class",
        "better": "n/a",
    },
}

SCALAR_METRICS = ("S_h", "S_v", "S_d", "A_x", "Delta_n", "DB", "CD", "n_classes")
PER_CLASS_METRICS = (
    "voxels_k", "centroid_n_k", "centroid_x_k",
    "sigma_n_k", "sigma_x_k", "E_k", "drift_k",
)


@dataclass
class MetricsRow:
    """Metrics for one scope — the global histogram or one timepoint."""

    scope: str                       # "global" or "timepoint"
    timepoint: Optional[int] = None
    scalars: Dict[str, Optional[float]] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "global" if self.timepoint is None else f"T{self.timepoint}"


# ── shape metrics, from the histogram alone ──────────────────────────────────

def _weighted_moments(counts: np.ndarray, centers: np.ndarray):
    """Mean and standard deviation of a binned 1-D distribution."""
    total = float(counts.sum())
    if total <= 0:
        return float("nan"), float("nan")
    mean = float(np.sum(counts * centers) / total)
    variance = float(np.sum(counts * (centers - mean) ** 2) / total)
    return mean, math.sqrt(max(variance, 0.0))


def _weighted_median(counts: np.ndarray, centers: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    cumulative = np.cumsum(counts) / total
    return float(centers[int(np.searchsorted(cumulative, 0.5))])


def compute_shape_metrics(
    histogram_data,
    reference=None,
) -> Dict[str, Optional[float]]:
    """Shape metrics of one bimodal histogram.

    *reference* is another ``HistogramData`` on the same grid (the first
    timepoint) used only for ``Delta_n``; without it that entry is None.
    """
    counts = np.asarray(histogram_data.histogram, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return {name: None for name in ("S_h", "S_v", "S_d", "A_x", "Delta_n")}

    density = counts / total
    # rows = X-ray bins, columns = neutron bins
    row_variance = density.var(axis=1)     # spread along neutron, fixed X-ray
    column_variance = density.var(axis=0)  # spread along X-ray, fixed neutron

    s_h = float(np.max(row_variance) / (np.mean(column_variance) + EPSILON))
    s_v = float(np.max(column_variance) / (np.mean(row_variance) + EPSILON))

    neutron_centers = np.asarray(histogram_data.x_centers, dtype=np.float64)
    xray_centers = np.asarray(histogram_data.y_centers, dtype=np.float64)
    neutron_marginal = counts.sum(axis=0)
    xray_marginal = counts.sum(axis=1)

    neutron_mean, neutron_sigma = _weighted_moments(
        neutron_marginal, neutron_centers
    )
    xray_mean, xray_sigma = _weighted_moments(xray_marginal, xray_centers)

    # Pearson correlation between the modalities, from the joint counts
    covariance = float(
        np.sum(
            density
            * (xray_centers[:, None] - xray_mean)
            * (neutron_centers[None, :] - neutron_mean)
        )
    )
    s_d = float(covariance / ((neutron_sigma * xray_sigma) + EPSILON))

    xray_median = _weighted_median(xray_marginal, xray_centers)
    a_x = float((xray_mean - xray_median) / (xray_sigma + EPSILON))

    delta_n = None
    if reference is not None:
        reference_counts = np.asarray(reference.histogram, dtype=np.float64)
        if reference_counts.sum() > 0:
            reference_mean, _ = _weighted_moments(
                reference_counts.sum(axis=0),
                np.asarray(reference.x_centers, dtype=np.float64),
            )
            delta_n = float(neutron_mean - reference_mean)

    return {"S_h": s_h, "S_v": s_v, "S_d": s_d, "A_x": a_x, "Delta_n": delta_n}


# ── class metrics, from the segmentation ─────────────────────────────────────

def compute_class_metrics(
    neutron_volume,
    xray_volume,
    class_masks: Dict[str, np.ndarray],
    reference_centroids: Optional[Dict[str, Sequence[float]]] = None,
    max_samples: int = 2_000_000,
) -> tuple:
    """Per-class statistics and the Davies–Bouldin index over the classes.

    Returns ``(scalars, per_class, centroids)``. *reference_centroids* maps a
    class name to its ``(neutron, xray)`` centroid at the first timepoint and
    produces the drift metrics; pass None for the reference timepoint itself.

    Classes larger than *max_samples* voxels are randomly subsampled — the
    mean, spread and elongation are unchanged within sampling error, and it
    keeps the cost bounded on large volumes.
    """
    neutron = np.asarray(neutron_volume)
    xray = np.asarray(xray_volume)
    rng = np.random.default_rng(0)

    per_class: Dict[str, Dict[str, float]] = {
        name: {} for name in PER_CLASS_METRICS
    }
    centroids: Dict[str, tuple] = {}
    spreads: Dict[str, float] = {}

    for name, mask in class_masks.items():
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != neutron.shape or not mask_bool.any():
            continue

        neutron_values = neutron[mask_bool].astype(np.float64, copy=False)
        xray_values = xray[mask_bool].astype(np.float64, copy=False)
        finite = np.isfinite(neutron_values) & np.isfinite(xray_values)
        neutron_values = neutron_values[finite]
        xray_values = xray_values[finite]
        if neutron_values.size == 0:
            continue

        voxels = int(neutron_values.size)
        if voxels > max_samples:
            chosen = rng.choice(voxels, max_samples, replace=False)
            neutron_values = neutron_values[chosen]
            xray_values = xray_values[chosen]

        centroid_n = float(np.mean(neutron_values))
        centroid_x = float(np.mean(xray_values))
        sigma_n = float(np.std(neutron_values))
        sigma_x = float(np.std(xray_values))
        largest = max(sigma_n, sigma_x)
        smallest = max(min(sigma_n, sigma_x), EPSILON)
        # A class with no spread at all is isotropic, not infinitely flat.
        elongation = 1.0 if largest <= EPSILON else largest / smallest

        per_class["voxels_k"][name] = float(voxels)
        per_class["centroid_n_k"][name] = centroid_n
        per_class["centroid_x_k"][name] = centroid_x
        per_class["sigma_n_k"][name] = sigma_n
        per_class["sigma_x_k"][name] = sigma_x
        per_class["E_k"][name] = float(elongation)

        centroids[name] = (centroid_n, centroid_x)
        spreads[name] = math.sqrt((sigma_n ** 2 + sigma_x ** 2) / 2.0)

        if reference_centroids and name in reference_centroids:
            reference_n, reference_x = reference_centroids[name]
            per_class["drift_k"][name] = float(
                math.hypot(centroid_n - reference_n, centroid_x - reference_x)
            )

    # Davies–Bouldin over the classes: mean over classes of the worst
    # (spread_i + spread_j) / distance_ij against every other class.
    names = list(centroids)
    davies_bouldin = None
    if len(names) >= 2:
        terms = []
        for i, name_i in enumerate(names):
            ratios = []
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                distance = math.dist(centroids[name_i], centroids[name_j])
                if distance > EPSILON:
                    ratios.append((spreads[name_i] + spreads[name_j]) / distance)
            if ratios:
                terms.append(max(ratios))
        if terms:
            davies_bouldin = float(np.mean(terms))

    drifts = list(per_class["drift_k"].values())
    scalars = {
        "DB": davies_bouldin,
        "CD": float(np.mean(drifts)) if drifts else None,
        "n_classes": float(len(names)),
    }
    return scalars, per_class, centroids


# ── output ───────────────────────────────────────────────────────────────────

def write_metrics_csv(rows: Sequence[MetricsRow], output_path) -> str:
    """Write every scalar and per-class value as one long-format CSV."""
    fieldnames = [
        "scope", "timepoint", "metric", "class", "value",
        "label", "unit", "meaning", "better_when",
    ]
    records = []
    for row in rows:
        for metric in SCALAR_METRICS:
            if metric not in row.scalars:
                continue
            info = METRIC_INFO[metric]
            records.append({
                "scope": row.scope,
                "timepoint": "" if row.timepoint is None else row.timepoint,
                "metric": metric,
                "class": "",
                "value": "" if row.scalars[metric] is None else row.scalars[metric],
                "label": info["label"],
                "unit": info["unit"],
                "meaning": info["meaning"],
                "better_when": info["better"],
            })
        for metric in PER_CLASS_METRICS:
            info = METRIC_INFO[metric]
            for class_name, value in row.per_class.get(metric, {}).items():
                records.append({
                    "scope": row.scope,
                    "timepoint": "" if row.timepoint is None else row.timepoint,
                    "metric": metric,
                    "class": class_name,
                    "value": value,
                    "label": info["label"],
                    "unit": info["unit"],
                    "meaning": info["meaning"],
                    "better_when": info["better"],
                })

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return str(output_path)


def plot_metric_evolution(rows: Sequence[MetricsRow], output_path,
                          dpi: int = 130) -> Optional[str]:
    """Plot every metric against time; one panel per metric.

    The global value, where one exists, is drawn as a dashed reference line.
    Returns None when there are fewer than two timepoints to plot.
    """
    timepoint_rows = sorted(
        (row for row in rows if row.scope == "timepoint"),
        key=lambda row: row.timepoint,
    )
    if len(timepoint_rows) < 2:
        return None

    global_row = next((row for row in rows if row.scope == "global"), None)
    timepoints = [row.timepoint for row in timepoint_rows]

    panels = []
    for metric in SCALAR_METRICS:
        if metric == "n_classes":
            continue
        values = [row.scalars.get(metric) for row in timepoint_rows]
        if any(value is not None and np.isfinite(value) for value in values):
            panels.append(("scalar", metric, values))
    for metric in PER_CLASS_METRICS:
        class_names = sorted({
            name for row in timepoint_rows for name in row.per_class.get(metric, {})
        })
        if class_names:
            panels.append(("per_class", metric, class_names))

    if not panels:
        return None

    columns = min(3, len(panels))
    plot_rows = math.ceil(len(panels) / columns)
    figure = Figure(figsize=(5.2 * columns, 3.6 * plot_rows), dpi=dpi)
    FigureCanvasAgg(figure)
    axes = figure.subplots(plot_rows, columns, squeeze=False)

    for index, (kind, metric, payload) in enumerate(panels):
        axis = axes[index // columns][index % columns]
        info = METRIC_INFO[metric]

        if kind == "scalar":
            series = [
                np.nan if value is None else value for value in payload
            ]
            axis.plot(timepoints, series, "o-", linewidth=1.8)
            if global_row is not None:
                global_value = global_row.scalars.get(metric)
                if global_value is not None and np.isfinite(global_value):
                    axis.axhline(
                        global_value, linestyle="--", color="grey",
                        linewidth=1.0, label="global",
                    )
                    axis.legend(fontsize=7)
        else:
            for class_name in payload:
                series = [
                    row.per_class.get(metric, {}).get(class_name, np.nan)
                    for row in timepoint_rows
                ]
                axis.plot(timepoints, series, "o-", linewidth=1.6,
                          label=class_name)
            axis.legend(fontsize=7)

        axis.set_title(f"{metric} — {info['label']}", fontsize=10)
        axis.set_xlabel("Timepoint")
        axis.set_ylabel(info["unit"])
        axis.grid(alpha=0.3)

    for index in range(len(panels), plot_rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    figure.suptitle(
        "Bimodal histogram and segmentation metrics over time", fontsize=14
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    return str(output_path)
