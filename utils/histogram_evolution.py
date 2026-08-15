"""Temporal histogram analyses on the shared (neutron, X-ray) bin grid.

Three views of how the joint histogram evolves:

* **cumulative** — each timepoint against the first,
  ``log10(h_t + 1) - log10(h_0 + 1)``. Shows total drift from the starting
  state.
* **incremental** — each timepoint against the one before it,
  ``log10(h_t + 1) - log10(h_{t-1} + 1)``. Shows *when* changes happen; a
  slow steady drift looks small here while a sudden event stands out.
* **marginal kymographs** — each modality's 1-D histogram stacked against
  time. One dimension down, so it separates a shift in neutron from a shift
  in X-ray, which the joint view can hide.

In every case red means bins gained voxels, blue means they lost them.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Reference modes for the panel figures
REFERENCE_FIRST = "first"
REFERENCE_PREVIOUS = "previous"


def compute_log_differences(
    histograms: Sequence,
    reference: str = REFERENCE_FIRST,
) -> list:
    """Log10 count differences between timepoints.

    With ``reference="first"`` returns log10(h_t+1) - log10(h_0+1) for every
    t >= 1; with ``reference="previous"`` returns
    log10(h_t+1) - log10(h_{t-1}+1) for every t >= 1.
    """
    if reference not in (REFERENCE_FIRST, REFERENCE_PREVIOUS):
        raise ValueError(
            f"reference must be {REFERENCE_FIRST!r} or {REFERENCE_PREVIOUS!r}"
        )
    if len(histograms) < 2:
        raise ValueError(
            "Histogram evolution needs at least two timepoints"
        )

    logs = [
        np.log10(h.histogram.astype(np.float64) + 1.0) for h in histograms
    ]
    if reference == REFERENCE_FIRST:
        return [current - logs[0] for current in logs[1:]]
    return [current - previous for previous, current in zip(logs, logs[1:])]


def save_histogram_evolution_image(
    histograms: Sequence,
    output_path: str,
    max_columns: int = 4,
    dpi: int = 150,
    reference_mode: str = REFERENCE_FIRST,
) -> str:
    """Render the evolution figure and save it as an image file.

    Parameters
    ----------
    histograms : sequence of HistogramData
        One local histogram per timepoint, in temporal order. All share the
        global bin grid, so their edges are identical.
    output_path : str
        Destination image path (extension selects the format, e.g. .png).
    reference_mode : "first" or "previous"
        Compare each timepoint with T0 (cumulative drift) or with the
        preceding timepoint (per-step change).

    The first panel shows the log-scaled T0 histogram for context; every
    following panel shows a log-difference with a symmetric diverging colour
    scale shared across panels.
    """
    differences = compute_log_differences(histograms, reference_mode)
    reference = histograms[0]
    extent = [
        reference.x_edges[0], reference.x_edges[-1],
        reference.y_edges[0], reference.y_edges[-1],
    ]

    limit = max(float(np.max(np.abs(d))) for d in differences)
    if limit <= 0:
        limit = 1e-6

    num_panels = len(differences) + 1  # + reference panel
    columns = max(1, min(int(max_columns), num_panels))
    rows = math.ceil(num_panels / columns)

    fig = Figure(figsize=(4.4 * columns, 3.8 * rows), dpi=dpi)
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, columns, squeeze=False)

    # Reference panel
    ref_ax = axes[0][0]
    ref_image = ref_ax.imshow(
        np.log10(reference.histogram.astype(np.float64) + 1.0),
        extent=extent, origin='lower', aspect='auto',
        cmap='viridis', interpolation='nearest',
    )
    ref_ax.set_title("T0 reference (log10 counts)")
    ref_ax.set_xlabel("Neutron Intensity")
    ref_ax.set_ylabel("X-ray Intensity")
    fig.colorbar(ref_image, ax=ref_ax, fraction=0.046)

    incremental = reference_mode == REFERENCE_PREVIOUS
    diff_image = None
    for index, diff in enumerate(differences):
        ax = axes[(index + 1) // columns][(index + 1) % columns]
        diff_image = ax.imshow(
            diff, extent=extent, origin='lower', aspect='auto',
            cmap='RdBu_r', vmin=-limit, vmax=limit,
            interpolation='nearest',
        )
        ax.set_title(
            f"T{index + 1} − T{index}" if incremental
            else f"T{index + 1} − T0"
        )
        ax.set_xlabel("Neutron Intensity")
        ax.set_ylabel("X-ray Intensity")

    # Hide unused grid cells
    for index in range(num_panels, rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    if diff_image is not None:
        fig.colorbar(
            diff_image,
            ax=[axes[i // columns][i % columns]
                for i in range(1, num_panels)],
            fraction=0.02,
            label=(
                "log10 count change vs previous timepoint" if incremental
                else "log10 count change vs T0"
            ),
        )

    fig.suptitle(
        "Histogram change per step (log scale, each timepoint vs the previous)"
        if incremental else
        "Histogram evolution (log scale, normalized to first timepoint)",
        fontsize=14,
    )
    fig.savefig(output_path, bbox_inches="tight")
    return output_path


def compute_marginals(histograms: Sequence) -> tuple:
    """Per-timepoint 1-D marginals of the joint histogram.

    Each timepoint is count-normalized first so that timepoints with
    different numbers of finite voxels stay comparable.

    Returns ``(neutron_marginals, xray_marginals)``, both shaped
    ``(T, bins)``. ``HistogramData.histogram`` is stored as
    ``[xray_bin, neutron_bin]``, so the neutron marginal sums over the X-ray
    axis (0) and the X-ray marginal sums over the neutron axis (1).
    """
    if len(histograms) < 2:
        raise ValueError("Marginal evolution needs at least two timepoints")

    neutron = []
    xray = []
    for hist in histograms:
        counts = hist.histogram.astype(np.float64)
        total = counts.sum()
        if total > 0:
            counts = counts / total
        neutron.append(counts.sum(axis=0))
        xray.append(counts.sum(axis=1))
    return np.asarray(neutron), np.asarray(xray)


def compute_marginal_changes(
    marginals: np.ndarray,
    reference: str = REFERENCE_FIRST,
) -> np.ndarray:
    """log2 change of a marginal stack, per timepoint.

    With ``reference="first"`` every row is compared with T0, giving
    cumulative drift and a first row of zeros. With ``reference="previous"``
    each row is compared with the row before it, giving the per-step change;
    the first row is NaN because it has no predecessor.

    Bins that are empty in the denominator become NaN rather than infinity,
    so they are drawn blank instead of saturating the colour scale.
    """
    marginals = np.asarray(marginals, dtype=np.float64)
    if reference == REFERENCE_FIRST:
        denominator = np.broadcast_to(marginals[0], marginals.shape)
        numerator = marginals
    elif reference == REFERENCE_PREVIOUS:
        denominator = marginals[:-1]
        numerator = marginals[1:]
    else:
        raise ValueError(
            f"reference must be {REFERENCE_FIRST!r} or {REFERENCE_PREVIOUS!r}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        change = np.log2(numerator / denominator)
    change = np.where(np.isfinite(change), change, np.nan)

    if reference == REFERENCE_PREVIOUS:
        # Keep one row per timepoint so the time axis still lines up; T0 has
        # no predecessor and stays blank.
        leading = np.full((1, marginals.shape[1]), np.nan)
        change = np.vstack([leading, change])
    return change


def save_marginal_evolution_image(
    histograms: Sequence,
    output_path: str,
    dpi: int = 150,
    reference_mode: str = REFERENCE_FIRST,
) -> str:
    """Save marginal kymographs: each modality's 1-D histogram versus time.

    Two panels (neutron, X-ray) with time on the x-axis and intensity on the
    y-axis, coloured by the log2 change of each intensity band. Red bands
    grew, blue bands shrank.

    ``reference_mode="first"`` compares every timepoint with T0 (cumulative
    drift); ``"previous"`` compares each timepoint with the one before it, so
    the steps where a band actually moves stand out. Bins empty in the
    denominator are left blank rather than saturating the scale, and the
    colour range is the 99th percentile of |change| so a few extreme bins do
    not flatten the rest.
    """
    neutron_marginals, xray_marginals = compute_marginals(histograms)
    reference = histograms[0]
    num_timepoints = len(histograms)
    incremental = reference_mode == REFERENCE_PREVIOUS

    fig = Figure(figsize=(13, 4.8), dpi=dpi)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, squeeze=False)[0]

    panels = (
        (axes[0], neutron_marginals, reference.x_centers, "Neutron Intensity"),
        (axes[1], xray_marginals, reference.y_centers, "X-ray Intensity"),
    )
    for ax, marginals, centers, label in panels:
        change = compute_marginal_changes(marginals, reference_mode)

        finite = np.abs(change[np.isfinite(change)])
        limit = float(np.percentile(finite, 99)) if finite.size else 0.0
        if not limit > 0:
            limit = 1.0

        image = ax.imshow(
            change.T, origin="lower", aspect="auto", cmap="RdBu_r",
            vmin=-limit, vmax=limit, interpolation="nearest",
            extent=[-0.5, num_timepoints - 0.5, centers[0], centers[-1]],
        )
        ax.set_xlabel("Timepoint")
        ax.set_ylabel(label)
        ax.set_title(
            f"{label} marginal — log2 vs previous" if incremental
            else f"{label} marginal — log2 vs T0"
        )
        fig.colorbar(image, ax=ax, label="log2 population change")

    fig.suptitle(
        "Marginal change per step (each timepoint vs the previous)"
        if incremental else
        "Marginal evolution of neutron and X-ray intensities over time",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return output_path
