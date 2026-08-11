"""Temporal histogram-evolution figure: log-scale change versus timepoint 0.

For each timepoint t the panel shows

    log10(h_t + 1) - log10(h_0 + 1)

on the shared (neutron, X-ray) histogram grid, i.e. the log-ratio of the
per-bin voxel counts relative to the first timepoint. Red regions gained
voxels over time, blue regions lost them, white is unchanged — a quick map
of *where in intensity space* the sample evolves.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def compute_log_differences(histograms: Sequence) -> list:
    """Return log10(h_t+1) - log10(h_0+1) for every t >= 1."""
    if len(histograms) < 2:
        raise ValueError(
            "Histogram evolution needs at least two timepoints"
        )
    reference = np.log10(histograms[0].histogram.astype(np.float64) + 1.0)
    return [
        np.log10(h.histogram.astype(np.float64) + 1.0) - reference
        for h in histograms[1:]
    ]


def save_histogram_evolution_image(
    histograms: Sequence,
    output_path: str,
    max_columns: int = 4,
    dpi: int = 150,
) -> str:
    """Render the evolution figure and save it as an image file.

    Parameters
    ----------
    histograms : sequence of HistogramData
        One local histogram per timepoint, in temporal order. All share the
        global bin grid, so their edges are identical.
    output_path : str
        Destination image path (extension selects the format, e.g. .png).

    The first panel shows the log-scaled reference histogram (T0); every
    following panel shows the log-difference of that timepoint against T0
    with a symmetric diverging colour scale shared across panels.
    """
    differences = compute_log_differences(histograms)
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

    diff_image = None
    for index, diff in enumerate(differences):
        ax = axes[(index + 1) // columns][(index + 1) % columns]
        diff_image = ax.imshow(
            diff, extent=extent, origin='lower', aspect='auto',
            cmap='RdBu_r', vmin=-limit, vmax=limit,
            interpolation='nearest',
        )
        ax.set_title(f"T{index + 1} − T0")
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
            label="log10 count change vs T0",
        )

    fig.suptitle(
        "Histogram evolution (log scale, normalized to first timepoint)",
        fontsize=14,
    )
    fig.savefig(output_path, bbox_inches="tight")
    return output_path
