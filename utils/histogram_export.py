"""Saving per-class bimodal histograms to disk.

Each exported histogram shares the global histogram's bin grid, so the files
can be stacked and compared bin-for-bin across classes and timepoints. The
counts go out as ``.npy`` (exact, compact, the layout the analysis notebook
expects) alongside a ``.png`` for quick viewing, plus the shared bin edges
written once per export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Counts are stored as [xray_bin, neutron_bin] — the orientation used
# throughout the application, so row/column meaning never changes.
COUNTS_LAYOUT = "rows = X-ray bins, columns = neutron bins"


def sanitize_name(name: str) -> str:
    """Turn a class name into something safe for a file name."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(name).strip()
    )
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned or "class"


def save_bin_edges(histogram_data, output_dir) -> list:
    """Write the shared bin edges once per export. Returns file names."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    neutron_path = directory / "histogram_edges_neutron.npy"
    xray_path = directory / "histogram_edges_xray.npy"
    np.save(neutron_path, np.asarray(histogram_data.x_edges, dtype=np.float64))
    np.save(xray_path, np.asarray(histogram_data.y_edges, dtype=np.float64))

    readme = directory / "histogram_edges_README.txt"
    readme.write_text(
        "Bimodal (neutron vs X-ray) histograms exported per class and "
        "timepoint.\n\n"
        f"Counts arrays (*_hist.npy): {COUNTS_LAYOUT}.\n"
        "histogram_edges_neutron.npy : neutron bin edges (len = bins + 1)\n"
        "histogram_edges_xray.npy    : X-ray bin edges  (len = bins + 1)\n\n"
        "All histograms use these same edges, taken from the global "
        "histogram, so they can be compared or stacked directly.\n",
        encoding="utf-8",
    )
    return [neutron_path.name, xray_path.name, readme.name]


def save_class_histogram(
    histogram_data,
    output_path,
    title: str = "",
    write_image: bool = True,
    dpi: int = 130,
) -> list:
    """Save one class histogram as ``<output_path>.npy`` (+ ``.png``).

    *output_path* is a path without extension. Returns the file names written.
    """
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []

    counts_path = base.with_suffix(".npy")
    np.save(counts_path, np.asarray(histogram_data.histogram))
    written.append(counts_path.name)

    if not write_image:
        return written

    figure = Figure(figsize=(5.2, 4.4), dpi=dpi)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    image = axes.imshow(
        histogram_data.to_log_scale(),
        extent=[
            histogram_data.x_edges[0], histogram_data.x_edges[-1],
            histogram_data.y_edges[0], histogram_data.y_edges[-1],
        ],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    axes.set_xlabel("Neutron Intensity")
    axes.set_ylabel("X-ray Intensity")
    axes.set_title(title or "Class histogram")
    figure.colorbar(image, ax=axes, label="log10(counts + 1)")
    figure.tight_layout()

    image_path = base.with_suffix(".png")
    figure.savefig(image_path)
    written.append(image_path.name)
    return written
