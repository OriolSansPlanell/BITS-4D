"""
figure_export.py - Side-by-side "histogram + slice" figure

Renders the local bimodal (neutron/X-ray) histogram with every label's
selection drawn on top, next to the volume slice currently on screen with the
same labels highlighted. Both panels use the colours the application shows, so
a label is the same colour in the histogram and in the slice.

Pure matplotlib (no pyplot, no Qt): the main window gathers what is on screen
and hands it over, which keeps this testable without a display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon

#: File types the figure can be saved as (matplotlib picks the writer from
#: the extension).
SUPPORTED_FORMATS = (".svg", ".png", ".pdf", ".tif", ".tiff", ".jpg", ".jpeg")

#: What the figure is saved as unless another format is asked for: SVG keeps
#: every outline, label and legend as editable vector graphics.
DEFAULT_FORMAT = ".svg"


@dataclass
class HistogramPanel:
    """What to draw in the left panel.

    ``overlays`` are ``(label, vertices_Nx2, color)`` in histogram data
    coordinates (neutron on x, X-ray on y) — the same tuples the histogram
    canvases draw.
    """

    histogram_data: object                     # histograms.HistogramData
    overlays: Sequence[Tuple[str, np.ndarray, object]] = ()
    log_scale: bool = True
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    title: str = "Local histogram"


@dataclass
class SlicePanel:
    """What to draw in the right panel.

    ``overlays`` are ``(label, mask_2d, color)`` with masks the same shape as
    ``image`` (already sliced for the current plane).
    """

    image: np.ndarray
    overlays: Sequence[Tuple[str, np.ndarray, object]] = ()
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    title: str = "Volume slice"
    note: str = ""


def _rgb(color, fallback=(1.0, 0.0, 0.0)):
    try:
        red, green, blue, _alpha = mcolors.to_rgba(color)
        return red, green, blue
    except (ValueError, TypeError):
        return fallback


def _alpha(color, default):
    if isinstance(color, (tuple, list)) and len(color) > 3:
        try:
            return float(color[3])
        except (TypeError, ValueError):
            pass
    return default


def _draw_histogram(axes, figure, panel: HistogramPanel, show_legend: bool):
    data = panel.histogram_data
    values = data.to_log_scale() if panel.log_scale else data.histogram
    vmin, vmax = panel.vmin, panel.vmax
    if vmin is not None and vmax is not None and not vmax > vmin:
        vmin = vmax = None

    image = axes.imshow(
        values,
        extent=[data.x_edges[0], data.x_edges[-1],
                data.y_edges[0], data.y_edges[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(
        image, ax=axes, fraction=0.046, pad=0.02,
        label="log10(counts + 1)" if panel.log_scale else "counts",
    )

    handles = []
    for label, vertices, color in panel.overlays:
        if vertices is None:
            continue
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or len(vertices) < 3:
            continue
        red, green, blue = _rgb(color)
        # Filled as well as outlined, like the on-screen canvas: the shaded
        # area is exactly the region that label selects.
        axes.add_patch(MplPolygon(
            vertices, closed=True, fill=True,
            facecolor=(red, green, blue, 0.18),
            edgecolor=(red, green, blue, 0.95),
            linewidth=1.8, linestyle="--",
        ))
        handles.append(Patch(
            facecolor=(red, green, blue, 0.18),
            edgecolor=(red, green, blue, 0.95),
            linestyle="--", label=label,
        ))

    axes.set_xlim(data.x_edges[0], data.x_edges[-1])
    axes.set_ylim(data.y_edges[0], data.y_edges[-1])
    axes.set_xlabel("Neutron intensity")
    axes.set_ylabel("X-ray intensity")
    axes.set_title(panel.title)
    axes.grid(True, alpha=0.25, linewidth=0.5)
    if show_legend and handles:
        axes.legend(handles=handles, loc="upper right", fontsize=8,
                    framealpha=0.85)
    return handles


def _draw_slice(axes, panel: SlicePanel, show_legend: bool, outline: bool):
    image = np.asarray(panel.image)
    vmin, vmax = panel.vmin, panel.vmax
    if vmin is None or vmax is None or not vmax > vmin:
        vmin = vmax = None
    extent = [0, image.shape[1], image.shape[0], 0]
    axes.imshow(
        image, cmap="gray", interpolation="nearest",
        vmin=vmin, vmax=vmax, extent=extent, origin="upper", aspect="equal",
    )

    handles = []
    for label, mask, color in panel.overlays:
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image.shape:
            continue
        pixels = int(np.count_nonzero(mask))
        red, green, blue = _rgb(color)
        alpha = _alpha(color, 0.5)
        if pixels:
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[mask] = [red, green, blue, alpha]
            axes.imshow(rgba, extent=extent, origin="upper", aspect="equal",
                        interpolation="nearest")
            if outline and 0 < pixels < mask.size:
                # Pixel-centre coordinates match the extent above
                axes.contour(
                    np.arange(mask.shape[1]) + 0.5,
                    np.arange(mask.shape[0]) + 0.5,
                    mask.astype(float), levels=[0.5],
                    colors=[(red, green, blue, 1.0)], linewidths=0.9,
                )
        share = 100.0 * pixels / mask.size if mask.size else 0.0
        handles.append(Patch(
            facecolor=(red, green, blue, alpha),
            edgecolor=(red, green, blue, 1.0),
            label=f"{label} ({share:.1f}% of slice)",
        ))

    axes.set_xlim(0, image.shape[1])
    axes.set_ylim(image.shape[0], 0)
    axes.set_title(panel.title)
    axes.set_xticks([])
    axes.set_yticks([])
    if panel.note:
        axes.set_xlabel(panel.note, fontsize=8)
    if show_legend and handles:
        axes.legend(handles=handles, loc="upper right", fontsize=8,
                    framealpha=0.85)
    return handles


def render_histogram_slice_figure(
    histogram: HistogramPanel,
    slice_panel: SlicePanel,
    show_legend: bool = True,
    outline_highlights: bool = True,
    suptitle: str = "",
    figsize=(13.0, 5.6),
    dpi: int = 100,
) -> Figure:
    """Build the two-panel figure and return it (not saved)."""
    figure = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(figure)
    left, right = figure.subplots(
        1, 2, gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    _draw_histogram(left, figure, histogram, show_legend)
    _draw_slice(right, slice_panel, show_legend, outline_highlights)
    if suptitle:
        figure.suptitle(suptitle)
    figure.tight_layout()
    return figure


def save_histogram_slice_figure(
    path,
    histogram: HistogramPanel,
    slice_panel: SlicePanel,
    dpi: int = 300,
    **kwargs,
) -> Path:
    """Render and save the figure; returns the path written.

    The format follows the extension; a path without a supported extension
    gets ``.svg``. In SVG the text is kept as text (not outlines), so labels
    stay editable in Inkscape or Illustrator; *dpi* sets the resolution of
    the two embedded images (histogram and slice).
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        path = path.with_name(path.name + DEFAULT_FORMAT)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = render_histogram_slice_figure(histogram, slice_panel, **kwargs)
    from utils.figure_io import save_figure
    save_figure(figure, path, dpi)
    return path


def write_histogram_slice_csv(path, histogram: HistogramPanel,
                              slice_panel: SlicePanel) -> str:
    """The numbers behind the two-panel figure, one row per label.

    Histogram labels: how many voxels of the local histogram fall inside the
    label's region (bins whose centre is inside the outline), and their share.
    Slice labels: how many pixels of the displayed slice are highlighted,
    and their share of the slice.
    """
    from matplotlib.path import Path as MplPath
    from utils.figure_io import write_csv

    data = histogram.histogram_data
    counts = np.asarray(data.histogram, dtype=np.float64)
    total = max(counts.sum(), 1.0)
    x_centers = 0.5 * (np.asarray(data.x_edges[:-1]) + np.asarray(data.x_edges[1:]))
    y_centers = 0.5 * (np.asarray(data.y_edges[:-1]) + np.asarray(data.y_edges[1:]))
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    centers = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    rows = []
    for label, vertices, color in histogram.overlays:
        if vertices is None or len(vertices) < 3:
            continue
        inside = MplPath(np.asarray(vertices, float)).contains_points(centers)
        voxels = float(counts.ravel()[inside].sum())
        rows.append(("histogram", label, mcolors.to_hex(_rgb(color)),
                     int(voxels), voxels / total))
    image = np.asarray(slice_panel.image)
    for label, mask, color in slice_panel.overlays:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image.shape:
            continue
        pixels = int(np.count_nonzero(mask))
        rows.append(("slice", label, mcolors.to_hex(_rgb(color)),
                     pixels, pixels / max(mask.size, 1)))
    return write_csv(
        path, ("panel", "label", "color", "count", "share"), rows,
    )

