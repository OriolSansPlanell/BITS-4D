"""The two-panel "histogram + slice" figure, rendered without Qt."""

import numpy as np
import pytest

from histograms.histogram_engine_4d import HistogramEngine4D
from utils.figure_export import (
    HistogramPanel,
    SlicePanel,
    render_histogram_slice_figure,
    save_histogram_slice_figure,
)


def _panels():
    neutron = np.full((4, 16, 16), 500.0)
    xray = np.full((4, 16, 16), 500.0)
    neutron[:, :6, :6] = 100.0
    xray[:, :6, :6] = 900.0
    histogram = HistogramEngine4D(bins=32, use_gpu=False).compute_global_histogram(
        neutron[None], xray[None]
    )
    square = np.array([[50, 850], [150, 850], [150, 950], [50, 950]], float)
    mask = np.zeros((16, 16), dtype=bool)
    mask[:6, :6] = True
    hist_panel = HistogramPanel(
        histogram_data=histogram,
        overlays=[("Class 1: Lithium", square, "#e6194b")],
    )
    slice_panel = SlicePanel(
        image=neutron[2],
        overlays=[("Lithium", mask, (0.9, 0.1, 0.3, 0.5))],
        title="XY Slice (Z=2, Neutron)",
    )
    return hist_panel, slice_panel


def test_figure_has_histogram_left_and_slice_right():
    hist_panel, slice_panel = _panels()
    figure = render_histogram_slice_figure(hist_panel, slice_panel)
    left, right = figure.axes[0], figure.axes[1]
    assert left.get_xlabel() == "Neutron intensity"
    assert left.get_ylabel() == "X-ray intensity"
    assert right.get_title() == "XY Slice (Z=2, Neutron)"
    # One outline per label on the histogram
    assert len(left.patches) == 1
    # Base slice + one highlight layer
    assert len(right.images) == 2


def test_label_colour_is_shared_between_panels():
    hist_panel, slice_panel = _panels()
    figure = render_histogram_slice_figure(hist_panel, slice_panel)
    left, right = figure.axes[0], figure.axes[1]
    edge = left.patches[0].get_edgecolor()[:3]
    highlight = right.images[1].get_array()
    painted = highlight[0, 0, :3]
    np.testing.assert_allclose(edge, (0xe6 / 255, 0x19 / 255, 0x4b / 255), atol=1e-6)
    np.testing.assert_allclose(painted, (0.9, 0.1, 0.3), atol=1e-6)
    # Highlight covers exactly the mask
    assert highlight[..., 3].astype(bool).sum() == 36


def test_legends_name_every_label():
    hist_panel, slice_panel = _panels()
    figure = render_histogram_slice_figure(hist_panel, slice_panel)
    left_labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
    right_labels = [t.get_text() for t in figure.axes[1].get_legend().get_texts()]
    assert left_labels == ["Class 1: Lithium"]
    assert right_labels == ["Lithium (14.1% of slice)"]


def test_masks_of_another_shape_are_skipped():
    hist_panel, slice_panel = _panels()
    slice_panel.overlays = [("Wrong plane", np.ones((4, 16), bool), "red")]
    figure = render_histogram_slice_figure(hist_panel, slice_panel)
    assert len(figure.axes[1].images) == 1


@pytest.mark.parametrize("suffix", [".png", ".pdf", ".svg", ".tif"])
def test_saves_in_the_requested_format(tmp_path, suffix):
    hist_panel, slice_panel = _panels()
    path = save_histogram_slice_figure(
        tmp_path / f"figure{suffix}", hist_panel, slice_panel, dpi=72
    )
    assert path.suffix == suffix
    assert path.stat().st_size > 0


def test_unknown_extension_falls_back_to_png(tmp_path):
    hist_panel, slice_panel = _panels()
    path = save_histogram_slice_figure(
        tmp_path / "figure", hist_panel, slice_panel, dpi=72
    )
    assert path.name == "figure.png"
    assert path.exists()
