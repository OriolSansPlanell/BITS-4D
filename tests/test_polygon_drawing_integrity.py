"""The drawn polygon must be the polygon that gets segmented.

Two failure modes are covered:

* stray vertices — pan/zoom drags and non-left clicks used to add points to
  the polygon being drawn, quietly turning it into a self-crossing shape;
* winding-rule mismatch — a self-crossing outline encloses, visually, more
  than ``Path.contains_points`` selects, so part of the region the user
  drew was never segmented. ROIs are now drawn filled, and matplotlib fills
  with the same rule that decides containment, so the shaded area is exactly
  what is selected.
"""

import os

import numpy as np
import pytest

from utils.roi_manager import ROIManager, polygon_self_intersects

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402


# ── self-intersection detection ──────────────────────────────────────────────

def test_simple_polygons_are_not_flagged():
    square = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    assert not polygon_self_intersects(square)
    triangle = np.array([[0., 0.], [10., 0.], [5., 10.]])
    assert not polygon_self_intersects(triangle)
    # Concave but not crossing
    arrow = np.array([[0., 0.], [10., 0.], [5., 4.], [10., 10.], [0., 10.]])
    assert not polygon_self_intersects(arrow)


def test_crossing_polygon_is_flagged():
    bow_tie = np.array([[0., 0.], [10., 10.], [0., 10.], [10., 0.]])
    assert polygon_self_intersects(bow_tie)


def test_stray_vertex_makes_a_crossing_polygon():
    """A vertex dropped elsewhere by a pan/zoom click, as used to happen.

    The closing edge back to the first vertex then cuts across the shape.
    """
    good = [[10., 10.], [40., 10.], [40., 40.], [10., 40.]]
    with_stray = np.array(good + [[25., 5.]])
    assert polygon_self_intersects(with_stray)


# ── containment matches the filled drawing ───────────────────────────────────

def _render_fill(vertices, probe, limit=10.0):
    """Rasterize the filled polygon and sample it at *probe* points."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Polygon as MplPolygon

    fig = Figure(figsize=(3, 3), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.axis("off")
    ax.add_patch(
        MplPolygon(vertices, closed=True, facecolor="black", edgecolor="none")
    )
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., 0]
    height, width = image.shape
    return [
        image[int((1 - p[1] / limit) * height) - 1,
              int(p[0] / limit * width)] < 128
        for p in probe
    ]


def test_filled_drawing_equals_the_segmented_region():
    """What is shaded is what is selected, even for a crossing polygon."""
    crossing = np.array([
        [0., 0.], [10., 0.], [10., 10.], [0., 10.],
        [0., 2.],
        [2., 2.], [2., 8.], [8., 8.], [8., 2.], [2., 2.],
        [0., 2.],
    ])
    probe = np.array([[5., 5.], [1., 5.], [9., 5.], [1., 9.]])

    manager = ROIManager()
    manager.set_polygon_roi(crossing)
    selected = manager.is_inside_roi(probe[:, 0], probe[:, 1]).tolist()

    assert _render_fill(crossing, probe) == selected
    # The centre is ringed by edges yet excluded — which is why an
    # outline-only drawing was misleading.
    assert selected[0] is False


def test_simple_polygon_selects_everything_it_encloses():
    square = np.array([[10., 10.], [40., 10.], [40., 40.], [10., 40.]])
    manager = ROIManager()
    manager.set_polygon_roi(square)
    probe = np.array([[25., 25.], [11., 39.], [5., 25.], [45., 25.]])
    np.testing.assert_array_equal(
        manager.is_inside_roi(probe[:, 0], probe[:, 1]),
        [True, True, False, False],
    )


# ── click handling on the canvas ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def canvas(qapp):
    from gui.dual_histogram_widget import HistogramCanvas
    from histograms import HistogramEngine4D

    rng = np.random.default_rng(0)
    neutron = rng.uniform(0, 1000, size=(1, 2, 8, 8))
    xray = rng.uniform(0, 1000, size=neutron.shape)
    engine = HistogramEngine4D(bins=32, use_gpu=False)

    c = HistogramCanvas("Test")
    c.roi_manager = ROIManager()
    c.set_histogram_data(engine.compute_global_histogram(neutron, xray))
    c.set_drawing_mode('polygon')
    return c


class _Click:
    def __init__(self, canvas, x, y, button=1):
        self.inaxes = canvas.ax
        self.xdata, self.ydata = x, y
        self.button = button


def test_left_clicks_add_vertices(canvas):
    for point in ((100.0, 100.0), (200.0, 100.0), (200.0, 200.0)):
        canvas.on_mouse_press(_Click(canvas, *point))
    assert len(canvas.polygon_points) == 3


def test_non_left_clicks_are_ignored(canvas):
    canvas.on_mouse_press(_Click(canvas, 100.0, 100.0, button=1))
    canvas.on_mouse_press(_Click(canvas, 500.0, 500.0, button=3))  # right
    canvas.on_mouse_press(_Click(canvas, 600.0, 600.0, button=2))  # middle
    assert len(canvas.polygon_points) == 1


def test_clicks_during_pan_or_zoom_do_not_add_vertices(canvas):
    """Zooming in to place a vertex precisely must not drop a stray one."""
    canvas.on_mouse_press(_Click(canvas, 100.0, 100.0))

    from contextlib import nullcontext

    class _Toolbar:
        """Minimal stand-in for NavigationToolbar2QT during a zoom drag."""
        mode = "zoom rect"

        def _wait_cursor_for_draw_cm(self):
            return nullcontext()

    canvas.toolbar = _Toolbar()
    canvas.on_mouse_press(_Click(canvas, 900.0, 900.0))
    assert len(canvas.polygon_points) == 1, "stray vertex added while zooming"

    canvas.toolbar.mode = ""     # tool released
    canvas.on_mouse_press(_Click(canvas, 300.0, 300.0))
    assert len(canvas.polygon_points) == 2


def test_clicks_outside_the_data_area_are_ignored(canvas):
    event = _Click(canvas, None, None)
    canvas.on_mouse_press(event)
    assert canvas.polygon_points == []
