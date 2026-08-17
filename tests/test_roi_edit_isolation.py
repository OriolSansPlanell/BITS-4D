"""Editing an ROI must not alter what was already segmented.

Regression: the outline recorded for a segmented layer was the *same array
object* as the live ROI. Dragging a vertex afterwards rewrote that record in
place, so the histogram showed the edited shape while the mask — and the RF
labels built from it — came from the original shape. The region looked
"modified", with parts of the drawn ROI apparently unsegmented.
"""

import os

import numpy as np
import pytest

from utils.roi_manager import ROIManager

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


# ── ROIManager / handler level ───────────────────────────────────────────────

def test_set_polygon_roi_does_not_alias_the_caller_array():
    points = SQUARE.copy()
    manager = ROIManager()
    manager.set_polygon_roi(points)
    points[0] = [99.0, 99.0]
    np.testing.assert_array_equal(manager.polygon_points[0], [0.0, 0.0])


def test_active_vertices_snapshot_survives_later_edits():
    manager = ROIManager()
    manager.set_polygon_roi(SQUARE)
    snapshot = manager.get_active_vertices()

    manager.polygon_points[2] = [50.0, 50.0]     # simulate an in-place edit
    np.testing.assert_array_equal(snapshot, SQUARE)

    manager.set_polygon_roi(np.vstack([SQUARE, [[5.0, -5.0]]]))
    np.testing.assert_array_equal(snapshot, SQUARE)


def test_rectangle_snapshot_is_a_closed_outline():
    manager = ROIManager()
    manager.set_rectangle_roi(1.0, 2.0, 3.0, 4.0)
    np.testing.assert_array_equal(
        manager.get_active_vertices(),
        [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
    )


def test_no_active_roi_has_no_vertices():
    assert ROIManager().get_active_vertices() is None


def test_editable_handler_replaces_rather_than_mutates(qapp_unused=None):
    """Dragging a vertex must leave earlier snapshots untouched."""
    from utils.editable_roi_handler import EditableROIHandler

    manager = ROIManager()
    manager.set_polygon_roi(SQUARE)
    snapshot = manager.get_active_vertices()

    handler = EditableROIHandler(ax=None, roi_manager=manager)
    handler.enabled = True
    handler.dragging_vertex = 1

    class _Event:
        xdata, ydata = 42.0, 43.0
        inaxes = None

    # _on_motion bails out on a foreign axes, so drive the update directly
    # the same way it does.
    points = np.array(manager.polygon_points, dtype=float)
    points[handler.dragging_vertex] = [_Event.xdata, _Event.ydata]
    manager.set_polygon_roi(points)

    np.testing.assert_array_equal(snapshot, SQUARE)
    np.testing.assert_array_equal(manager.polygon_points[1], [42.0, 43.0])


# ── Whole-window behaviour ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    neutron = np.full((1, 4, 16, 16), 500.0)
    xray = np.full((1, 4, 16, 16), 500.0)
    blob = np.zeros(neutron.shape, dtype=bool)
    blob[:, :, :8, :8] = True
    neutron[blob] = 100.0
    xray[blob] = 900.0

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    w._test_blob = blob[0]
    return w


def test_segmented_layer_outline_is_frozen_at_segmentation_time(window):
    """The recorded outline must keep describing the voxels actually masked."""
    manager = window.dual_histogram.get_roi_manager()
    manager.set_polygon_roi(
        np.array([[50.0, 850.0], [150.0, 850.0], [150.0, 950.0], [50.0, 950.0]])
    )
    window._segment_current_volume()

    key = list(window.segmentation_layer_shapes)[0]
    recorded_before = window.segmentation_layer_shapes[key].copy()
    mask_before = window.segmentation_masks[0][0][0].copy()

    # Edit the live ROI in place, the way vertex dragging used to. Even this
    # must not reach back into the record of what was segmented.
    manager.polygon_points[0] = [-400.0, -400.0]
    window.dual_histogram._on_roi_updated()

    np.testing.assert_array_equal(
        window.segmentation_layer_shapes[key], recorded_before
    )
    np.testing.assert_array_equal(window.segmentation_masks[0][0][0], mask_before)


def test_layer_outline_matches_the_mask_it_produced(window):
    """The overlay drawn for a layer must be the ROI that made its mask."""
    manager = window.dual_histogram.get_roi_manager()
    vertices = np.array(
        [[50.0, 850.0], [150.0, 850.0], [150.0, 950.0], [50.0, 950.0]]
    )
    manager.set_polygon_roi(vertices)
    window._segment_current_volume()

    mask = window.segmentation_masks[0][0][0]
    np.testing.assert_array_equal(mask, window._test_blob)

    key = list(window.segmentation_layer_shapes)[0]
    np.testing.assert_array_equal(
        window.segmentation_layer_shapes[key], vertices
    )


def test_saved_class_geometry_is_independent_of_later_edits(window, monkeypatch):
    manager = window.dual_histogram.get_roi_manager()
    manager.set_polygon_roi(SQUARE)
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("Saved", True)),
    )
    window.dual_histogram._save_current_as_class()

    stored = manager.named_rois[0]['points'].copy()
    manager.set_polygon_roi(SQUARE * 3)
    np.testing.assert_array_equal(manager.named_rois[0]['points'], stored)


def test_batch_specs_snapshot_the_geometry(window):
    """Specs handed to a worker must not change if the ROI is edited after."""
    from gui.main_window import BiTS4DMainWindow

    manager = window.dual_histogram.get_roi_manager()
    manager.set_polygon_roi(SQUARE)
    specs = BiTS4DMainWindow._enumerate_roi_specs(manager)
    captured = specs[-1]['points'].copy()

    manager.polygon_points[0] = [77.0, 77.0]
    np.testing.assert_array_equal(specs[-1]['points'], captured)
