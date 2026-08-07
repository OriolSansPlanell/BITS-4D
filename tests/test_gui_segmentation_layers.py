"""GUI-level regression tests for segmentation layers and histogram overlays.

Covers the reported issues:
- the histogram overlay of a segmented ROI must be the exact drawn shape,
  not a smaller/rounder convex hull re-derived from the voxel intensities;
- pressing "Segment Current" after clearing the ROI and drawing a new one
  must not silently resurrect previous ROIs — the user chooses to keep or
  replace them.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    neutron = np.full((1, 4, 24, 24), 500.0)
    xray = np.full((1, 4, 24, 24), 500.0)
    blob_a = np.zeros(neutron.shape, dtype=bool)
    blob_a[:, :, :8, :8] = True
    blob_b = np.zeros(neutron.shape, dtype=bool)
    blob_b[:, :, 16:, 16:] = True
    neutron[blob_a] = 100
    xray[blob_a] = 900
    neutron[blob_b] = 900
    xray[blob_b] = 100

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    w._test_blobs = (blob_a[0], blob_b[0])
    return w


def test_histogram_overlay_is_exact_roi_shape(window):
    rm = window.dual_histogram.get_roi_manager()
    rm.set_rectangle_roi(50, 850, 150, 950)
    window._segment_current_volume()

    overlays = window.dual_histogram.global_canvas.roi_overlays
    seg_overlays = [o for o in overlays if o[0].startswith("Seg:")]
    assert len(seg_overlays) == 1
    _name, vertices, _color = seg_overlays[0]
    expected = np.array(
        [[50.0, 850.0], [150.0, 850.0], [150.0, 950.0], [50.0, 950.0]]
    )
    np.testing.assert_array_equal(np.asarray(vertices), expected)


def test_replace_previous_layers_on_request(window, monkeypatch):
    blob_a, blob_b = window._test_blobs
    rm = window.dual_histogram.get_roi_manager()

    rm.set_rectangle_roi(50, 850, 150, 950)          # material A
    window._segment_current_volume()
    assert len(window.segmentation_masks[0]) == 1

    rm.clear_roi()
    rm.set_rectangle_roi(850, 50, 950, 150)          # material B
    # Answer "No" → replace previous layers with the new ROI only
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )
    window._segment_current_volume()

    layers = window.segmentation_masks[0]
    assert len(layers) == 1, "previous ROI layer must be replaced"
    np.testing.assert_array_equal(layers[0][0], blob_b)
    # The replaced layer's recorded outline must be gone as well
    names = {name for (_t, name) in window.segmentation_layer_shapes}
    assert names == {layers[0][2]}


def test_keep_previous_layers_on_request(window, monkeypatch):
    blob_a, blob_b = window._test_blobs
    rm = window.dual_histogram.get_roi_manager()

    rm.set_rectangle_roi(50, 850, 150, 950)
    window._segment_current_volume()
    rm.clear_roi()
    rm.set_rectangle_roi(850, 50, 950, 150)
    # Answer "Yes" → keep both layers, under distinct names
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    window._segment_current_volume()

    layers = window.segmentation_masks[0]
    assert len(layers) == 2
    names = [name for _m, _c, name in layers]
    assert len(set(names)) == 2, "kept layers must have distinct names"
    masks = {name: mask for mask, _c, name in layers}
    collected = list(masks.values())
    np.testing.assert_array_equal(collected[0] | collected[1], blob_a | blob_b)


def test_cancel_leaves_layers_untouched(window, monkeypatch):
    rm = window.dual_histogram.get_roi_manager()
    rm.set_rectangle_roi(50, 850, 150, 950)
    window._segment_current_volume()
    before = list(window.segmentation_masks[0])

    rm.clear_roi()
    rm.set_rectangle_roi(850, 50, 950, 150)
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.Cancel),
    )
    window._segment_current_volume()
    assert window.segmentation_masks[0] == before
