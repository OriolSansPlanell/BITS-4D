"""Unticking a class must hide the segmentation it produced.

Regression: the tick controlled the ROI (drawing, and what future
segmentation covers) but not the layer already computed from it. Segmenting
one ROI, unticking it, then segmenting a second left the first region still
highlighted in the slice viewer.
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
def window(qapp, monkeypatch):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    # Two materials with disjoint, asymmetric signatures
    neutron = np.full((1, 4, 20, 20), 500.0)
    xray = np.full((1, 4, 20, 20), 500.0)
    blob_a = np.zeros(neutron.shape, dtype=bool)
    blob_a[:, :, :8, :8] = True
    blob_b = np.zeros(neutron.shape, dtype=bool)
    blob_b[:, :, 12:, 12:] = True
    neutron[blob_a] = 100.0
    xray[blob_a] = 900.0
    neutron[blob_b] = 900.0
    xray[blob_b] = 100.0

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    w._blob_a, w._blob_b = blob_a[0], blob_b[0]
    return w


def _save_class(window, monkeypatch, name, rect):
    """Draw an ROI, save it as a named class, and segment it."""
    manager = window.dual_histogram.get_roi_manager()
    manager.set_rectangle_roi(*rect)
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        staticmethod(lambda *a, **k: (name, True)),
    )
    window.dual_histogram._save_current_as_class()
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    window._segment_current_volume()


RECT_A = (50, 850, 150, 950)     # material A
RECT_B = (850, 50, 950, 150)     # material B


def _shown_layer_names(window):
    return {entry[0] for entry in window.slice_viewer.mask_overlays}


def test_unticking_a_class_hides_its_segmentation(window, monkeypatch):
    _save_class(window, monkeypatch, "A", RECT_A)
    assert "A" in _shown_layer_names(window)

    # Untick it in the selection panel
    window.dual_histogram.roi_manager.set_named_roi_visible(0, False)
    window.dual_histogram._on_roi_updated()

    assert "A" not in _shown_layer_names(window), (
        "the segmentation of an unticked class is still highlighted"
    )
    # The mask itself is kept, so ticking it back restores the layer
    assert any(l[2] == "A" for l in window.segmentation_masks[0])
    window.dual_histogram.roi_manager.set_named_roi_visible(0, True)
    window.dual_histogram._on_roi_updated()
    assert "A" in _shown_layer_names(window)


def test_second_roi_does_not_resurrect_the_hidden_one(window, monkeypatch):
    """The exact reported sequence: segment, untick, segment a second ROI."""
    _save_class(window, monkeypatch, "A", RECT_A)
    window.dual_histogram.roi_manager.set_named_roi_visible(0, False)
    window.dual_histogram._on_roi_updated()

    _save_class(window, monkeypatch, "B", RECT_B)

    shown = _shown_layer_names(window)
    assert "B" in shown
    assert "A" not in shown, "the unticked first ROI reappeared"

    # Only material B is highlighted on the displayed slice
    viewer = window.slice_viewer
    viewer.current_axis = 'z'
    viewer.current_slice_index = 2
    viewer._update_display()
    rendered = [
        viewer._slice_mask_for_display(entry[1],
                                       entry[3] if len(entry) > 3 else None)
        for entry in viewer.mask_overlays
    ]
    combined = np.zeros(window._blob_b[2].shape, dtype=bool)
    for mask in rendered:
        if mask is not None:
            combined |= mask
    np.testing.assert_array_equal(combined, window._blob_b[2])


def test_hidden_classes_are_excluded_from_rf_training(window, monkeypatch):
    _save_class(window, monkeypatch, "A", RECT_A)
    _save_class(window, monkeypatch, "B", RECT_B)
    assert len(window._visible_layers(0)) == 2

    window.dual_histogram.roi_manager.set_named_roi_visible(0, False)
    visible = window._visible_layers(0)
    assert [l[2] for l in visible] == ["B"]
    # ... while the mask is still stored for when it is ticked back on
    assert len(window.segmentation_masks[0]) == 2


def test_histogram_outlines_drop_hidden_classes(window, monkeypatch):
    _save_class(window, monkeypatch, "A", RECT_A)
    _save_class(window, monkeypatch, "B", RECT_B)
    window._update_rf_histogram_overlays(0)
    labels = " ".join(
        name for name, _v, _c in window.dual_histogram.global_canvas.roi_overlays
    )
    assert "A" in labels and "B" in labels

    window.dual_histogram.roi_manager.set_named_roi_visible(0, False)
    window._update_rf_histogram_overlays(0)
    labels = " ".join(
        name for name, _v, _c in window.dual_histogram.global_canvas.roi_overlays
    )
    assert "B" in labels
    assert "A" not in labels, "hidden class still outlined on the histogram"


def test_class_outline_is_not_drawn_twice(window, monkeypatch):
    """A segmented class must not get both a class outline and a hull."""
    _save_class(window, monkeypatch, "A", RECT_A)
    window._update_rf_histogram_overlays(0)
    names = [n for n, _v, _c in window.dual_histogram.global_canvas.roi_overlays]
    matching = [n for n in names if "A" in n]
    assert len(matching) == 1, f"duplicate outlines for one class: {matching}"
