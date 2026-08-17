"""Removing a class asks what to do with the segmentation it produced.

Unticking a class hides its layer; removing the class leaves nothing in the
panel controlling that layer, so the user is asked whether to discard it
rather than having either outcome chosen silently.
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
    return w


RECT_A = (50, 850, 150, 950)
RECT_B = (850, 50, 950, 150)


def _save_and_segment(window, monkeypatch, name, rect):
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


def _answer(monkeypatch, response):
    """Make the next QMessageBox.question return *response*."""
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: response)
    )


def _layer_names(window):
    return [layer[2] for layer in window.segmentation_masks.get(0, [])]


def test_layer_count_is_reported_to_the_panel(window, monkeypatch):
    _save_and_segment(window, monkeypatch, "A", RECT_A)
    assert window._count_layers_for_class("A") == 1
    assert window._count_layers_for_class("nonexistent") == 0
    # The panel is wired to ask this window
    assert window.dual_histogram.layer_count_provider is not None


def test_removing_and_discarding_deletes_the_layer(window, monkeypatch):
    _save_and_segment(window, monkeypatch, "A", RECT_A)
    _save_and_segment(window, monkeypatch, "B", RECT_B)
    assert sorted(_layer_names(window)) == ["A", "B"]

    window.dual_histogram.roi_list_widget.setCurrentRow(0)
    _answer(monkeypatch, QMessageBox.Yes)          # discard its segmentation
    window.dual_histogram._remove_selected_class()

    assert _layer_names(window) == ["B"]
    assert [r['name'] for r in window.dual_histogram.roi_manager.named_rois] == ["B"]
    # Its recorded outline is gone as well
    assert (0, "A") not in window.segmentation_layer_shapes


def test_removing_and_keeping_leaves_the_layer(window, monkeypatch):
    _save_and_segment(window, monkeypatch, "A", RECT_A)

    window.dual_histogram.roi_list_widget.setCurrentRow(0)
    _answer(monkeypatch, QMessageBox.No)           # keep its segmentation
    window.dual_histogram._remove_selected_class()

    assert _layer_names(window) == ["A"]
    assert window.dual_histogram.roi_manager.named_rois == []


def test_cancel_keeps_the_class_and_the_layer(window, monkeypatch):
    _save_and_segment(window, monkeypatch, "A", RECT_A)

    window.dual_histogram.roi_list_widget.setCurrentRow(0)
    _answer(monkeypatch, QMessageBox.Cancel)
    window.dual_histogram._remove_selected_class()

    assert _layer_names(window) == ["A"]
    assert [r['name'] for r in window.dual_histogram.roi_manager.named_rois] == ["A"]


def test_class_without_segmentation_is_removed_with_a_plain_confirm(
    window, monkeypatch
):
    manager = window.dual_histogram.get_roi_manager()
    manager.set_rectangle_roi(*RECT_A)
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("Unsegmented", True)),
    )
    window.dual_histogram._save_current_as_class()
    assert window._count_layers_for_class("Unsegmented") == 0

    asked = {}

    def _question(parent, title, text, *args, **kwargs):
        asked['text'] = text
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    window.dual_histogram.roi_list_widget.setCurrentRow(0)
    window.dual_histogram._remove_selected_class()

    assert window.dual_histogram.roi_manager.named_rois == []
    assert "segmentation layer" not in asked['text'], (
        "should not offer to discard segmentation when there is none"
    )


def test_clear_all_offers_the_same_choice(window, monkeypatch):
    _save_and_segment(window, monkeypatch, "A", RECT_A)
    _save_and_segment(window, monkeypatch, "B", RECT_B)

    _answer(monkeypatch, QMessageBox.Cancel)
    window.dual_histogram._clear_all_classes()
    assert len(window.dual_histogram.roi_manager.named_rois) == 2
    assert sorted(_layer_names(window)) == ["A", "B"]

    _answer(monkeypatch, QMessageBox.No)           # keep the segmentation
    window.dual_histogram._clear_all_classes()
    assert window.dual_histogram.roi_manager.named_rois == []
    assert sorted(_layer_names(window)) == ["A", "B"]


def test_clear_all_can_discard_every_layer(window, monkeypatch):
    _save_and_segment(window, monkeypatch, "A", RECT_A)
    _save_and_segment(window, monkeypatch, "B", RECT_B)

    _answer(monkeypatch, QMessageBox.Yes)
    window.dual_histogram._clear_all_classes()

    assert window.dual_histogram.roi_manager.named_rois == []
    assert _layer_names(window) == []
    assert window.segmentation_layer_shapes == {}
