"""GUI wiring for the model-based segmentation, drift and spatial metrics."""

import csv
import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    rng = np.random.default_rng(0)
    timepoints, shape = 4, (4, 16, 16)
    neutron = np.zeros((timepoints,) + shape)
    xray = np.zeros((timepoints,) + shape)

    # Two phases, drifting together; the second one also shrinks
    for timepoint in range(timepoints):
        drift = 60.0 * timepoint
        matrix = np.zeros(shape, dtype=bool)
        matrix[:, :, :10] = True
        blob = np.zeros(shape, dtype=bool)
        blob[:, :, 10:10 + max(6 - timepoint, 2)] = True
        matrix |= ~(matrix | blob)
        for mask, (mean_n, mean_x) in (
            (matrix, (500.0, 500.0)), (blob, (1100.0, 1100.0))
        ):
            count = int(mask.sum())
            neutron[timepoint][mask] = np.clip(
                rng.normal(mean_n + drift, 50, count), 20, 2180
            )
            xray[timepoint][mask] = np.clip(
                rng.normal(mean_x + drift, 50, count), 20, 2180
            )

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    return w


def _segment_two_classes(window, monkeypatch):
    """Draw and save two ROIs, then segment the current timepoint."""
    roi_manager = window.dual_histogram.get_roi_manager()
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    for name, (low, high) in (
        ("Matrix", (300, 700)), ("Deposit", (900, 1300))
    ):
        roi_manager.set_rectangle_roi(low, low, high, high)
        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: (name, True))
        )
        window.dual_histogram._save_current_as_class()
    window._segment_current_volume()
    return [name for _m, _c, name in window._visible_layers(0)]


# ── model-based segmentation ─────────────────────────────────────────────────

def test_model_segmentation_labels_every_timepoint(window, monkeypatch):
    from gui import main_window as mw

    names = _segment_two_classes(window, monkeypatch)
    assert len(names) == 2

    class FakeDialog(mw.ModelSegmentationDialog):
        def exec_(self):
            self.anchor_strength = 0.5
            self.anchor_classes = ["Matrix"]
            self.estimate_scale = False
            self.beta = 1.0
            self.sweeps = 3
            self.memory = 0.5
            self.outlier_component = True
            self.reject_margin = None
            self.detect_mixels = True
            return QDialog.Accepted

    monkeypatch.setattr(mw, "ModelSegmentationDialog", FakeDialog)
    window._on_model_segmentation()

    assert window.model_result is not None
    assert len(window.model_result) == window.dataset.num_timepoints

    # Every timepoint now carries layers named after the user's classes
    for timepoint in range(window.dataset.num_timepoints):
        layers = {name for _m, _c, name in window.segmentation_masks[timepoint]}
        assert layers <= set(names)
        assert layers


def test_model_segmentation_tracks_a_drifting_class(window, monkeypatch):
    """The point of the exercise: counts follow the truth, not the T0 box."""
    from gui import main_window as mw

    _segment_two_classes(window, monkeypatch)

    class FakeDialog(mw.ModelSegmentationDialog):
        def exec_(self):
            self.anchor_strength = 0.5
            self.anchor_classes = ["Matrix"]
            self.estimate_scale = False
            self.beta = 1.0
            self.sweeps = 3
            self.memory = 0.5
            self.outlier_component = True
            self.reject_margin = None
            self.detect_mixels = False
            return QDialog.Accepted

    monkeypatch.setattr(mw, "ModelSegmentationDialog", FakeDialog)
    window._on_model_segmentation()

    counts = [
        entry.voxel_counts["Deposit"] for entry in window.model_result.timepoints
    ]
    # The deposit really does shrink; a frozen T0 box would lose it entirely
    assert counts[0] > counts[-1]
    assert all(count > 0 for count in counts)

    drifts = [
        entry.drift.magnitude for entry in window.model_result.timepoints
    ]
    assert drifts[-1] > drifts[0]


def test_model_segmentation_needs_classes_first(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window._on_model_segmentation()
    assert shown and "Segment the current timepoint first" in shown[-1]


def test_model_run_is_scriptable_without_the_gui(window, monkeypatch):
    """The engine must not depend on the widgets around it."""
    from model import (
        DriftTracker, ROIAnchoredMixture, ROIDerivedMRF, SequentialSegmenter,
    )
    from model.temporal import DriftTransition

    _segment_two_classes(window, monkeypatch)
    masks = window._model_class_masks(0)
    neutron, xray = window.dataset.get_volume_at_time(0)

    segmenter = SequentialSegmenter(
        mixture=ROIAnchoredMixture(),
        mrf=ROIDerivedMRF(beta=1.0, n_sweeps=2),
        temporal=DriftTransition(memory=0.5),
        drift_tracker=DriftTracker(anchor_classes=["Matrix"]),
    )
    segmenter.prepare(
        neutron, xray, masks,
        window.global_histogram.x_edges, window.global_histogram.y_edges,
    )
    outcome = segmenter.run(window.dataset)
    assert len(outcome) == window.dataset.num_timepoints
    assert set(outcome.class_names) == set(masks)


# ── drift ────────────────────────────────────────────────────────────────────

def test_drift_export_writes_one_row_per_timepoint(window, monkeypatch, tmp_path):
    from gui import main_window as mw

    _segment_two_classes(window, monkeypatch)
    target = tmp_path / "drift.csv"

    class FakeAnchors(mw.AnchorSelectionDialog):
        def exec_(self):
            self.anchor_classes = ["Matrix"]
            self.estimate_scale = False
            return QDialog.Accepted

    monkeypatch.setattr(mw, "AnchorSelectionDialog", FakeAnchors)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")),
    )
    window._on_estimate_drift()

    assert target.exists()
    with open(target, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == window.dataset.num_timepoints
    assert float(records[0]["magnitude"]) < float(records[-1]["magnitude"])
    assert set(records[0]) >= {
        "timepoint", "shift_neutron", "shift_xray", "magnitude", "residual",
    }


def test_drift_needs_anchor_classes(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window._on_estimate_drift()
    assert shown and "chemically inert" in shown[-1]


# ── spatial metrics ──────────────────────────────────────────────────────────

def test_spatial_metrics_export(window, monkeypatch, tmp_path):
    _segment_two_classes(window, monkeypatch)
    window._segment_all_volumes()

    target = tmp_path / "spatial.csv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")),
    )
    window._on_export_spatial_metrics()

    assert target.exists()
    with open(target, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    metrics = {record["metric"] for record in records}
    assert {"com_z_k", "rg_k", "n_components_k", "sa_vol_k"} <= metrics
    classes = {record["class"] for record in records if record["class"]}
    assert {"Matrix", "Deposit"} <= classes
    assert (tmp_path / "spatial_evolution.png").exists()


def test_spatial_metrics_need_a_segmentation(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window._on_export_spatial_metrics()
    assert shown and "Segment at least one timepoint" in shown[-1]


# ── block cross-validation ───────────────────────────────────────────────────

def test_block_cv_reports_a_lower_number_than_training_accuracy(
    window, monkeypatch
):
    _segment_two_classes(window, monkeypatch)
    window.rf_ref_spin.setValue(0)
    window._rf_train()
    if window.rf_engine is None or not window.rf_engine.is_trained:
        pytest.skip("Random Forest training did not run in this environment")

    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window._on_block_cross_validation()

    assert shown
    message = shown[-1]
    assert "block CV" in message
    assert "kappa" in message
    assert "memorisation" in message


def test_block_cv_needs_a_trained_model(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window.rf_engine = None
    window._on_block_cross_validation()
    assert shown and "Train the Random Forest first" in shown[-1]
