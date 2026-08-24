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


def _build_window(drift_per_step=0.0):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    rng = np.random.default_rng(0)
    timepoints, shape = 4, (4, 16, 16)
    neutron = np.zeros((timepoints,) + shape)
    xray = np.zeros((timepoints,) + shape)

    # Two materials; the second shrinks over time. *drift_per_step* moves
    # both of them together, which is an instrument effect rather than a
    # sample one — locked mode is explicitly not built to absorb it.
    for timepoint in range(timepoints):
        drift = drift_per_step * timepoint
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


@pytest.fixture()
def window(qapp):
    """A stable instrument: only the sample changes."""
    return _build_window(drift_per_step=0.0)


@pytest.fixture()
def drifting_window(qapp):
    """An instrument that moves under the sample."""
    return _build_window(drift_per_step=60.0)


def _segment_two_classes(window, monkeypatch):
    """Draw and save two ROIs, then segment the current timepoint."""
    roi_manager = window.dual_histogram.get_roi_manager()
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    # Problems are reported through warning(); unstubbed it blocks forever
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: None)
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

def _fake_dialog(monkeypatch, **overrides):
    """Drive the tracking dialog without a user."""
    from gui import main_window as mw

    settings = {
        "control_materials": ["Matrix"],
        "smoothing_mode": "auto",
        "smoothing_strength": None,
        "find_mixed_boundaries": False,
        "lock_definitions": True,
    }
    settings.update(overrides)

    class FakeDialog(mw.MaterialTrackingDialog):
        def exec_(self):
            for key, value in settings.items():
                setattr(self, key, value)
            return QDialog.Accepted

    monkeypatch.setattr(mw, "MaterialTrackingDialog", FakeDialog)
    return settings


def test_model_segmentation_labels_every_timepoint(window, monkeypatch):
    names = _segment_two_classes(window, monkeypatch)
    assert len(names) == 2

    _fake_dialog(monkeypatch)
    window._on_model_segmentation()

    assert window.model_result is not None
    assert len(window.model_result) == window.dataset.num_timepoints

    # Every timepoint now carries layers named after the user's classes
    for timepoint in range(window.dataset.num_timepoints):
        layers = {name for _m, _c, name in window.segmentation_masks[timepoint]}
        assert layers <= set(names)
        assert layers


def test_model_segmentation_follows_a_shrinking_material(window, monkeypatch):
    """A real change in the sample has to come through as a real change."""
    _segment_two_classes(window, monkeypatch)
    _fake_dialog(monkeypatch)
    window._on_model_segmentation()

    counts = [
        entry.voxel_counts["Deposit"] for entry in window.model_result.timepoints
    ]
    assert counts[0] > counts[-1]
    assert all(count > 0 for count in counts)

    # The control material was declared unchanging, and stays so
    control = [
        entry.voxel_counts["Matrix"] for entry in window.model_result.timepoints
    ]
    assert max(control) - min(control) < 0.25 * max(control)


def test_results_are_the_same_run_backwards(window, monkeypatch):
    """Timepoints are independent, so order cannot matter."""
    from model import ClassLibrary, LockedSegmenter
    from model.spatial_prior import ROIDerivedMRF
    from model.validity import build_valid_mask

    _segment_two_classes(window, monkeypatch)
    masks = window._model_class_masks(0)
    neutron, xray = window.dataset.get_volume_at_time(0)
    valid = build_valid_mask(neutron, xray)

    library = ClassLibrary.from_masks(
        neutron, xray, masks, valid_mask=valid, inert=["Matrix"]
    )
    segmenter = LockedSegmenter(library, prior=ROIDerivedMRF(beta=1.0, n_sweeps=3))
    segmenter.set_grid(
        window.global_histogram.x_edges, window.global_histogram.y_edges
    )
    order = list(range(window.dataset.num_timepoints))
    forwards = segmenter.segment_series(window.dataset, timepoints=order)
    backwards = segmenter.segment_series(
        window.dataset, timepoints=order[::-1]
    )
    for entry in forwards.timepoints:
        other = next(
            e for e in backwards.timepoints if e.timepoint == entry.timepoint
        )
        assert np.array_equal(entry.labels, other.labels)


def test_model_segmentation_needs_classes_first(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window._on_model_segmentation()
    assert shown and "Draw and segment at least one material first" in shown[-1]


def test_drift_makes_locked_mode_refuse_and_say_why(drifting_window, monkeypatch):
    """Locked definitions cannot absorb a moving instrument — and say so.

    The refusal is the feature: the alternative is a results screen full of
    numbers that quietly describe the wrong voxels.
    """
    window = drifting_window
    _segment_two_classes(window, monkeypatch)

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: warned.append(a[-1])),
    )
    _fake_dialog(monkeypatch)
    window._on_model_segmentation()

    assert warned, "a drifting series should not produce a silent result"
    message = warned[-1]
    assert "not reliable" in message
    assert "have not been applied" in message
    # It must distinguish drift from a missing material, and name the remedy
    assert "drifted" in message
    assert "Check Instrument Stability" in message
    assert window.model_result is None


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
    assert shown and "should not change during the experiment" in shown[-1]


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
    assert "never saw during training" in message
    assert "overlap" in message
    assert "memorised" in message


def test_check_data_reports_field_of_view_overlap(window, monkeypatch):
    """The fact that broke the previous run has to be visible on load."""
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    # Blank the X-ray channel over part of the volume: different fields of view
    window.dataset.xray_data[:, :, :, :6] = 0.0
    window._on_check_data()

    assert shown
    message = shown[-1]
    assert "only one instrument" in message
    assert "neutron data but no X-ray data" in message
    assert "excluded" in message


def test_check_data_is_quiet_when_the_views_agree(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window._on_check_data()
    assert shown and "only one instrument" not in shown[-1]


def test_block_cv_needs_a_trained_model(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window.rf_engine = None
    window._on_block_cross_validation()
    assert shown and "Train the Random Forest first" in shown[-1]
