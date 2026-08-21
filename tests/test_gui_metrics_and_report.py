"""GUI wiring for the metrics export and the segmentation text report.

Checks that the window collects one metrics row per timepoint plus a global
row, that the rows carry the classes the user actually segmented, and that
the exported report names each class, its label value and its per-timepoint
voxel counts.
"""

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

    # Three timepoints; the low-neutron/high-X-ray blob drifts along neutron
    neutron = np.full((3, 4, 16, 16), 500.0)
    xray = np.full((3, 4, 16, 16), 500.0)
    blob = np.zeros(neutron.shape, dtype=bool)
    blob[:, :, :6, :6] = True
    xray[blob] = 900.0
    for timepoint in range(3):
        neutron[timepoint][blob[timepoint]] = 100.0 + 20.0 * timepoint

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=32, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    return w


def _segment_all_as(window, name, monkeypatch):
    roi_manager = window.dual_histogram.get_roi_manager()
    roi_manager.set_rectangle_roi(50, 850, 200, 950)
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: (name, True))
    )
    window.dual_histogram._save_current_as_class()
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    window._segment_all_volumes()


# ── metrics collection ───────────────────────────────────────────────────────

def test_metrics_rows_cover_the_global_scope_and_every_timepoint(window):
    rows = window._collect_metrics_rows()

    assert len(rows) == 1 + window.dataset.num_timepoints
    assert rows[0].scope == "global" and rows[0].timepoint is None
    assert [row.timepoint for row in rows[1:]] == [0, 1, 2]
    assert rows[0].label == "global" and rows[1].label == "T0"


def test_shape_metrics_are_available_without_any_segmentation(window):
    rows = window._collect_metrics_rows()

    for row in rows:
        assert np.isfinite(row.scalars["S_h"])
        assert np.isfinite(row.scalars["S_d"])
    # No classes -> no class metrics, and no invented ones
    assert all(not row.per_class for row in rows)
    assert rows[0].scalars.get("DB") is None


def test_delta_n_is_zero_at_the_reference_and_grows_after(window):
    rows = window._collect_metrics_rows()
    timepoint_rows = rows[1:]

    assert timepoint_rows[0].scalars["Delta_n"] == pytest.approx(0.0, abs=1e-9)
    # The blob drifts up in neutron, so the marginal mean follows it
    assert timepoint_rows[1].scalars["Delta_n"] > 0
    assert timepoint_rows[2].scalars["Delta_n"] > timepoint_rows[1].scalars["Delta_n"]


def test_class_metrics_follow_the_segmented_classes(window, monkeypatch):
    _segment_all_as(window, "Lithium", monkeypatch)
    rows = window._collect_metrics_rows()

    for row in rows:
        assert "Lithium" in row.per_class["voxels_k"], row.label
        assert row.scalars["n_classes"] == 1
        # A single class has no other class to separate from
        assert row.scalars["DB"] is None

    timepoint_rows = rows[1:]
    assert timepoint_rows[0].per_class["drift_k"] == {}       # the reference
    assert timepoint_rows[1].per_class["drift_k"]["Lithium"] > 0
    assert (
        timepoint_rows[2].per_class["drift_k"]["Lithium"]
        > timepoint_rows[1].per_class["drift_k"]["Lithium"]
    )
    assert timepoint_rows[2].scalars["CD"] == pytest.approx(
        timepoint_rows[2].per_class["drift_k"]["Lithium"]
    )


def test_hidden_classes_are_left_out_of_the_metrics(window, monkeypatch):
    _segment_all_as(window, "Lithium", monkeypatch)
    roi_manager = window.dual_histogram.get_roi_manager()
    roi_manager.set_named_roi_visible(0, False)

    rows = window._collect_metrics_rows()
    assert all(not row.per_class for row in rows)


def test_metrics_export_writes_csv_and_plot(window, monkeypatch, tmp_path):
    _segment_all_as(window, "Lithium", monkeypatch)
    target = tmp_path / "metrics.csv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")),
    )
    window._on_export_histogram_metrics()

    assert target.exists()
    plot = tmp_path / "metrics_evolution.png"
    assert plot.exists() and plot.stat().st_size > 5000

    with open(target, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    metrics = {record["metric"] for record in records}
    assert {"S_h", "S_v", "S_d", "A_x", "Delta_n"} <= metrics
    assert {"voxels_k", "centroid_n_k", "drift_k"} <= metrics
    assert {record["timepoint"] for record in records} == {"", "0", "1", "2"}


def test_metrics_export_appends_the_csv_suffix(window, monkeypatch, tmp_path):
    target = tmp_path / "metrics"
    shown = []
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: shown.append(a[-1])),
    )
    window._on_export_histogram_metrics()

    assert (tmp_path / "metrics.csv").exists()
    # Without classes the user is told which metrics could not be computed
    assert shown and "need" in shown[-1] and "segmented classes" in shown[-1]


# ── segmentation report ──────────────────────────────────────────────────────

def _export_all(window, monkeypatch, outdir):
    from gui import main_window as mw

    class FakeDialog(mw.ExportOptionsDialog):
        def exec_(self):
            self.selected_layers = list(self.layers)
            self.export_mask = False
            self.export_neutron = False
            self.export_xray = False
            self.export_labels = True
            self.export_histogram = False
            self.export_report = True
            return QDialog.Accepted

    monkeypatch.setattr(mw, "ExportOptionsDialog", FakeDialog)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: str(outdir)),
    )
    window._export_all_timepoints()


def test_report_records_names_values_and_counts(window, monkeypatch, tmp_path):
    _segment_all_as(window, "Lithium", monkeypatch)
    _export_all(window, monkeypatch, tmp_path)

    report_path = tmp_path / "segmentation_report.txt"
    assert report_path.exists()
    report = report_path.read_text()

    assert "Lithium" in report
    assert "Class legend" in report and "value" in report
    assert "Voxels per class and timepoint" in report
    for timepoint in range(window.dataset.num_timepoints):
        voxels = int(window._visible_layers(timepoint)[0][0].sum())
        assert f"{voxels:,}" in report
    # Provenance
    assert "histogram bins" in report
    assert "neutron range" in report
    assert "rectangle ROI" in report


def test_report_label_value_matches_the_exported_label_volume(
    window, monkeypatch, tmp_path
):
    tifffile = pytest.importorskip("tifffile")
    _segment_all_as(window, "Lithium", monkeypatch)
    _export_all(window, monkeypatch, tmp_path)

    report = (tmp_path / "segmentation_report.txt").read_text()
    legend = [line for line in report.splitlines() if "Lithium" in line]
    assert legend and legend[0].split()[0] == "1"

    labels = [
        name for name in os.listdir(tmp_path)
        if "labels" in name and name.endswith((".tif", ".tiff"))
    ]
    assert labels
    volume = tifffile.imread(os.path.join(tmp_path, sorted(labels)[0]))
    assert set(np.unique(volume)) == {0, 1}


def test_label_values_are_stable_across_timepoints(window, monkeypatch, tmp_path):
    """A label value must mean the same class at every timepoint."""
    tifffile = pytest.importorskip("tifffile")
    _segment_all_as(window, "Lithium", monkeypatch)
    _export_all(window, monkeypatch, tmp_path)

    labels = sorted(
        name for name in os.listdir(tmp_path)
        if "labels" in name and name.endswith((".tif", ".tiff"))
    )
    assert len(labels) == window.dataset.num_timepoints
    for name in labels:
        volume = tifffile.imread(os.path.join(tmp_path, name))
        assert set(np.unique(volume)) <= {0, 1}
        assert (volume == 1).any()
