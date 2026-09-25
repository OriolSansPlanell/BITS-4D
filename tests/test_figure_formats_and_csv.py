"""Every analysis figure: chosen format (SVG by default) and optional CSV.

The pure writers are tested without Qt; the pop-up and the menu actions
that use it are tested offscreen.
"""

import csv
import os

import numpy as np
import pytest

from histograms.histogram_engine_4d import HistogramEngine4D
from utils.figure_io import figure_path, save_figure, write_csv


def _histograms(timepoints=3, seed=0):
    rng = np.random.default_rng(seed)
    neutron = rng.normal(500, 30, (timepoints, 4, 16, 16))
    xray = rng.normal(500, 30, (timepoints, 4, 16, 16))
    # A phase that grows over time, so the differences are not zero
    for t in range(timepoints):
        neutron[t, :, : 2 + 3 * t] = 200.0
        xray[t, :, : 2 + 3 * t] = 800.0
    engine = HistogramEngine4D(bins=32, use_gpu=False)
    engine.compute_global_histogram(neutron, xray)
    return [engine.compute_local_histogram(neutron[t], xray[t], t)
            for t in range(timepoints)]


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


# ── the shared writer ────────────────────────────────────────────────────────

@pytest.mark.parametrize("base,fmt,expected", [
    ("plot", "svg", "plot.svg"),
    ("plot.png", "svg", "plot.svg"),
    ("plot.csv", "pdf", "plot.pdf"),
    ("my.data.v2", "tif", "my.data.v2.tif"),
])
def test_figure_path_swaps_the_extension(base, fmt, expected):
    assert figure_path(base, fmt).name == expected


@pytest.mark.parametrize("fmt", ["svg", "pdf", "png", "tif"])
def test_save_figure_writes_every_format(tmp_path, fmt):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure()
    FigureCanvasAgg(figure)
    figure.add_subplot(111).plot([0, 1], [0, 1], label="Lithium")
    path = save_figure(figure, tmp_path / f"f.{fmt}", dpi=72)
    assert os.path.getsize(path) > 0
    if fmt == "svg":
        # Text stays text, so labels can be edited afterwards
        figure.axes[0].set_title("Editable title")
        save_figure(figure, tmp_path / "t.svg")
        assert "Editable title" in (tmp_path / "t.svg").read_text()


def test_write_csv(tmp_path):
    path = write_csv(tmp_path / "d.csv", ("a", "b"), [(1, 2.5), (3, "")])
    assert _read(path) == [{"a": "1", "b": "2.5"}, {"a": "3", "b": ""}]


# ── the data behind each analysis figure ─────────────────────────────────────

def test_evolution_csv_summary_and_bins(tmp_path):
    from utils.histogram_evolution import (
        REFERENCE_FIRST, compute_log_differences, write_evolution_csv,
    )
    histograms = _histograms()
    summary_path, bins_path = tmp_path / "e.csv", tmp_path / "e_bins.csv"
    write_evolution_csv(histograms, summary_path, bins_path, REFERENCE_FIRST)

    summary = _read(summary_path)
    assert [row["timepoint"] for row in summary] == ["1", "2"]
    assert all(row["reference_timepoint"] == "0" for row in summary)
    changed = [float(row["share_of_voxels_changed"]) for row in summary]
    assert 0 < changed[0] < changed[1] <= 1

    bins = _read(bins_path)
    differences = compute_log_differences(histograms, REFERENCE_FIRST)
    # Every non-empty bin, with the plotted value
    expected = sum(
        int(np.count_nonzero((h.histogram > 0) | (histograms[0].histogram > 0)))
        for h in histograms[1:]
    )
    assert len(bins) == expected
    row = bins[0]
    x_index = int(np.argmin(np.abs(histograms[0].x_centers
                                   - float(row["neutron_center"]))))
    y_index = int(np.argmin(np.abs(histograms[0].y_centers
                                   - float(row["xray_center"]))))
    assert float(row["log10_difference"]) == pytest.approx(
        differences[int(row["timepoint"]) - 1][y_index, x_index])


def test_incremental_evolution_csv_uses_the_previous_timepoint(tmp_path):
    from utils.histogram_evolution import REFERENCE_PREVIOUS, write_evolution_csv
    write_evolution_csv(_histograms(), tmp_path / "i.csv", None,
                        REFERENCE_PREVIOUS)
    rows = _read(tmp_path / "i.csv")
    assert [(r["timepoint"], r["reference_timepoint"]) for r in rows] == [
        ("1", "0"), ("2", "1")]


def test_marginal_csv(tmp_path):
    from utils.histogram_evolution import write_marginal_csv
    histograms = _histograms()
    rows = _read(write_marginal_csv(histograms, tmp_path / "m.csv"))
    bins = len(histograms[0].x_centers)
    assert len(rows) == 3 * 2 * bins
    # Each timepoint's shares of one modality add up to 1
    shares = sum(float(r["share_of_voxels"]) for r in rows
                 if r["timepoint"] == "1" and r["modality"] == "neutron")
    assert shares == pytest.approx(1.0)
    # T0 against itself: no change
    assert all(float(r["log2_change"]) == 0 for r in rows
               if r["timepoint"] == "0" and r["log2_change"] != "")


def test_class_histogram_in_any_format_with_csv(tmp_path):
    from utils.histogram_export import save_class_histogram
    histogram = _histograms()[0]
    written = save_class_histogram(histogram, tmp_path / "Lithium_hist",
                                   image_format="svg", write_csv=True)
    assert set(written) == {"Lithium_hist.npy", "Lithium_hist.svg",
                            "Lithium_hist.csv"}
    rows = _read(tmp_path / "Lithium_hist.csv")
    assert sum(int(r["count"]) for r in rows) == int(histogram.histogram.sum())


def test_histogram_slice_csv(tmp_path):
    from utils.figure_export import (
        HistogramPanel, SlicePanel, write_histogram_slice_csv,
    )
    histogram = _histograms()[0]
    square = np.array([[100, 700], [300, 700], [300, 900], [100, 900]], float)
    mask = np.zeros((16, 16), bool)
    mask[:4] = True
    write_histogram_slice_csv(
        tmp_path / "hs.csv",
        HistogramPanel(histogram, [("Class 1: Lithium", square, "#e6194b")]),
        SlicePanel(np.zeros((16, 16)), [("Lithium", mask, "#e6194b")]),
    )
    rows = {(r["panel"], r["label"]): r for r in _read(tmp_path / "hs.csv")}
    hist_row = rows[("histogram", "Class 1: Lithium")]
    assert int(hist_row["count"]) == 4 * 2 * 16      # the 200/800 voxels at T0
    assert rows[("slice", "Lithium")]["count"] == "64"
    assert float(rows[("slice", "Lithium")]["share"]) == pytest.approx(0.25)


# ── the pop-up and the menu actions ──────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _answer_popup(monkeypatch, fmt="svg", save_csv=True, dpi=100):
    """Stub the pop-up: pick *fmt*, the CSV choice and the resolution."""
    from PyQt5.QtWidgets import QDialog
    from gui.figure_save_dialog import FigureSaveDialog

    def exec_(self):
        self._format_buttons[fmt].setChecked(True)
        self._csv_cb.setChecked(save_csv)
        self._dpi_spin.setValue(dpi)
        return QDialog.Accepted

    monkeypatch.setattr(FigureSaveDialog, "exec_", exec_)


def _answer_save_as(monkeypatch, path):
    from PyQt5.QtWidgets import QFileDialog
    asked = []

    def get(*args, **_kwargs):
        asked.append(args)
        return str(path), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(get))
    return asked


def test_popup_offers_svg_first_and_remembers_the_choice(qapp, monkeypatch,
                                                         tmp_path):
    from gui.figure_save_dialog import FigureSaveDialog, ask_figure_output

    dialog = FigureSaveDialog("t", csv_label="csv")
    assert dialog.fmt == "svg"
    _answer_popup(monkeypatch, fmt="png", save_csv=False, dpi=150)
    asked = _answer_save_as(monkeypatch, tmp_path / "chosen")
    output = ask_figure_output(None, "Save", "default_name")
    assert output.figure_path == tmp_path / "chosen.png"
    assert not output.save_csv and output.dpi == 150
    # The save dialog suggested the chosen format
    assert asked[0][2].endswith("default_name.png")
    assert "*.png" in asked[0][3]
    # Next time the pop-up starts from the last answer
    assert FigureSaveDialog("t").fmt == "png"


def test_popup_without_data_hides_the_csv_option(qapp):
    from gui.figure_save_dialog import FigureSaveDialog
    dialog = FigureSaveDialog("t", csv_label=None)
    assert dialog._csv_cb.isHidden() and not dialog.save_csv


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow
    from data import Dataset4D

    rng = np.random.default_rng(1)
    neutron = rng.normal(500, 30, (3, 4, 16, 16))
    xray = rng.normal(500, 30, (3, 4, 16, 16))
    for t in range(3):
        neutron[t, :, : 2 + 3 * t] = 200.0
        xray[t, :, : 2 + 3 * t] = 800.0
    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=32, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    return w


@pytest.fixture()
def quiet(monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))


@pytest.mark.parametrize("action,extra_csv", [
    ("_on_export_histogram_evolution", ["_bins"]),
    ("_on_export_histogram_increment", ["_bins"]),
    ("_on_export_marginal_evolution", []),
    ("_on_export_marginal_increment", []),
])
def test_time_analyses_save_svg_and_csv(window, monkeypatch, quiet, tmp_path,
                                        action, extra_csv):
    _answer_popup(monkeypatch, fmt="svg", save_csv=True)
    _answer_save_as(monkeypatch, tmp_path / "analysis")
    getattr(window, action)()
    assert (tmp_path / "analysis.svg").stat().st_size > 0
    assert (tmp_path / "analysis.csv").exists()
    for suffix in extra_csv:
        assert (tmp_path / f"analysis{suffix}.csv").exists()


def test_time_analysis_without_csv_writes_only_the_figure(
        window, monkeypatch, quiet, tmp_path):
    _answer_popup(monkeypatch, fmt="pdf", save_csv=False)
    _answer_save_as(monkeypatch, tmp_path / "analysis")
    window._on_export_marginal_evolution()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["analysis.pdf"]


def test_metrics_figure_in_the_chosen_format(window, monkeypatch, quiet,
                                             tmp_path):
    _answer_popup(monkeypatch, fmt="png", save_csv=True)
    _answer_save_as(monkeypatch, tmp_path / "metrics")
    window._on_export_histogram_metrics()
    assert (tmp_path / "metrics.png").exists()
    assert (tmp_path / "metrics.csv").exists()


def test_histogram_slice_figure_with_label_csv(window, monkeypatch, quiet,
                                               tmp_path):
    import gui.main_window as main_window
    from PyQt5.QtWidgets import QDialog

    manager = window.dual_histogram.get_roi_manager()
    manager.set_rectangle_roi(150, 750, 250, 850)
    manager.add_named_roi("Lithium")
    manager.clear_roi()
    window.dual_histogram._apply_roi_change()
    monkeypatch.setattr(main_window.FigureExportDialog, "exec_",
                        lambda self: QDialog.Accepted)
    _answer_save_as(monkeypatch, tmp_path / "figure")
    window._on_export_histogram_slice_figure()
    assert (tmp_path / "figure.svg").exists()
    rows = _read(tmp_path / "figure.csv")
    assert [r["label"] for r in rows if r["panel"] == "histogram"] == [
        "Class 1: Lithium"]


def test_export_dialog_passes_the_histogram_format(qapp):
    from gui.main_window import ExportOptionsDialog
    mask = np.zeros((2, 2, 2), bool)
    mask[0] = True
    dialog = ExportOptionsDialog([(mask, (1, 0, 0, 0.5), "Lithium")])
    dialog._histogram_cb.setChecked(True)
    dialog._histogram_format.setCurrentIndex(
        dialog._histogram_format.findData("pdf"))
    dialog._histogram_csv_cb.setChecked(True)
    dialog._on_accept()
    assert dialog.export_histogram
    assert dialog.histogram_format == "pdf" and dialog.histogram_csv


def test_kmeans_timeline_in_the_chosen_format(window, monkeypatch, quiet,
                                              tmp_path):
    window.kmeans_scope_combo.setCurrentIndex(
        window.kmeans_scope_combo.findData("series"))
    window.kmeans_clusters_spin.setValue(2)
    window._run_kmeans()
    _answer_popup(monkeypatch, fmt="svg", save_csv=True)
    _answer_save_as(monkeypatch, tmp_path / "timeline")
    window._export_kmeans_timeline()
    assert (tmp_path / "timeline.svg").exists()
    rows = _read(tmp_path / "timeline.csv")
    assert {r["timepoint"] for r in rows} == {"0", "1", "2"}
