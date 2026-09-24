"""The histogram + slice figure export, and the fixes made alongside it.

* Otsu imported the classifier module from where it used to live, so the
  button always failed.
* The "definitions allowed to move" tracking path called a helper that no
  longer existed, so it crashed after running.
* Toggling saved selections replaced the histogram overlays, wiping the class
  outlines; changing timepoint left the previous timepoint's layer outlines.
* Recalling a saved selection kept a stale 3-D region-grow mask, which
  "Create Histogram ROI" then used instead of the recalled region.
* Saving a 3- or 4-slice uint8 volume wrote it as an RGB picture.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

RECT_A = (50, 850, 150, 950)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp, monkeypatch):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    neutron = np.full((2, 4, 20, 20), 500.0)
    xray = np.full((2, 4, 20, 20), 500.0)
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
    w._blob = blob[0]
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )

    # An error dialog would block the test run; fail with its text instead
    def _error_dialog(*args, **_kwargs):
        raise AssertionError(" | ".join(str(a) for a in args[1:]))

    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_error_dialog))
    return w


def _save_class(window, monkeypatch, name, rect=RECT_A):
    manager = window.dual_histogram.get_roi_manager()
    manager.set_rectangle_roi(*rect)
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        staticmethod(lambda *a, **k: (name, True)),
    )
    window.dual_histogram._save_current_as_class()
    window._segment_current_volume()


# ── figure export ────────────────────────────────────────────────────────────

def test_panels_carry_the_labels_on_screen(window, monkeypatch):
    _save_class(window, monkeypatch, "Lithium")
    viewer = window.slice_viewer
    viewer.current_axis = 'z'
    viewer.current_slice_index = 1
    viewer._update_display()

    hist_panel, slice_panel = window._histogram_slice_figure_panels()
    assert hist_panel.histogram_data is window.dual_histogram.local_canvas.histogram_data
    assert [o[0] for o in hist_panel.overlays] == ["Class 1: Lithium"]
    assert [o[0] for o in slice_panel.overlays] == ["Lithium"]
    np.testing.assert_array_equal(slice_panel.overlays[0][1], window._blob[1])
    np.testing.assert_array_equal(slice_panel.image, viewer.current_slice)
    assert "Z=1" in slice_panel.title


def test_active_roi_is_optional(window):
    window.dual_histogram.get_roi_manager().set_rectangle_roi(*RECT_A)
    with_active, _ = window._histogram_slice_figure_panels(True)
    without, _ = window._histogram_slice_figure_panels(False)
    assert "Active ROI (unsaved)" in [o[0] for o in with_active.overlays]
    assert "Active ROI (unsaved)" not in [o[0] for o in without.overlays]


def test_hidden_class_is_left_out_of_both_panels(window, monkeypatch):
    _save_class(window, monkeypatch, "Lithium")
    window.dual_histogram.get_roi_manager().set_named_roi_visible(0, False)
    window.dual_histogram._apply_roi_change()
    hist_panel, slice_panel = window._histogram_slice_figure_panels(False)
    assert hist_panel.overlays == []
    assert slice_panel.overlays == []


def test_export_action_writes_the_file(window, monkeypatch, tmp_path):
    from PyQt5.QtWidgets import QDialog, QFileDialog
    import gui.main_window as main_window

    _save_class(window, monkeypatch, "Lithium")
    monkeypatch.setattr(
        main_window.FigureExportDialog, "exec_",
        lambda self: (self._on_accept(), QDialog.Accepted)[1],
    )
    target = tmp_path / "figure.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")),
    )
    window._on_export_histogram_slice_figure()
    assert target.exists() and target.stat().st_size > 0


# ── fixes ────────────────────────────────────────────────────────────────────

def test_otsu_runs(window):
    window.otsu_classes_spin.setValue(2)
    window._run_otsu_segment()
    names = {layer[2] for layer in window.segmentation_masks[0]}
    assert any(name.startswith("Otsu class") for name in names)


def test_adaptive_tracking_installs_its_layers(window, monkeypatch):
    import gui.main_window as main_window

    labels = np.zeros((4, 20, 20), dtype=np.int32)
    labels[:, :8, :8] = 1
    names = ["Background", "Lithium"]
    entry = SimpleNamespace(
        timepoint=0, class_names=names, fit=None,
        mask_for=lambda name: labels == names.index(name),
    )
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))

    class _Outcome(list):
        """What SequentialSegmenter.run returns, reduced to what is read."""
        timepoints = [entry]

    monkeypatch.setattr(main_window, "run_with_progress",
                        lambda *a, **k: _Outcome([entry]))
    settings = {"control_materials": [], "smoothing_strength": 0.0}
    window._run_adaptive_tracking(settings, 0, {"Lithium": window._blob})
    assert {layer[2] for layer in window.segmentation_masks[0]} == set(names)


def test_selections_do_not_wipe_class_outlines(window, monkeypatch):
    _save_class(window, monkeypatch, "Lithium")
    window.selection_manager.add_selection(
        name="picked", spatial_mask=None,
        histogram_roi=np.array([[0, 0], [10, 0], [10, 10]], float),
    )
    for show_all in (True, False):
        window.selection_manager.show_all_cb.setChecked(show_all)
        window._update_histogram_overlays()
        names = [o[0] for o in window.dual_histogram.local_canvas.roi_overlays]
        assert "Class 1: Lithium" in names
        assert ("picked" in names) == show_all


def test_layer_outlines_follow_the_timepoint(window):
    window.segmentation_masks[1] = [
        (np.broadcast_to(window._blob, (4, 20, 20)).copy(),
         (1, 0, 0, 0.5), "Only at T1"),
    ]
    window._update_current_timepoint(1)
    names = [o[0] for o in window.dual_histogram.local_canvas.roi_overlays]
    assert "Seg: Only at T1" in names
    window._update_current_timepoint(0)
    names = [o[0] for o in window.dual_histogram.local_canvas.roi_overlays]
    assert "Seg: Only at T1" not in names


def test_recalling_a_selection_drops_a_stale_3d_grow(window):
    viewer = window.slice_viewer
    viewer.region_grow_mask_3d = np.ones((4, 20, 20), dtype=bool)
    flat = np.zeros(viewer.current_slice.shape, dtype=bool)
    flat[:2, :2] = True
    window.selection_manager.add_selection(
        name="flat", spatial_mask=flat, histogram_roi=None,
        source_axis='z', source_slice_index=viewer.current_slice_index,
    )
    window._on_selection_recalled(window.selection_manager.selections[-1])
    assert viewer.region_grow_mask_3d is None
    np.testing.assert_array_equal(viewer.region_grow_mask, flat)


def test_region_grow_highlight_survives_a_redraw(window):
    viewer = window.slice_viewer
    viewer.current_axis = 'z'
    viewer.current_slice_index = 1
    viewer._update_display()
    viewer.region_grow_mask = window._blob[1].copy()
    viewer.region_grow_plane = ('z', 1)
    viewer._display_mask_overlay()
    assert viewer.mask_overlay is not None

    viewer._update_display()            # e.g. a contrast change
    assert viewer.mask_overlay is not None

    viewer.current_slice_index = 2      # another slice: not drawn there
    viewer._update_display()
    assert viewer.mask_overlay is None


def test_small_uint8_volumes_stay_greyscale_stacks(tmp_path):
    import tifffile
    from data.tiff_io import write_volume_tiff

    volume = (np.arange(4 * 5 * 6) % 256).astype(np.uint8).reshape(4, 5, 6)
    path = tmp_path / "labels.tif"
    write_volume_tiff(path, volume)
    np.testing.assert_array_equal(tifffile.imread(path), volume)
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].photometric == tifffile.PHOTOMETRIC.MINISBLACK
