"""One material, one colour, everywhere it appears.

A class shows up in three places — the histogram overlay, the selection
panel, and the highlight in the slice viewer — and a user reads those three
as the same object. If they disagree, the colour stops being a way to
identify anything, which is the whole reason it is there.

The colour therefore has to be resolved from the class itself, never from
where the class happens to sit in some list. Positions drift: reloading from
a file, hiding a class, or re-running a segmentation can all reorder the
layers relative to the saved regions.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.colors as mcolors  # noqa: E402
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from utils.roi_manager import ROIManager  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    rng = np.random.default_rng(0)
    shape = (2, 4, 16, 16)
    neutron = np.zeros(shape)
    xray = np.zeros(shape)
    bands = [(300.0, 300.0), (800.0, 800.0), (1400.0, 1400.0)]
    for timepoint in range(shape[0]):
        for index, (mean_n, mean_x) in enumerate(bands):
            mask = np.zeros(shape[1:], dtype=bool)
            mask[:, :, index * 5:(index + 1) * 5] = True
            count = int(mask.sum())
            neutron[timepoint][mask] = np.clip(
                rng.normal(mean_n, 40, count), 20, 1900
            )
            xray[timepoint][mask] = np.clip(
                rng.normal(mean_x, 40, count), 20, 1900
            )

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    return w


def _rgb(colour):
    """Any matplotlib-accepted colour as a comparable (r, g, b)."""
    red, green, blue, _alpha = mcolors.to_rgba(colour)
    return (round(red, 4), round(green, 4), round(blue, 4))


def _save_classes(window, monkeypatch, names_and_boxes):
    roi_manager = window.dual_histogram.get_roi_manager()
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )
    for name, (low, high) in names_and_boxes:
        roi_manager.set_rectangle_roi(low, low, high, high)
        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: (name, True))
        )
        window.dual_histogram._save_current_as_class()


CLASSES = [
    ("Lithium", (200, 400)),
    ("Separator", (700, 900)),
    ("Aluminium", (1300, 1500)),
]


# ── the colour comes from the class, not from a list position ────────────────

def test_layer_colour_matches_the_region_that_produced_it(window, monkeypatch):
    _save_classes(window, monkeypatch, CLASSES)
    window._segment_current_volume()

    roi_manager = window.dual_histogram.get_roi_manager()
    saved = {roi["name"]: roi["color"] for roi in roi_manager.named_rois}

    for _mask, colour, name in window._visible_layers(0):
        assert name in saved, name
        assert _rgb(colour) == _rgb(saved[name]), (
            f"{name} is {colour} in the slice viewer but {saved[name]} on "
            f"the histogram"
        )


def test_colours_survive_a_save_and_reload(window, monkeypatch, tmp_path):
    """The reported bug: after reloading, the three views disagreed."""
    _save_classes(window, monkeypatch, CLASSES)
    roi_manager = window.dual_histogram.get_roi_manager()
    before = {roi["name"]: roi["color"] for roi in roi_manager.named_rois}

    path = tmp_path / "rois.json"
    roi_manager.save_to_file(str(path))
    roi_manager.clear_named_rois()
    roi_manager.load_from_file(str(path))

    after = {roi["name"]: roi["color"] for roi in roi_manager.named_rois}
    assert after == before

    window._segment_current_volume()
    for _mask, colour, name in window._visible_layers(0):
        assert _rgb(colour) == _rgb(after[name]), name


def test_hiding_a_class_does_not_recolour_the_others(window, monkeypatch):
    """Hiding shifts every later class's position in the visible list."""
    _save_classes(window, monkeypatch, CLASSES)
    window._segment_current_volume()
    original = {
        name: _rgb(colour)
        for _mask, colour, name in window._visible_layers(0)
    }

    window.dual_histogram.get_roi_manager().set_named_roi_visible(0, False)
    window._segment_current_volume()

    for _mask, colour, name in window._visible_layers(0):
        assert _rgb(colour) == original[name], (
            f"{name} changed colour because another class was hidden"
        )


def test_tracking_the_series_keeps_each_material_its_own_colour(
    window, monkeypatch
):
    """The path that produced the reported mismatch."""
    _save_classes(window, monkeypatch, CLASSES)
    window._segment_current_volume()

    roi_manager = window.dual_histogram.get_roi_manager()
    saved = {roi["name"]: _rgb(roi["color"]) for roi in roi_manager.named_rois}

    window._refresh_material_panel()
    window._run_material_tracking(preview=False)
    if window.model_result is None:
        pytest.skip("the run was refused on this fixture")

    for timepoint in range(window.dataset.num_timepoints):
        for _mask, colour, name in window.segmentation_masks[timepoint]:
            assert _rgb(colour) == saved[name], (
                f"{name} is a different colour at timepoint {timepoint}"
            )


def test_the_histogram_outline_uses_the_same_colour(window, monkeypatch):
    _save_classes(window, monkeypatch, CLASSES)
    window._segment_current_volume()
    window._update_class_histogram_overlays(0)

    roi_manager = window.dual_histogram.get_roi_manager()
    saved = {roi["name"]: _rgb(roi["color"]) for roi in roi_manager.named_rois}

    for label, _vertices, colour in window.dual_histogram.global_canvas.roi_overlays:
        for name, expected in saved.items():
            if name in label:
                assert _rgb(colour) == expected, label


# ── the panel shows the colour legibly ───────────────────────────────────────

def test_the_panel_shows_a_solid_colour_swatch(window, monkeypatch):
    """A washed-out background reads as a different colour entirely."""
    _save_classes(window, monkeypatch, CLASSES)
    panel = window.dual_histogram

    for row, roi in enumerate(panel.roi_manager.named_rois):
        item = panel.roi_list_widget.item(row)
        icon = item.icon()
        assert not icon.isNull(), f"row {row} has no colour swatch"

        pixmap = icon.pixmap(16, 16)
        image = pixmap.toImage()
        centre = image.pixelColor(pixmap.width() // 2, pixmap.height() // 2)
        expected = _rgb(roi["color"])
        actual = (
            round(centre.red() / 255, 4),
            round(centre.green() / 255, 4),
            round(centre.blue() / 255, 4),
        )
        assert all(
            abs(a - b) < 0.02 for a, b in zip(actual, expected)
        ), f"swatch for {roi['name']} is {actual}, region colour is {expected}"


def test_a_hidden_class_keeps_its_colour_but_looks_hidden(window, monkeypatch):
    _save_classes(window, monkeypatch, CLASSES)
    panel = window.dual_histogram
    panel.roi_manager.set_named_roi_visible(0, False)
    panel._update_roi_list()

    item = panel.roi_list_widget.item(0)
    pixmap = item.icon().pixmap(16, 16)
    centre = pixmap.toImage().pixelColor(
        pixmap.width() // 2, pixmap.height() // 2
    )
    expected = _rgb(panel.roi_manager.named_rois[0]["color"])
    actual = (
        round(centre.red() / 255, 4),
        round(centre.green() / 255, 4),
        round(centre.blue() / 255, 4),
    )
    # Same hue — identity must not depend on visibility
    assert all(abs(a - b) < 0.02 for a, b in zip(actual, expected))
    # ...but the row is clearly marked as off
    assert item.checkState() == 0


# ── one palette ──────────────────────────────────────────────────────────────

def test_there_is_only_one_class_palette():
    """Two palettes is how the same class ends up two colours."""
    from gui.main_window import BiTS4DMainWindow
    from utils.roi_manager import CLASS_COLORS

    overlay = [_rgb(colour) for colour in BiTS4DMainWindow._OVERLAY_COLORS]
    canonical = [_rgb(colour) for colour in CLASS_COLORS]
    assert overlay == canonical[:len(overlay)]


def test_the_fallback_colour_is_keyed_on_the_class_not_the_row():
    """Two classes must not collide just because one was hidden."""
    from gui.main_window import BiTS4DMainWindow

    roi_manager = ROIManager()
    roi_manager.set_rectangle_roi(0, 0, 1, 1)
    roi_manager.add_named_roi("A")
    roi_manager.set_rectangle_roi(0, 0, 1, 1)
    roi_manager.add_named_roi("B")

    resolve = BiTS4DMainWindow._colour_for_layer
    assert _rgb(resolve(roi_manager, "A")) == _rgb(
        roi_manager.named_rois[0]["color"]
    )
    assert _rgb(resolve(roi_manager, "B")) == _rgb(
        roi_manager.named_rois[1]["color"]
    )
    # An unknown name still gets a stable colour, derived from the name
    first = resolve(roi_manager, "Otsu 3")
    assert _rgb(first) == _rgb(resolve(roi_manager, "Otsu 3"))
    assert _rgb(first) != _rgb(resolve(roi_manager, "Otsu 4"))
