"""Several unsaved ROIs; selecting, moving and removing ROIs on the histogram.

* Drawing a second ROI keeps the first; each unsaved ROI has its own colour
  (the colour it will have as a class).
* "Save as Class" with two ROIs drawn makes two classes, one name each.
* Clicking an ROI on the histogram selects it; the arrow keys move it by one
  histogram bin (Shift: ten) and Backspace/Delete removes it.
"""

import os

import numpy as np
import pytest

from utils.roi_manager import ROIManager

RECT_A = (50.0, 850.0, 150.0, 950.0)
RECT_B = (850.0, 50.0, 950.0, 150.0)


# ── the model (no Qt) ───────────────────────────────────────────────────────

def _two_unsaved():
    manager = ROIManager()
    manager.set_rectangle_roi(*RECT_A)
    manager.stash_active()
    manager.set_rectangle_roi(*RECT_B)
    return manager


def test_drawing_a_second_roi_keeps_the_first():
    manager = _two_unsaved()
    assert manager.unsaved_count() == 2
    unsaved = manager.get_unsaved_rois()
    assert [roi['rectangle'] for roi in unsaved] == [RECT_A, RECT_B]
    # Each in its own colour
    assert unsaved[0]['color'] != unsaved[1]['color']
    # Both are segmented
    x = np.array([100.0, 900.0, 500.0])
    y = np.array([900.0, 100.0, 500.0])
    np.testing.assert_array_equal(manager.is_inside_roi(x, y),
                                  [True, True, False])


def test_saving_two_unsaved_rois_makes_two_classes():
    manager = _two_unsaved()
    colors = [roi['color'] for roi in manager.get_unsaved_rois()]
    ids = manager.save_unsaved_as_classes(["Lithium", "Steel"])
    assert ids == [1, 2]
    assert [roi['name'] for roi in manager.named_rois] == ["Lithium", "Steel"]
    assert [roi['rectangle'] for roi in manager.named_rois] == [RECT_A, RECT_B]
    # Saved in the colours they were drawn in
    assert [roi['color'] for roi in manager.named_rois] == colors
    assert manager.unsaved_count() == 0


def test_saving_needs_one_name_per_roi():
    manager = _two_unsaved()
    with pytest.raises(ValueError):
        manager.save_unsaved_as_classes(["Only one"])
    assert manager.unsaved_count() == 2


def test_hit_test_finds_the_topmost_roi():
    manager = _two_unsaved()
    assert manager.hit_test(100, 900) == ('pending', 0)
    assert manager.hit_test(900, 100) == ('active', 0)
    assert manager.hit_test(500, 500) is None


def test_translate_moves_only_the_chosen_roi():
    manager = _two_unsaved()
    manager.translate(('pending', 0), 10.0, -5.0)
    assert manager.pending_rois[0]['rectangle'] == (60.0, 845.0, 160.0, 945.0)
    assert manager.rectangle == RECT_B


def test_translate_moves_a_saved_class():
    manager = ROIManager()
    manager.set_polygon_roi(np.array([[0, 0], [10, 0], [10, 10]], float))
    manager.add_named_roi("Tri")
    manager.clear_roi()
    manager.translate(('named', 0), 1.0, 2.0)
    np.testing.assert_allclose(manager.named_rois[0]['points'],
                               [[1, 2], [11, 2], [11, 12]])


def test_removing_the_active_roi_promotes_the_previous_one():
    manager = _two_unsaved()
    manager.remove_unsaved(('active', 0))
    assert manager.unsaved_count() == 1
    assert manager.rectangle == RECT_A
    assert manager.pending_rois == []


def test_selection_is_dropped_when_its_roi_goes():
    manager = _two_unsaved()
    manager.select(('pending', 0))
    manager.remove_unsaved(('pending', 0))
    assert manager.get_selected() is None


def test_editing_a_class_keeps_what_was_being_drawn():
    manager = ROIManager()
    manager.set_rectangle_roi(*RECT_A)
    manager.add_named_roi("A")
    manager.clear_roi()
    manager.set_rectangle_roi(*RECT_B)
    manager.take_named_roi(0)
    assert manager.unsaved_count() == 2
    assert manager.rectangle == RECT_A          # the class, now editable
    assert manager.pending_rois[0]['rectangle'] == RECT_B


def test_unsaved_rois_survive_save_and_load():
    manager = _two_unsaved()
    restored = ROIManager()
    restored.load_from_dict(manager.save_to_dict())
    assert [r['rectangle'] for r in restored.get_unsaved_rois()] == \
        [RECT_A, RECT_B]


# ── the histogram panel (Qt) ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def panel(qapp):
    from gui.dual_histogram_widget import DualHistogramWidget
    from histograms import HistogramEngine4D

    neutron = np.full((1, 4, 16, 16), 500.0)
    xray = np.full((1, 4, 16, 16), 500.0)
    neutron[:, :, :4, :4] = 100.0
    xray[:, :, :4, :4] = 900.0
    neutron[:, :, -4:, -4:] = 900.0
    xray[:, :, -4:, -4:] = 100.0
    histogram = HistogramEngine4D(bins=64, use_gpu=False) \
        .compute_global_histogram(neutron, xray)
    widget = DualHistogramWidget()
    widget.set_global_histogram(histogram)
    widget.set_local_histogram(histogram)
    widget.resize(900, 700)
    return widget


def _draw_rectangle(canvas, rect):
    """Drive the rectangle tool the way the mouse does."""
    from matplotlib.backend_bases import MouseEvent

    canvas.set_drawing_mode('rectangle')
    canvas.draw()
    x1, y1, x2, y2 = rect
    for name, (x, y) in (("button_press_event", (x1, y1)),
                         ("button_release_event", (x2, y2))):
        px, py = canvas.ax.transData.transform((x, y))
        canvas.callbacks.process(
            name, MouseEvent(name, canvas, px, py, button=1)
        )


def _click(canvas, x, y):
    from matplotlib.backend_bases import MouseEvent

    canvas.draw()
    px, py = canvas.ax.transData.transform((x, y))
    canvas.callbacks.process(
        "button_press_event",
        MouseEvent("button_press_event", canvas, px, py, button=1),
    )


def _key(canvas, key):
    from matplotlib.backend_bases import KeyEvent

    canvas.callbacks.process("key_press_event",
                             KeyEvent("key_press_event", canvas, key))


def test_two_drawn_rois_are_both_listed_and_both_saved(panel, monkeypatch):
    canvas = panel.global_canvas
    _draw_rectangle(canvas, (120, 800, 250, 890))
    _draw_rectangle(canvas, (800, 120, 890, 250))
    manager = panel.get_roi_manager()
    assert manager.unsaved_count() == 2
    assert panel.roi_list_widget.count() == 2

    names = iter(["Lithium", "Steel"])
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        staticmethod(lambda *a, **k: (next(names), True)),
    )
    panel._save_current_as_class()
    assert [roi['name'] for roi in manager.named_rois] == ["Lithium", "Steel"]
    assert manager.unsaved_count() == 0
    colours = {roi['color'] for roi in manager.named_rois}
    assert len(colours) == 2


def test_cancelling_a_name_saves_nothing(panel, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    # A "No ROI" warning would block the run; fail instead
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: pytest.fail("no ROI was drawn")))
    canvas = panel.global_canvas
    _draw_rectangle(canvas, (120, 800, 250, 890))
    _draw_rectangle(canvas, (800, 120, 890, 250))
    answers = iter([("Lithium", True), ("", False)])
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        staticmethod(lambda *a, **k: next(answers)),
    )
    panel._save_current_as_class()
    manager = panel.get_roi_manager()
    assert manager.named_rois == []
    assert manager.unsaved_count() == 2


def test_click_selects_and_arrows_move_by_one_bin(panel):
    canvas = panel.global_canvas
    _draw_rectangle(canvas, (120, 800, 250, 890))
    manager = panel.get_roi_manager()
    before = manager.get_active_vertices()

    _click(canvas, 180, 850)
    assert manager.get_selected() == ('active', 0)

    step_x, step_y = canvas.key_step()
    _key(canvas, 'right')
    _key(canvas, 'up')
    np.testing.assert_allclose(manager.get_active_vertices(),
                               before + [step_x, step_y])
    _key(canvas, 'shift+left')
    np.testing.assert_allclose(manager.get_active_vertices(),
                               before + [step_x - 10 * step_x, step_y])


def test_click_on_empty_space_deselects(panel):
    canvas = panel.global_canvas
    _draw_rectangle(canvas, (120, 800, 250, 890))
    _click(canvas, 180, 850)
    _click(canvas, 600, 400)
    assert panel.get_roi_manager().get_selected() is None
    # Arrow keys then do nothing
    before = panel.get_roi_manager().get_active_vertices()
    _key(canvas, 'right')
    np.testing.assert_allclose(panel.get_roi_manager().get_active_vertices(),
                               before)


def test_backspace_removes_the_selected_unsaved_roi(panel):
    canvas = panel.global_canvas
    _draw_rectangle(canvas, (120, 800, 250, 890))
    _draw_rectangle(canvas, (800, 120, 890, 250))
    _click(canvas, 180, 850)
    _key(canvas, 'backspace')
    manager = panel.get_roi_manager()
    assert manager.unsaved_count() == 1
    np.testing.assert_allclose(manager.rectangle[:2], (800, 120), atol=1e-6)


def test_backspace_on_a_class_asks_then_removes_it(panel, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    manager = panel.get_roi_manager()
    manager.set_rectangle_roi(120, 800, 250, 890)
    manager.add_named_roi("Lithium")
    manager.clear_roi()
    panel._update_roi_list()
    panel._apply_roi_change()

    asked = []
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: asked.append(a) or QMessageBox.Yes),
    )
    canvas = panel.local_canvas          # works on either histogram
    _click(canvas, 180, 850)
    assert manager.get_selected() == ('named', 0)
    _key(canvas, 'delete')
    assert asked, "removing a saved class must be confirmed"
    assert manager.named_rois == []


def test_moving_a_class_updates_both_histograms(panel):
    manager = panel.get_roi_manager()
    manager.set_rectangle_roi(120, 800, 250, 890)
    manager.add_named_roi("Lithium")
    manager.clear_roi()
    panel._apply_roi_change()
    _click(panel.global_canvas, 180, 850)
    _key(panel.global_canvas, 'down')
    moved = manager.named_rois[0]['rectangle']
    assert moved[1] < 800
    for canvas in (panel.global_canvas, panel.local_canvas):
        outlines = [np.asarray(v) for _n, v, _c in canvas.roi_overlays]
        assert any(np.isclose(v[:, 1].min(), moved[1]) for v in outlines)


def test_clicking_a_list_row_selects_the_roi(panel):
    canvas = panel.global_canvas
    _draw_rectangle(canvas, (120, 800, 250, 890))
    _draw_rectangle(canvas, (800, 120, 890, 250))
    panel.roi_list_widget.setCurrentRow(0)
    assert panel.get_roi_manager().get_selected() == ('pending', 0)
