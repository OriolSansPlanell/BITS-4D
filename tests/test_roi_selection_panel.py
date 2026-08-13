"""Managing multiple histogram selections: visibility, edit, remove.

Hidden classes must be excluded from *both* the histogram overlays and the
segmentation, so what is displayed always equals what is segmented.
"""

import os

import numpy as np
import pytest

from utils.roi_manager import ROIManager

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402


def _two_class_manager():
    """Manager with two saved classes covering disjoint intensity boxes."""
    manager = ROIManager()
    manager.set_rectangle_roi(0, 0, 10, 10)
    manager.add_named_roi("A")
    manager.clear_roi()
    manager.set_rectangle_roi(100, 100, 110, 110)
    manager.add_named_roi("B")
    manager.clear_roi()
    return manager


def _points():
    neutron = np.array([5.0, 105.0, 500.0])
    xray = np.array([5.0, 105.0, 500.0])
    return neutron, xray


# ── ROIManager model ─────────────────────────────────────────────────────────

def test_new_classes_are_visible_by_default():
    manager = _two_class_manager()
    assert len(manager.get_visible_named_rois()) == 2


def test_hidden_class_is_excluded_from_segmentation_and_overlays():
    manager = _two_class_manager()
    neutron, xray = _points()
    np.testing.assert_array_equal(
        manager.is_inside_roi(neutron, xray), [True, True, False]
    )

    manager.set_named_roi_visible(1, False)   # hide class B
    np.testing.assert_array_equal(
        manager.is_inside_roi(neutron, xray), [True, False, False]
    )
    # ... and it disappears from the histogram overlays too, so display and
    # segmentation still agree
    labels = [name for name, _v, _c in manager.get_named_roi_overlays()]
    assert len(labels) == 1 and "A" in labels[0]


def test_hidden_classes_do_not_appear_in_multi_class_labels():
    manager = _two_class_manager()
    neutron, xray = _points()
    manager.set_named_roi_visible(0, False)
    labels = manager.get_multi_class_labels(neutron, xray)
    assert set(np.unique(labels)) == {0, 2}


def test_has_roi_false_when_everything_hidden():
    manager = _two_class_manager()
    manager.set_named_roi_visible(0, False)
    manager.set_named_roi_visible(1, False)
    assert not manager.has_roi()
    # An active ROI still counts
    manager.set_rectangle_roi(0, 0, 1, 1)
    assert manager.has_roi()


def test_take_named_roi_moves_a_class_into_the_active_slot():
    manager = _two_class_manager()
    entry = manager.take_named_roi(0)
    assert entry['name'] == "A"
    assert len(manager.named_rois) == 1
    assert manager.roi_type == 'rectangle'
    assert manager.rectangle == (0, 0, 10, 10)
    # No double counting: the class is gone from the list, present as active
    neutron, xray = _points()
    np.testing.assert_array_equal(
        manager.is_inside_roi(neutron, xray), [True, True, False]
    )


def test_take_named_roi_rejects_bad_index():
    manager = _two_class_manager()
    with pytest.raises(IndexError):
        manager.take_named_roi(7)


def test_visibility_survives_a_save_load_round_trip(tmp_path):
    manager = _two_class_manager()
    manager.set_named_roi_visible(1, False)
    path = tmp_path / "rois.json"
    manager.save_to_file(str(path))

    restored = ROIManager()
    restored.load_from_file(str(path))
    assert restored.named_rois[0]['visible'] is True
    assert restored.named_rois[1]['visible'] is False


def test_files_without_visibility_default_to_visible(tmp_path):
    """Older ROI files have no 'visible' key and must still work."""
    manager = _two_class_manager()
    path = tmp_path / "legacy.json"
    manager.save_to_file(str(path))

    import json
    data = json.loads(path.read_text())
    for entry in data['named_rois']:
        entry.pop('visible', None)
    path.write_text(json.dumps(data))

    restored = ROIManager()
    restored.load_from_file(str(path))
    assert len(restored.get_visible_named_rois()) == 2


# ── Panel widget ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def widget(qapp):
    from gui.dual_histogram_widget import DualHistogramWidget

    w = DualHistogramWidget()
    w.roi_manager.set_rectangle_roi(0, 0, 10, 10)
    w.roi_manager.add_named_roi("A")
    w.roi_manager.clear_roi()
    w.roi_manager.set_rectangle_roi(100, 100, 110, 110)
    w.roi_manager.add_named_roi("B")
    w.roi_manager.clear_roi()
    w._update_roi_list()
    return w


def test_panel_lists_every_class_with_a_checkbox(widget):
    assert widget.roi_list_widget.count() == 2
    for row in range(2):
        item = widget.roi_list_widget.item(row)
        assert item.checkState() == Qt.Checked
        assert item.flags() & Qt.ItemIsUserCheckable


def test_unticking_a_row_hides_that_class(widget):
    widget.roi_list_widget.item(1).setCheckState(Qt.Unchecked)
    assert widget.roi_manager.named_rois[1]['visible'] is False
    assert len(widget.roi_manager.get_visible_named_rois()) == 1
    # Re-ticking restores it
    widget.roi_list_widget.item(1).setCheckState(Qt.Checked)
    assert len(widget.roi_manager.get_visible_named_rois()) == 2


def test_isolate_shows_only_the_highlighted_class(widget):
    widget.roi_list_widget.setCurrentRow(1)
    widget._isolate_selected_class()
    visible = widget.roi_manager.get_visible_named_rois()
    assert len(visible) == 1 and visible[0]['name'] == "B"
    assert widget.roi_list_widget.currentRow() == 1


def test_show_and_hide_all(widget):
    widget._set_all_classes_visible(False)
    assert widget.roi_manager.get_visible_named_rois() == []
    widget._set_all_classes_visible(True)
    assert len(widget.roi_manager.get_visible_named_rois()) == 2


def test_edit_round_trip_preserves_class_identity(widget, monkeypatch):
    original = dict(widget.roi_manager.named_rois[0])

    widget.roi_list_widget.setCurrentRow(0)
    widget._edit_selected_class()

    # It became the active ROI and left the class list
    assert widget.roi_manager.roi_type == 'rectangle'
    assert len(widget.roi_manager.named_rois) == 1
    assert widget._editing_class['class_id'] == original['class_id']

    # Reshape it, then save it back under the same name
    widget.roi_manager.set_rectangle_roi(0, 0, 20, 20)
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        staticmethod(lambda *a, **k: (original['name'], True)),
    )
    widget._save_current_as_class()

    assert len(widget.roi_manager.named_rois) == 2
    restored = [r for r in widget.roi_manager.named_rois
                if r['name'] == original['name']][0]
    assert restored['class_id'] == original['class_id']
    assert restored['color'] == original['color']
    assert restored['rectangle'] == (0, 0, 20, 20)   # the edit was kept
    assert widget._editing_class is None
    assert widget.roi_manager.roi_type is None       # active slot cleared


def test_remove_selected_deletes_only_that_class(widget, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    widget.roi_list_widget.setCurrentRow(0)
    widget._remove_selected_class()
    remaining = [r['name'] for r in widget.roi_manager.named_rois]
    assert remaining == ["B"]
    assert widget.roi_list_widget.count() == 1


def test_remove_can_be_cancelled(widget, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )
    widget.roi_list_widget.setCurrentRow(0)
    widget._remove_selected_class()
    assert len(widget.roi_manager.named_rois) == 2


def test_segmentation_enumerates_only_visible_classes(qapp):
    """Hidden classes must not be segmented, so the layers created match the
    selections drawn on the histogram."""
    from gui.main_window import BiTS4DMainWindow

    manager = _two_class_manager()
    specs = BiTS4DMainWindow._enumerate_roi_specs(manager)
    assert [s['name'] for s in specs] == ["A", "B"]

    manager.set_named_roi_visible(0, False)
    specs = BiTS4DMainWindow._enumerate_roi_specs(manager)
    assert [s['name'] for s in specs] == ["B"]

    # An active ROI is still segmented alongside the visible classes
    manager.set_rectangle_roi(200, 200, 210, 210)
    specs = BiTS4DMainWindow._enumerate_roi_specs(manager)
    assert [s['name'] for s in specs] == ["B", "Active ROI"]


def test_buttons_follow_the_current_selection(widget):
    widget.roi_list_widget.setCurrentRow(-1)
    widget._update_selection_buttons()
    assert not widget.remove_class_btn.isEnabled()
    assert not widget.edit_class_btn.isEnabled()
    assert widget.clear_all_classes_btn.isEnabled()

    widget.roi_list_widget.setCurrentRow(0)
    assert widget.remove_class_btn.isEnabled()
    assert widget.edit_class_btn.isEnabled()
    assert widget.only_selected_btn.isEnabled()
