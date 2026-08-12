"""The segmentation highlight must follow slice index, plane and timepoint.

Regression: the viewer stored pre-sliced 2-D masks, so scrolling within a
plane showed a stale slice's voxels and switching plane dropped the overlay
entirely (shape mismatch). Segmentation layers are 3-D, so the viewer must
re-slice them on every redraw.
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

    # A blob at a known, asymmetric 3-D location so slicing errors show up:
    # z 2..5, y 4..11, x 6..17 — different extents on every axis.
    neutron = np.full((3, 8, 16, 24), 500.0)
    xray = np.full((3, 8, 16, 24), 500.0)
    blob = np.zeros(neutron.shape, dtype=bool)
    blob[:, 2:6, 4:12, 6:18] = True
    neutron[blob] = 100.0
    xray[blob] = 900.0

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    w._test_blob = blob
    return w


def _expected_slice(blob_3d, axis, index):
    if axis == 'z':
        return blob_3d[index, :, :]
    if axis == 'y':
        return blob_3d[:, index, :]
    return blob_3d[:, :, index]


def _rendered_slice(viewer):
    """The 2-D highlight the viewer would draw for the current view."""
    assert len(viewer.mask_overlays) == 1
    _name, mask, _color = viewer.mask_overlays[0]
    return viewer._slice_mask_for_display(mask)


def _segment_active_roi(window):
    rm = window.dual_histogram.get_roi_manager()
    rm.set_rectangle_roi(50, 850, 150, 950)
    window._segment_current_volume()


def test_viewer_receives_three_dimensional_layers(window):
    _segment_active_roi(window)
    _name, mask, _color = window.slice_viewer.mask_overlays[0]
    assert mask.ndim == 3, "viewer must hold the 3-D layer, not a 2-D slice"


def test_highlight_follows_slice_index_within_a_plane(window):
    blob = window._test_blob[0]
    _segment_active_roi(window)
    viewer = window.slice_viewer

    # Every Z slice, including ones outside the blob, must match exactly
    for index in range(window.dataset.neutron_data.shape[1]):
        viewer.current_slice_index = index
        viewer._update_display()
        rendered = _rendered_slice(viewer)
        expected = _expected_slice(blob, 'z', index)
        if expected.any():
            np.testing.assert_array_equal(rendered, expected)
        else:
            # Nothing to draw on this slice
            assert rendered is None or not rendered.any()


def test_highlight_follows_plane_changes(window):
    blob = window._test_blob[0]
    _segment_active_roi(window)
    viewer = window.slice_viewer

    for axis, index in (('z', 3), ('y', 6), ('x', 10)):
        viewer.current_axis = axis
        viewer.current_slice_index = index
        viewer._update_display()
        rendered = _rendered_slice(viewer)
        expected = _expected_slice(blob, axis, index)
        assert rendered is not None, f"overlay vanished on axis {axis}"
        np.testing.assert_array_equal(rendered, expected)
        # An artist must actually be drawn for a non-empty slice
        assert len(viewer.overlay_artists) == 1


def test_highlight_absent_on_planes_outside_the_blob(window):
    _segment_active_roi(window)
    viewer = window.slice_viewer
    viewer.current_axis = 'y'
    viewer.current_slice_index = 0   # blob starts at y=4
    viewer._update_display()
    assert viewer.overlay_artists == []
    assert viewer._visible_mask_pixels == 0


def test_segment_all_timepoints_highlights_every_timepoint_and_plane(
    window, monkeypatch
):
    blob = window._test_blob
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    rm = window.dual_histogram.get_roi_manager()
    rm.set_rectangle_roi(50, 850, 150, 950)
    window._segment_all_volumes()

    num_timepoints = window.dataset.num_timepoints
    assert set(window.segmentation_masks) == set(range(num_timepoints))

    viewer = window.slice_viewer
    for timepoint in range(num_timepoints):
        window._update_current_timepoint(timepoint)
        for axis, index in (('z', 4), ('y', 5), ('x', 9)):
            viewer.current_axis = axis
            viewer.current_slice_index = index
            viewer._update_display()
            rendered = _rendered_slice(viewer)
            expected = _expected_slice(blob[timepoint], axis, index)
            assert rendered is not None, (
                f"no highlight at T={timepoint}, axis {axis}"
            )
            np.testing.assert_array_equal(rendered, expected)


def test_selection_changes_do_not_erase_segmentation_highlight(window):
    """Regression: the selection manager replaced the whole overlay list,
    wiping segmentation layers whenever selections changed or "Show All"
    was toggled."""
    _segment_active_roi(window)
    viewer = window.slice_viewer
    seg_names = {name for _m, _c, name in window.segmentation_masks[0]}
    assert seg_names

    # Add a saved selection and turn on "Show All on Histogram"
    flat_mask = np.zeros(viewer.current_slice.shape, dtype=bool)
    flat_mask[0:3, 0:3] = True
    window.selection_manager.add_selection(
        name="picked", spatial_mask=flat_mask, histogram_roi=None
    )
    window.selection_manager.show_all_cb.setChecked(True)
    window._update_histogram_overlays()

    shown = {entry[0] for entry in viewer.mask_overlays}
    assert seg_names <= shown, "segmentation layers were erased by selections"
    assert "picked" in shown

    # Turning it back off must keep the segmentation layers
    window.selection_manager.show_all_cb.setChecked(False)
    window._update_histogram_overlays()
    shown = {entry[0] for entry in viewer.mask_overlays}
    assert seg_names <= shown, "segmentation layers lost when hiding selections"
    assert "picked" not in shown


def test_two_dimensional_selection_masks_still_supported(window):
    """Single-slice masks (region growing) must show only on matching slices."""
    viewer = window.slice_viewer
    viewer.current_axis = 'z'
    viewer.current_slice_index = 3
    viewer._update_display()
    flat_mask = np.zeros(viewer.current_slice.shape, dtype=bool)
    flat_mask[1:4, 1:4] = True
    viewer.set_mask_overlays([("region", flat_mask, (1.0, 0.0, 0.0, 0.5))])
    np.testing.assert_array_equal(
        viewer._slice_mask_for_display(flat_mask), flat_mask
    )
    # A plane whose slice has a different shape must not raise, just skip
    viewer.current_axis = 'x'
    viewer.current_slice_index = 5
    viewer._update_display()
    assert viewer._slice_mask_for_display(flat_mask) is None
