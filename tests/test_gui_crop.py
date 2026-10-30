"""GUI-level tests of the virtual crop and the axial profile.

A rectangle drawn on the slice viewer is propagated along the view axis;
the local histogram, the axial profile and Segment Current then use the
box only, and a range dragged on the profile bounds the box along the
stack axis. Display binning must be undone when the box is built.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

SHAPE = (2, 12, 32, 32)      # (T, Z, Y, X)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _layered_dataset():
    """A pellet (y 4..28, x 4..28) with a bright layer in z 4..8."""
    neutron = np.full(SHAPE, 500.0, dtype=np.float32)
    xray = np.full(SHAPE, 500.0, dtype=np.float32)
    neutron[:, 4:8, 4:28, 4:28] = 900.0
    xray[:, 4:8, 4:28, 4:28] = 100.0
    return neutron, xray


def _make_window(monkeypatch, display_budget=None):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D
    from utils import config

    neutron, xray = _layered_dataset()
    if display_budget is not None:
        monkeypatch.setattr(config, "DISPLAY_MAX_VOLUME_BYTES", display_budget)
    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._prepare_display_volumes()
    w._update_current_timepoint(0)
    return w


@pytest.fixture()
def window(qapp, monkeypatch):
    return _make_window(monkeypatch)


@pytest.fixture()
def binned_window(qapp, monkeypatch):
    # budget below one volume (12*32*32*4 = 48 KiB) forces display binning
    w = _make_window(monkeypatch, display_budget=8_192)
    assert w.display_bin_factor > 1
    return w


def test_no_crop_shows_whole_volume_and_a_profile(window):
    assert window.crop_box is None
    local = window.dual_histogram.local_canvas.histogram_data
    assert local.num_voxels == int(np.prod(SHAPE[1:]))
    prof = window.axial_profile.profile
    assert prof is not None and prof['axis'] == 'z'
    assert prof['index'].tolist() == list(range(SHAPE[1]))
    assert "whole cross-section" in window.axial_profile.title_label.text()
    assert not window.slice_viewer.clear_crop_btn.isEnabled()


def test_crop_rectangle_is_propagated_and_drives_the_local_histogram(window):
    # drawn on the XY plane (view axis z): columns = x, rows = y
    window._on_crop_rectangle_drawn((4.0, 4.0, 28.0, 28.0), 'z')
    box = window.crop_box
    assert box is not None
    assert box.bounds('x') == (4, 28) and box.bounds('y') == (4, 28)
    assert box.bounds('z') == (0, 12)                            # the whole stack

    local = window.dual_histogram.local_canvas.histogram_data
    assert local.num_voxels == box.num_voxels == 12 * 24 * 24
    # only pellet voxels remain: values 500 (host) and 900 (layer) in neutron
    assert "crop" in window.dual_histogram.local_canvas.title
    assert window.slice_viewer.crop_box == box
    assert window.slice_viewer.clear_crop_btn.isEnabled()
    assert window.slice_viewer._crop_artists                      # drawn on the slice
    assert "inside the crop" in window.axial_profile.title_label.text()

    # the profile now sees the bright layer cleanly
    prof = window.axial_profile.profile
    assert prof['neutron'][4:8].min() > 850 and prof['neutron'][:4].max() < 550


def test_profile_range_bounds_the_crop_along_the_stack_axis(window):
    window._on_crop_rectangle_drawn((4.0, 4.0, 28.0, 28.0), 'z')
    window._on_profile_range_selected('z', 4, 8)
    box = window.crop_box
    assert box.bounds('z') == (4, 8)
    local = window.dual_histogram.local_canvas.histogram_data
    assert local.num_voxels == 4 * 24 * 24
    # the histogram holds the layer only: every voxel at neutron 900
    xs = local.x_centers
    occupied = xs[local.histogram.sum(axis=0) > 0]
    assert occupied.min() > 850
    # the profile keeps the whole axis in view but marks the range
    assert window.axial_profile.selected_range == (4, 8)
    assert window.axial_profile.profile['index'].size == SHAPE[1]
    assert window.axial_profile.clear_range_btn.isEnabled()

    # a new rectangle on the same plane keeps the chosen range
    window._on_crop_rectangle_drawn((6.0, 6.0, 26.0, 26.0), 'z')
    assert window.crop_box.bounds('z') == (4, 8)
    assert window.crop_box.bounds('x') == (6, 26)

    # releasing the range restores the full stack
    window._on_profile_range_cleared('z')
    assert window.crop_box.bounds('z') == (0, 12)


def test_segment_current_is_limited_to_the_crop(window):
    rm = window.dual_histogram.get_roi_manager()
    rm.set_rectangle_roi(0, 0, 2000, 2000)                        # everything
    window._on_crop_rectangle_drawn((4.0, 4.0, 28.0, 28.0), 'z')
    window._on_profile_range_selected('z', 4, 8)
    window._segment_current_volume()
    layers = window.segmentation_masks[0]
    assert len(layers) == 1
    mask = layers[0][0]
    assert mask.shape == SHAPE[1:]
    np.testing.assert_array_equal(mask, window.crop_box.mask())


def test_clearing_the_crop_restores_the_whole_volume(window):
    window._on_crop_rectangle_drawn((4.0, 4.0, 28.0, 28.0), 'z')
    window.slice_viewer._on_clear_crop_clicked()                  # the ✕ Crop button
    assert window.crop_box is None
    local = window.dual_histogram.local_canvas.histogram_data
    assert local.num_voxels == int(np.prod(SHAPE[1:]))
    assert window.dual_histogram.local_canvas.title == "Local Histogram"
    assert window.slice_viewer.crop_box is None


def test_crop_follows_the_timepoint(window):
    window._on_crop_rectangle_drawn((4.0, 4.0, 28.0, 28.0), 'z')
    window._update_current_timepoint(1)
    local = window.dual_histogram.local_canvas.histogram_data
    assert local.num_voxels == window.crop_box.num_voxels
    # the whole-volume cache is intact for both timepoints
    assert window.histogram_engine.get_cached_local_histogram(0).num_voxels == int(np.prod(SHAPE[1:]))


def test_crop_on_other_planes_and_profile_axis_follow_the_viewer(window):
    viewer = window.slice_viewer
    viewer.y_axis_btn.setChecked(True)                            # XZ plane: cols x, rows z
    assert viewer.current_axis == 'y'
    assert window.axial_profile.axis == 'y'
    window._on_crop_rectangle_drawn((4.0, 4.0, 28.0, 8.0), 'y')
    box = window.crop_box
    assert box.bounds('x') == (4, 28) and box.bounds('z') == (4, 8)
    assert box.bounds('y') == (0, 32)
    # its footprint on the XY plane is the x range over all rows
    assert box.rectangle_on_plane('z') == (4.0, 0.0, 28.0, 32.0)
    # a click on the profile moves the viewer to that slice of that axis
    window._on_profile_index_clicked('z', 6)
    assert viewer.current_axis == 'z' and viewer.current_slice_index == 6


def test_display_binning_is_undone_when_the_box_is_built(binned_window):
    w = binned_window
    f = w.display_bin_factor
    # the rectangle is drawn in display pixels of the binned slice
    w._on_crop_rectangle_drawn((4.0 / f, 4.0 / f, 28.0 / f, 28.0 / f), 'z')
    box = w.crop_box
    assert box.shape == SHAPE[1:]                                 # full resolution
    assert box.bounds('x') == (4, 28) and box.bounds('y') == (4, 28)
    # and it is drawn back on the binned slice at the drawn position
    assert box.rectangle_on_plane('z', f) == (4.0 / f, 4.0 / f, 28.0 / f, 28.0 / f)
    local = w.dual_histogram.local_canvas.histogram_data
    assert local.num_voxels == box.num_voxels                     # full-res voxels
    # profile indices are full-resolution; the viewer's slice is scaled to them
    w._on_view_slice_changed('z', 2)
    assert w.axial_profile.current_index == 2 * f
