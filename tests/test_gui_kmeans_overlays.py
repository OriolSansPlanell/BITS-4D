"""K-means cluster highlights must follow the viewing plane.

Regression: cluster selections stored only the 2-D mask slice extracted when
clustering ran. On an isotropic volume every plane yields the same slice
shape, so the shape check passed and the stale mask was drawn over a
different plane — the highlight showed the previous plane's voxels.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    # Deliberately isotropic (12, 12, 12): every plane's slice is 12x12, so a
    # shape check alone cannot detect a mask drawn on the wrong plane.
    neutron = np.full((1, 12, 12, 12), 500.0)
    xray = np.full((1, 12, 12, 12), 500.0)
    blob = np.zeros(neutron.shape, dtype=bool)
    blob[:, 2:5, 6:10, 1:4] = True   # different extent on each axis
    neutron[blob] = 100.0
    xray[blob] = 900.0

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=32, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    w.selection_manager.show_all_cb.setChecked(True)
    w._test_blob = blob[0]
    return w


def _slice_of(volume_mask, axis, index):
    if axis == 'z':
        return volume_mask[index, :, :]
    if axis == 'y':
        return volume_mask[:, index, :]
    return volume_mask[:, :, index]


def _emit_three_d_cluster(window):
    """Mimic _auto_detect_3d's clusters_detected payload (6-tuple)."""
    viewer = window.slice_viewer
    viewer.current_axis = 'z'
    viewer.current_slice_index = 3
    viewer._update_display()

    mask_3d = window._test_blob
    mask_2d = _slice_of(mask_3d, 'z', 3)
    payload = [(
        "3D Cluster 0", mask_2d, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        0, (1.0, 0.0, 0.0, 0.5), mask_3d,
    )]
    window._on_clusters_detected(payload)
    window._update_histogram_overlays()


def _rendered(viewer, name):
    for entry in viewer.mask_overlays:
        if entry[0] == name:
            plane = entry[3] if len(entry) > 3 else None
            return viewer._slice_mask_for_display(entry[1], plane)
    return None


def test_three_d_cluster_keeps_its_volume_mask(window):
    _emit_three_d_cluster(window)
    selection = window.selection_manager.selections[-1]
    assert selection.spatial_mask_3d is not None
    assert selection.spatial_mask_3d.ndim == 3


def test_three_d_cluster_highlight_follows_plane_changes(window):
    _emit_three_d_cluster(window)
    viewer = window.slice_viewer
    blob = window._test_blob

    for axis, index in (('z', 3), ('y', 7), ('x', 2)):
        viewer.current_axis = axis
        viewer.current_slice_index = index
        viewer._update_display()
        rendered = _rendered(viewer, "3D Cluster 0")
        expected = _slice_of(blob, axis, index)
        assert rendered is not None, f"highlight vanished on axis {axis}"
        np.testing.assert_array_equal(
            rendered, expected,
            err_msg=f"highlight does not match slice on axis {axis}",
        )


def test_three_d_cluster_highlight_absent_outside_the_blob(window):
    _emit_three_d_cluster(window)
    viewer = window.slice_viewer
    viewer.current_axis = 'z'
    viewer.current_slice_index = 11   # blob spans z 2..5 only
    viewer._update_display()
    rendered = _rendered(viewer, "3D Cluster 0")
    assert rendered is None or not rendered.any()


def test_two_d_cluster_is_pinned_to_its_own_plane(window):
    """A 2-D cluster mask must not be drawn on a different plane, even when
    the slice shapes happen to match."""
    viewer = window.slice_viewer
    viewer.current_axis = 'z'
    viewer.current_slice_index = 3
    viewer._update_display()

    mask_2d = _slice_of(window._test_blob, 'z', 3)
    window._on_clusters_detected([(
        "Cluster 0", mask_2d, None, 0, (0.0, 1.0, 0.0, 0.5),
    )])
    window._update_histogram_overlays()

    # Shown on the plane and slice it was made on
    np.testing.assert_array_equal(_rendered(viewer, "Cluster 0"), mask_2d)

    # Not on another slice of the same plane
    viewer.current_slice_index = 8
    viewer._update_display()
    assert _rendered(viewer, "Cluster 0") is None

    # Nor on a different plane (same 12x12 shape — the isotropic trap)
    viewer.current_axis = 'y'
    viewer.current_slice_index = 3
    viewer._update_display()
    assert viewer.current_slice.shape == mask_2d.shape
    assert _rendered(viewer, "Cluster 0") is None
