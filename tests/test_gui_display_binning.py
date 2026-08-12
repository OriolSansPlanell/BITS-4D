"""GUI-level tests for binned display volumes on large datasets.

Forces a small display budget so the display pyramid kicks in, then checks
that the slice viewer shows binned volumes, that segmentation masks are
scaled to the display grid for overlay, and that segmentation itself still
runs at full resolution.
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
def window(qapp, monkeypatch):
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D
    from utils import config

    neutron = np.full((2, 8, 32, 32), 500.0, dtype=np.float32)
    xray = np.full((2, 8, 32, 32), 500.0, dtype=np.float32)
    blob = np.zeros(neutron.shape, dtype=bool)
    blob[:, :, :16, :16] = True
    neutron[blob] = 100.0
    xray[blob] = 900.0

    # Force binning: budget smaller than one volume (8*32*32*4 = 32 KiB)
    monkeypatch.setattr(config, "DISPLAY_MAX_VOLUME_BYTES", 8_192)

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._prepare_display_volumes()
    w._update_current_timepoint(0)
    w._test_blob = blob
    return w


def test_display_volumes_are_binned_under_budget(window):
    from utils import config

    assert window.display_bin_factor > 1
    disp_n, disp_x = window._current_display_volumes()
    assert disp_n.nbytes <= config.DISPLAY_MAX_VOLUME_BYTES
    assert disp_n.shape == disp_x.shape
    factor = window.display_bin_factor
    assert disp_n.shape == tuple(
        s // factor for s in window.dataset.neutron_data.shape[1:]
    )
    # The slice viewer must be fed the binned copies
    viewer_n, _viewer_x = window.slice_viewer.current_slice_data
    assert viewer_n.shape == disp_n.shape


def test_segmentation_runs_full_resolution_and_overlay_is_binned(window):
    blob = window._test_blob
    rm = window.dual_histogram.get_roi_manager()
    rm.set_rectangle_roi(50, 850, 150, 950)   # the blob's histogram location
    window._segment_current_volume()

    layers = window.segmentation_masks[0]
    assert len(layers) == 1
    mask, _color, name = layers[0]
    # Stored mask is full resolution and exact
    assert mask.shape == window.dataset.neutron_data.shape[1:]
    np.testing.assert_array_equal(mask, blob[0])

    # The viewer receives the whole 3-D layer on the display grid, so it can
    # re-slice it for any plane/index without the main window recomputing.
    disp_n, _ = window._current_display_volumes()
    overlays = window.slice_viewer.mask_overlays
    assert len(overlays) == 1
    _oname, overlay_mask, _ocolor = overlays[0]
    assert overlay_mask.ndim == 3
    assert overlay_mask.shape == disp_n.shape

    # And the slice it renders for the current view is the binned blob
    viewer = window.slice_viewer
    assert viewer.current_axis == 'z'
    rendered = viewer._slice_mask_for_display(overlay_mask)
    assert rendered.shape == disp_n.shape[1:]
    factor = window.display_bin_factor
    assert rendered[:16 // factor, :16 // factor].all()


def test_local_histograms_served_from_cache(window):
    engine = window.histogram_engine
    engine.precompute_all_local_histograms(
        window.dataset.neutron_data, window.dataset.xray_data
    )
    for timepoint in range(window.dataset.num_timepoints):
        assert engine.get_cached_local_histogram(timepoint) is not None
    # Timepoint switch must not clear or miss the cache
    window._update_current_timepoint(1)
    stats = engine.get_cache_stats()
    assert sorted(stats["cached_timepoints"]) == [0, 1]
