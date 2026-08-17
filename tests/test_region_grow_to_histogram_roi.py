"""A 3-D region grow must build its histogram ROI from the whole region.

Regression: the slice viewer computed a 3-D mask (thousands of voxels) but
emitted only the 2-D slice currently displayed, so the convex-hull ROI
described the handful of voxels visible on that slice. The resulting ROI was
far narrower than the region actually selected, and segmenting with it
recovered only a fraction of it.
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

    # A material whose intensities drift along Y. Viewed on the y axis, a
    # single slice therefore shows only a slim part of the region's full
    # intensity spread — which is exactly when using one slice goes wrong.
    neutron = np.full((1, 12, 20, 20), 5000.0)
    xray = np.full((1, 12, 20, 20), 5000.0)
    blob = np.zeros(neutron.shape, dtype=bool)
    blob[:, :, 6:14, 6:14] = True
    for y in range(6, 14):
        plane = np.zeros(neutron.shape, dtype=bool)
        plane[:, :, y, :] = True
        here = blob & plane
        neutron[here] = 40000.0 + 800.0 * (y - 6)
        xray[here] = 40000.0 + 800.0 * (y - 6)

    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = w.histogram_engine.compute_global_histogram(neutron, xray)
    w.global_histogram = hist
    w.dual_histogram.set_global_histogram(hist)
    w._update_current_timepoint(0)
    w._test_blob = blob[0]
    return w


def _grow_3d(window, axis='y', index=10):
    """Run a 3-D bivariate region grow the way the viewer does."""
    from utils.region_growing_3d import RegionGrowing3D

    viewer = window.slice_viewer
    viewer.current_axis = axis
    viewer.current_slice_index = index
    viewer._update_display()

    neutron_vol, xray_vol = viewer.current_slice_data
    seed = (5, 10, 10)          # inside the blob
    mask_3d = RegionGrowing3D.bivariate_region_growing_3d(
        neutron_vol, xray_vol, seed, 6000.0, 6000.0, connectivity=1
    )
    viewer.region_grow_mask_3d = mask_3d
    viewer.region_grow_mask = RegionGrowing3D.extract_slice_from_3d_mask(
        mask_3d, axis, index
    )
    return mask_3d


def test_three_d_grow_emits_the_volume_mask(window):
    mask_3d = _grow_3d(window)
    viewer = window.slice_viewer

    emitted = {}
    viewer.spatial_roi_to_histogram.connect(
        lambda coords, axis, index: emitted.update(
            coords=coords, axis=axis, index=index
        )
    )
    viewer._create_histogram_roi_from_spatial()

    assert emitted["coords"][0] == 'mask'
    sent = np.asarray(emitted["coords"][1])
    assert sent.ndim == 3, "the 2-D slice was emitted instead of the volume"
    assert np.count_nonzero(sent) == np.count_nonzero(mask_3d)
    # The slice really is a small part of the region, so this matters
    assert np.count_nonzero(viewer.region_grow_mask) < np.count_nonzero(mask_3d)


def test_roi_covers_the_whole_grown_region_not_one_slice(window):
    mask_3d = _grow_3d(window)
    viewer = window.slice_viewer

    neutron_vol, xray_vol = viewer.current_slice_data
    full_span = neutron_vol[mask_3d].max() - neutron_vol[mask_3d].min()
    slice_mask = viewer.region_grow_mask
    neutron_slice = neutron_vol[:, viewer.current_slice_index, :]
    slice_span = (neutron_slice[slice_mask].max()
                  - neutron_slice[slice_mask].min())
    assert slice_span < full_span, "test setup: slice must be narrower"

    viewer._create_histogram_roi_from_spatial()
    roi = window.dual_histogram.get_roi_manager()
    assert roi.roi_type == 'polygon'

    # Every grown voxel must fall inside the ROI built from it
    inside = roi.is_inside_roi(neutron_vol, xray_vol)
    covered = np.count_nonzero(inside & mask_3d) / np.count_nonzero(mask_3d)
    assert covered > 0.99, f"ROI covers only {covered:.1%} of the grown region"

    # And the ROI must span the whole region, not just the displayed slice
    x_min, y_min, x_max, y_max = roi.get_roi_bounds()
    assert (x_max - x_min) >= full_span * 0.9


def test_two_d_grow_still_uses_the_slice(window):
    """A 2-D region grow has no volume mask and must keep working."""
    from utils.region_growing import RegionGrowing

    viewer = window.slice_viewer
    viewer.current_axis = 'z'
    viewer.current_slice_index = 5
    viewer._update_display()

    neutron_vol, xray_vol = viewer.current_slice_data
    neutron_slice = neutron_vol[5]
    xray_slice = xray_vol[5]
    viewer.region_grow_mask = RegionGrowing.flood_fill_bivariate(
        neutron_slice, xray_slice, (10, 10), 6000.0, 6000.0, connectivity=2
    )
    viewer.region_grow_mask_3d = None

    emitted = {}
    viewer.spatial_roi_to_histogram.connect(
        lambda coords, axis, index: emitted.update(coords=coords)
    )
    viewer._create_histogram_roi_from_spatial()

    assert np.asarray(emitted["coords"][1]).ndim == 2
    roi = window.dual_histogram.get_roi_manager()
    assert roi.roi_type == 'polygon'
    inside = roi.is_inside_roi(neutron_slice, xray_slice)
    assert np.all(inside[viewer.region_grow_mask])


def test_convex_hull_deduplicates_large_point_clouds():
    """Millions of voxels collapse to few distinct intensity pairs."""
    from utils.region_growing import RegionGrowing

    rng = np.random.default_rng(0)
    neutron = rng.integers(1000, 1010, size=200_000).astype(float)
    xray = rng.integers(2000, 2010, size=200_000).astype(float)
    hull = RegionGrowing.create_convex_hull_roi(neutron, xray)
    assert len(hull) >= 3
    # Every input point must fall inside the (slightly grown) hull. Points
    # exactly on a boundary are not "contained", which is why the hull is
    # expanded about its centroid.
    from matplotlib.path import Path
    pts = np.column_stack([neutron, xray])
    assert Path(hull).contains_points(pts).all()


def test_uniform_region_still_yields_a_usable_roi():
    """A single-valued region must not collapse to a zero-area ROI."""
    from utils.region_growing import RegionGrowing
    from utils.roi_manager import ROIManager

    neutron = np.full(500, 44000.0)
    xray = np.full(500, 31000.0)
    hull = RegionGrowing.create_convex_hull_roi(neutron, xray)

    manager = ROIManager()
    manager.set_polygon_roi(hull)
    assert manager.get_roi_area() > 0
    assert bool(manager.is_inside_roi(neutron, xray).all()), (
        "a uniform region selected none of its own voxels"
    )
