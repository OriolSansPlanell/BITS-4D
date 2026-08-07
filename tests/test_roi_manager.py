"""Regression tests for ROI containment and selection/segmentation parity."""

import numpy as np
from matplotlib.path import Path

from utils.roi_manager import ROIManager


def _volume_with_two_materials():
    """Volumes with asymmetric signatures so any axis swap is detectable.

    Material A: neutron ~ 100, xray ~ 900.
    Material B: neutron ~ 900, xray ~ 100.
    """
    rng = np.random.default_rng(0)
    neutron = np.full((4, 16, 16), 500.0)
    xray = np.full((4, 16, 16), 500.0)
    blob_a = np.zeros(neutron.shape, dtype=bool)
    blob_a[:, :5, :5] = True
    blob_b = np.zeros(neutron.shape, dtype=bool)
    blob_b[:, 10:, 10:] = True
    neutron[blob_a] = 100 + rng.normal(0, 5, blob_a.sum())
    xray[blob_a] = 900 + rng.normal(0, 5, blob_a.sum())
    neutron[blob_b] = 900 + rng.normal(0, 5, blob_b.sum())
    xray[blob_b] = 100 + rng.normal(0, 5, blob_b.sum())
    return neutron, xray, blob_a, blob_b


def test_rectangle_roi_selects_displayed_region():
    neutron, xray, blob_a, _ = _volume_with_two_materials()
    manager = ROIManager()
    # Drawn around material A as it appears on the histogram display
    # (x = neutron ~ 100, y = xray ~ 900).
    manager.set_rectangle_roi(50, 850, 150, 950)
    mask = manager.is_inside_roi(neutron, xray)
    np.testing.assert_array_equal(mask, blob_a)


def test_active_roi_participates_in_union_with_named_rois():
    """A freshly drawn ROI must be segmented even after classes were saved.

    Regression: the active ROI was silently ignored once named ROIs existed,
    so the region selected on the histogram differed from the segmented area.
    """
    neutron, xray, blob_a, blob_b = _volume_with_two_materials()
    manager = ROIManager()
    manager.set_rectangle_roi(50, 850, 150, 950)      # material A
    manager.add_named_roi("A")
    manager.clear_roi()
    manager.set_rectangle_roi(850, 50, 950, 150)      # new active ROI: B

    union = manager.is_inside_roi(neutron, xray)
    np.testing.assert_array_equal(union, blob_a | blob_b)


def test_multi_class_labels_include_active_roi_as_next_class():
    neutron, xray, blob_a, blob_b = _volume_with_two_materials()
    manager = ROIManager()
    manager.set_rectangle_roi(50, 850, 150, 950)
    class_a = manager.add_named_roi("A")
    manager.clear_roi()
    manager.set_rectangle_roi(850, 50, 950, 150)

    labels = manager.get_multi_class_labels(neutron, xray)
    np.testing.assert_array_equal(labels == class_a, blob_a)
    np.testing.assert_array_equal(labels == class_a + 1, blob_b)


def test_polygon_mask_matches_unfiltered_path_containment():
    """The bounding-box prefilter must not change polygon membership."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, size=(32, 32))
    y = rng.uniform(0, 100, size=(32, 32))
    polygon = np.array([[20.0, 10.0], [80.0, 25.0], [55.0, 90.0], [15.0, 60.0]])

    manager = ROIManager()
    manager.set_polygon_roi(polygon)
    fast = manager.is_inside_roi(x, y)

    reference = (
        Path(polygon)
        .contains_points(np.column_stack([x.ravel(), y.ravel()]))
        .reshape(x.shape)
    )
    np.testing.assert_array_equal(fast, reference)


def test_named_roi_round_trip_preserves_masks(tmp_path):
    neutron, xray, blob_a, blob_b = _volume_with_two_materials()
    manager = ROIManager()
    manager.set_rectangle_roi(50, 850, 150, 950)
    manager.add_named_roi("A")
    manager.clear_roi()
    manager.set_polygon_roi(
        np.array([[800.0, 40.0], [960.0, 40.0], [960.0, 160.0], [800.0, 160.0]])
    )

    path = tmp_path / "roi.json"
    manager.save_to_file(str(path))

    restored = ROIManager()
    restored.load_from_file(str(path))
    np.testing.assert_array_equal(
        restored.is_inside_roi(neutron, xray),
        manager.is_inside_roi(neutron, xray),
    )
    assert np.count_nonzero(restored.is_inside_roi(neutron, xray)) == (
        blob_a.sum() + blob_b.sum()
    )
