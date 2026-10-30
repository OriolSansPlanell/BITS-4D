"""The virtual crop: box geometry, the axial profile, and the engines' use
of a box. All headless — none of this needs Qt."""

import numpy as np
import pytest

from utils.crop_box import CropBox, axial_profile, PLANE_AXES


SHAPE = (12, 20, 30)   # (Z, Y, X)


@pytest.fixture
def volumes():
    rng = np.random.default_rng(0)
    neutron = rng.normal(100.0, 5.0, SHAPE).astype(np.float32)
    xray = rng.normal(200.0, 5.0, SHAPE).astype(np.float32)
    # a brighter layer in z 4..8 (a "cathode") inside a pellet y 4..16, x 5..25
    neutron[4:8, 4:16, 5:25] += 50.0
    xray[4:8, 4:16, 5:25] += 80.0
    return neutron, xray


# ── geometry ───────────────────────────────────────────────────────────────
def test_full_box_covers_everything():
    box = CropBox.full(SHAPE)
    assert box.is_full() and box.num_voxels == np.prod(SHAPE)
    assert box.slices() == (slice(0, 12), slice(0, 20), slice(0, 30))
    assert CropBox.full((3,) + SHAPE).shape == SHAPE          # (T, Z, Y, X) accepted


@pytest.mark.parametrize("axis", ["z", "y", "x"])
def test_rectangle_is_propagated_along_the_view_axis(axis):
    # rectangle from display column 2.3 to 5.6 and row 1.2 to 3.9 covers
    # columns 2..5 and rows 1..3 (half-open 2..6 and 1..4)
    box = CropBox.from_plane_rectangle(axis, (2.3, 1.2, 5.6, 3.9), SHAPE)
    row_axis, col_axis = PLANE_AXES[axis]
    assert box.bounds(row_axis) == (1, 4)
    assert box.bounds(col_axis) == (2, 6)
    size_along = SHAPE["zyx".index(axis)]
    assert box.bounds(axis) == (0, size_along)                  # full stack
    # the footprint on the drawing plane is the rectangle itself
    assert box.rectangle_on_plane(axis) == (2.0, 1.0, 6.0, 4.0)


def test_corner_order_and_degenerate_rectangles_do_not_matter():
    a = CropBox.from_plane_rectangle('z', (10, 3, 4, 9), SHAPE)
    b = CropBox.from_plane_rectangle('z', (4, 9, 10, 3), SHAPE)
    assert a == b
    # a click without a drag still selects one pixel
    c = CropBox.from_plane_rectangle('z', (7.0, 7.0, 7.0, 7.0), SHAPE)
    assert c.bounds('x') == (7, 8) and c.bounds('y') == (7, 8)


def test_display_binning_is_undone():
    # drawn on a x2 median-binned display: display pixel p = full-res [2p, 2p+2)
    box = CropBox.from_plane_rectangle('y', (3, 1, 6, 2), SHAPE, bin_factor=2)
    assert box.bounds('x') == (6, 12) and box.bounds('z') == (2, 4)
    assert box.bounds('y') == (0, 20)
    # and the footprint drawn back on the binned display is the drawn rectangle
    assert box.rectangle_on_plane('y', bin_factor=2) == (3.0, 1.0, 6.0, 2.0)


def test_rectangle_beyond_the_volume_is_clipped():
    box = CropBox.from_plane_rectangle('z', (-5, -5, 100, 100), SHAPE)
    assert box.is_full()


def test_range_along_the_stack_axis_and_footprints_on_other_planes():
    box = CropBox.from_plane_rectangle('z', (5, 4, 25, 16), SHAPE)
    box = box.with_range('z', 4, 8)
    assert box.bounds('z') == (4, 8)
    assert box.num_voxels == 4 * 12 * 20
    # seen on the XZ plane (view axis y): columns = x, rows = z
    assert box.rectangle_on_plane('y') == (5.0, 4.0, 25.0, 8.0)
    # seen on the YZ plane (view axis x): columns = y, rows = z
    assert box.rectangle_on_plane('x') == (4.0, 4.0, 16.0, 8.0)
    assert box.with_full_range('z').bounds('z') == (0, 12)
    assert "z 4–7" in box.describe()


def test_crop_mask_embed_and_dict_roundtrip(volumes):
    neutron, _ = volumes
    box = CropBox(SHAPE, (4, 8), (4, 16), (5, 25))
    view = box.crop(neutron)
    assert view.shape == box.size_zyx
    assert np.shares_memory(view, neutron)                      # a view, not a copy
    assert box.crop(np.stack([neutron, neutron])).shape == (2,) + box.size_zyx
    mask = box.mask()
    assert mask.sum() == box.num_voxels and mask[5, 10, 10] and not mask[0, 0, 0]
    back = box.embed(np.ones(box.size_zyx, dtype=bool), fill=False)
    assert np.array_equal(back, mask)
    assert CropBox.from_dict(box.to_dict()) == box


def test_invalid_boxes_are_refused():
    with pytest.raises(ValueError):
        CropBox(SHAPE, (5, 5), (0, 20), (0, 30))
    with pytest.raises(ValueError):
        CropBox(SHAPE, (0, 13), (0, 20), (0, 30))
    with pytest.raises(ValueError):
        CropBox.from_plane_rectangle('t', (0, 0, 1, 1), SHAPE)


# ── axial profile ──────────────────────────────────────────────────────────
def test_axial_profile_traces_the_layer(volumes):
    neutron, xray = volumes
    box = CropBox(SHAPE, (0, 12), (4, 16), (5, 25))          # pellet cross-section
    prof = axial_profile(neutron, xray, 'z', box)
    assert prof['index'].tolist() == list(range(12))
    assert prof['count'].tolist() == [12 * 20] * 12
    inside = prof['neutron'][4:8]
    outside = np.concatenate([prof['neutron'][:4], prof['neutron'][8:]])
    assert inside.min() > outside.max() + 40
    assert prof['xray'][4:8].min() > prof['xray'][:4].max() + 70
    # a range along the stack axis restricts the profile to that range
    part = axial_profile(neutron, xray, 'z', box.with_range('z', 4, 8))
    assert part['index'].tolist() == [4, 5, 6, 7]
    np.testing.assert_allclose(part['neutron'], prof['neutron'][4:8])


def test_axial_profile_other_axes_and_nan_handling(volumes):
    neutron, xray = volumes
    neutron = neutron.copy()
    neutron[:, 0, :] = np.nan                                   # a dead row
    prof_y = axial_profile(neutron, xray, 'y')
    assert prof_y['index'].size == 20
    assert np.isnan(prof_y['neutron'][0]) and prof_y['count'][0] == 0
    assert np.isfinite(prof_y['neutron'][1:]).all()
    prof_x = axial_profile(neutron, xray, 'x', statistic='median')
    assert prof_x['index'].size == 30
    with pytest.raises(ValueError):
        axial_profile(neutron, xray, 'z', CropBox.full((2, 2, 2)))


# ── engines ────────────────────────────────────────────────────────────────
def test_box_histogram_equals_masked_histogram_and_keeps_caches_apart(volumes):
    from histograms import HistogramEngine4D

    neutron, xray = volumes
    engine = HistogramEngine4D(bins=32, use_gpu=False, chunk_voxels=1000)
    engine.compute_global_histogram(neutron[None], xray[None])
    whole = engine.compute_local_histogram(neutron, xray, 0)
    box = CropBox(SHAPE, (4, 8), (4, 16), (5, 25))
    cropped = engine.compute_box_histogram(neutron, xray, box, 0)
    masked = engine.compute_masked_histogram(neutron, xray, box.mask())

    assert cropped.num_voxels == box.num_voxels
    np.testing.assert_array_equal(cropped.histogram, masked.histogram)
    np.testing.assert_array_equal(cropped.x_edges, whole.x_edges)      # same bin grid
    # the whole-volume cache is untouched by crop histograms
    assert engine.get_cached_local_histogram(0) is whole
    assert engine.compute_box_histogram(neutron, xray, box, 0) is cropped   # cached
    with pytest.raises(ValueError):
        engine.compute_box_histogram(neutron, xray, CropBox.full((2, 2, 2)), 0)


def test_segmentation_is_limited_to_the_box(volumes):
    from segmentation.segmentation_engine_4d import SegmentationEngine4D
    from utils.roi_manager import ROIManager

    neutron, xray = volumes
    rm = ROIManager()
    rm.set_rectangle_roi(0, 0, 1000, 1000)                      # everything
    engine = SegmentationEngine4D()
    box = CropBox(SHAPE, (4, 8), (4, 16), (5, 25))
    mask = engine.segment_volume(neutron, xray, rm, box=box)
    assert mask.shape == SHAPE and mask.dtype == bool
    assert np.array_equal(mask, box.mask())
    assert engine.segment_volume(neutron, xray, rm).all()
