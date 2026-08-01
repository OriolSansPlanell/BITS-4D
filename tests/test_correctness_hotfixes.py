from types import SimpleNamespace

import numpy as np

from histograms.histogram_engine_4d import HistogramEngine4D
from segmentation.segmentation_engine_4d import SegmentationEngine4D
from utils.region_growing_3d import RegionGrowing3D
from utils.selection_library import SelectionLibrary


def test_uint16_region_growing_does_not_underflow():
    volume = np.array([[[1000, 0, 2000]]], dtype=np.uint16)
    mask = RegionGrowing3D.univariate_region_growing_3d(
        volume, (0, 0, 0), tolerance=1000, connectivity=1
    )
    assert mask.tolist() == [[[True, True, True]]]


def test_overlay_is_vectorized_and_constant_safe():
    image = np.full((3, 3), 7, dtype=np.uint16)
    mask = np.eye(3, dtype=bool)
    result = SegmentationEngine4D.apply_overlay(image, mask, (255, 0, 0), 0.5)
    assert result.shape == (3, 3, 3)
    assert result.dtype == np.uint8
    assert np.all(result[mask, 0] > 0)


def test_selection_colour_round_trip_preserves_hex(tmp_path):
    selection = SimpleNamespace(
        name="Lithium",
        spatial_mask=np.array([[True, False]]),
        histogram_roi=np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        cluster_id=1,
        color="#e6194b",
        visible=False,
    )
    path = SelectionLibrary.save_selections([selection], tmp_path / "classes.bits")
    loaded = SelectionLibrary.load_selections(path)[0]
    assert loaded["color"] == "#e6194b"
    assert loaded["visible"] is False


def test_histogram_ignores_invalid_pairs_and_handles_constant_values():
    neutron = np.ones((1, 2, 2, 2), dtype=float)
    xray = np.ones_like(neutron) * 5
    neutron.ravel()[0] = np.nan
    engine = HistogramEngine4D(bins=8, use_gpu=False)
    result = engine.compute_global_histogram(neutron, xray)
    assert result.num_voxels == 7
    assert result.ignored_voxels == 1
    assert int(result.histogram.sum()) == 7
    assert result.data_range[0][0] < result.data_range[0][1]
