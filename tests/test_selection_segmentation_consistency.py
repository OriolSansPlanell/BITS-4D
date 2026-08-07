"""End-to-end parity between the displayed histogram and segmentation.

The histogram canvas displays ``HistogramData.histogram`` with
``imshow(..., origin='lower', extent=[x_edges[0], x_edges[-1],
y_edges[0], y_edges[-1]])`` so that pixel ``histogram[row, col]`` is drawn
at ``x = x_centers[col]`` (neutron) and ``y = y_centers[row]`` (X-ray).
An ROI drawn in those display coordinates must select exactly the voxels
whose (neutron, xray) pairs fall inside it.
"""

import numpy as np

from histograms.histogram_engine_4d import HistogramEngine4D
from segmentation.segmentation_engine_4d import SegmentationEngine4D
from utils.roi_manager import ROIManager


def test_histogram_peak_location_matches_segmented_material():
    rng = np.random.default_rng(0)
    neutron = np.full((1, 8, 32, 32), 500.0)
    xray = np.full((1, 8, 32, 32), 500.0)

    # A material with an asymmetric signature: neutron ~ 100, xray ~ 900.
    blob = np.zeros(neutron.shape, dtype=bool)
    blob[:, :, :10, :10] = True
    neutron[blob] = 100 + rng.normal(0, 5, blob.sum())
    xray[blob] = 900 + rng.normal(0, 5, blob.sum())

    engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = engine.compute_global_histogram(neutron, xray)

    # Locate the second-highest peak (the blob) on the displayed histogram.
    flat_order = np.argsort(hist.histogram, axis=None)[::-1]
    row, col = np.unravel_index(flat_order[1], hist.histogram.shape)
    display_x = hist.x_centers[col]   # neutron axis
    display_y = hist.y_centers[row]   # X-ray axis
    assert abs(display_x - 100) < 20, "blob must appear at neutron~100 on x-axis"
    assert abs(display_y - 900) < 20, "blob must appear at xray~900 on y-axis"

    # A rectangle drawn around that displayed peak must segment the blob.
    manager = ROIManager()
    manager.set_rectangle_roi(
        display_x - 50, display_y - 50, display_x + 50, display_y + 50
    )
    mask = SegmentationEngine4D().segment_volume(neutron[0], xray[0], manager)
    np.testing.assert_array_equal(mask, blob[0])


def test_segment_all_volumes_matches_single_volume_results():
    rng = np.random.default_rng(2)
    neutron = rng.uniform(0, 1000, size=(3, 4, 8, 8))
    xray = rng.uniform(0, 1000, size=neutron.shape)

    manager = ROIManager()
    manager.set_rectangle_roi(200, 300, 700, 800)

    seg = SegmentationEngine4D()
    mask_4d = seg.segment_all_volumes(neutron, xray, manager)
    for timepoint in range(neutron.shape[0]):
        np.testing.assert_array_equal(
            mask_4d[timepoint],
            seg.segment_volume(neutron[timepoint], xray[timepoint], manager),
        )
