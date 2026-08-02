import numpy as np
import tifffile

from data.data_loader_4d import TIFF4DLoader
from histograms.histogram_engine_4d import HistogramEngine4D
from segmentation.random_forest_4d import RandomForestSegmentation4D
from utils.clustering_3d import KMeans3D


def _has_memmap_base(array):
    current = array
    while current is not None:
        if isinstance(current, np.memmap):
            return True
        current = getattr(current, "base", None)
    return False


def test_chunked_histogram_matches_numpy_reference():
    rng = np.random.default_rng(2)
    neutron = rng.normal(size=(2, 4, 5, 6))
    xray = rng.normal(loc=3, scale=2, size=neutron.shape)
    engine = HistogramEngine4D(bins=12, use_gpu=False, chunk_voxels=17)
    result = engine.compute_global_histogram(neutron, xray)
    reference, _, _ = np.histogram2d(
        neutron.ravel(),
        xray.ravel(),
        bins=12,
        range=[result.data_range[0], result.data_range[1]],
    )
    np.testing.assert_array_equal(result.histogram, reference.T.astype(np.uint64))


def test_kmeans_predicts_unsampled_voxels_instead_of_upsampling_blocks():
    neutron = np.indices((4, 4, 4)).sum(axis=0) % 2
    xray = neutron * 100
    labels, centers, stats = KMeans3D.cluster_volume(
        neutron,
        xray,
        n_clusters=2,
        subsample_factor=2,
        max_fit_samples=64,
        prediction_chunk_size=7,
    )
    assert len(np.unique(labels)) == 2
    # Each intensity phase must map consistently despite being interleaved below
    # the spatial subsampling grid.
    assert len(np.unique(labels[neutron == 0])) == 1
    assert len(np.unique(labels[neutron == 1])) == 1
    assert stats["scaled_features"] is True


def test_random_forest_balances_training_and_predicts_in_chunks():
    neutron = np.zeros((8, 8, 8), dtype=np.float32)
    xray = np.zeros_like(neutron)
    labels = np.zeros(neutron.shape, dtype=np.int32)
    labels[:, :, 4:] = 1
    neutron[labels == 1] = 10
    xray[labels == 1] = 3

    model = RandomForestSegmentation4D(
        n_estimators=30,
        max_samples_per_class=40,
        prediction_chunk_size=31,
    )
    stats = model._fit(neutron, xray, labels, "basic", verbose=False)
    predicted, confidence = model.predict_timepoint(neutron, xray)
    assert stats["sampled_voxels"] <= 80
    assert stats["class_balanced"] is True
    assert np.mean(predicted == labels) > 0.98
    assert confidence.shape == labels.shape


def test_loader_enforces_memmap_above_threshold(tmp_path):
    neutron = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
    xray = neutron + 2
    neutron_path = tmp_path / "neutron.tif"
    xray_path = tmp_path / "xray.tif"
    tifffile.imwrite(neutron_path, neutron)
    tifffile.imwrite(xray_path, xray)

    dataset = TIFF4DLoader.load(
        neutron_path,
        xray_path,
        use_memmap=False,
        memmap_threshold_gb=0.0,
    )
    assert dataset.metadata["memory_mapped"] is True
    assert _has_memmap_base(dataset.neutron_data)
