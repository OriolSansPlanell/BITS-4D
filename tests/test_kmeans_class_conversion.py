import numpy as np
import pytest

from segmentation.kmeans_class_conversion import (
    build_material_layers_from_cluster_selections,
)


def _payload(cluster_id, mask):
    return (
        f"3D Cluster {cluster_id}",
        mask[0],
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        cluster_id,
        (1.0, 0.0, 0.0, 0.5),
        mask,
    )


def test_kmeans_clusters_become_sorted_rf_layers():
    shape = (2, 2, 2)
    cluster_0 = np.zeros(shape, dtype=bool)
    cluster_0[:, :, :1] = True
    cluster_1 = ~cluster_0

    layers, summary = build_material_layers_from_cluster_selections(
        [_payload(1, cluster_1), _payload(0, cluster_0)],
        shape,
    )

    assert [layer[2] for layer in layers] == [
        "K-means cluster 0",
        "K-means cluster 1",
    ]
    np.testing.assert_array_equal(layers[0][0], cluster_0)
    np.testing.assert_array_equal(layers[1][0], cluster_1)
    assert summary.cluster_ids == (0, 1)
    assert summary.covered_voxels == 8
    assert summary.uncovered_voxels == 0
    assert summary.overlapping_voxels == 0


def test_conversion_reports_overlap_and_uncovered_voxels():
    shape = (1, 2, 3)
    first = np.array([[[True, True, False], [False, False, False]]])
    second = np.array([[[False, True, False], [False, True, False]]])

    _, summary = build_material_layers_from_cluster_selections(
        [_payload(0, first), _payload(1, second)],
        shape,
    )

    assert summary.covered_voxels == 3
    assert summary.uncovered_voxels == 3
    assert summary.overlapping_voxels == 1


def test_conversion_requires_3d_payloads():
    five_field_payload = (
        "2D Cluster 0",
        np.ones((2, 2), dtype=bool),
        np.ones((3, 2), dtype=float),
        0,
        "red",
    )

    with pytest.raises(ValueError, match="No 3-D K-means"):
        build_material_layers_from_cluster_selections(
            [five_field_payload],
            (2, 2, 2),
        )


def test_conversion_rejects_wrong_mask_shape():
    with pytest.raises(ValueError, match="does not match"):
        build_material_layers_from_cluster_selections(
            [_payload(0, np.ones((2, 2, 2), dtype=bool))],
            (3, 2, 2),
        )
