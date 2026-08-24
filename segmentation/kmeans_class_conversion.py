"""Conversion of 3-D K-means cluster payloads into material layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class KMeansClassConversionSummary:
    """Summary of a K-means-to-material conversion."""

    cluster_ids: tuple[int, ...]
    covered_voxels: int
    uncovered_voxels: int
    overlapping_voxels: int
    total_voxels: int


def build_material_layers_from_cluster_selections(
    cluster_selections: Iterable[Sequence],
    volume_shape: tuple[int, int, int],
) -> tuple[list[tuple[np.ndarray, object, str]], KMeansClassConversionSummary]:
    """
    Convert 3-D K-means cluster payloads into segmentation layers.

    Each input payload must use the six-field form emitted by the 3-D
    auto-detection workflow:

        (name, mask_2d, histogram_roi, cluster_id, color, mask_3d)

    The output layer format matches ``BiTS4DMainWindow.segmentation_masks``:

        (mask_3d, color, class_name)

    Classes are ordered by the original K-means cluster id, so the material
    order — and therefore every label value downstream — is deterministic.
    Every cluster becomes a material; voxels outside every cluster remain
    background.
    """
    shape = tuple(int(value) for value in volume_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"volume_shape must be a positive 3-D shape, got {shape}")

    parsed: dict[int, tuple[np.ndarray, object]] = {}

    for payload in cluster_selections:
        if len(payload) != 6:
            continue

        _name, _mask_2d, _histogram_roi, cluster_id, color, mask_3d = payload
        cluster_id = int(cluster_id)

        if cluster_id in parsed:
            raise ValueError(f"Duplicate K-means cluster id: {cluster_id}")

        mask = np.asarray(mask_3d, dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"Cluster {cluster_id} mask shape {mask.shape} does not match "
                f"volume shape {shape}"
            )
        if not np.any(mask):
            raise ValueError(f"Cluster {cluster_id} contains no voxels")

        parsed[cluster_id] = (mask.copy(), color)

    if not parsed:
        raise ValueError(
            "No 3-D K-means cluster payloads are available. "
            "Run Auto-Detect in 3-D clustering mode first."
        )

    occupancy = np.zeros(shape, dtype=np.uint32)
    layers: list[tuple[np.ndarray, object, str]] = []

    for cluster_id in sorted(parsed):
        mask, color = parsed[cluster_id]
        occupancy += mask.astype(np.uint32, copy=False)
        layers.append((mask, color, f"K-means cluster {cluster_id}"))

    covered = int(np.count_nonzero(occupancy))
    overlapping = int(np.count_nonzero(occupancy > 1))
    total = int(np.prod(shape, dtype=np.int64))

    summary = KMeansClassConversionSummary(
        cluster_ids=tuple(sorted(parsed)),
        covered_voxels=covered,
        uncovered_voxels=total - covered,
        overlapping_voxels=overlapping,
        total_voxels=total,
    )
    return layers, summary


#: Name this function had while the classifier was the destination.
build_rf_layers_from_cluster_selections = build_material_layers_from_cluster_selections
