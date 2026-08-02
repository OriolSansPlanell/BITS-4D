"""Memory-bounded, scale-aware K-means clustering for paired 3-D volumes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeans3D:
    @staticmethod
    def _validate(neutron_vol, xray_vol, n_clusters, subsample_factor):
        neutron = np.asarray(neutron_vol)
        xray = np.asarray(xray_vol)
        if neutron.shape != xray.shape:
            raise ValueError(
                f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}"
            )
        if neutron.ndim != 3:
            raise ValueError("K-means requires paired 3-D volumes")
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if subsample_factor < 1:
            raise ValueError("subsample_factor must be positive")
        return neutron, xray

    @staticmethod
    def _sample_features(
        neutron: np.ndarray,
        xray: np.ndarray,
        factor: int,
        max_fit_samples: int,
        random_state: int,
        n_clusters: int,
    ) -> np.ndarray:
        n_sample = neutron[::factor, ::factor, ::factor].reshape(-1)
        x_sample = xray[::factor, ::factor, ::factor].reshape(-1)
        features = np.column_stack((n_sample, x_sample))
        finite = np.all(np.isfinite(features), axis=1)
        features = features[finite]

        rng = np.random.default_rng(random_state)
        if len(features) > max_fit_samples:
            indices = rng.choice(len(features), max_fit_samples, replace=False)
            features = features[indices]

        # A regular grid can alias a periodic microstructure and contain fewer
        # distinct phases than requested.  Fall back to a random full-resolution
        # sample in that case.
        if len(features) < n_clusters or len(np.unique(features, axis=0)) < n_clusters:
            full_n = neutron.reshape(-1)
            full_x = xray.reshape(-1)
            finite = np.isfinite(full_n) & np.isfinite(full_x)
            valid_indices = np.flatnonzero(finite)
            if len(valid_indices) < n_clusters:
                raise ValueError("Not enough finite distinct voxels for K-means")
            count = min(max_fit_samples, len(valid_indices))
            chosen = rng.choice(valid_indices, count, replace=False)
            features = np.column_stack((full_n[chosen], full_x[chosen]))
        return features.astype(np.float64, copy=False)

    @staticmethod
    def cluster_volume(
        neutron_vol,
        xray_vol,
        n_clusters: int = 3,
        subsample_factor: int = 1,
        random_state: int = 42,
        max_fit_samples: int = 250_000,
        prediction_chunk_size: int = 2_000_000,
        progress_callback=None,
        cancel_check=None,
    ):
        neutron, xray = KMeans3D._validate(
            neutron_vol, xray_vol, n_clusters, subsample_factor
        )
        if max_fit_samples < n_clusters:
            raise ValueError("max_fit_samples must be at least n_clusters")

        fit_features = KMeans3D._sample_features(
            neutron,
            xray,
            subsample_factor,
            max_fit_samples,
            random_state,
            n_clusters,
        )
        scaler = StandardScaler()
        fit_scaled = scaler.fit_transform(fit_features)
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        model.fit(fit_scaled)
        if progress_callback:
            progress_callback(30, f"Fitted K-means on {len(fit_features):,} voxels")

        n_flat = neutron.reshape(-1)
        x_flat = xray.reshape(-1)
        labels_flat = np.zeros(n_flat.size, dtype=np.int32)
        finite_all = np.isfinite(n_flat) & np.isfinite(x_flat)
        valid_indices = np.flatnonzero(finite_all)
        for start in range(0, len(valid_indices), prediction_chunk_size):
            if cancel_check:
                cancel_check()
            chunk_indices = valid_indices[start : start + prediction_chunk_size]
            chunk = np.column_stack((n_flat[chunk_indices], x_flat[chunk_indices]))
            labels_flat[chunk_indices] = model.predict(scaler.transform(chunk))
            if progress_callback:
                progress_callback(
                    30 + int(70 * min(start + len(chunk_indices), len(valid_indices)) / max(len(valid_indices), 1)),
                    "Classifying full-resolution voxels...",
                )

        labels_3d = labels_flat.reshape(neutron.shape)
        centers_original = scaler.inverse_transform(model.cluster_centers_)
        stats = {
            "n_clusters": int(n_clusters),
            "subsample_factor": int(subsample_factor),
            "fit_voxels": int(len(fit_features)),
            "total_voxels": int(neutron.size),
            "finite_voxels": int(len(valid_indices)),
            "cluster_centers": centers_original,
            "inertia_scaled": float(model.inertia_),
            "scaled_features": True,
        }
        for cluster_id in range(n_clusters):
            count = int(np.count_nonzero(labels_3d == cluster_id))
            stats[f"cluster_{cluster_id}_voxels"] = count
            stats[f"cluster_{cluster_id}_percent"] = 100.0 * count / labels_3d.size
        return labels_3d, centers_original, stats

    @staticmethod
    def _upsample_labels(labels_sub, target_shape, factor):
        """Legacy helper retained for API compatibility; prediction no longer uses it."""
        zoom_factors = tuple(target_shape[i] / labels_sub.shape[i] for i in range(3))
        labels_full = zoom(labels_sub, zoom_factors, order=0)
        result = np.zeros(target_shape, dtype=labels_sub.dtype)
        slices = tuple(
            slice(0, min(labels_full.shape[i], target_shape[i])) for i in range(3)
        )
        result[slices] = labels_full[slices]
        return result

    @staticmethod
    def create_cluster_masks(labels_3d, n_clusters):
        return [np.asarray(labels_3d) == index for index in range(n_clusters)]

    @staticmethod
    def extract_slice_from_labels(labels_3d, axis, slice_index):
        if axis == "z":
            return labels_3d[slice_index, :, :]
        if axis == "y":
            return labels_3d[:, slice_index, :]
        if axis == "x":
            return labels_3d[:, :, slice_index]
        raise ValueError("axis must be 'z', 'y', or 'x'")

    @staticmethod
    def get_cluster_statistics(labels_3d, cluster_id, neutron_vol, xray_vol):
        mask = np.asarray(labels_3d) == cluster_id
        if not np.any(mask):
            return None
        neutron_values = np.asarray(neutron_vol)[mask]
        xray_values = np.asarray(xray_vol)[mask]
        return {
            "cluster_id": int(cluster_id),
            "voxels": int(np.count_nonzero(mask)),
            "volume_fraction": float(np.mean(mask)),
            "neutron_mean": float(np.mean(neutron_values)),
            "neutron_std": float(np.std(neutron_values)),
            "xray_mean": float(np.mean(xray_values)),
            "xray_std": float(np.std(xray_values)),
        }

    @staticmethod
    def create_convex_hull_roi_3d(
        neutron_vals,
        xray_vals,
        percentile: float = 98,
        density_aware: bool = True,
        random_state: int = 42,
    ):
        from scipy.spatial import ConvexHull, QhullError

        neutron = np.asarray(neutron_vals).reshape(-1)
        xray = np.asarray(xray_vals).reshape(-1)
        finite = np.isfinite(neutron) & np.isfinite(xray)
        neutron = neutron[finite]
        xray = xray[finite]
        if len(neutron) == 0:
            raise ValueError("Cannot build an ROI from an empty cluster")

        rng = np.random.default_rng(random_state)
        count = min(5000, len(neutron))
        if len(neutron) > count:
            indices = rng.choice(len(neutron), count, replace=False)
            neutron = neutron[indices]
            xray = xray[indices]

        lower_q = max(0.0, 100.0 - float(percentile))
        upper_q = min(100.0, float(percentile))
        n_lower, n_upper = np.percentile(neutron, [lower_q, upper_q])
        x_lower, x_upper = np.percentile(xray, [lower_q, upper_q])
        keep = (
            (neutron >= n_lower)
            & (neutron <= n_upper)
            & (xray >= x_lower)
            & (xray <= x_upper)
        )
        points = np.column_stack((neutron[keep], xray[keep]))

        if len(points) >= 3 and len(np.unique(points, axis=0)) >= 3:
            try:
                hull = ConvexHull(points)
                return points[hull.vertices]
            except QhullError:
                pass
        return np.array(
            [
                [float(np.min(neutron)), float(np.min(xray))],
                [float(np.max(neutron)), float(np.min(xray))],
                [float(np.max(neutron)), float(np.max(xray))],
                [float(np.min(neutron)), float(np.max(xray))],
            ]
        )
