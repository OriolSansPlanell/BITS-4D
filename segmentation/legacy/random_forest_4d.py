"""Memory-bounded Random Forest segmentation for paired 3-D/4-D datasets."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from segmentation.features import (
    LEGACY_SPECS,
    PRESETS,
    FeatureSpec,
    extract_features_at_indices,
    resolve_spec,
)

# Legacy names, kept so saved models and existing notebooks keep working.
# New code should pass a FeatureSpec or a preset name instead — see
# segmentation/features.py for why the ladder was a problem.
_VALID_LEVELS = set(LEGACY_SPECS)


def _validate_pair(neutron_vol: np.ndarray, xray_vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neutron = np.asarray(neutron_vol)
    xray = np.asarray(xray_vol)
    if neutron.shape != xray.shape:
        raise ValueError(f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}")
    if neutron.ndim != 3:
        raise ValueError("Feature extraction requires paired 3-D volumes")
    return neutron, xray


def _extract_features_at_indices(
    neutron_vol: np.ndarray,
    xray_vol: np.ndarray,
    indices: np.ndarray,
    level="basic",
) -> np.ndarray:
    """Feature matrix for the voxels at *indices*.

    *level* may be a legacy name (``'basic'``, ``'advanced'``, ``'expert'``),
    a preset name, or a :class:`~segmentation.features.FeatureSpec`. The
    legacy names produce exactly the columns they always did, in the same
    order.
    """
    neutron, xray = _validate_pair(neutron_vol, xray_vol)
    return extract_features_at_indices(neutron, xray, indices, resolve_spec(level))


def _extract_features(neutron_vol, xray_vol, level="basic") -> np.ndarray:
    """Compatibility helper; callers should prefer chunked indexed extraction."""
    neutron, _ = _validate_pair(neutron_vol, xray_vol)
    return _extract_features_at_indices(
        neutron_vol, xray_vol, np.arange(neutron.size, dtype=np.int64), level
    )


def labels_from_manual(mask, multi_class_masks: Optional[Dict[int, np.ndarray]] = None):
    mask_array = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask_array.shape, dtype=np.int32)
    if multi_class_masks:
        for class_id, class_mask in sorted(multi_class_masks.items()):
            class_mask = np.asarray(class_mask, dtype=bool)
            if class_mask.shape != labels.shape:
                raise ValueError("All class masks must have the same shape")
            labels[class_mask] = int(class_id)
    else:
        labels[mask_array] = 1
    return labels


def labels_from_kmeans(
    cluster_map,
    neutron_vol,
    xray_vol,
    background_cluster: Optional[int] = None,
):
    clusters = np.asarray(cluster_map)
    neutron, xray = _validate_pair(neutron_vol, xray_vol)
    if clusters.shape != neutron.shape:
        raise ValueError("cluster_map must match the volume shape")
    cluster_ids = np.unique(clusters)
    if background_cluster is None:
        means = {}
        for cluster_id in cluster_ids:
            mask = clusters == cluster_id
            means[int(cluster_id)] = float(
                np.mean(neutron[mask].astype(np.float64))
                + np.mean(xray[mask].astype(np.float64))
            )
        background_cluster = min(means, key=means.get)
    if background_cluster not in cluster_ids:
        raise ValueError("background_cluster is not present in cluster_map")

    labels = np.zeros(clusters.shape, dtype=np.int32)
    next_label = 1
    for cluster_id in sorted(cluster_ids):
        if cluster_id == background_cluster:
            continue
        labels[clusters == cluster_id] = next_label
        next_label += 1
    return labels


def labels_from_otsu(neutron_vol, xray_vol, n_classes=3, channel="both"):
    try:
        from skimage.filters import threshold_multiotsu, threshold_otsu
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-image is required for Otsu segmentation") from exc

    neutron, xray = _validate_pair(neutron_vol, xray_vol)
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if channel == "neutron":
        data = neutron.astype(np.float32, copy=False)
    elif channel == "xray":
        data = xray.astype(np.float32, copy=False)
    elif channel == "both":
        def normalize(volume):
            array = volume.astype(np.float32, copy=False)
            minimum = float(np.nanmin(array))
            maximum = float(np.nanmax(array))
            if maximum <= minimum:
                return np.zeros(array.shape, dtype=np.float32)
            return (array - minimum) / (maximum - minimum)

        data = (normalize(neutron) + normalize(xray)) / 2.0
    else:
        raise ValueError("channel must be 'neutron', 'xray', or 'both'")

    finite_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        thresholds = threshold_multiotsu(finite_data, classes=n_classes)
    except ValueError:
        thresholds = [threshold_otsu(finite_data)]
    return np.digitize(finite_data, bins=thresholds).astype(np.int32)


class RandomForestSegmentation4D:
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples_per_class: int = 100_000,
        prediction_chunk_size: int = 1_000_000,
        random_state: int = 42,
        min_samples_leaf: int = 50,
        max_depth: Optional[int] = 20,
    ) -> None:
        """
        Parameters
        ----------
        min_samples_leaf, max_depth
            Capacity limits. Neighbouring voxels are near-duplicates, so an
            unrestricted forest grows leaves around individual voxels and
            memorises the training volume — which inflates in-sample accuracy
            without improving anything that transfers to another timepoint.
            The defaults (50, 20) are deliberately tighter than scikit-learn's
            (1, unlimited); pass ``min_samples_leaf=5, max_depth=None`` to
            reproduce a model trained before this changed.
        """
        self.n_estimators = int(n_estimators)
        self.max_samples_per_class = int(max_samples_per_class)
        self.prediction_chunk_size = int(prediction_chunk_size)
        self.random_state = int(random_state)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.rf: Optional[RandomForestClassifier] = None
        self.scaler = None  # retained for backward-compatible model payloads
        self.feature_level = "basic"
        self.trained = False
        self.train_stats: dict = {}
        self.classes_: Optional[np.ndarray] = None

    def _sample_indices(self, labels_flat: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        sampled = []
        for class_id in np.unique(labels_flat):
            class_indices = np.flatnonzero(labels_flat == class_id)
            count = min(len(class_indices), self.max_samples_per_class)
            if count == 0:
                continue
            if len(class_indices) > count:
                class_indices = rng.choice(class_indices, count, replace=False)
            sampled.append(np.asarray(class_indices, dtype=np.int64))
        if len(sampled) < 2:
            raise ValueError("Random Forest training requires at least two classes")
        result = np.concatenate(sampled)
        rng.shuffle(result)
        return result

    def _fit(
        self,
        neutron_vol,
        xray_vol,
        labels,
        feature_level,
        verbose=True,
    ) -> dict:
        neutron, xray = _validate_pair(neutron_vol, xray_vol)
        labels_array = np.asarray(labels, dtype=np.int32)
        if labels_array.shape != neutron.shape:
            raise ValueError("labels must match the training volume shape")
        spec = resolve_spec(feature_level)

        labels_flat = labels_array.reshape(-1)
        sampled_indices = self._sample_indices(labels_flat)
        x_train = _extract_features_at_indices(
            neutron, xray, sampled_indices, feature_level
        )
        y_train = labels_flat[sampled_indices]
        self.feature_level = feature_level

        # The capacity limit is a ceiling, not a fixed size: a leaf floor of
        # 50 is right for a million sampled voxels and fatal for a few
        # hundred, where it would stop the forest splitting at all. Scale it
        # to the training set so the intent — leaves that hold a
        # neighbourhood rather than a voxel — survives at every dataset size.
        n_train = int(len(y_train))
        n_classes_train = max(int(np.unique(y_train).size), 1)
        effective_leaf = int(
            max(1, min(self.min_samples_leaf, n_train // (n_classes_train * 20)))
        )

        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_split=max(2 * effective_leaf, 10),
            min_samples_leaf=effective_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
            oob_score=True,
            max_samples=0.8,
        )
        self.rf.fit(x_train, y_train)
        self.classes_ = self.rf.classes_
        self.trained = True

        train_accuracy = float(np.mean(self.rf.predict(x_train) == y_train))

        # How much of the decision function rests on features that are
        # identical at every timepoint. Impurity importance over-weights
        # continuous, high-cardinality features — which normalised
        # coordinates are — so read this as an upper bound and confirm with
        # utils.validation.permutation_anchoring_index on held-out blocks.
        feature_names = spec.feature_names()
        anchored = spec.anchored_features()
        anchoring = 0.0
        if anchored:
            from utils.validation import anchoring_index

            anchoring = float(
                anchoring_index(
                    self.rf.feature_importances_, feature_names, anchored
                )
            )

        unique_all, counts_all = np.unique(labels_flat, return_counts=True)
        unique_sample, counts_sample = np.unique(y_train, return_counts=True)
        self.train_stats = {
            "n_voxels": int(labels_flat.size),
            "sampled_voxels": int(len(sampled_indices)),
            "n_features": int(x_train.shape[1]),
            "n_classes": int(len(unique_all)),
            "classes": unique_all.tolist(),
            "class_counts": {
                int(class_id): int(count)
                for class_id, count in zip(unique_all, counts_all)
            },
            "sampled_class_counts": {
                int(class_id): int(count)
                for class_id, count in zip(unique_sample, counts_sample)
            },
            "training_accuracy": train_accuracy,
            "oob_score": float(getattr(self.rf, "oob_score_", np.nan)),
            "feature_level": feature_level,
            "feature_spec": spec.describe(),
            "feature_names": feature_names,
            "anchored_features": anchored,
            "anchoring_index": anchoring,
            "time_invariant": bool(spec.is_time_invariant),
            "min_samples_leaf": effective_leaf,
            "min_samples_leaf_requested": self.min_samples_leaf,
            "max_depth": self.max_depth,
            "class_balanced": True,
        }
        return dict(self.train_stats)

    def train_from_manual(
        self,
        neutron_vol,
        xray_vol,
        mask,
        feature_level="basic",
        multi_class_masks=None,
        verbose=True,
    ):
        return self._fit(
            neutron_vol,
            xray_vol,
            labels_from_manual(mask, multi_class_masks),
            feature_level,
            verbose,
        )

    def train_from_kmeans(
        self,
        neutron_vol,
        xray_vol,
        cluster_map,
        feature_level="basic",
        background_cluster=None,
        verbose=True,
    ):
        return self._fit(
            neutron_vol,
            xray_vol,
            labels_from_kmeans(
                cluster_map, neutron_vol, xray_vol, background_cluster
            ),
            feature_level,
            verbose,
        )

    def train_from_otsu(
        self,
        neutron_vol,
        xray_vol,
        n_classes=3,
        feature_level="basic",
        channel="both",
        verbose=True,
    ):
        return self._fit(
            neutron_vol,
            xray_vol,
            labels_from_otsu(neutron_vol, xray_vol, n_classes, channel),
            feature_level,
            verbose,
        )

    def predict_timepoint(
        self,
        neutron_vol,
        xray_vol,
        return_confidence=True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.trained or self.rf is None:
            raise RuntimeError("Model not trained")
        neutron, xray = _validate_pair(neutron_vol, xray_vol)
        total = neutron.size
        labels_flat = np.empty(total, dtype=np.int32)
        confidence_flat = (
            np.empty(total, dtype=np.float32) if return_confidence else None
        )

        for start in range(0, total, self.prediction_chunk_size):
            if cancel_check:
                cancel_check()
            stop = min(start + self.prediction_chunk_size, total)
            indices = np.arange(start, stop, dtype=np.int64)
            features = _extract_features_at_indices(
                neutron, xray, indices, self.feature_level
            )
            labels_flat[start:stop] = self.rf.predict(features).astype(np.int32)
            if return_confidence:
                probabilities = self.rf.predict_proba(features)
                confidence_flat[start:stop] = np.max(probabilities, axis=1).astype(
                    np.float32
                )
            if progress_callback:
                progress_callback(
                    int(100 * stop / total),
                    f"Predicting voxels {stop:,}/{total:,}",
                )

        labels = labels_flat.reshape(neutron.shape)
        confidence = (
            None
            if confidence_flat is None
            else confidence_flat.reshape(neutron.shape)
        )
        return labels, confidence

    def predict_all_timepoints(
        self,
        neutron_4d,
        xray_4d,
        confidence_threshold=0.70,
        progress_callback=None,
        verbose=True,
        cancel_check=None,
    ) -> Dict[int, dict]:
        neutron = np.asarray(neutron_4d)
        xray = np.asarray(xray_4d)
        if neutron.shape != xray.shape or neutron.ndim != 4:
            raise ValueError("Prediction inputs must be matching 4-D arrays")
        results = {}
        for timepoint in range(neutron.shape[0]):
            if cancel_check:
                cancel_check()

            def local_progress(value, message):
                if progress_callback:
                    overall = int(
                        100 * (timepoint + value / 100.0) / neutron.shape[0]
                    )
                    progress_callback(overall, f"T={timepoint}: {message}")

            labels, confidence = self.predict_timepoint(
                neutron[timepoint],
                xray[timepoint],
                return_confidence=True,
                progress_callback=local_progress,
                cancel_check=cancel_check,
            )
            uncertain_fraction = float(
                np.mean(confidence < float(confidence_threshold))
            )
            results[timepoint] = {
                "labels": labels,
                "confidence": confidence,
                "mean_confidence": float(np.mean(confidence)),
                "uncertain_fraction": uncertain_fraction,
                "needs_review": uncertain_fraction > 0.10,
            }
        if progress_callback:
            progress_callback(100, "RF prediction complete")
        return results

    def save_model(self, filepath: str) -> None:
        if not self.trained:
            raise RuntimeError("No trained model to save")
        payload = {
            "rf": self.rf,
            "scaler": self.scaler,
            "feature_level": self.feature_level,
            "train_stats": self.train_stats,
            "classes": self.classes_,
            "max_samples_per_class": self.max_samples_per_class,
            "prediction_chunk_size": self.prediction_chunk_size,
            "random_state": self.random_state,
            "min_samples_leaf": self.min_samples_leaf,
            "max_depth": self.max_depth,
        }
        with Path(filepath).open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filepath: str) -> None:
        with Path(filepath).open("rb") as handle:
            payload = pickle.load(handle)
        self.rf = payload["rf"]
        self.scaler = payload.get("scaler")
        self.feature_level = payload["feature_level"]
        self.train_stats = payload.get("train_stats", {})
        self.classes_ = payload.get("classes")
        self.max_samples_per_class = payload.get(
            "max_samples_per_class", self.max_samples_per_class
        )
        self.prediction_chunk_size = payload.get(
            "prediction_chunk_size", self.prediction_chunk_size
        )
        self.random_state = payload.get("random_state", self.random_state)
        # Models saved before capacity limits existed carry the old values
        self.min_samples_leaf = payload.get("min_samples_leaf", self.min_samples_leaf)
        self.max_depth = payload.get("max_depth", self.max_depth)
        self.trained = True

    @property
    def is_trained(self) -> bool:
        return self.trained

    def get_train_stats(self) -> dict:
        return dict(self.train_stats)

    def __repr__(self) -> str:
        status = "trained" if self.trained else "untrained"
        return (
            "RandomForestSegmentation4D("
            f"n_estimators={self.n_estimators}, status={status})"
        )
