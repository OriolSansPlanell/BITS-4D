"""
random_forest_4d.py - Random Forest Segmentation for 4D Multimodal Datasets

Trains a Random Forest classifier using labels derived from:
  - Manual ROI segmentation
  - K-means clustering
  - Multi-level Otsu thresholding

Background (label = 0) is always included as an explicit class during training.
Predictions propagate to all timepoints using 3D spatial + intensity features.

Version: 15.2  (ported from BiTS_v16 / improved_rf.py)
"""

import numpy as np
import sys
import pickle
from typing import Optional, Callable, Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import sobel, uniform_filter, generic_filter, laplace


# ─────────────────────────────────────────────────────────────
#  Feature extraction helpers
# ─────────────────────────────────────────────────────────────

def _extract_features(
    neutron_vol: np.ndarray,
    xray_vol: np.ndarray,
    level: str = "basic",
) -> np.ndarray:
    """
    Build a feature matrix from a single 3-D volume pair.

    Parameters
    ----------
    neutron_vol, xray_vol : (Z, Y, X) float arrays
    level : 'basic' | 'advanced' | 'expert'

    Feature layout
    --------------
    basic (6):
        neutron, xray, ratio (n/x), sum, difference, magnitude

    advanced (+3):
        normalised Z, Y, X spatial coordinates

    expert (+6):
        local mean & std of each channel (3×3×3 window)
        laplacian of neutron channel
    """
    n = neutron_vol.astype(np.float32)
    x = xray_vol.astype(np.float32)

    feats = [
        n.ravel(),
        x.ravel(),
        (n / (x + 1e-6)).ravel(),
        (n + x).ravel(),
        (n - x).ravel(),
        np.sqrt(n ** 2 + x ** 2).ravel(),
    ]

    if level in ("advanced", "expert"):
        Z, Y, X = np.meshgrid(
            np.linspace(0, 1, n.shape[0]),
            np.linspace(0, 1, n.shape[1]),
            np.linspace(0, 1, n.shape[2]),
            indexing="ij",
        )
        feats += [Z.ravel(), Y.ravel(), X.ravel()]

    if level == "expert":
        for vol in (n, x):
            feats.append(uniform_filter(vol, size=3).ravel())
            feats.append(generic_filter(vol, np.std, size=3).ravel())
        feats.append(laplace(n).ravel())

    return np.column_stack(feats)


# ─────────────────────────────────────────────────────────────
#  Label-generation helpers
# ─────────────────────────────────────────────────────────────

def labels_from_manual(
    mask: np.ndarray,
    multi_class_masks: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """
    Convert a binary ROI mask (or dict of class masks) to an integer label array.

    Parameters
    ----------
    mask : (Z, Y, X) bool  – single foreground mask (class 1); rest = background (0)
    multi_class_masks : optional dict {class_id -> bool mask}
        If provided, overrides `mask` and assigns each class its own integer label.
        Unassigned voxels get label 0 (background).

    Returns
    -------
    labels : (Z, Y, X) int32
        0 = background, 1+ = foreground classes
    """
    labels = np.zeros(mask.shape, dtype=np.int32)

    if multi_class_masks is not None and len(multi_class_masks) > 0:
        for class_id, m in sorted(multi_class_masks.items()):
            labels[m.astype(bool)] = int(class_id)
    else:
        labels[mask.astype(bool)] = 1

    print(
        f"  Manual labels: {np.sum(labels > 0):,} foreground voxels, "
        f"{np.sum(labels == 0):,} background",
        file=sys.stderr,
    )
    return labels


def labels_from_kmeans(
    cluster_map: np.ndarray,
    neutron_vol: np.ndarray,
    xray_vol: np.ndarray,
    background_cluster: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a k-means cluster map to a label array with explicit background class.

    Parameters
    ----------
    cluster_map : (Z, Y, X) int – cluster ids 0 … K-1
    neutron_vol, xray_vol : volumes for auto-detecting background cluster
    background_cluster : if given, that cluster id is designated class 0;
                         otherwise the cluster with the lowest combined
                         mean intensity is chosen automatically.

    Returns
    -------
    labels : (Z, Y, X) int32
        0 = background cluster, 1+ = remaining clusters (renumbered sequentially)
    """
    cluster_ids = np.unique(cluster_map)

    # Auto-detect background = cluster with lowest mean (neutron + xray) intensity
    if background_cluster is None:
        mean_intensity = {}
        for cid in cluster_ids:
            m = cluster_map == cid
            mean_intensity[cid] = (
                float(np.mean(neutron_vol[m])) + float(np.mean(xray_vol[m]))
            ) / 2.0
        background_cluster = min(mean_intensity, key=mean_intensity.get)
        print(
            f"  K-means: auto-selected cluster {background_cluster} as background "
            f"(mean intensity {mean_intensity[background_cluster]:.1f})",
            file=sys.stderr,
        )

    # Build label map: background → 0, rest → 1, 2, …
    labels = np.zeros_like(cluster_map, dtype=np.int32)
    new_id = 1
    for cid in sorted(cluster_ids):
        if cid == background_cluster:
            continue  # stays 0
        labels[cluster_map == cid] = new_id
        new_id += 1

    for lbl in sorted(np.unique(labels)):
        count = int(np.sum(labels == lbl))
        name = "Background" if lbl == 0 else f"Class {lbl}"
        print(f"    {name}: {count:,} voxels", file=sys.stderr)

    return labels


def labels_from_otsu(
    neutron_vol: np.ndarray,
    xray_vol: np.ndarray,
    n_classes: int = 3,
    channel: str = "both",
) -> np.ndarray:
    """
    Apply multi-level Otsu thresholding and return a label array.

    Parameters
    ----------
    neutron_vol, xray_vol : (Z, Y, X) arrays
    n_classes : total number of classes (including background = 0).
                Multi-level Otsu will find (n_classes - 1) thresholds.
    channel : 'neutron' | 'xray' | 'both'
        Which channel to threshold. 'both' averages the two.

    Returns
    -------
    labels : (Z, Y, X) int32
        0 = lowest-intensity class (background), 1+ = brighter classes
    """
    try:
        from skimage.filters import threshold_multiotsu
    except ImportError:
        raise ImportError(
            "scikit-image is required for Otsu segmentation. "
            "Install it with: pip install scikit-image"
        )

    if channel == "neutron":
        data = neutron_vol.astype(np.float32)
    elif channel == "xray":
        data = xray_vol.astype(np.float32)
    else:  # both
        data = (
            neutron_vol.astype(np.float32) / (neutron_vol.max() + 1e-6)
            + xray_vol.astype(np.float32) / (xray_vol.max() + 1e-6)
        ) / 2.0

    n_thresholds = max(1, n_classes - 1)

    try:
        thresholds = threshold_multiotsu(data, classes=n_classes)
    except Exception as e:
        print(f"  Multi-Otsu failed ({e}), falling back to single Otsu", file=sys.stderr)
        from skimage.filters import threshold_otsu
        thresholds = [threshold_otsu(data)]

    labels = np.digitize(data, bins=thresholds).astype(np.int32)
    # digitize returns 0…N-1 → already 0-based; class 0 = below first threshold = background

    print(f"  Otsu thresholds: {[f'{t:.2f}' for t in thresholds]}", file=sys.stderr)
    for lbl in sorted(np.unique(labels)):
        count = int(np.sum(labels == lbl))
        name = "Background" if lbl == 0 else f"Class {lbl}"
        print(f"    {name}: {count:,} voxels", file=sys.stderr)

    return labels


# ─────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────

class RandomForestSegmentation4D:
    """
    Random Forest segmentation for 4D dual-modality (neutron + X-ray) datasets.

    Workflow
    --------
    1. Generate training labels via one of:
         - ``train_from_manual()``  – uses existing ROI segmentation mask(s)
         - ``train_from_kmeans()``  – uses 3-D k-means cluster map
         - ``train_from_otsu()``    – runs multi-level Otsu on the reference volume

    2. Call ``predict_timepoint()`` or ``predict_all_timepoints()`` to propagate.

    In every case **background (label 0) is always included** as a training class.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    """

    def __init__(self, n_estimators: int = 100):
        self.n_estimators = n_estimators
        self.rf: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_level: str = "basic"
        self.trained: bool = False
        self.train_stats: dict = {}
        self.classes_: Optional[np.ndarray] = None

    # ── training ──────────────────────────────────────────────

    def _fit(
        self,
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
        labels: np.ndarray,
        feature_level: str,
        verbose: bool,
    ) -> dict:
        """
        Internal training routine shared by all ``train_from_*`` methods.

        ``labels`` must be a (Z, Y, X) int32 array where
        0 = background and positive integers are foreground classes.
        ALL labelled voxels (including background=0) are used for training.
        """
        if verbose:
            print(
                f"\nRandom Forest training  [{feature_level} features]",
                file=sys.stderr,
            )

        self.feature_level = feature_level

        # Feature matrix for the training timepoint
        X_all = _extract_features(neutron_vol, xray_vol, feature_level)
        y_all = labels.ravel().astype(np.int32)

        # Statistics
        unique_cls, counts = np.unique(y_all, return_counts=True)
        class_dist = {int(c): int(n) for c, n in zip(unique_cls, counts)}

        if verbose:
            print(
                f"  Total voxels: {len(y_all):,}  |  "
                f"features: {X_all.shape[1]}",
                file=sys.stderr,
            )
            for cls_id, cnt in class_dist.items():
                name = "Background" if cls_id == 0 else f"Class {cls_id}"
                print(
                    f"    {name}: {cnt:,} ({100*cnt/len(y_all):.1f}%)",
                    file=sys.stderr,
                )

        # Normalise
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_all)

        # Train
        if verbose:
            print("  Training Random Forest …", file=sys.stderr)

        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        self.rf.fit(X_scaled, y_all)
        self.classes_ = self.rf.classes_
        self.trained = True

        # Training accuracy
        acc = float(np.mean(self.rf.predict(X_scaled) == y_all))
        if verbose:
            print(f"  Training accuracy: {acc * 100:.2f}%", file=sys.stderr)
            print("  ✓ Training complete!", file=sys.stderr)

        self.train_stats = {
            "n_voxels": int(len(y_all)),
            "n_features": int(X_all.shape[1]),
            "n_classes": int(len(unique_cls)),
            "classes": unique_cls.tolist(),
            "class_counts": class_dist,
            "training_accuracy": acc,
            "feature_level": feature_level,
        }
        return self.train_stats

    # ── public training entry-points ─────────────────────────

    def train_from_manual(
        self,
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
        mask: np.ndarray,
        feature_level: str = "basic",
        multi_class_masks: Optional[Dict[int, np.ndarray]] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train using an existing binary (or multi-class) ROI segmentation mask.

        Parameters
        ----------
        neutron_vol, xray_vol : (Z, Y, X) arrays  – reference timepoint volumes
        mask : (Z, Y, X) bool – True = foreground
        feature_level : 'basic' | 'advanced' | 'expert'
        multi_class_masks : optional dict {class_id (int) -> bool mask}
            For multi-class manual segmentation; overrides ``mask``.
        """
        if verbose:
            print("\n[RF] Source: Manual ROI segmentation", file=sys.stderr)
        labels = labels_from_manual(mask, multi_class_masks)
        return self._fit(neutron_vol, xray_vol, labels, feature_level, verbose)

    def train_from_kmeans(
        self,
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
        cluster_map: np.ndarray,
        feature_level: str = "basic",
        background_cluster: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train using a k-means cluster label map.

        Parameters
        ----------
        cluster_map : (Z, Y, X) int  – cluster ids 0 … K-1 (from KMeans3D)
        background_cluster : cluster id to designate as background (0).
            If None the cluster with the lowest combined mean intensity is used.
        """
        if verbose:
            print("\n[RF] Source: K-means cluster map", file=sys.stderr)
        labels = labels_from_kmeans(
            cluster_map, neutron_vol, xray_vol, background_cluster
        )
        return self._fit(neutron_vol, xray_vol, labels, feature_level, verbose)

    def train_from_otsu(
        self,
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
        n_classes: int = 3,
        feature_level: str = "basic",
        channel: str = "both",
        verbose: bool = True,
    ) -> dict:
        """
        Train using multi-level Otsu thresholding.

        Parameters
        ----------
        n_classes : total number of output classes (including background = 0)
        channel : 'neutron' | 'xray' | 'both' – channel used for thresholding
        """
        if verbose:
            print("\n[RF] Source: Multi-level Otsu thresholding", file=sys.stderr)
        labels = labels_from_otsu(neutron_vol, xray_vol, n_classes, channel)
        return self._fit(neutron_vol, xray_vol, labels, feature_level, verbose)

    # ── prediction ────────────────────────────────────────────

    def predict_timepoint(
        self,
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
        return_confidence: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict segmentation labels for a single 3-D volume pair.

        Returns
        -------
        labels : (Z, Y, X) int32
        confidence : (Z, Y, X) float32 or None
            Maximum predicted-class probability at each voxel.
        """
        if not self.trained:
            raise RuntimeError("Model not trained – call one of the train_from_*() methods first.")

        X = _extract_features(neutron_vol, xray_vol, self.feature_level)
        X_scaled = self.scaler.transform(X)

        labels_flat = self.rf.predict(X_scaled).astype(np.int32)
        labels = labels_flat.reshape(neutron_vol.shape)

        if return_confidence:
            proba = self.rf.predict_proba(X_scaled)
            conf_flat = np.max(proba, axis=1).astype(np.float32)
            confidence = conf_flat.reshape(neutron_vol.shape)
            return labels, confidence

        return labels, None

    def predict_all_timepoints(
        self,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
        confidence_threshold: float = 0.70,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        verbose: bool = True,
    ) -> Dict[int, dict]:
        """
        Predict labels for every timepoint in a 4-D dataset.

        Parameters
        ----------
        neutron_4d, xray_4d : (T, Z, Y, X) arrays
        confidence_threshold : voxels below this confidence are flagged for review
        progress_callback : optional callable(percent: int, message: str)

        Returns
        -------
        results : dict  {timepoint_index -> {
            'labels'            : (Z,Y,X) int32,
            'confidence'        : (Z,Y,X) float32,
            'mean_confidence'   : float,
            'uncertain_fraction': float,
            'needs_review'      : bool,
        }}
        """
        if not self.trained:
            raise RuntimeError("Model not trained – call one of the train_from_*() methods first.")

        T = neutron_4d.shape[0]
        results: Dict[int, dict] = {}

        for t in range(T):
            pct = int((t / T) * 100)
            msg = f"RF predicting timepoint {t + 1}/{T}"
            if progress_callback:
                progress_callback(pct, msg)
            if verbose:
                print(f"  {msg} …", file=sys.stderr)

            lbl, conf = self.predict_timepoint(
                neutron_4d[t], xray_4d[t], return_confidence=True
            )
            mean_conf = float(np.mean(conf))
            unc_frac = float(np.sum(conf < confidence_threshold) / conf.size)

            results[t] = {
                "labels": lbl,
                "confidence": conf,
                "mean_confidence": mean_conf,
                "uncertain_fraction": unc_frac,
                "needs_review": unc_frac > 0.10,
            }

            if verbose:
                flag = " ⚠ REVIEW" if results[t]["needs_review"] else ""
                print(
                    f"    confidence {mean_conf * 100:.1f}% "
                    f"| uncertain {unc_frac * 100:.1f}%{flag}",
                    file=sys.stderr,
                )

        if progress_callback:
            progress_callback(100, "RF prediction complete")
        if verbose:
            avg_conf = np.mean([r["mean_confidence"] for r in results.values()])
            n_review = sum(1 for r in results.values() if r["needs_review"])
            print(
                f"\n  RF summary: {T} timepoints | "
                f"avg confidence {avg_conf * 100:.1f}% | "
                f"{n_review} flagged for review",
                file=sys.stderr,
            )

        return results

    # ── model persistence ─────────────────────────────────────

    def save_model(self, filepath: str) -> None:
        """Serialise trained model to disk (pickle)."""
        if not self.trained:
            raise RuntimeError("No trained model to save.")
        payload = {
            "rf": self.rf,
            "scaler": self.scaler,
            "feature_level": self.feature_level,
            "train_stats": self.train_stats,
            "classes": self.classes_,
        }
        with open(filepath, "wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        print(f"[RF] Model saved → {filepath}", file=sys.stderr)

    def load_model(self, filepath: str) -> None:
        """Deserialise a previously saved model from disk."""
        with open(filepath, "rb") as fh:
            payload = pickle.load(fh)
        self.rf = payload["rf"]
        self.scaler = payload["scaler"]
        self.feature_level = payload["feature_level"]
        self.train_stats = payload.get("train_stats", {})
        self.classes_ = payload.get("classes", None)
        self.trained = True
        print(
            f"[RF] Model loaded ← {filepath}  "
            f"(feature level: {self.feature_level}, "
            f"classes: {self.train_stats.get('classes', '?')})",
            file=sys.stderr,
        )

    # ── convenience ───────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self.trained

    def get_train_stats(self) -> dict:
        return dict(self.train_stats)

    def __repr__(self) -> str:
        status = "trained" if self.trained else "untrained"
        return (
            f"RandomForestSegmentation4D("
            f"n_estimators={self.n_estimators}, "
            f"status={status})"
        )
