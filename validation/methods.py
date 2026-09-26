"""BiTS 4D and the methods it is compared against, on a phantom.

Every method returns a ``(T, Z, Y, X)`` int array whose value k+1 means
``phantom.classes[k]`` and 0 means unassigned.

Supervision is the same for every supervised method: the class
**definitions** returned by :func:`definitions` — each class's truth, eroded
by one voxel (a region clear of partial volume, as a careful user would
draw), at the first timepoint where that leaves anything. A phase that
appears during the experiment is therefore defined at the timepoint it
appears, for every method alike.

Unsupervised methods (Otsu, K-means, GMM) get their clusters named by an
oracle (:func:`validation.scoring.oracle_match`) — the best naming possible,
which in practice must be done by hand. Their scores are an upper bound.

A U-Net is not included: training one needs labelled volumes beyond the
ROIs every other method gets, plus a deep-learning stack this project does
not depend on. The random-forest pixel classifier on ilastik's default
feature set is the standard interactive-learning baseline in its place.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from validation.phantom import Phantom, PhantomDataset
from validation.scoring import oracle_match


# ── shared set-up ────────────────────────────────────────────────────────────

def definitions(phantom: Phantom, reference: int = 0,
                birth: bool = True) -> Dict[str, Tuple[int, np.ndarray]]:
    """``{class: (timepoint, roi_mask)}`` as described in the module doc.

    With ``birth=False`` every class must be defined at *reference*; a class
    whose ROI is empty there keeps its empty ROI, so the library reports it
    as undefinable rather than it being left out unseen.
    """
    result = {}
    order = [reference] + ([t for t in range(phantom.num_timepoints)
                            if t != reference] if birth else [])
    for name in phantom.classes:
        for timepoint in order:
            mask = phantom.roi_masks(timepoint)[name]
            if mask.any():
                result[name] = (timepoint, mask)
                break
        else:
            result[name] = (reference, phantom.roi_masks(reference)[name])
    return result


def histogram_edges(phantom: Phantom, bins: int = 128):
    neutron = phantom.neutron[np.isfinite(phantom.neutron)]
    xray = phantom.xray[np.isfinite(phantom.xray)]
    return (np.linspace(neutron.min(), neutron.max(), bins + 1),
            np.linspace(xray.min(), xray.max(), bins + 1))


def _to_class_order(library_labels, library_names, classes):
    """Library label k+1 → phantom label of the same class name."""
    lookup = np.zeros(len(library_names) + 1, dtype=np.int16)
    for index, name in enumerate(library_names, start=1):
        lookup[index] = classes.index(name) + 1
    return lookup[library_labels]


# ── BiTS 4D ──────────────────────────────────────────────────────────────────

def attenuation_coefficients(phantom: Phantom) -> Dict[str, Tuple[float, float]]:
    """Stand-in 'tabulated' coefficients for the phantom's classes.

    The phantom's grey values are an unknown linear map of these (a
    different gain and offset per modality), as a scanner's are of the real
    coefficients — which is exactly what the calibration has to undo.
    """
    from validation.phantom import SIGNAL
    return {name: ((SIGNAL[name][0] - 150.0) / 800.0,
                   (SIGNAL[name][1] - 50.0) / 1200.0)
            for name in phantom.classes}


def run_bits(phantom: Phantom, smoothing="auto", max_components: int = 1,
             birth: bool = True, bins: int = 128, n_sweeps: int = 6,
             predict: Sequence[str] = (), references=("Air", "Aluminium"),
             coefficient_error: float = 0.0,
             return_details: bool = False):
    """Locked mode, set up exactly as the application sets it up.

    Classes in *predict* get no region: they are placed from their
    attenuation coefficients, calibrated on *references* (see
    :mod:`model.calibration`); *coefficient_error* scales their
    coefficients by ``1 + error`` to see what an inaccurate table costs.
    """
    import warnings
    from model import ClassLibrary, LockedSegmenter
    from model.spatial_prior import ROIDerivedMRF
    from model.validity import build_valid_mask

    defs = definitions(phantom, birth=birth)
    defs = {name: entry for name, entry in defs.items() if name not in predict}
    sources = {}
    for name, (t, mask) in defs.items():
        n_t, x_t = phantom.neutron[t], phantom.xray[t]
        sources[name] = (n_t, x_t, mask, build_valid_mask(n_t, x_t), t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        library = ClassLibrary.from_sources(sources,
                                            max_components=max_components)
    if predict:
        from model.calibration import (
            fit_calibration, merge_libraries, predicted_classes,
            reference_from_region,
        )
        table = attenuation_coefficients(phantom)
        calibration = fit_calibration([
            reference_from_region(name, table[name], *sources[name][:2],
                                  sources[name][2] & sources[name][3])
            for name in references])
        targets = {name: tuple(v * (1.0 + coefficient_error)
                               for v in table[name]) for name in predict}
        library = merge_libraries(library,
                                  predicted_classes(targets, calibration))

    segmenter = LockedSegmenter(library, prior=ROIDerivedMRF(beta=1.0,
                                                             n_sweeps=n_sweeps),
                                bins=bins)
    segmenter.set_grid(*histogram_edges(phantom, bins))
    neutron0, xray0 = phantom.neutron[0], phantom.xray[0]
    valid0 = build_valid_mask(neutron0, xray0)
    raw = segmenter.segment_timepoint(neutron0, xray0, beta=0.0)
    segmenter.learn_boundaries(np.maximum(raw.labels - 1, 0), valid_mask=valid0)

    sweep = None
    if smoothing == "auto":
        from model.locked import smoothing_check_timepoints
        checked = smoothing_check_timepoints(
            phantom.num_timepoints, 0, [t for t, _m in defs.values()])
        strength, sweep = segmenter.auto_smoothing(
            neutron0, xray0, extra_volumes=[
                (phantom.neutron[t], phantom.xray[t], t) for t in checked[1:]
            ])
    else:
        strength = float(smoothing)
    outcome = segmenter.segment_series(PhantomDataset(phantom), beta=strength,
                                       enforce_guards=False)
    outcome.smoothing_sweep = sweep
    outcome.smoothing_report = segmenter.last_smoothing_report
    labels = np.stack([
        _to_class_order(entry.labels, library.names, phantom.classes)
        for entry in outcome.timepoints
    ])
    if return_details:
        return labels, dict(library=library, outcome=outcome, sweep=sweep,
                            smoothing=strength, segmenter=segmenter)
    return labels


def run_bits_adaptive(phantom: Phantom, controls=("Air", "Aluminium"),
                      track_drift: bool = True, bins: int = 128):
    """The advanced mode: definitions may move, anchored on *controls*.

    ``track_drift=False`` with a static transition is the frozen-boundary
    control the design argument compares against.
    """
    from model import (DriftTracker, ROIAnchoredMixture, ROIDerivedMRF,
                       SequentialSegmenter)
    from model.temporal import DriftTransition, StaticTransition

    defs = definitions(phantom, birth=False)
    masks = {name: mask for name, (_t, mask) in defs.items() if mask.any()}
    segmenter = SequentialSegmenter(
        mixture=ROIAnchoredMixture(outlier_component=True),
        mrf=ROIDerivedMRF(beta=1.0, n_sweeps=3),
        temporal=DriftTransition(memory=0.5) if track_drift else StaticTransition(),
        drift_tracker=DriftTracker(anchor_classes=controls) if track_drift else None,
        bins=bins,
    )
    segmenter.prepare(phantom.neutron[0], phantom.xray[0], masks,
                      *histogram_edges(phantom, bins), anchor_strength=0.5)
    outcome = segmenter.run(PhantomDataset(phantom))
    names = list(outcome.class_names)
    labels = []
    for entry in outcome.timepoints:
        per_voxel = np.zeros(entry.labels.shape, dtype=np.int16)
        for name in names:
            if name in phantom.classes:
                per_voxel[entry.mask_for(name)] = phantom.classes.index(name) + 1
        labels.append(per_voxel)
    return np.stack(labels)


# ── baselines ────────────────────────────────────────────────────────────────

def run_otsu(phantom: Phantom, channel: str = "neutron", classes: int = 2):
    """Global (2 classes) or multi-level Otsu on one channel, per timepoint."""
    from skimage.filters import threshold_multiotsu, threshold_otsu

    data = phantom.neutron if channel == "neutron" else phantom.xray
    result = []
    for t in range(phantom.num_timepoints):
        volume = data[t]
        if classes == 2:
            regions = (volume > threshold_otsu(volume)).astype(np.int16)
        else:
            regions = np.digitize(volume, threshold_multiotsu(volume, classes))
        result.append(oracle_match(regions, phantom.labels[t],
                                   len(phantom.classes)))
    return np.stack(result)


def _features_2d(neutron, xray):
    return np.column_stack((np.ravel(neutron), np.ravel(xray))).astype(np.float64)


def run_kmeans(phantom: Phantom, seed: int = 0):
    """Per-timepoint K-means (standardised channels), oracle-named."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    k = len(phantom.classes)
    result = []
    for t in range(phantom.num_timepoints):
        features = StandardScaler().fit_transform(
            _features_2d(phantom.neutron[t], phantom.xray[t]))
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(features)
        result.append(oracle_match(labels.reshape(phantom.labels[t].shape),
                                   phantom.labels[t], k))
    return np.stack(result)


def run_gmm(phantom: Phantom, seed: int = 0):
    """Per-timepoint Gaussian mixture (full covariance), oracle-named."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    k = len(phantom.classes)
    result = []
    for t in range(phantom.num_timepoints):
        features = StandardScaler().fit_transform(
            _features_2d(phantom.neutron[t], phantom.xray[t]))
        model = GaussianMixture(k, covariance_type="full", n_init=3,
                                random_state=seed).fit(features)
        labels = model.predict(features)
        result.append(oracle_match(labels.reshape(phantom.labels[t].shape),
                                   phantom.labels[t], k))
    return np.stack(result)


def _seeds_at(phantom: Phantom, defs, timepoint: int) -> np.ndarray:
    """ROI seeds usable at *timepoint*: every class defined at or before it."""
    seeds = np.zeros(phantom.labels.shape[1:], dtype=np.int32)
    for name, (defined_at, mask) in defs.items():
        if defined_at <= timepoint:
            seeds[mask] = phantom.classes.index(name) + 1
    return seeds


def run_random_walker(phantom: Phantom, beta: float = 130.0):
    """Random walker on both channels, seeded with the class ROIs.

    The seeds are the same definitions every supervised method gets, placed
    where they were drawn; a class is seeded from the timepoint it was
    defined at onwards.
    """
    from skimage.segmentation import random_walker

    defs = definitions(phantom)
    result = []
    for t in range(phantom.num_timepoints):
        image = np.stack([phantom.neutron[t], phantom.xray[t]], axis=-1)
        image = (image - image.mean(axis=(0, 1, 2))) / image.std(axis=(0, 1, 2))
        seeds = _seeds_at(phantom, defs, t)
        labels = random_walker(image, seeds, beta=beta, mode="cg_j",
                               channel_axis=-1)
        result.append(labels.astype(np.int16))
    return np.stack(result)


def _pixel_features(neutron, xray):
    """ilastik's default-style filter bank on both channels."""
    features = []
    for volume in (neutron, xray):
        volume = np.asarray(volume, dtype=np.float32)
        features.append(volume)
        for sigma in (0.7, 1.6, 3.5):
            features.append(ndimage.gaussian_filter(volume, sigma))
        features.append(ndimage.gaussian_gradient_magnitude(volume, 1.0))
        features.append(ndimage.gaussian_laplace(volume, 1.6))
    return np.stack([f.ravel() for f in features], axis=1)


def run_pixel_classifier(phantom: Phantom, seed: int = 0,
                         max_samples_per_class: int = 4000):
    """Random forest on filter-bank features, trained on the ROIs.

    Trained once on every class definition (each at its own timepoint),
    then applied to every timepoint — the usual ilastik workflow.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    defs = definitions(phantom)
    features_by_t = {}
    samples, targets = [], []
    for name, (t, mask) in defs.items():
        if t not in features_by_t:
            features_by_t[t] = _pixel_features(phantom.neutron[t], phantom.xray[t])
        index = np.flatnonzero(mask.ravel())
        if len(index) > max_samples_per_class:
            index = rng.choice(index, max_samples_per_class, replace=False)
        samples.append(features_by_t[t][index])
        targets.append(np.full(len(index), phantom.classes.index(name) + 1))
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                   random_state=seed)
    model.fit(np.concatenate(samples), np.concatenate(targets))
    result = []
    for t in range(phantom.num_timepoints):
        features = features_by_t.get(t)
        if features is None:
            features = _pixel_features(phantom.neutron[t], phantom.xray[t])
        result.append(model.predict(features).reshape(
            phantom.labels[t].shape).astype(np.int16))
    return np.stack(result)


BASELINES = {
    "Otsu (neutron, 2 classes)": lambda p: run_otsu(p, "neutron", 2),
    "Multi-Otsu (neutron)": lambda p: run_otsu(p, "neutron", len(p.classes)),
    "Multi-Otsu (X-ray)": lambda p: run_otsu(p, "xray", len(p.classes)),
    "K-means per timepoint*": run_kmeans,
    "GMM per timepoint*": run_gmm,
    "Random walker": run_random_walker,
    "Pixel classifier (RF, ilastik features)": run_pixel_classifier,
}
