"""K-means at three scales: slice, volume and time series.

The time-series level must find a phase that exists in only one timepoint
and holds a fraction of a percent of it — which K-means alone cannot do, as
it spends its centres on the bulk of the voxels — without inventing phases
in series that have none.
"""

import os

import numpy as np
import pytest

from histograms.histogram_engine_4d import HistogramEngine4D
from utils.kmeans_levels import (
    cluster_series,
    cluster_slice,
    describe_presence,
    plot_timeline,
    run_series_clustering,
)


def _series(seed=0, broad=90.0, transient_at=3, transient_slab=2, drift=0.0,
            timepoints=6):
    """Broad majority phase, a steady minor phase, optionally a tiny phase
    present at one timepoint only."""
    rng = np.random.default_rng(seed)
    shape = (8, 40, 40)
    neutron = np.empty((timepoints,) + shape)
    xray = np.empty((timepoints,) + shape)
    for t in range(timepoints):
        neutron[t] = rng.normal(500 + drift * t, broad, shape)
        xray[t] = rng.normal(500, broad, shape)
    steady = np.zeros(shape, bool)
    steady[:, :6, :] = True
    neutron[:, steady] = rng.normal(150, 20, (timepoints, steady.sum()))
    xray[:, steady] = rng.normal(900, 20, (timepoints, steady.sum()))
    transient = np.zeros(shape, bool)
    if transient_at is not None:
        transient[:transient_slab, 30:34, 30:34] = True
        neutron[transient_at][transient] = rng.normal(900, 15, transient.sum())
        xray[transient_at][transient] = rng.normal(120, 15, transient.sum())
    engine = HistogramEngine4D(bins=128, use_gpu=False)
    engine.compute_global_histogram(neutron, xray)
    return neutron, xray, engine, transient


def _histograms(neutron, xray, engine):
    return [engine.compute_local_histogram(neutron[t], xray[t], t)
            for t in range(len(neutron))]


# ── slice ────────────────────────────────────────────────────────────────────

def test_slice_clustering_separates_phases_and_scales_channels():
    rng = np.random.default_rng(0)
    # X-ray values a thousand times larger than neutron: unscaled K-means
    # would split along X-ray only
    neutron = rng.normal(1.0, 0.05, (60, 60))
    xray = rng.normal(1000.0, 30.0, (60, 60))
    neutron[:, :20] = rng.normal(2.0, 0.05, (60, 20))
    result = cluster_slice(neutron, xray, 2)
    left, right = result.labels[:, :20], result.labels[:, 20:]
    assert len(np.unique(left)) == 1 and len(np.unique(right)) == 1
    assert left[0, 0] != right[0, 0]
    assert result.pixel_counts.sum() == neutron.size


def test_slice_clustering_skips_non_finite_pixels():
    neutron = np.tile(np.array([1.0, 5.0]), (10, 5))
    xray = neutron.copy()
    neutron[0, 0] = np.nan
    result = cluster_slice(neutron, xray, 2)
    assert result.labels[0, 0] == -1


def test_slice_needs_enough_distinct_values():
    with pytest.raises(ValueError):
        cluster_slice(np.ones((5, 5)), np.ones((5, 5)), 3)


# ── time series ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(4))
def test_series_finds_a_phase_present_in_one_timepoint(seed):
    neutron, xray, engine, transient = _series(seed)
    result = cluster_series(_histograms(neutron, xray, engine), 3)
    found = result.transient_clusters()
    assert len(found) == 1, "exactly one transient phase expected"
    phase = found[0]
    assert result.is_transient_phase(phase)
    assert result.present_at(phase) == [3]
    # Its voxels are labelled as that phase, at that timepoint only
    labels = result.label_volume(neutron[3], xray[3])
    assert np.mean(labels[transient] == phase) > 0.9
    assert not np.any(result.label_volume(neutron[0], xray[0]) == phase)


def test_kmeans_alone_misses_that_phase():
    """Why the transient stage exists: volume K-means at that very
    timepoint spends its centres on the broad phase instead."""
    from utils.clustering_3d import KMeans3D

    neutron, xray, _engine, _transient = _series(0)
    _labels, centers, _stats = KMeans3D.cluster_volume(neutron[3], xray[3], 3)
    near = np.hypot(centers[:, 0] - 900, centers[:, 1] - 120) < 60
    assert not near.any()


@pytest.mark.parametrize("broad,drift", [(25, 0), (90, 0), (120, 0), (60, 3)])
def test_series_invents_no_phase_when_there_is_none(broad, drift):
    neutron, xray, engine, _ = _series(1, broad=broad, transient_at=None,
                                       drift=drift, timepoints=8)
    result = cluster_series(_histograms(neutron, xray, engine), 3)
    assert result.n_clusters == 3
    assert result.transient_clusters() == []


def test_transient_search_can_be_switched_off():
    neutron, xray, engine, _ = _series(0)
    result = cluster_series(_histograms(neutron, xray, engine), 3,
                            find_transient=False)
    assert result.n_clusters == 3 and result.n_persistent == 3


def test_every_voxel_is_labelled_and_counts_match_the_labels():
    neutron, xray, engine, _ = _series(2)
    result = cluster_series(_histograms(neutron, xray, engine), 3)
    for t in (0, 3):
        labels = result.label_volume(neutron[t], xray[t])
        assert labels.min() >= 0
        counted = np.bincount(labels.ravel(), minlength=result.n_clusters)
        np.testing.assert_array_equal(counted, result.counts[t].astype(int))


def test_clusters_keep_their_identity_through_time():
    neutron, xray, engine, _ = _series(0)
    result = cluster_series(_histograms(neutron, xray, engine), 3)
    steady_voxel = (0, 0, 0)
    ids = {int(result.label_volume(neutron[t], xray[t])[steady_voxel])
           for t in range(len(neutron))}
    assert len(ids) == 1


def test_outline_encloses_the_cluster_centre():
    from matplotlib.path import Path

    neutron, xray, engine, _ = _series(0)
    result = cluster_series(_histograms(neutron, xray, engine), 3)
    for cluster in range(result.n_clusters):
        outline = result.outline(cluster)
        assert outline is not None
        assert Path(outline).contains_point(result.centers[cluster])


def test_run_on_a_dataset_and_write_the_timeline(tmp_path):
    from data import Dataset4D

    neutron, xray, engine, _ = _series(0)
    result, labels = run_series_clustering(Dataset4D(neutron, xray), engine, 3)
    assert sorted(labels) == list(range(6))
    csv_path = tmp_path / "timeline.csv"
    result.write_timeline_csv(csv_path)
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 1 + 6 * result.n_clusters
    assert "transient_phase" in lines[0]
    plot_timeline(result, tmp_path / "timeline.svg")
    assert (tmp_path / "timeline.svg").stat().st_size > 0


def test_histograms_on_different_grids_are_refused():
    neutron, xray, engine, _ = _series(0)
    first = engine.compute_local_histogram(neutron[0], xray[0], 0)
    other = HistogramEngine4D(bins=64, use_gpu=False)
    other.compute_global_histogram(neutron, xray)
    second = other.compute_local_histogram(neutron[1], xray[1], 1)
    with pytest.raises(ValueError):
        cluster_series([first, second], 2)


@pytest.mark.parametrize("present,expected", [
    ([0, 1, 2, 3], "all timepoints"),
    ([3], "T3"),
    ([2, 3, 4], "T2–T4"),
    ([0, 3, 4, 5], "T0, T3–T5"),
    ([], "no timepoint"),
])
def test_presence_is_described_as_ranges(present, expected):
    all_t = [0, 1, 2, 3] if expected == "all timepoints" else list(range(8))
    assert describe_presence(present, all_t) == expected


# ── in the application ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow
    from data import Dataset4D

    neutron, xray, engine, transient = _series(0, broad=25, transient_slab=3)
    w = BiTS4DMainWindow()
    w.dataset = Dataset4D(neutron, xray)
    w.histogram_engine = engine
    w.global_histogram = engine.get_global_histogram()
    w.dual_histogram.set_global_histogram(w.global_histogram)
    w._update_current_timepoint(3)
    w._transient_mask = transient
    return w


@pytest.fixture()
def quiet_dialogs(monkeypatch):
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a[2])))

    def _fail(*args, **_kwargs):
        raise AssertionError(" | ".join(str(a) for a in args[1:]))

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_fail))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_fail))
    return shown


def _run(window, scope, clusters=3):
    index = window.kmeans_scope_combo.findData(scope)
    window.kmeans_scope_combo.setCurrentIndex(index)
    window.kmeans_clusters_spin.setValue(clusters)
    window._run_kmeans()


def test_three_scopes_are_offered(window):
    scopes = [window.kmeans_scope_combo.itemData(i)
              for i in range(window.kmeans_scope_combo.count())]
    assert scopes == ["slice", "volume", "series"]


def test_slice_scope_saves_pinned_selections(window, quiet_dialogs):
    _run(window, "slice")
    names = [s.name for s in window.selection_manager.selections]
    assert names == ["Slice cluster 0", "Slice cluster 1", "Slice cluster 2"]
    selection = window.selection_manager.selections[0]
    assert selection.spatial_mask.ndim == 2
    assert selection.source_axis == window.slice_viewer.current_axis
    # Running again replaces them rather than adding duplicates
    _run(window, "slice")
    assert len(window.selection_manager.selections) == 3


def test_volume_scope_can_be_copied_to_materials(window, quiet_dialogs):
    _run(window, "volume")
    assert window.copy_clusters_btn.isEnabled()
    window._convert_kmeans_clusters_to_materials()
    names = {layer[2] for layer in window.segmentation_masks[3]}
    assert {"K-means cluster 0", "K-means cluster 1",
            "K-means cluster 2"} <= names


def test_series_scope_writes_layers_for_every_timepoint(window, quiet_dialogs):
    _run(window, "series")
    result = window.kmeans_series_result
    assert result is not None and result.transient_clusters()
    transient_name = result.cluster_name(result.transient_clusters()[0])
    for t in range(window.dataset.num_timepoints):
        names = [layer[2] for layer in window.segmentation_masks[t]]
        assert "Series cluster 0" in names
        # The transient phase is a layer only where it exists
        assert (transient_name in names) == (t == 3)
    layer = next(l for l in window.segmentation_masks[3]
                 if l[2] == transient_name)
    assert np.mean(layer[0][window._transient_mask]) > 0.9
    assert window.kmeans_timeline_btn.isEnabled()
    assert any("only in some timepoints" in text for text in quiet_dialogs)


def test_series_rerun_replaces_its_layers(window, quiet_dialogs):
    _run(window, "series")
    before = len(window.segmentation_masks[0])
    _run(window, "series")
    assert len(window.segmentation_masks[0]) == before


def test_auto_detect_button_opens_the_kmeans_settings(window):
    window.slice_viewer.auto_detect_btn.click()
    page = window.right_tabs.currentWidget()
    assert page.isAncestorOf(window.kmeans_run_btn)
