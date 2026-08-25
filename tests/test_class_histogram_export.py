"""Per-class bimodal histograms, and class names in exported file names."""

import os

import numpy as np
import pytest

from histograms.histogram_engine_4d import HistogramEngine4D
from utils.histogram_export import (
    sanitize_name,
    save_bin_edges,
    save_class_histogram,
)


def _paired_volumes():
    """Two materials with disjoint, asymmetric signatures."""
    neutron = np.full((1, 4, 16, 16), 500.0)
    xray = np.full((1, 4, 16, 16), 500.0)
    blob_a = np.zeros(neutron.shape, dtype=bool)
    blob_a[:, :, :6, :6] = True
    blob_b = np.zeros(neutron.shape, dtype=bool)
    blob_b[:, :, 10:, 10:] = True
    neutron[blob_a] = 100.0
    xray[blob_a] = 900.0
    neutron[blob_b] = 900.0
    xray[blob_b] = 100.0
    return neutron, xray, blob_a[0], blob_b[0]


# ── masked histogram on the shared grid ──────────────────────────────────────

def test_class_histogram_shares_the_global_bin_grid():
    neutron, xray, blob_a, _ = _paired_volumes()
    engine = HistogramEngine4D(bins=32, use_gpu=False)
    global_hist = engine.compute_global_histogram(neutron, xray)

    class_hist = engine.compute_masked_histogram(neutron[0], xray[0], blob_a)

    np.testing.assert_array_equal(class_hist.x_edges, global_hist.x_edges)
    np.testing.assert_array_equal(class_hist.y_edges, global_hist.y_edges)
    assert class_hist.data_range == global_hist.data_range
    assert class_hist.histogram.shape == global_hist.histogram.shape


def test_class_histogram_counts_only_the_masked_voxels():
    neutron, xray, blob_a, blob_b = _paired_volumes()
    engine = HistogramEngine4D(bins=32, use_gpu=False)
    engine.compute_global_histogram(neutron, xray)

    hist_a = engine.compute_masked_histogram(neutron[0], xray[0], blob_a)
    hist_b = engine.compute_masked_histogram(neutron[0], xray[0], blob_b)

    assert int(hist_a.histogram.sum()) == int(blob_a.sum())
    assert int(hist_b.histogram.sum()) == int(blob_b.sum())
    # Disjoint materials occupy disjoint bins on the shared grid
    assert not np.any((hist_a.histogram > 0) & (hist_b.histogram > 0))


def test_class_histogram_is_positioned_where_the_material_lives():
    neutron, xray, blob_a, _ = _paired_volumes()
    engine = HistogramEngine4D(bins=32, use_gpu=False)
    engine.compute_global_histogram(neutron, xray)
    hist = engine.compute_masked_histogram(neutron[0], xray[0], blob_a)

    row, col = np.unravel_index(np.argmax(hist.histogram), hist.histogram.shape)
    assert abs(hist.x_centers[col] - 100) < 40    # neutron on the x axis
    assert abs(hist.y_centers[row] - 900) < 40    # X-ray on the y axis


def test_masked_histogram_matches_chunked_and_unchunked():
    neutron, xray, blob_a, _ = _paired_volumes()
    reference = HistogramEngine4D(bins=32, use_gpu=False)
    reference.compute_global_histogram(neutron, xray)
    chunked = HistogramEngine4D(bins=32, use_gpu=False, chunk_voxels=7)
    chunked.compute_global_histogram(neutron, xray)

    np.testing.assert_array_equal(
        reference.compute_masked_histogram(neutron[0], xray[0], blob_a).histogram,
        chunked.compute_masked_histogram(neutron[0], xray[0], blob_a).histogram,
    )


def test_masked_histogram_requires_a_global_histogram():
    neutron, xray, blob_a, _ = _paired_volumes()
    engine = HistogramEngine4D(bins=16, use_gpu=False)
    with pytest.raises(RuntimeError):
        engine.compute_masked_histogram(neutron[0], xray[0], blob_a)


def test_masked_histogram_rejects_a_mismatched_mask():
    neutron, xray, _, _ = _paired_volumes()
    engine = HistogramEngine4D(bins=16, use_gpu=False)
    engine.compute_global_histogram(neutron, xray)
    with pytest.raises(ValueError):
        engine.compute_masked_histogram(
            neutron[0], xray[0], np.ones((2, 2, 2), dtype=bool)
        )


# ── file output ──────────────────────────────────────────────────────────────

def test_saving_writes_counts_and_image(tmp_path):
    neutron, xray, blob_a, _ = _paired_volumes()
    engine = HistogramEngine4D(bins=32, use_gpu=False)
    engine.compute_global_histogram(neutron, xray)
    hist = engine.compute_masked_histogram(neutron[0], xray[0], blob_a)

    written = save_class_histogram(hist, tmp_path / "t000_Lithium_hist",
                                   title="Lithium")
    assert sorted(written) == ["t000_Lithium_hist.npy", "t000_Lithium_hist.png"]

    counts = np.load(tmp_path / "t000_Lithium_hist.npy")
    np.testing.assert_array_equal(counts, hist.histogram)
    assert (tmp_path / "t000_Lithium_hist.png").stat().st_size > 1000


def test_bin_edges_are_written_once_and_describe_the_grid(tmp_path):
    neutron, xray, _, _ = _paired_volumes()
    engine = HistogramEngine4D(bins=32, use_gpu=False)
    global_hist = engine.compute_global_histogram(neutron, xray)

    written = save_bin_edges(global_hist, tmp_path)
    assert "histogram_edges_neutron.npy" in written
    assert "histogram_edges_xray.npy" in written

    np.testing.assert_array_equal(
        np.load(tmp_path / "histogram_edges_neutron.npy"), global_hist.x_edges
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "histogram_edges_xray.npy"), global_hist.y_edges
    )
    readme = (tmp_path / "histogram_edges_README.txt").read_text()
    assert "neutron" in readme and "X-ray" in readme


# ── file naming ──────────────────────────────────────────────────────────────

def test_sanitize_keeps_the_class_name_readable():
    assert sanitize_name("Lithium") == "Lithium"
    assert sanitize_name("Solid Electrolyte") == "Solid_Electrolyte"
    assert sanitize_name("Li/graphite") == "Li_graphite"
    assert sanitize_name("  spaced  ") == "spaced"
    assert sanitize_name("") == "class"
    assert sanitize_name("RF: Lithium") == "RF_Lithium"


@pytest.mark.parametrize("name", ["Lithium", "Solid Electrolyte", "Li/graphite"])
def test_named_classes_appear_in_file_names(tmp_path, name):
    neutron, xray, blob_a, _ = _paired_volumes()
    engine = HistogramEngine4D(bins=16, use_gpu=False)
    engine.compute_global_histogram(neutron, xray)
    hist = engine.compute_masked_histogram(neutron[0], xray[0], blob_a)

    prefix = tmp_path / f"timepoint_000_{sanitize_name(name)}_hist"
    written = save_class_histogram(hist, prefix, title=name)
    stem = sanitize_name(name)
    assert all(stem in filename for filename in written)
    assert "Class" not in " ".join(written)
