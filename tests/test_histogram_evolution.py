"""Tests for the temporal histogram-evolution export."""

import numpy as np
import pytest

from histograms.histogram_engine_4d import HistogramEngine4D
from utils.histogram_evolution import (
    REFERENCE_FIRST,
    REFERENCE_PREVIOUS,
    compute_log_differences,
    compute_marginals,
    save_histogram_evolution_image,
    save_marginal_evolution_image,
)


def _histograms_for(neutron_4d, xray_4d, bins=32):
    engine = HistogramEngine4D(bins=bins, use_gpu=False)
    engine.compute_global_histogram(neutron_4d, xray_4d)
    return [
        engine.compute_local_histogram(neutron_4d[t], xray_4d[t], t)
        for t in range(neutron_4d.shape[0])
    ]


def test_log_differences_are_zero_for_identical_timepoints():
    rng = np.random.default_rng(0)
    volume = rng.uniform(0, 100, size=(1, 4, 8, 8))
    neutron = np.concatenate([volume, volume])
    xray = np.concatenate([volume * 2, volume * 2])
    diffs = compute_log_differences(_histograms_for(neutron, xray))
    assert len(diffs) == 1
    np.testing.assert_allclose(diffs[0], 0.0)


def test_log_differences_show_signed_population_change():
    # T0 has material at (n~10, x~10); T1 moves it to (n~90, x~90)
    neutron = np.full((2, 2, 8, 8), 10.0)
    xray = np.full((2, 2, 8, 8), 10.0)
    neutron[1] = 90.0
    xray[1] = 90.0
    hists = _histograms_for(neutron, xray)
    diff = compute_log_differences(hists)[0]
    assert diff.min() < 0  # voxels vanished from the T0 bin
    assert diff.max() > 0  # and appeared in the T1 bin


def test_single_timepoint_rejected():
    volume = np.zeros((1, 2, 4, 4))
    hists = _histograms_for(volume + 1.0, volume + 2.0)
    with pytest.raises(ValueError):
        compute_log_differences(hists)


def test_incremental_differences_compare_against_previous_timepoint():
    """A value that only moves between T1 and T2 must show up in the T2
    increment and nowhere else, while the cumulative view keeps showing it."""
    neutron = np.full((3, 2, 8, 8), 10.0)
    xray = np.full((3, 2, 8, 8), 10.0)
    neutron[2] = 90.0          # nothing changes T0->T1; a jump at T2
    xray[2] = 90.0

    hists = _histograms_for(neutron, xray)
    incremental = compute_log_differences(hists, REFERENCE_PREVIOUS)
    assert len(incremental) == 2
    np.testing.assert_allclose(incremental[0], 0.0)   # T1 == T0
    assert np.abs(incremental[1]).max() > 0           # T2 moved

    cumulative = compute_log_differences(hists, REFERENCE_FIRST)
    np.testing.assert_allclose(cumulative[0], 0.0)
    assert np.abs(cumulative[1]).max() > 0


def test_steady_drift_looks_smaller_incrementally_than_cumulatively():
    neutron = np.stack([np.full((2, 8, 8), v) for v in (10.0, 40.0, 70.0)])
    xray = neutron.copy()
    hists = _histograms_for(neutron, xray)

    cumulative = compute_log_differences(hists, REFERENCE_FIRST)
    incremental = compute_log_differences(hists, REFERENCE_PREVIOUS)
    # The last cumulative panel spans two steps of drift, the incremental one
    # only the final step, so the cumulative view touches more bins.
    assert np.count_nonzero(cumulative[-1]) >= np.count_nonzero(incremental[-1])


def test_invalid_reference_mode_rejected():
    volume = np.zeros((2, 2, 4, 4))
    hists = _histograms_for(volume + 1.0, volume + 2.0)
    with pytest.raises(ValueError):
        compute_log_differences(hists, "sideways")


def test_marginals_use_the_correct_axis_for_each_modality():
    """HistogramData.histogram is [xray_bin, neutron_bin]; a change in only
    one modality must appear in that modality's marginal alone."""
    # Neutron moves 10 -> 90, X-ray stays at 50 throughout.
    neutron = np.stack([np.full((2, 8, 8), 10.0), np.full((2, 8, 8), 90.0)])
    xray = np.full((2, 2, 8, 8), 50.0)

    hists = _histograms_for(neutron, xray)
    neutron_marginals, xray_marginals = compute_marginals(hists)

    assert neutron_marginals.shape == (2, hists[0].histogram.shape[1])
    assert xray_marginals.shape == (2, hists[0].histogram.shape[0])

    # Each marginal is a normalized distribution
    np.testing.assert_allclose(neutron_marginals.sum(axis=1), 1.0)
    np.testing.assert_allclose(xray_marginals.sum(axis=1), 1.0)

    # The neutron marginal's occupied bin moves; the X-ray one does not.
    neutron_peaks = [int(np.argmax(m)) for m in neutron_marginals]
    xray_peaks = [int(np.argmax(m)) for m in xray_marginals]
    assert neutron_peaks[0] != neutron_peaks[1], "neutron change not detected"
    assert xray_peaks[0] == xray_peaks[1], "X-ray marginal changed unexpectedly"


def test_marginals_normalize_away_differing_voxel_counts():
    """Timepoints with different finite-voxel counts stay comparable."""
    neutron = np.full((2, 2, 8, 8), 20.0)
    xray = np.full((2, 2, 8, 8), 20.0)
    neutron[1, 0, 0, 0] = np.nan   # one fewer finite pair at T1
    xray[1, 0, 0, 0] = np.nan

    neutron_marginals, xray_marginals = compute_marginals(
        _histograms_for(neutron, xray)
    )
    np.testing.assert_allclose(neutron_marginals[0], neutron_marginals[1])
    np.testing.assert_allclose(xray_marginals[0], xray_marginals[1])


def test_marginal_image_file_is_written(tmp_path):
    rng = np.random.default_rng(3)
    neutron = rng.uniform(0, 100, size=(4, 2, 8, 8))
    xray = rng.uniform(0, 100, size=neutron.shape)
    output = tmp_path / "marginals.png"
    saved = save_marginal_evolution_image(_histograms_for(neutron, xray),
                                          str(output))
    assert saved == str(output)
    assert output.stat().st_size > 1000


def test_incremental_image_file_is_written(tmp_path):
    rng = np.random.default_rng(4)
    neutron = rng.uniform(0, 100, size=(3, 2, 8, 8))
    xray = rng.uniform(0, 100, size=neutron.shape)
    output = tmp_path / "incremental.png"
    saved = save_histogram_evolution_image(
        _histograms_for(neutron, xray), str(output),
        reference_mode=REFERENCE_PREVIOUS,
    )
    assert saved == str(output)
    assert output.stat().st_size > 1000


def test_image_file_is_written(tmp_path):
    rng = np.random.default_rng(1)
    neutron = rng.uniform(0, 100, size=(3, 2, 8, 8))
    xray = rng.uniform(0, 100, size=neutron.shape)
    hists = _histograms_for(neutron, xray)
    output = tmp_path / "evolution.png"
    saved = save_histogram_evolution_image(hists, str(output))
    assert saved == str(output)
    assert output.exists()
    assert output.stat().st_size > 1000
