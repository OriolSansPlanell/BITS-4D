"""Tests for the temporal histogram-evolution export."""

import numpy as np
import pytest

from histograms.histogram_engine_4d import HistogramEngine4D
from utils.histogram_evolution import (
    compute_log_differences,
    save_histogram_evolution_image,
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
