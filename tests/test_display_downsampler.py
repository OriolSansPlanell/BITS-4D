"""Tests for median-binned display volumes and display-grid masks."""

import numpy as np
import pytest

from utils.display_downsampler import DisplayDownsampler


def test_bin_factor_one_for_small_volumes():
    assert DisplayDownsampler.choose_bin_factor((10, 10, 10), 2, 1 << 30) == 1


def test_bin_factor_brings_display_volume_under_limit():
    shape = (100, 200, 200)  # 4M voxels * 4B float32 = 16 MB unbinned
    max_bytes = 3_000_000
    factor = DisplayDownsampler.choose_bin_factor(shape, 4, max_bytes)
    assert factor > 1
    binned_bytes = int(np.prod([s // factor for s in shape])) * 4
    assert binned_bytes <= max_bytes
    # Must be the smallest such factor
    if factor > 2:
        smaller = factor - 1
        assert int(np.prod([s // smaller for s in shape])) * 4 > max_bytes


def test_median_binning_matches_blockwise_median():
    rng = np.random.default_rng(0)
    volume = rng.uniform(0, 100, size=(6, 8, 10)).astype(np.float32)
    binned = DisplayDownsampler.bin_volume_median(volume, 2)
    assert binned.shape == (3, 4, 5)
    assert binned.dtype == np.float32
    reference = np.median(
        volume.reshape(3, 2, 4, 2, 5, 2), axis=(1, 3, 5)
    ).astype(np.float32)
    np.testing.assert_allclose(binned, reference)


def test_median_binning_crops_partial_blocks():
    volume = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)
    binned = DisplayDownsampler.bin_volume_median(volume, 2)
    assert binned.shape == (3, 4, 5)


def test_factor_one_returns_volume_unchanged():
    volume = np.zeros((4, 4, 4), dtype=np.uint16)
    assert DisplayDownsampler.bin_volume_median(volume, 1) is volume


def test_mask_binning_preserves_presence():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1, 1, 1] = True  # single voxel must survive binning
    binned = DisplayDownsampler.bin_mask(mask, 2)
    assert binned.shape == (3, 3, 3)
    assert binned[0, 0, 0]
    assert binned.sum() == 1


def test_mask_upscale_round_trip_covers_original():
    rng = np.random.default_rng(1)
    mask = rng.random((6, 8, 10)) > 0.7
    binned = DisplayDownsampler.bin_mask(mask, 2)
    upscaled = DisplayDownsampler.upscale_mask(binned, 2, mask.shape)
    assert upscaled.shape == mask.shape
    # Presence-preserving down + block-replicating up ⇒ superset of original
    assert np.all(upscaled[mask])


def test_mask_upscale_pads_cropped_edges():
    mask = np.ones((3, 4, 5), dtype=bool)
    upscaled = DisplayDownsampler.upscale_mask(mask, 2, (7, 9, 11))
    assert upscaled.shape == (7, 9, 11)
    assert upscaled.all()


def test_bin_dataset_processes_every_timepoint():
    rng = np.random.default_rng(2)
    neutron = rng.uniform(size=(3, 4, 6, 6))
    xray = rng.uniform(size=neutron.shape)
    messages = []
    n_binned, x_binned = DisplayDownsampler.bin_dataset(
        neutron, xray, 2,
        progress_callback=lambda v, m: messages.append((v, m)),
    )
    assert len(n_binned) == 3 and len(x_binned) == 3
    assert n_binned[0].shape == (2, 3, 3)
    assert messages[-1][0] == 100


def test_non_3d_input_rejected():
    with pytest.raises(ValueError):
        DisplayDownsampler.bin_volume_median(np.zeros((4, 4)), 2)
    with pytest.raises(ValueError):
        DisplayDownsampler.bin_mask(np.zeros((4, 4), dtype=bool), 2)
