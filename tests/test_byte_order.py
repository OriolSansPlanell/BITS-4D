"""Regression tests for big-endian TIFF datasets.

Big-endian TIFFs yield NumPy arrays whose byte order differs from the
machine's native order; PyTorch refuses to build tensors from them
("given numpy array has byte order different from the native byte order").
The loader must normalize in-RAM data, and the GPU histogram path must
convert chunks defensively for read-only memory maps.
"""

import numpy as np
import tifffile

from data.data_loader_4d import TIFF4DLoader
from histograms.histogram_engine_4d import HistogramEngine4D


def _write_big_endian_pair(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 60000, size=(2, 3, 8, 8), dtype=np.uint16)
    neutron_path = tmp_path / "neutron_be.tif"
    xray_path = tmp_path / "xray_be.tif"
    tifffile.imwrite(neutron_path, data, byteorder=">",
                     photometric="minisblack")
    tifffile.imwrite(xray_path, data + 2, byteorder=">",
                     photometric="minisblack")
    return neutron_path, xray_path, data


def test_loader_normalizes_big_endian_to_native(tmp_path):
    neutron_path, xray_path, original = _write_big_endian_pair(tmp_path)
    dataset = TIFF4DLoader.load(neutron_path, xray_path, use_memmap=False)
    assert dataset.neutron_data.dtype.isnative
    assert dataset.xray_data.dtype.isnative
    # Byte swapping must not change any values
    np.testing.assert_array_equal(dataset.neutron_data, original)
    np.testing.assert_array_equal(dataset.xray_data, original + 2)


def test_native_arrays_pass_through_unchanged():
    array = np.arange(8, dtype=np.uint16)
    assert TIFF4DLoader._to_native_byte_order(array) is array


def test_read_only_arrays_are_left_for_chunkwise_conversion():
    array = np.arange(8, dtype=np.dtype(">u2"))
    array.setflags(write=False)
    result = TIFF4DLoader._to_native_byte_order(array)
    assert result is array  # unchanged; consumers convert chunk-wise


def test_histogram_engine_handles_big_endian_input():
    """Even unnormalized big-endian arrays must histogram correctly (the
    situation for read-only big-endian memory maps)."""
    rng = np.random.default_rng(1)
    native = rng.uniform(0, 1000, size=(2, 3, 8, 8))
    big_endian = native.astype(np.dtype(">f8"))
    assert not big_endian.dtype.isnative

    engine_be = HistogramEngine4D(bins=16, use_gpu=False)
    result_be = engine_be.compute_global_histogram(big_endian, big_endian * 2)
    engine_native = HistogramEngine4D(bins=16, use_gpu=False)
    result_native = engine_native.compute_global_histogram(native, native * 2)
    np.testing.assert_array_equal(result_be.histogram, result_native.histogram)


def test_gpu_chunk_conversion_produces_native_arrays():
    """The exact conversion used before torch.as_tensor must yield native
    float64 regardless of the chunk's byte order."""
    chunk = np.arange(32, dtype=np.dtype(">f8"))
    finite = np.isfinite(chunk)
    converted = np.asarray(chunk[finite], dtype=np.float64)
    assert converted.dtype.isnative
    np.testing.assert_array_equal(converted, np.arange(32, dtype=np.float64))
