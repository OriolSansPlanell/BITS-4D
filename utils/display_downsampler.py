"""Median binning of large volumes for fast, memory-bounded visualization.

Segmentation always runs on the original full-resolution volumes; these
helpers produce lighter *display* copies (and matching mask copies) so the
slice viewer stays responsive on datasets far larger than the screen can
meaningfully show.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

Progress = Optional[Callable[[int, str], None]]
CancelCheck = Optional[Callable[[], None]]

# Display volumes are produced as float32 (4 bytes per voxel)
_DISPLAY_ITEMSIZE = 4


class DisplayDownsampler:
    """Block-median binning of 3-D volumes and matching boolean masks."""

    @staticmethod
    def choose_bin_factor(
        shape: Tuple[int, int, int],
        itemsize: int,
        max_bytes: int,
    ) -> int:
        """Smallest integer factor so one display volume fits in max_bytes.

        Factor 1 means the original volume is small enough to display
        directly. For factors >= 2 the binned volume is float32, so its
        size is estimated with 4 bytes per voxel.
        """
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        voxels = int(np.prod(shape, dtype=np.int64))
        if voxels * int(itemsize) <= max_bytes:
            return 1
        factor = 2
        while True:
            binned_voxels = int(
                np.prod([max(s // factor, 1) for s in shape], dtype=np.int64)
            )
            if binned_voxels * _DISPLAY_ITEMSIZE <= max_bytes:
                return factor
            factor += 1

    @staticmethod
    def _effective_factors(shape, factor):
        """Per-axis factors, clamped so tiny axes keep at least one block."""
        return tuple(min(int(factor), int(s)) for s in shape)

    @staticmethod
    def bin_volume_median(volume: np.ndarray, factor: int) -> np.ndarray:
        """Bin a 3-D volume by block median.

        Processes one output Z-slab at a time so peak memory stays at a
        single slab of float32 blocks even for memory-mapped inputs.
        Trailing voxels that do not fill a complete block are cropped.
        Returns the input unchanged for factor <= 1.
        """
        vol = np.asarray(volume)
        if vol.ndim != 3:
            raise ValueError(f"Expected a 3-D volume, got {vol.ndim}D")
        if factor <= 1:
            return vol

        fz, fy, fx = DisplayDownsampler._effective_factors(vol.shape, factor)
        out_z = vol.shape[0] // fz
        out_y = vol.shape[1] // fy
        out_x = vol.shape[2] // fx
        out = np.empty((out_z, out_y, out_x), dtype=np.float32)
        for zi in range(out_z):
            slab = np.asarray(
                vol[zi * fz:(zi + 1) * fz, :out_y * fy, :out_x * fx],
                dtype=np.float32,
            )
            blocks = slab.reshape(fz, out_y, fy, out_x, fx)
            out[zi] = np.median(blocks, axis=(0, 2, 4))
        return out

    @staticmethod
    def bin_mask(mask: np.ndarray, factor: int) -> np.ndarray:
        """Bin a boolean 3-D mask to the display grid.

        A display voxel is set when *any* original voxel in its block is
        set, so thin segmented structures stay visible after binning.
        """
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 3:
            raise ValueError(f"Expected a 3-D mask, got {mask_array.ndim}D")
        if factor <= 1:
            return mask_array

        fz, fy, fx = DisplayDownsampler._effective_factors(
            mask_array.shape, factor
        )
        out_z = mask_array.shape[0] // fz
        out_y = mask_array.shape[1] // fy
        out_x = mask_array.shape[2] // fx
        cropped = mask_array[:out_z * fz, :out_y * fy, :out_x * fx]
        return cropped.reshape(out_z, fz, out_y, fy, out_x, fx).any(
            axis=(1, 3, 5)
        )

    @staticmethod
    def upscale_mask(
        mask: np.ndarray,
        factor: int,
        target_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Expand a display-grid boolean mask back to the full-resolution grid.

        Each display voxel is replicated over its original block; voxels
        cropped away during binning are filled by replicating the nearest
        block (edge padding). This is the inverse used when masks produced
        in display space (e.g. from 3-D clustering of the binned volumes)
        must drive computations on the original volumes.
        """
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 3 or len(target_shape) != 3:
            raise ValueError("upscale_mask expects 3-D mask and target shape")
        if factor <= 1:
            if mask_array.shape != tuple(target_shape):
                raise ValueError(
                    f"Mask shape {mask_array.shape} does not match target "
                    f"{tuple(target_shape)} at factor 1"
                )
            return mask_array

        fz, fy, fx = DisplayDownsampler._effective_factors(
            target_shape, factor
        )
        up = mask_array.repeat(fz, axis=0).repeat(fy, axis=1).repeat(fx, axis=2)
        up = up[:target_shape[0], :target_shape[1], :target_shape[2]]
        pad = [(0, t - s) for s, t in zip(up.shape, target_shape)]
        if any(p[1] for p in pad):
            up = np.pad(up, pad, mode="edge")
        return up

    @staticmethod
    def bin_dataset(
        neutron_4d,
        xray_4d,
        factor: int,
        progress_callback: Progress = None,
        cancel_check: CancelCheck = None,
    ) -> Tuple[list, list]:
        """Bin every timepoint of a paired 4-D dataset for display.

        Returns two lists of 3-D float32 volumes (one entry per timepoint).
        """
        num_timepoints = int(np.asarray(neutron_4d).shape[0])
        neutron_binned = []
        xray_binned = []
        for timepoint in range(num_timepoints):
            if cancel_check:
                cancel_check()
            neutron_binned.append(
                DisplayDownsampler.bin_volume_median(
                    neutron_4d[timepoint], factor
                )
            )
            xray_binned.append(
                DisplayDownsampler.bin_volume_median(xray_4d[timepoint], factor)
            )
            if progress_callback:
                progress_callback(
                    int(100 * (timepoint + 1) / num_timepoints),
                    f"Median-binning volume {timepoint + 1}/{num_timepoints} "
                    f"(x{factor})",
                )
        return neutron_binned, xray_binned
