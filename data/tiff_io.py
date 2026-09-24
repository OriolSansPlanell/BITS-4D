"""
tiff_io.py - Writing volumes as TIFF

tifffile guesses the photometric interpretation from the array shape: a
uint8 volume with 3 or 4 slices (or a trailing axis of 3/4) is written as an
RGB(A) image, so it reads back as one colour picture instead of a stack of
greyscale slices. Every volume written here is a greyscale stack, so say so.
"""

import numpy as np
import tifffile


def write_volume_tiff(path, volume) -> None:
    """Write *volume* as a greyscale TIFF stack (one page per slice)."""
    tifffile.imwrite(path, np.asarray(volume), photometric="minisblack")
