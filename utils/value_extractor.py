"""
value_extractor.py - Extract neutron/X-ray values from spatial ROI

Supports extracting values from 2D spatial selections for histogram ROI creation
"""

import numpy as np
from typing import Tuple, Optional
import sys


class ValueExtractor:
    """
    Extract neutron and X-ray values from spatial regions
    """
    
    @staticmethod
    def extract_from_rectangle(
        neutron_vol: np.ndarray,
        xray_vol: np.ndarray,
        rect_coords: Tuple[float, float, float, float],
        axis: str = 'z',
        slice_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract values from rectangular spatial ROI
        
        Args:
            neutron_vol: 3D neutron volume (Z, Y, X)
            xray_vol: 3D X-ray volume (Z, Y, X)
            rect_coords: (x1, y1, x2, y2) in pixel coordinates
            axis: 'z', 'y', or 'x' - which axis to slice
            slice_index: Index of slice to extract from
            
        Returns:
            neutron_values: 1D array of neutron values
            xray_values: 1D array of X-ray values
        """
        print(f"ValueExtractor.extract_from_rectangle:", file=sys.stderr)
        print(f"  rect_coords: {rect_coords}", file=sys.stderr)
        print(f"  axis: {axis}, slice_index: {slice_index}", file=sys.stderr)
        
        # Extract the appropriate slice
        if axis == 'z':
            neutron_slice = neutron_vol[slice_index, :, :]
            xray_slice = xray_vol[slice_index, :, :]
        elif axis == 'y':
            neutron_slice = neutron_vol[:, slice_index, :]
            xray_slice = xray_vol[:, slice_index, :]
        else:  # 'x'
            neutron_slice = neutron_vol[:, :, slice_index]
            xray_slice = xray_vol[:, :, slice_index]
        
        print(f"  slice shape: {neutron_slice.shape}", file=sys.stderr)
        
        # Parse rectangle coordinates
        x1, y1, x2, y2 = rect_coords
        
        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Convert to integer pixel indices
        # Note: matplotlib coordinates are in data space, need to map to pixels
        height, width = neutron_slice.shape
        
        # Clamp to valid range
        x1_idx = max(0, min(int(x1), width - 1))
        x2_idx = max(0, min(int(x2), width - 1))
        y1_idx = max(0, min(int(y1), height - 1))
        y2_idx = max(0, min(int(y2), height - 1))
        
        print(f"  pixel indices: x[{x1_idx}:{x2_idx}], y[{y1_idx}:{y2_idx}]", file=sys.stderr)
        
        # Ensure we have at least 1 pixel
        if x2_idx <= x1_idx or y2_idx <= y1_idx:
            print(f"  WARNING: Invalid rectangle (zero area)", file=sys.stderr)
            return np.array([]), np.array([])
        
        # Extract values from rectangle
        neutron_roi = neutron_slice[y1_idx:y2_idx, x1_idx:x2_idx]
        xray_roi = xray_slice[y1_idx:y2_idx, x1_idx:x2_idx]
        
        # Flatten to 1D arrays
        neutron_values = neutron_roi.flatten()
        xray_values = xray_roi.flatten()
        
        print(f"  extracted {len(neutron_values)} values", file=sys.stderr)
        
        return neutron_values, xray_values
    
    @staticmethod
    def create_histogram_roi_from_values(
        neutron_values: np.ndarray,
        xray_values: np.ndarray,
        margin: float = 0.05
    ) -> Tuple[float, float, float, float]:
        """
        Create rectangular histogram ROI from extracted values
        
        Args:
            neutron_values: Array of neutron intensities
            xray_values: Array of X-ray intensities
            margin: Percentage margin to add around values (default 5%)
            
        Returns:
            (n_min, x_min, n_max, x_max) - histogram ROI coordinates
        """
        if len(neutron_values) == 0 or len(xray_values) == 0:
            print(f"  WARNING: Empty value arrays", file=sys.stderr)
            return (0, 0, 1, 1)
        
        # Compute value ranges
        n_min = float(np.min(neutron_values))
        n_max = float(np.max(neutron_values))
        x_min = float(np.min(xray_values))
        x_max = float(np.max(xray_values))
        
        print(f"  value ranges:", file=sys.stderr)
        print(f"    neutron: [{n_min:.2f}, {n_max:.2f}]", file=sys.stderr)
        print(f"    xray: [{x_min:.2f}, {x_max:.2f}]", file=sys.stderr)
        
        # Add margin
        n_range = n_max - n_min
        x_range = x_max - x_min
        
        # Handle zero range (all values the same)
        if n_range == 0:
            n_range = abs(n_min) * 0.1 if n_min != 0 else 1
        if x_range == 0:
            x_range = abs(x_min) * 0.1 if x_min != 0 else 1
        
        n_min -= n_range * margin
        n_max += n_range * margin
        x_min -= x_range * margin
        x_max += x_range * margin
        
        print(f"  with {margin*100}% margin:", file=sys.stderr)
        print(f"    neutron: [{n_min:.2f}, {n_max:.2f}]", file=sys.stderr)
        print(f"    xray: [{x_min:.2f}, {x_max:.2f}]", file=sys.stderr)
        
        return (n_min, x_min, n_max, x_max)
