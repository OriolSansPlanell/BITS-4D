"""
dataset_4d.py - 4D Dataset Management

Manages 4D multivariate tomography data with temporal evolution
"""

import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import tifffile


class Dataset4D:
    """
    Manages 4D multivariate tomography data (T, Z, Y, X)
    
    Attributes:
        neutron_data: 4D numpy array for neutron tomography
        xray_data: 4D numpy array for X-ray tomography
        current_timepoint: Currently displayed timepoint
        num_timepoints: Total number of temporal volumes
        shape: Full 4D shape (T, Z, Y, X)
        metadata: Dict containing pixel size, time intervals, etc.
    """
    
    def __init__(self, neutron_data: np.ndarray, xray_data: np.ndarray, 
                 metadata: Optional[Dict] = None):
        """
        Initialize 4D dataset
        
        Args:
            neutron_data: 4D array of shape (T, Z, Y, X)
            xray_data: 4D array of shape (T, Z, Y, X)
            metadata: Optional metadata dictionary
            
        Raises:
            ValueError: If data shapes don't match or invalid dimensions
        """
        # Validate inputs
        if neutron_data.shape != xray_data.shape:
            raise ValueError(
                f"Shape mismatch: neutron {neutron_data.shape} vs "
                f"xray {xray_data.shape}"
            )
        
        if neutron_data.ndim != 4:
            raise ValueError(
                f"Expected 4D data, got {neutron_data.ndim}D"
            )
        
        self.neutron_data = neutron_data
        self.xray_data = xray_data
        self.shape = neutron_data.shape
        self.num_timepoints = self.shape[0]
        self.current_timepoint = 0
        self.metadata = metadata or {}
        
        # Initialize caches
        self._histogram_cache = {}
        self._segmentation_cache = {}
        
    def get_current_volume(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get neutron and X-ray volumes at current timepoint
        
        Returns:
            Tuple of (neutron_3d, xray_3d) arrays
        """
        return (
            self.neutron_data[self.current_timepoint],
            self.xray_data[self.current_timepoint]
        )
    
    def get_volume_at_time(self, timepoint: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get neutron and X-ray volumes at specific timepoint
        
        Args:
            timepoint: Temporal index (0 to num_timepoints-1)
            
        Returns:
            Tuple of (neutron_3d, xray_3d) arrays
            
        Raises:
            IndexError: If timepoint is out of range
        """
        if not 0 <= timepoint < self.num_timepoints:
            raise IndexError(
                f"Timepoint {timepoint} out of range [0, {self.num_timepoints})"
            )
        
        return (
            self.neutron_data[timepoint],
            self.xray_data[timepoint]
        )
    
    def set_timepoint(self, timepoint: int) -> None:
        """
        Set current timepoint for viewing
        
        Args:
            timepoint: Temporal index to set as current
            
        Raises:
            IndexError: If timepoint is out of range
        """
        if not 0 <= timepoint < self.num_timepoints:
            raise IndexError(
                f"Timepoint {timepoint} out of range [0, {self.num_timepoints})"
            )
        
        self.current_timepoint = timepoint
    
    def next_timepoint(self) -> bool:
        """
        Advance to next timepoint
        
        Returns:
            True if advanced, False if already at last timepoint
        """
        if self.current_timepoint < self.num_timepoints - 1:
            self.current_timepoint += 1
            return True
        return False
    
    def previous_timepoint(self) -> bool:
        """
        Go back to previous timepoint
        
        Returns:
            True if moved back, False if already at first timepoint
        """
        if self.current_timepoint > 0:
            self.current_timepoint -= 1
            return True
        return False
    
    def get_slice(self, axis: str = 'z', 
                  index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D slice from current volume
        
        Args:
            axis: Axis to slice ('x', 'y', or 'z')
            index: Slice index (defaults to middle)
            
        Returns:
            Tuple of (neutron_slice, xray_slice)
        """
        neutron_vol, xray_vol = self.get_current_volume()
        
        # Default to middle slice
        if index is None:
            axis_map = {'z': 0, 'y': 1, 'x': 2}
            index = neutron_vol.shape[axis_map[axis.lower()]] // 2
        
        # Extract slice based on axis
        axis = axis.lower()
        if axis == 'z':
            neutron_slice = neutron_vol[index, :, :]
            xray_slice = xray_vol[index, :, :]
        elif axis == 'y':
            neutron_slice = neutron_vol[:, index, :]
            xray_slice = xray_vol[:, index, :]
        elif axis == 'x':
            neutron_slice = neutron_vol[:, :, index]
            xray_slice = xray_vol[:, :, index]
        else:
            raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'")
        
        return neutron_slice, xray_slice
    
    def get_memory_usage(self) -> Dict:
        """
        Calculate memory usage of dataset
        
        Returns:
            Dict with memory statistics in MB
        """
        neutron_mb = self.neutron_data.nbytes / (1024**2)
        xray_mb = self.xray_data.nbytes / (1024**2)
        cache_mb = sum(
            arr.nbytes / (1024**2) 
            for arr in self._histogram_cache.values()
            if isinstance(arr, np.ndarray)
        )
        
        return {
            'neutron_data': neutron_mb,
            'xray_data': xray_mb,
            'cache': cache_mb,
            'total': neutron_mb + xray_mb + cache_mb
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data to free memory"""
        self._histogram_cache.clear()
        self._segmentation_cache.clear()
    
    def get_temporal_statistics(self) -> Dict:
        """
        Compute statistics across temporal dimension
        
        Returns:
            Dict with temporal statistics
        """
        return {
            'num_timepoints': self.num_timepoints,
            'current_timepoint': self.current_timepoint,
            'neutron_mean_per_time': np.mean(
                self.neutron_data.reshape(self.num_timepoints, -1), axis=1
            ),
            'xray_mean_per_time': np.mean(
                self.xray_data.reshape(self.num_timepoints, -1), axis=1
            ),
            'neutron_std_per_time': np.std(
                self.neutron_data.reshape(self.num_timepoints, -1), axis=1
            ),
            'xray_std_per_time': np.std(
                self.xray_data.reshape(self.num_timepoints, -1), axis=1
            ),
        }
    
    def export_timepoint(self, timepoint: int, 
                        output_dir: Path, 
                        prefix: str = 'volume') -> None:
        """
        Export specific timepoint as separate TIFF files
        
        Args:
            timepoint: Timepoint to export
            output_dir: Directory to save files
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        neutron_vol, xray_vol = self.get_volume_at_time(timepoint)
        
        neutron_path = output_dir / f"{prefix}_t{timepoint:04d}_neutron.tif"
        xray_path = output_dir / f"{prefix}_t{timepoint:04d}_xray.tif"
        
        tifffile.imwrite(neutron_path, neutron_vol)
        tifffile.imwrite(xray_path, xray_vol)
    
    def __repr__(self) -> str:
        return (
            f"Dataset4D(shape={self.shape}, "
            f"current_t={self.current_timepoint}, "
            f"dtype={self.neutron_data.dtype})"
        )
