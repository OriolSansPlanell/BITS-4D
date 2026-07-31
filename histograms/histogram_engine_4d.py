"""
histogram_engine_4d.py - 4D Histogram Computation with Progress Reporting

Computes and caches 2D histograms for 4D datasets
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from collections import OrderedDict

# Try to import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class HistogramData:
    """Container for histogram data and metadata"""
    histogram: np.ndarray  # 2D histogram array
    x_edges: np.ndarray    # Bin edges for x-axis (neutron)
    y_edges: np.ndarray    # Bin edges for y-axis (x-ray)
    x_centers: np.ndarray  # Bin centers for x-axis
    y_centers: np.ndarray  # Bin centers for y-axis
    data_range: Tuple[Tuple[float, float], Tuple[float, float]]
    num_voxels: int
    
    def to_log_scale(self) -> np.ndarray:
        """Return histogram in log scale for better visualization"""
        return np.log10(self.histogram + 1)


class HistogramEngine4D:
    """
    Compute and manage 2D histograms for 4D bivariate datasets
    """
    
    def __init__(self, bins: int = 256, cache_size: int = 10, use_gpu: bool = True):
        """
        Initialize histogram engine
        
        Args:
            bins: Number of bins for 2D histogram
            cache_size: Number of local histograms to cache
            use_gpu: Whether to use GPU acceleration if available
        """
        self.bins = bins
        self.cache_size = cache_size
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        
        # Cache for local histograms (LRU cache)
        self._local_cache: OrderedDict[int, HistogramData] = OrderedDict()
        
        # Global histogram (computed once)
        self._global_histogram: Optional[HistogramData] = None
        
        # Data range for consistent binning
        self._data_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        
        if self.use_gpu:
            print("GPU acceleration enabled for histogram computation")
        else:
            print("Using CPU for histogram computation")
    
    def compute_global_histogram(
        self, 
        neutron_4d: np.ndarray, 
        xray_4d: np.ndarray,
        subsample: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> HistogramData:
        """
        Compute global histogram from entire 4D dataset
        
        Args:
            neutron_4d: 4D neutron data (T, Z, Y, X)
            xray_4d: 4D X-ray data (T, Z, Y, X)
            subsample: If provided, subsample data by this factor
            progress_callback: Optional callback(percentage, message)
            
        Returns:
            HistogramData object containing global histogram
        """
        def report_progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
        
        report_progress(0, f"Computing global histogram ({self.bins}x{self.bins})...")
        
        # Apply subsampling if requested
        if subsample and subsample > 1:
            neutron_data = neutron_4d[::subsample, ::subsample, ::subsample, ::subsample]
            xray_data = xray_4d[::subsample, ::subsample, ::subsample, ::subsample]
            report_progress(20, f"Subsampled to {neutron_data.shape}")
        else:
            neutron_data = neutron_4d
            xray_data = xray_4d
        
        report_progress(30, "Flattening data...")
        
        # Flatten to 1D arrays
        neutron_flat = neutron_data.ravel()
        xray_flat = xray_data.ravel()
        
        report_progress(50, "Determining data range...")
        
        # Determine data range for consistent binning
        neutron_range = (float(neutron_flat.min()), float(neutron_flat.max()))
        xray_range = (float(xray_flat.min()), float(xray_flat.max()))
        self._data_range = (neutron_range, xray_range)
        
        report_progress(60, "Computing 2D histogram...")
        
        # Compute histogram
        if self.use_gpu:
            hist_data = self._compute_histogram_gpu(
                neutron_flat, xray_flat, neutron_range, xray_range
            )
        else:
            hist_data = self._compute_histogram_cpu(
                neutron_flat, xray_flat, neutron_range, xray_range
            )
        
        # Cache global histogram
        self._global_histogram = hist_data
        
        report_progress(100, f"Global histogram complete: {hist_data.num_voxels:,} voxels")
        
        print(f"compute_global_histogram returning: {type(hist_data)}")
        print(f"  histogram shape: {hist_data.histogram.shape}")
        print(f"  data_range: {hist_data.data_range}")
        
        return hist_data
    
    def compute_local_histogram(
        self, 
        neutron_3d: np.ndarray, 
        xray_3d: np.ndarray,
        timepoint: int,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> HistogramData:
        """
        Compute histogram for single timepoint
        
        Args:
            neutron_3d: 3D neutron volume at timepoint
            xray_3d: 3D X-ray volume at timepoint
            timepoint: Timepoint index (for caching)
            progress_callback: Optional callback(percentage, message)
            
        Returns:
            HistogramData object for this timepoint
        """
        # Check cache first
        if timepoint in self._local_cache:
            self._local_cache.move_to_end(timepoint)
            if progress_callback:
                progress_callback(100, f"Retrieved from cache")
            return self._local_cache[timepoint]
        
        def report_progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
        
        # Use global data range for consistent binning
        if self._data_range is None:
            raise RuntimeError("Must compute global histogram first")
        
        neutron_range, xray_range = self._data_range
        
        report_progress(20, f"Flattening timepoint {timepoint}...")
        
        # Flatten to 1D
        neutron_flat = neutron_3d.ravel()
        xray_flat = xray_3d.ravel()
        
        report_progress(50, "Computing histogram...")
        
        # Compute histogram
        if self.use_gpu:
            hist_data = self._compute_histogram_gpu(
                neutron_flat, xray_flat, neutron_range, xray_range
            )
        else:
            hist_data = self._compute_histogram_cpu(
                neutron_flat, xray_flat, neutron_range, xray_range
            )
        
        report_progress(80, "Caching result...")
        
        # Add to cache
        self._local_cache[timepoint] = hist_data
        
        # Enforce cache size limit (LRU eviction)
        if len(self._local_cache) > self.cache_size:
            self._local_cache.popitem(last=False)
        
        report_progress(100, f"Local histogram complete")
        
        return hist_data
    
    def precompute_all_local_histograms(
        self,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> None:
        """
        Pre-compute histograms for all timepoints
        
        Args:
            neutron_4d: 4D neutron data
            xray_4d: 4D X-ray data
            progress_callback: Optional callback(percentage, message)
        """
        num_timepoints = neutron_4d.shape[0]
        
        for t in range(num_timepoints):
            pct = int((t / num_timepoints) * 100)
            
            self.compute_local_histogram(
                neutron_4d[t], xray_4d[t], timepoint=t
            )
            
            if progress_callback:
                progress_callback(pct, f"Computing histogram {t+1}/{num_timepoints}")
        
        if progress_callback:
            progress_callback(100, "All histograms computed")
    
    def _compute_histogram_cpu(
        self,
        neutron_flat: np.ndarray,
        xray_flat: np.ndarray,
        neutron_range: Tuple[float, float],
        xray_range: Tuple[float, float]
    ) -> HistogramData:
        """Compute 2D histogram using NumPy (CPU)"""
        hist, x_edges, y_edges = np.histogram2d(
            neutron_flat,
            xray_flat,
            bins=self.bins,
            range=[neutron_range, xray_range]
        )
        
        # Transpose to get correct orientation
        hist = hist.T
        
        # Calculate bin centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        return HistogramData(
            histogram=hist,
            x_edges=x_edges,
            y_edges=y_edges,
            x_centers=x_centers,
            y_centers=y_centers,
            data_range=(neutron_range, xray_range),
            num_voxels=len(neutron_flat)
        )
    
    def _compute_histogram_gpu(
        self,
        neutron_flat: np.ndarray,
        xray_flat: np.ndarray,
        neutron_range: Tuple[float, float],
        xray_range: Tuple[float, float]
    ) -> HistogramData:
        """Compute 2D histogram using PyTorch (GPU)"""
        # Transfer to GPU
        neutron_gpu = torch.from_numpy(neutron_flat).cuda()
        xray_gpu = torch.from_numpy(xray_flat).cuda()
        
        # Compute bin indices
        neutron_norm = (neutron_gpu - neutron_range[0]) / (neutron_range[1] - neutron_range[0])
        xray_norm = (xray_gpu - xray_range[0]) / (xray_range[1] - xray_range[0])
        
        neutron_bins = torch.clamp(
            (neutron_norm * self.bins).long(), 0, self.bins - 1
        )
        xray_bins = torch.clamp(
            (xray_norm * self.bins).long(), 0, self.bins - 1
        )
        
        # Compute 2D histogram
        indices = xray_bins * self.bins + neutron_bins
        counts = torch.bincount(indices, minlength=self.bins * self.bins)
        hist = counts.reshape(self.bins, self.bins).float()
        
        # Transfer back to CPU
        hist = hist.cpu().numpy()
        
        # Create bin edges
        x_edges = np.linspace(neutron_range[0], neutron_range[1], self.bins + 1)
        y_edges = np.linspace(xray_range[0], xray_range[1], self.bins + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        return HistogramData(
            histogram=hist,
            x_edges=x_edges,
            y_edges=y_edges,
            x_centers=x_centers,
            y_centers=y_centers,
            data_range=(neutron_range, xray_range),
            num_voxels=len(neutron_flat)
        )
    
    def get_global_histogram(self) -> Optional[HistogramData]:
        """Get cached global histogram"""
        return self._global_histogram
    
    def get_cached_local_histogram(self, timepoint: int) -> Optional[HistogramData]:
        """Get cached local histogram for specific timepoint"""
        return self._local_cache.get(timepoint)
    
    def clear_cache(self) -> None:
        """Clear all cached local histograms"""
        self._local_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cache usage"""
        return {
            'cache_size': len(self._local_cache),
            'cache_limit': self.cache_size,
            'cached_timepoints': list(self._local_cache.keys()),
            'global_cached': self._global_histogram is not None
        }
