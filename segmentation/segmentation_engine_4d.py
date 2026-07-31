"""
segmentation_engine_4d.py - Segmentation for 4D datasets

Applies ROI-based segmentation to 4D volumes
"""

import numpy as np
from typing import Optional, Callable, Tuple
import sys
sys.path.append('..')
from utils.roi_manager import ROIManager


class SegmentationEngine4D:
    """
    Apply ROI-based segmentation to 4D datasets
    """
    
    def __init__(self):
        """Initialize segmentation engine"""
        self.segmentation_masks = {}  # Cache for segmentation masks
    
    def segment_volume(
        self,
        neutron_3d: np.ndarray,
        xray_3d: np.ndarray,
        roi_manager: ROIManager,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> np.ndarray:
        """
        Segment a single 3D volume using ROI
        
        Args:
            neutron_3d: 3D neutron volume
            xray_3d: 3D X-ray volume
            roi_manager: ROI manager with defined ROI
            progress_callback: Optional callback(percentage, message)
            
        Returns:
            Boolean 3D mask (True = inside ROI)
        """
        def report_progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
        
        if not roi_manager.has_roi():
            raise ValueError("No ROI defined in ROI manager")
        
        report_progress(10, "Checking voxels against ROI...")
        
        # Check each voxel against ROI
        mask = roi_manager.is_inside_roi(neutron_3d, xray_3d)
        
        report_progress(90, "Computing statistics...")
        
        # Calculate statistics
        num_segmented = np.sum(mask)
        total_voxels = mask.size
        percentage = (num_segmented / total_voxels) * 100
        
        report_progress(100, f"Segmented {num_segmented:,} voxels ({percentage:.1f}%)")
        
        return mask
    
    def segment_all_volumes(
        self,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray,
        roi_manager: ROIManager,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> np.ndarray:
        """
        Segment all volumes in 4D dataset
        
        Args:
            neutron_4d: 4D neutron data (T, Z, Y, X)
            xray_4d: 4D X-ray data (T, Z, Y, X)
            roi_manager: ROI manager with defined ROI
            progress_callback: Optional callback(percentage, message)
            
        Returns:
            Boolean 4D mask (same shape as input)
        """
        if not roi_manager.has_roi():
            raise ValueError("No ROI defined in ROI manager")
        
        num_timepoints = neutron_4d.shape[0]
        mask_4d = np.zeros(neutron_4d.shape, dtype=bool)
        
        for t in range(num_timepoints):
            pct = int((t / num_timepoints) * 100)
            
            if progress_callback:
                progress_callback(pct, f"Segmenting timepoint {t+1}/{num_timepoints}")
            
            # Segment this timepoint
            mask_4d[t] = roi_manager.is_inside_roi(
                neutron_4d[t], 
                xray_4d[t]
            )
        
        if progress_callback:
            total_segmented = np.sum(mask_4d)
            total_voxels = mask_4d.size
            percentage = (total_segmented / total_voxels) * 100
            progress_callback(
                100, 
                f"Complete: {total_segmented:,} voxels ({percentage:.1f}%)"
            )
        
        return mask_4d
    
    def get_segmentation_statistics(
        self,
        mask: np.ndarray,
        neutron_data: np.ndarray,
        xray_data: np.ndarray
    ) -> dict:
        """
        Compute statistics for segmented region
        
        Args:
            mask: Boolean mask
            neutron_data: Neutron intensity values
            xray_data: X-ray intensity values
            
        Returns:
            Dict with segmentation statistics
        """
        num_segmented = np.sum(mask)
        total_voxels = mask.size
        percentage = (num_segmented / total_voxels) * 100
        
        # Statistics for segmented voxels
        if num_segmented > 0:
            neutron_segmented = neutron_data[mask]
            xray_segmented = xray_data[mask]
            
            stats = {
                'num_voxels': int(num_segmented),
                'total_voxels': int(total_voxels),
                'percentage': float(percentage),
                'neutron_mean': float(np.mean(neutron_segmented)),
                'neutron_std': float(np.std(neutron_segmented)),
                'neutron_min': float(np.min(neutron_segmented)),
                'neutron_max': float(np.max(neutron_segmented)),
                'xray_mean': float(np.mean(xray_segmented)),
                'xray_std': float(np.std(xray_segmented)),
                'xray_min': float(np.min(xray_segmented)),
                'xray_max': float(np.max(xray_segmented)),
            }
        else:
            stats = {
                'num_voxels': 0,
                'total_voxels': int(total_voxels),
                'percentage': 0.0,
            }
        
        return stats
    
    def get_temporal_statistics(
        self,
        mask_4d: np.ndarray,
        neutron_4d: np.ndarray,
        xray_4d: np.ndarray
    ) -> dict:
        """
        Compute temporal statistics for 4D segmentation
        
        Args:
            mask_4d: 4D boolean mask
            neutron_4d: 4D neutron data
            xray_4d: 4D X-ray data
            
        Returns:
            Dict with temporal statistics
        """
        num_timepoints = mask_4d.shape[0]
        
        volumes_per_time = np.zeros(num_timepoints)
        neutron_mean_per_time = np.zeros(num_timepoints)
        xray_mean_per_time = np.zeros(num_timepoints)
        
        for t in range(num_timepoints):
            volumes_per_time[t] = np.sum(mask_4d[t])
            
            if volumes_per_time[t] > 0:
                neutron_mean_per_time[t] = np.mean(neutron_4d[t][mask_4d[t]])
                xray_mean_per_time[t] = np.mean(xray_4d[t][mask_4d[t]])
        
        return {
            'num_timepoints': num_timepoints,
            'volumes_per_time': volumes_per_time,
            'neutron_mean_per_time': neutron_mean_per_time,
            'xray_mean_per_time': xray_mean_per_time,
            'total_segmented_voxels': int(np.sum(mask_4d)),
            'mean_volume': float(np.mean(volumes_per_time)),
            'std_volume': float(np.std(volumes_per_time)),
        }
    
    def apply_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Apply segmentation mask as colored overlay on image
        
        Args:
            image: 2D grayscale image
            mask: 2D boolean mask
            color: RGB color for overlay
            alpha: Transparency (0-1)
            
        Returns:
            RGB image with overlay
        """
        # Normalize image to 0-255
        img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Create RGB image
        img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
        
        # Apply colored overlay where mask is True
        overlay = img_rgb.copy()
        overlay[mask] = [
            int(overlay[mask, 0] * (1 - alpha) + color[0] * alpha),
            int(overlay[mask, 1] * (1 - alpha) + color[1] * alpha),
            int(overlay[mask, 2] * (1 - alpha) + color[2] * alpha)
        ]
        
        return overlay
