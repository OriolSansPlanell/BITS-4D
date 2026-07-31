"""
data_loader_4d.py - 4D TIFF Loading with Progress Reporting

Loads 4D TIFF stacks for neutron and X-ray data with progress feedback
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict
from .dataset_4d import Dataset4D


class TIFF4DLoader:
    """
    Loads 4D TIFF stacks for neutron and X-ray data
    """
    
    @staticmethod
    def estimate_memory(neutron_path: str, xray_path: str) -> Dict:
        """
        Estimate memory required to load datasets
        
        Args:
            neutron_path: Path to neutron TIFF
            xray_path: Path to X-ray TIFF
            
        Returns:
            Dict with memory estimates in MB and GB
        """
        neutron_path = Path(neutron_path)
        xray_path = Path(xray_path)
        
        if not neutron_path.exists():
            raise FileNotFoundError(f"Neutron file not found: {neutron_path}")
        if not xray_path.exists():
            raise FileNotFoundError(f"X-ray file not found: {xray_path}")
        
        # Get file info without loading full data
        with tifffile.TiffFile(neutron_path) as tif:
            neutron_shape = tif.series[0].shape
            neutron_dtype = tif.series[0].dtype
        
        with tifffile.TiffFile(xray_path) as tif:
            xray_shape = tif.series[0].shape
            xray_dtype = tif.series[0].dtype
        
        neutron_bytes = np.prod(neutron_shape) * np.dtype(neutron_dtype).itemsize
        xray_bytes = np.prod(xray_shape) * np.dtype(xray_dtype).itemsize
        total_bytes = neutron_bytes + xray_bytes
        
        return {
            'neutron_mb': neutron_bytes / (1024**2),
            'xray_mb': xray_bytes / (1024**2),
            'total_mb': total_bytes / (1024**2),
            'total_gb': total_bytes / (1024**3),
            'neutron_shape': neutron_shape,
            'xray_shape': xray_shape,
            'neutron_dtype': str(neutron_dtype),
            'xray_dtype': str(xray_dtype)
        }
    
    @staticmethod
    def load(neutron_path: str, xray_path: str, 
             use_memmap: bool = False,
             progress_callback: Optional[Callable[[int, str], None]] = None) -> Dataset4D:
        """
        Load two 4D TIFF files into Dataset4D object
        
        Args:
            neutron_path: Path to neutron TIFF stack
            xray_path: Path to X-ray TIFF stack
            use_memmap: If True, use memory-mapped arrays (for large files)
            progress_callback: Optional callback(percentage, message)
            
        Returns:
            Dataset4D object
            
        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If files have mismatched dimensions
        """
        def report_progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
        
        # Check files exist
        neutron_path = Path(neutron_path)
        xray_path = Path(xray_path)
        
        if not neutron_path.exists():
            raise FileNotFoundError(f"Neutron file not found: {neutron_path}")
        if not xray_path.exists():
            raise FileNotFoundError(f"X-ray file not found: {xray_path}")
        
        report_progress(10, f"Loading neutron data: {neutron_path.name}...")
        
        # Load neutron data
        if use_memmap:
            neutron_data = tifffile.memmap(neutron_path, mode='r')
        else:
            neutron_data = tifffile.imread(neutron_path)
        
        report_progress(40, f"Loading X-ray data: {xray_path.name}...")
        
        # Load X-ray data
        if use_memmap:
            xray_data = tifffile.memmap(xray_path, mode='r')
        else:
            xray_data = tifffile.imread(xray_path)
        
        report_progress(70, "Validating data dimensions...")
        
        # Ensure 4D
        if neutron_data.ndim == 3:
            print("Warning: 3D data detected, adding temporal dimension")
            neutron_data = neutron_data[np.newaxis, ...]
            xray_data = xray_data[np.newaxis, ...]
        
        # Validate shapes match
        if neutron_data.shape != xray_data.shape:
            raise ValueError(
                f"Shape mismatch: neutron {neutron_data.shape} vs "
                f"xray {xray_data.shape}"
            )
        
        report_progress(85, "Extracting metadata...")
        
        # Extract metadata
        metadata = TIFF4DLoader._extract_metadata(neutron_path, xray_path)
        
        report_progress(95, "Creating dataset object...")
        
        # Create dataset
        dataset = Dataset4D(neutron_data, xray_data, metadata)
        
        report_progress(100, f"Dataset loaded: {dataset.shape}")
        
        mem_usage = dataset.get_memory_usage()
        print(f"Dataset loaded: {dataset}")
        print(f"Memory usage: {mem_usage['total']:.1f} MB")
        
        return dataset
    
    @staticmethod
    def _extract_metadata(neutron_path: Path, xray_path: Path) -> Dict:
        """
        Extract metadata from TIFF files
        
        Args:
            neutron_path: Path to neutron TIFF file
            xray_path: Path to X-ray TIFF file
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'neutron_file': str(neutron_path),
            'xray_file': str(xray_path)
        }
        
        try:
            with tifffile.TiffFile(neutron_path) as tif:
                # Try to extract ImageJ metadata
                if tif.imagej_metadata:
                    metadata['neutron_imagej'] = tif.imagej_metadata
                
                # Try to extract resolution
                first_page = tif.pages[0]
                if hasattr(first_page, 'tags'):
                    x_res = first_page.tags.get('XResolution')
                    y_res = first_page.tags.get('YResolution')
                    if x_res and y_res:
                        metadata['neutron_resolution'] = {
                            'x': x_res.value,
                            'y': y_res.value
                        }
        except Exception as e:
            print(f"Could not extract neutron metadata: {e}")
        
        try:
            with tifffile.TiffFile(xray_path) as tif:
                if tif.imagej_metadata:
                    metadata['xray_imagej'] = tif.imagej_metadata
                
                first_page = tif.pages[0]
                if hasattr(first_page, 'tags'):
                    x_res = first_page.tags.get('XResolution')
                    y_res = first_page.tags.get('YResolution')
                    if x_res and y_res:
                        metadata['xray_resolution'] = {
                            'x': x_res.value,
                            'y': y_res.value
                        }
        except Exception as e:
            print(f"Could not extract X-ray metadata: {e}")
        
        return metadata
    
    @staticmethod
    def validate_files(neutron_path: str, xray_path: str) -> Tuple[bool, str]:
        """
        Validate TIFF files before loading
        
        Args:
            neutron_path: Path to neutron TIFF
            xray_path: Path to X-ray TIFF
            
        Returns:
            Tuple of (is_valid, message)
        """
        neutron_path = Path(neutron_path)
        xray_path = Path(xray_path)
        
        # Check existence
        if not neutron_path.exists():
            return False, f"Neutron file not found: {neutron_path}"
        if not xray_path.exists():
            return False, f"X-ray file not found: {xray_path}"
        
        # Check file extension
        if neutron_path.suffix.lower() not in ['.tif', '.tiff']:
            return False, f"Neutron file is not a TIFF: {neutron_path.suffix}"
        if xray_path.suffix.lower() not in ['.tif', '.tiff']:
            return False, f"X-ray file is not a TIFF: {xray_path.suffix}"
        
        try:
            # Check if files can be opened
            with tifffile.TiffFile(neutron_path) as tif:
                neutron_shape = tif.series[0].shape
                neutron_ndim = len(neutron_shape)
            
            with tifffile.TiffFile(xray_path) as tif:
                xray_shape = tif.series[0].shape
                xray_ndim = len(xray_shape)
            
            # Check dimensions
            if neutron_ndim not in [3, 4]:
                return False, f"Neutron data must be 3D or 4D, got {neutron_ndim}D"
            if xray_ndim not in [3, 4]:
                return False, f"X-ray data must be 3D or 4D, got {xray_ndim}D"
            
            # For 3D data, shapes must match exactly
            # For 4D data, shapes must match exactly
            if neutron_shape != xray_shape:
                return False, (
                    f"Shape mismatch: neutron {neutron_shape} vs "
                    f"xray {xray_shape}"
                )
            
            return True, "Files are valid"
            
        except Exception as e:
            return False, f"Error validating files: {str(e)}"
