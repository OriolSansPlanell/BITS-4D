"""
config.py - Configuration and constants for BiTS 4D

Author: BiTS Development Team
Version: 2.0.0-beta
"""

# Application metadata
APP_NAME = "BiTS"
APP_VERSION = "15.1"
APP_TITLE = f"{APP_NAME} - Bivariate Tomography Segmentation"

# Histogram settings
DEFAULT_BINS = 256
MAX_BINS = 512
MIN_BINS = 64

# Cache settings
DEFAULT_HISTOGRAM_CACHE_SIZE = 10  # Number of local histograms to cache
DEFAULT_SEGMENTATION_CACHE_SIZE = 5  # Number of segmentation masks to cache

# Performance settings
USE_GPU_DEFAULT = True
SUBSAMPLE_THRESHOLD_GB = 16  # Dataset size above which to offer subsampling
MEMMAP_THRESHOLD_GB = 32  # Dataset size above which to use memory mapping

# UI settings
DEFAULT_FPS = 10
MAX_FPS = 60
MIN_FPS = 1

# File formats
SUPPORTED_TIFF_EXTENSIONS = ['.tif', '.tiff', '.TIF', '.TIFF']
SUPPORTED_EXPORT_FORMATS = ['TIFF', 'HDF5', 'NumPy']

# Colors for visualization
ROI_COLOR = '#00FF00'  # Green
SEGMENTATION_OVERLAY_COLOR = '#FF0000'  # Red
SEGMENTATION_OVERLAY_ALPHA = 0.3

# Progress dialog settings
PROGRESS_UPDATE_INTERVAL_MS = 100  # Update progress every 100ms
