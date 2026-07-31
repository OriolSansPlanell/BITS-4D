"""
feature_detection.py - Automatic feature detection for spatial ROI selection

Detects features (peaks, blobs, edges) in slices for automated analysis
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import sys


class FeatureDetector:
    """
    Automatic feature detection in 2D slices
    """
    
    @staticmethod
    def detect_peaks(image, threshold_percentile=90, min_distance=10):
        """
        Detect local maxima (peaks) in image
        
        Args:
            image: 2D array
            threshold_percentile: Percentile for thresholding (default 90)
            min_distance: Minimum distance between peaks (pixels)
            
        Returns:
            peaks: List of (y, x) coordinates
        """
        print(f"FeatureDetector.detect_peaks:", file=sys.stderr)
        print(f"  threshold_percentile: {threshold_percentile}", file=sys.stderr)
        print(f"  min_distance: {min_distance}", file=sys.stderr)
        
        # Smooth image to reduce noise
        smoothed = gaussian_filter(image, sigma=2)
        
        # Find local maxima
        # A pixel is a local max if it's >= all neighbors
        max_filter = ndimage.maximum_filter(smoothed, size=min_distance)
        is_local_max = (smoothed == max_filter)
        
        # Threshold to find significant peaks
        threshold_value = np.percentile(smoothed, threshold_percentile)
        print(f"  threshold value: {threshold_value}", file=sys.stderr)
        
        # Combine conditions
        peaks_mask = is_local_max & (smoothed > threshold_value)
        
        # Get coordinates
        y_coords, x_coords = np.where(peaks_mask)
        peaks = list(zip(y_coords, x_coords))
        
        print(f"  detected {len(peaks)} peaks", file=sys.stderr)
        
        return peaks
    
    @staticmethod
    def detect_valleys(image, threshold_percentile=10, min_distance=10):
        """
        Detect local minima (valleys/pores) in image
        
        Args:
            image: 2D array
            threshold_percentile: Percentile for thresholding (default 10)
            min_distance: Minimum distance between valleys (pixels)
            
        Returns:
            valleys: List of (y, x) coordinates
        """
        print(f"FeatureDetector.detect_valleys:", file=sys.stderr)
        
        # Smooth image
        smoothed = gaussian_filter(image, sigma=2)
        
        # Find local minima
        min_filter = ndimage.minimum_filter(smoothed, size=min_distance)
        is_local_min = (smoothed == min_filter)
        
        # Threshold
        threshold_value = np.percentile(smoothed, threshold_percentile)
        print(f"  threshold value: {threshold_value}", file=sys.stderr)
        
        # Combine conditions
        valleys_mask = is_local_min & (smoothed < threshold_value)
        
        # Get coordinates
        y_coords, x_coords = np.where(valleys_mask)
        valleys = list(zip(y_coords, x_coords))
        
        print(f"  detected {len(valleys)} valleys", file=sys.stderr)
        
        return valleys
    
    @staticmethod
    def detect_blobs(image, min_sigma=2, max_sigma=20, num_sigma=10, threshold_percentile=80):
        """
        Detect blob-like structures using Laplacian of Gaussian
        
        Args:
            image: 2D array
            min_sigma: Minimum blob size
            max_sigma: Maximum blob size
            num_sigma: Number of scales to try
            threshold_percentile: Response threshold
            
        Returns:
            blobs: List of (y, x, radius) tuples
        """
        print(f"FeatureDetector.detect_blobs:", file=sys.stderr)
        
        from scipy.ndimage import gaussian_laplace
        
        # Normalize image
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
        
        # Try different scales
        sigmas = np.linspace(min_sigma, max_sigma, num_sigma)
        responses = []
        
        for sigma in sigmas:
            # Laplacian of Gaussian
            response = -gaussian_laplace(image_norm, sigma) * sigma**2
            responses.append(response)
        
        # Stack responses
        response_stack = np.stack(responses, axis=0)
        
        # Find maximum response across scales
        max_response = np.max(response_stack, axis=0)
        max_scale = np.argmax(response_stack, axis=0)
        
        # Threshold
        threshold = np.percentile(max_response, threshold_percentile)
        
        # Find local maxima in response
        max_filter = ndimage.maximum_filter(max_response, size=3)
        is_local_max = (max_response == max_filter)
        
        # Apply threshold
        blobs_mask = is_local_max & (max_response > threshold)
        
        # Get coordinates and scales
        y_coords, x_coords = np.where(blobs_mask)
        scales = max_scale[blobs_mask]
        radii = sigmas[scales]
        
        blobs = [(int(y), int(x), float(r)) for y, x, r in zip(y_coords, x_coords, radii)]
        
        print(f"  detected {len(blobs)} blobs", file=sys.stderr)
        
        return blobs
    
    @staticmethod
    def detect_bivariate_features(neutron_slice, xray_slice, method='correlation', threshold_percentile=90):
        """
        Detect features using BOTH modalities
        
        Args:
            neutron_slice: 2D neutron image
            xray_slice: 2D X-ray image
            method: 'correlation', 'product', or 'both'
            threshold_percentile: Percentile threshold
            
        Returns:
            features: List of (y, x) coordinates
        """
        print(f"FeatureDetector.detect_bivariate_features:", file=sys.stderr)
        print(f"  method: {method}", file=sys.stderr)
        
        # Normalize both images
        n_norm = (neutron_slice - neutron_slice.mean()) / (neutron_slice.std() + 1e-10)
        x_norm = (xray_slice - xray_slice.mean()) / (xray_slice.std() + 1e-10)
        
        if method == 'correlation':
            # Find regions with high correlation
            combined = n_norm * x_norm
        elif method == 'product':
            # Find regions bright in both
            combined = (neutron_slice / neutron_slice.max()) * (xray_slice / xray_slice.max())
        else:  # 'both'
            # Combine correlation and product
            corr = n_norm * x_norm
            prod = (neutron_slice / neutron_slice.max()) * (xray_slice / xray_slice.max())
            combined = corr * prod
        
        # Smooth
        combined_smooth = gaussian_filter(combined, sigma=2)
        
        # Find peaks in combined response
        max_filter = ndimage.maximum_filter(combined_smooth, size=10)
        is_local_max = (combined_smooth == max_filter)
        
        # Threshold
        threshold = np.percentile(combined_smooth, threshold_percentile)
        
        # Get features
        features_mask = is_local_max & (combined_smooth > threshold)
        y_coords, x_coords = np.where(features_mask)
        features = list(zip(y_coords, x_coords))
        
        print(f"  detected {len(features)} bivariate features", file=sys.stderr)
        
        return features
