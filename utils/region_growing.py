"""
region_growing.py - Region growing / watershed for spatial ROI selection

Implements flood-fill based region growing from a seed point
"""

import numpy as np
from scipy import ndimage
import sys


class RegionGrowing:
    """
    Region growing algorithm for spatial feature selection
    """
    
    @staticmethod
    def flood_fill_tolerance(image, seed_point, tolerance, connectivity=1):
        """
        Flood fill from seed point with intensity tolerance (single modality)
        
        Args:
            image: 2D array (slice image)
            seed_point: (y, x) coordinates of seed
            tolerance: Intensity tolerance (absolute difference)
            connectivity: 1 (4-connected) or 2 (8-connected)
            
        Returns:
            mask: Boolean 2D array of selected region
        """
        print(f"RegionGrowing.flood_fill_tolerance:", file=sys.stderr)
        print(f"  seed_point: {seed_point}", file=sys.stderr)
        print(f"  tolerance: {tolerance}", file=sys.stderr)
        print(f"  image shape: {image.shape}", file=sys.stderr)
        
        # Get seed intensity
        seed_y, seed_x = int(seed_point[0]), int(seed_point[1])
        
        # Boundary check
        if seed_y < 0 or seed_y >= image.shape[0] or seed_x < 0 or seed_x >= image.shape[1]:
            print(f"  ERROR: Seed point out of bounds", file=sys.stderr)
            return np.zeros(image.shape, dtype=bool)
        
        seed_intensity = float(image[seed_y, seed_x])
        print(f"  seed intensity: {seed_intensity}", file=sys.stderr)
        
        # Create mask of pixels within tolerance
        lower = seed_intensity - tolerance
        upper = seed_intensity + tolerance
        
        within_tolerance = (image >= lower) & (image <= upper)
        print(f"  pixels within tolerance: {np.sum(within_tolerance)}", file=sys.stderr)
        
        # Label connected components
        structure = ndimage.generate_binary_structure(2, connectivity)
        labeled, num_features = ndimage.label(within_tolerance, structure=structure)
        
        print(f"  found {num_features} connected components", file=sys.stderr)
        
        # Get the label at seed point
        seed_label = labeled[seed_y, seed_x]
        
        if seed_label == 0:
            print(f"  WARNING: Seed point not in any region", file=sys.stderr)
            return np.zeros(image.shape, dtype=bool)
        
        # Create mask of the component containing seed
        mask = (labeled == seed_label)
        
        num_pixels = np.sum(mask)
        print(f"  selected region: {num_pixels} pixels", file=sys.stderr)
        
        return mask
    
    @staticmethod
    def flood_fill_bivariate(neutron_slice, xray_slice, seed_point, neutron_tolerance, xray_tolerance, connectivity=1):
        """
        Bivariate flood fill - considers BOTH neutron AND X-ray intensities
        
        A pixel is selected only if:
        - It's connected to the seed
        - |neutron - neutron_seed| <= neutron_tolerance
        - |xray - xray_seed| <= xray_tolerance
        
        Args:
            neutron_slice: 2D array (neutron image)
            xray_slice: 2D array (X-ray image)
            seed_point: (y, x) coordinates of seed
            neutron_tolerance: Tolerance for neutron intensity
            xray_tolerance: Tolerance for X-ray intensity
            connectivity: 1 (4-connected) or 2 (8-connected)
            
        Returns:
            mask: Boolean 2D array of selected region
        """
        print(f"RegionGrowing.flood_fill_bivariate:", file=sys.stderr)
        print(f"  seed_point: {seed_point}", file=sys.stderr)
        print(f"  neutron_tolerance: {neutron_tolerance}", file=sys.stderr)
        print(f"  xray_tolerance: {xray_tolerance}", file=sys.stderr)
        
        # Get seed coordinates
        seed_y, seed_x = int(seed_point[0]), int(seed_point[1])
        
        # Boundary check
        if (seed_y < 0 or seed_y >= neutron_slice.shape[0] or 
            seed_x < 0 or seed_x >= neutron_slice.shape[1]):
            print(f"  ERROR: Seed point out of bounds", file=sys.stderr)
            return np.zeros(neutron_slice.shape, dtype=bool)
        
        # Get seed intensities for BOTH modalities
        neutron_seed = float(neutron_slice[seed_y, seed_x])
        xray_seed = float(xray_slice[seed_y, seed_x])
        print(f"  seed neutron: {neutron_seed}, seed xray: {xray_seed}", file=sys.stderr)
        
        # Create mask of pixels within tolerance for BOTH modalities
        neutron_in_range = np.abs(neutron_slice - neutron_seed) <= neutron_tolerance
        xray_in_range = np.abs(xray_slice - xray_seed) <= xray_tolerance
        
        # BOTH must be satisfied
        within_tolerance = neutron_in_range & xray_in_range
        
        print(f"  pixels within neutron tolerance: {np.sum(neutron_in_range)}", file=sys.stderr)
        print(f"  pixels within xray tolerance: {np.sum(xray_in_range)}", file=sys.stderr)
        print(f"  pixels within BOTH tolerances: {np.sum(within_tolerance)}", file=sys.stderr)
        
        # Label connected components
        structure = ndimage.generate_binary_structure(2, connectivity)
        labeled, num_features = ndimage.label(within_tolerance, structure=structure)
        
        print(f"  found {num_features} connected components", file=sys.stderr)
        
        # Get the label at seed point
        seed_label = labeled[seed_y, seed_x]
        
        if seed_label == 0:
            print(f"  WARNING: Seed point not in any region (bivariate constraints too strict)", file=sys.stderr)
            return np.zeros(neutron_slice.shape, dtype=bool)
        
        # Create mask of the component containing seed
        mask = (labeled == seed_label)
        
        num_pixels = np.sum(mask)
        print(f"  selected region: {num_pixels} pixels (bivariate)", file=sys.stderr)
        
        return mask
    
    @staticmethod
    def extract_values_from_mask(neutron_slice, xray_slice, mask):
        """
        Extract neutron and X-ray values from masked region
        
        Args:
            neutron_slice: 2D array (neutron data)
            xray_slice: 2D array (X-ray data)
            mask: Boolean 2D array
            
        Returns:
            neutron_values: 1D array of values
            xray_values: 1D array of values
        """
        neutron_values = neutron_slice[mask]
        xray_values = xray_slice[mask]
        
        print(f"RegionGrowing.extract_values_from_mask:", file=sys.stderr)
        print(f"  extracted {len(neutron_values)} pixel pairs", file=sys.stderr)
        
        return neutron_values, xray_values
    
    @staticmethod
    def create_convex_hull_roi(neutron_values, xray_values, margin=0.05):
        """
        Create convex hull polygon ROI from point cloud in histogram space
        
        Args:
            neutron_values: 1D array of neutron intensities
            xray_values: 1D array of X-ray intensities
            margin: Percentage margin to add
            
        Returns:
            polygon_points: Nx2 array of (neutron, xray) vertices
        """
        from scipy.spatial import ConvexHull
        
        print(f"RegionGrowing.create_convex_hull_roi:", file=sys.stderr)
        
        if len(neutron_values) < 3:
            print(f"  ERROR: Need at least 3 points for convex hull", file=sys.stderr)
            # Fall back to bounding box
            n_min, n_max = float(np.min(neutron_values)), float(np.max(neutron_values))
            x_min, x_max = float(np.min(xray_values)), float(np.max(xray_values))
            
            # Add margin
            n_range = n_max - n_min
            x_range = x_max - x_min
            n_min -= n_range * margin
            n_max += n_range * margin
            x_min -= x_range * margin
            x_max += x_range * margin
            
            # Return rectangle vertices
            return np.array([
                [n_min, x_min],
                [n_max, x_min],
                [n_max, x_max],
                [n_min, x_max]
            ])
        
        # Create point cloud
        points = np.column_stack([neutron_values, xray_values])
        
        # Compute convex hull
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            print(f"  convex hull: {len(hull_points)} vertices", file=sys.stderr)
            
            # Add margin to hull
            centroid = np.mean(hull_points, axis=0)
            expanded_points = centroid + (hull_points - centroid) * (1 + margin)
            
            return expanded_points
            
        except Exception as e:
            print(f"  ERROR computing convex hull: {e}", file=sys.stderr)
            # Fall back to bounding box
            n_min, n_max = float(np.min(neutron_values)), float(np.max(neutron_values))
            x_min, x_max = float(np.min(xray_values)), float(np.max(xray_values))
            
            n_range = n_max - n_min
            x_range = x_max - x_min
            n_min -= n_range * margin
            n_max += n_range * margin
            x_min -= x_range * margin
            x_max += x_range * margin
            
            return np.array([
                [n_min, x_min],
                [n_max, x_min],
                [n_max, x_max],
                [n_min, x_max]
            ])
