"""
morphological_analysis.py - Morphological Feature Analysis

Computes shape, connectivity, and texture features for selections
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
import sys


class MorphologicalAnalyzer:
    """
    Analyzes morphological features of selections
    """
    
    @staticmethod
    def analyze_selection(mask):
        """
        Comprehensive morphological analysis of a selection
        
        Args:
            mask: 2D boolean array
            
        Returns:
            dict: Morphological features
        """
        if mask is None or np.sum(mask) == 0:
            return None
        
        features = {}
        
        # Basic properties
        features['area'] = np.sum(mask)
        features['bbox'] = MorphologicalAnalyzer._compute_bounding_box(mask)
        
        # Perimeter and shape
        features['perimeter'] = MorphologicalAnalyzer._compute_perimeter(mask)
        features['circularity'] = MorphologicalAnalyzer._compute_circularity(
            features['area'], features['perimeter']
        )
        features['compactness'] = MorphologicalAnalyzer._compute_compactness(
            features['area'], features['perimeter']
        )
        
        # Convex hull properties
        hull_features = MorphologicalAnalyzer._compute_convex_hull_features(mask)
        features.update(hull_features)
        
        # Moments and shape descriptors
        moments = MorphologicalAnalyzer._compute_moments(mask)
        features.update(moments)
        
        # Connectivity
        connectivity = MorphologicalAnalyzer._analyze_connectivity(mask)
        features.update(connectivity)
        
        # Texture (if large enough)
        if features['area'] > 100:
            texture = MorphologicalAnalyzer._analyze_texture(mask)
            features.update(texture)
        
        return features
    
    @staticmethod
    def _compute_bounding_box(mask):
        """Compute bounding box"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return {
            'min_row': int(rmin),
            'max_row': int(rmax),
            'min_col': int(cmin),
            'max_col': int(cmax),
            'height': int(rmax - rmin + 1),
            'width': int(cmax - cmin + 1)
        }
    
    @staticmethod
    def _compute_perimeter(mask):
        """Compute perimeter using contour detection"""
        # Dilate and subtract to get boundary
        dilated = ndimage.binary_dilation(mask)
        boundary = dilated & ~mask
        return np.sum(boundary)
    
    @staticmethod
    def _compute_circularity(area, perimeter):
        """Compute circularity (4π*area/perimeter²)"""
        if perimeter == 0:
            return 0
        return (4 * np.pi * area) / (perimeter ** 2)
    
    @staticmethod
    def _compute_compactness(area, perimeter):
        """Compute compactness (perimeter²/area)"""
        if area == 0:
            return 0
        return (perimeter ** 2) / area
    
    @staticmethod
    def _compute_convex_hull_features(mask):
        """Compute convex hull related features"""
        features = {}
        
        try:
            # Get coordinates
            coords = np.column_stack(np.where(mask))
            
            if len(coords) < 3:
                features['convex_area'] = np.sum(mask)
                features['solidity'] = 1.0
                features['convexity'] = 1.0
                return features
            
            # Compute convex hull
            hull = ConvexHull(coords)
            
            # Convex area (approximate)
            features['convex_area'] = hull.volume  # In 2D, volume = area
            
            # Solidity = area / convex_area
            area = np.sum(mask)
            features['solidity'] = area / features['convex_area'] if features['convex_area'] > 0 else 0
            
            # Convexity (ratio of perimeters)
            perimeter = MorphologicalAnalyzer._compute_perimeter(mask)
            convex_perimeter = hull.area  # In 2D, area = perimeter
            features['convexity'] = convex_perimeter / perimeter if perimeter > 0 else 0
            
        except Exception as e:
            print(f"Convex hull computation failed: {e}", file=sys.stderr)
            features['convex_area'] = np.sum(mask)
            features['solidity'] = 1.0
            features['convexity'] = 1.0
        
        return features
    
    @staticmethod
    def _compute_moments(mask):
        """Compute image moments for shape descriptors"""
        features = {}
        
        # Get coordinates
        coords = np.column_stack(np.where(mask))
        
        if len(coords) == 0:
            return features
        
        # Centroid
        centroid = np.mean(coords, axis=0)
        features['centroid_row'] = float(centroid[0])
        features['centroid_col'] = float(centroid[1])
        
        # Second moments for orientation
        coords_centered = coords - centroid
        
        # Moments
        mu20 = np.sum(coords_centered[:, 0] ** 2)
        mu02 = np.sum(coords_centered[:, 1] ** 2)
        mu11 = np.sum(coords_centered[:, 0] * coords_centered[:, 1])
        
        # Orientation (angle of major axis)
        if mu20 - mu02 == 0:
            features['orientation'] = 0
        else:
            features['orientation'] = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        
        # Eccentricity
        # Eigenvalues of covariance matrix
        a = mu20 + mu02
        b = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
        lambda1 = (a + b) / 2
        lambda2 = (a - b) / 2
        
        if lambda1 > 0:
            features['eccentricity'] = np.sqrt(1 - lambda2 / lambda1)
        else:
            features['eccentricity'] = 0
        
        # Aspect ratio (major/minor axis)
        if lambda2 > 0:
            features['aspect_ratio'] = np.sqrt(lambda1 / lambda2)
        else:
            features['aspect_ratio'] = 1.0
        
        return features
    
    @staticmethod
    def _analyze_connectivity(mask):
        """Analyze connectivity - number of connected components"""
        features = {}
        
        # Label connected components
        labeled, num_features = ndimage.label(mask)
        
        features['num_components'] = num_features
        
        # If multiple components, analyze them
        if num_features > 1:
            component_sizes = []
            for i in range(1, num_features + 1):
                component_sizes.append(np.sum(labeled == i))
            
            features['largest_component'] = max(component_sizes)
            features['smallest_component'] = min(component_sizes)
            features['mean_component_size'] = np.mean(component_sizes)
        else:
            features['largest_component'] = np.sum(mask)
            features['smallest_component'] = np.sum(mask)
            features['mean_component_size'] = np.sum(mask)
        
        return features
    
    @staticmethod
    def _analyze_texture(mask):
        """Analyze texture features"""
        features = {}
        
        # For now, simple roughness measure
        # Based on boundary irregularity
        
        # Erode and dilate to get internal and external boundaries
        eroded = ndimage.binary_erosion(mask)
        dilated = ndimage.binary_dilation(mask)
        
        internal_boundary = mask & ~eroded
        external_boundary = dilated & ~mask
        
        total_boundary = internal_boundary | external_boundary
        
        features['roughness'] = np.sum(total_boundary) / np.sum(mask) if np.sum(mask) > 0 else 0
        
        return features
    
    @staticmethod
    def compare_morphologies(features1, features2):
        """
        Compare two sets of morphological features
        
        Args:
            features1: Features dict for selection 1
            features2: Features dict for selection 2
            
        Returns:
            dict: Comparison metrics
        """
        comparison = {}
        
        # Numeric features to compare
        numeric_features = [
            'area', 'perimeter', 'circularity', 'compactness',
            'solidity', 'eccentricity', 'aspect_ratio',
            'num_components'
        ]
        
        for key in numeric_features:
            if key in features1 and key in features2:
                val1 = features1[key]
                val2 = features2[key]
                
                # Absolute difference
                comparison[f'{key}_diff'] = abs(val1 - val2)
                
                # Relative difference (%)
                if val1 > 0:
                    comparison[f'{key}_rel_diff'] = 100 * abs(val1 - val2) / val1
                else:
                    comparison[f'{key}_rel_diff'] = 0
        
        return comparison
