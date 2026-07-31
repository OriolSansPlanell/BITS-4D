"""
kmeans_3d.py - 3D K-means Clustering

K-means clustering on full 3D volumes
"""

import numpy as np
from sklearn.cluster import KMeans
import sys


class KMeans3D:
    """
    3D K-means clustering for volumetric bivariate data
    """
    
    @staticmethod
    def cluster_volume(neutron_vol, xray_vol, n_clusters=3, 
                      subsample_factor=1, random_state=42):
        """
        Perform K-means clustering on 3D volume
        
        Args:
            neutron_vol: 3D neutron array
            xray_vol: 3D X-ray array
            n_clusters: Number of clusters
            subsample_factor: Subsample for speed (1=no subsample, 2=every 2nd voxel, etc.)
            random_state: Random seed
            
        Returns:
            labels_3d: 3D array of cluster labels
            cluster_centers: (n_clusters, 2) array of centers
            stats: dict with clustering info
        """
        print(f"3D K-means clustering: {n_clusters} clusters, subsample={subsample_factor}", 
              file=sys.stderr)
        
        shape = neutron_vol.shape
        
        # Subsample if needed (for speed on large volumes)
        if subsample_factor > 1:
            neutron_sub = neutron_vol[::subsample_factor, ::subsample_factor, ::subsample_factor]
            xray_sub = xray_vol[::subsample_factor, ::subsample_factor, ::subsample_factor]
        else:
            neutron_sub = neutron_vol
            xray_sub = xray_vol
        
        # Flatten to 2D array for sklearn
        n_flat = neutron_sub.flatten()
        x_flat = xray_sub.flatten()
        
        # Create feature matrix
        features = np.column_stack([n_flat, x_flat])
        
        print(f"  Clustering {features.shape[0]:,} voxels...", file=sys.stderr)
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels_flat = kmeans.fit_predict(features)
        
        # Reshape to 3D
        labels_sub = labels_flat.reshape(neutron_sub.shape)
        
        # Upsample if needed
        if subsample_factor > 1:
            print(f"  Upsampling from {neutron_sub.shape} to {shape}...", file=sys.stderr)
            labels_3d = KMeans3D._upsample_labels(labels_sub, shape, subsample_factor)
        else:
            labels_3d = labels_sub
        
        # Statistics
        stats = {
            'n_clusters': n_clusters,
            'subsample_factor': subsample_factor,
            'total_voxels': np.prod(shape),
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
        
        # Cluster sizes
        for i in range(n_clusters):
            count = np.sum(labels_3d == i)
            pct = 100 * count / labels_3d.size
            stats[f'cluster_{i}_voxels'] = count
            stats[f'cluster_{i}_percent'] = pct
            print(f"  Cluster {i}: {count:,} voxels ({pct:.1f}%)", file=sys.stderr)
        
        return labels_3d, kmeans.cluster_centers_, stats
    
    @staticmethod
    def _upsample_labels(labels_sub, target_shape, factor):
        """
        Upsample cluster labels to full resolution using nearest neighbor
        
        Args:
            labels_sub: Subsampled 3D labels
            target_shape: Target shape tuple
            factor: Subsample factor
            
        Returns:
            Upsampled 3D labels
        """
        from scipy.ndimage import zoom
        
        # Compute zoom factors for each dimension
        zoom_factors = tuple(target_shape[i] / labels_sub.shape[i] for i in range(3))
        
        # Upsample using nearest neighbor (order=0)
        labels_full = zoom(labels_sub, zoom_factors, order=0)
        
        # Ensure exact target shape (zoom may be off by 1)
        if labels_full.shape != target_shape:
            # Crop or pad as needed
            slices = tuple(slice(0, min(labels_full.shape[i], target_shape[i])) 
                          for i in range(3))
            labels_cropped = labels_full[slices]
            
            # If too small, pad
            if labels_cropped.shape != target_shape:
                labels_padded = np.zeros(target_shape, dtype=labels_sub.dtype)
                labels_padded[slices] = labels_cropped
                return labels_padded
            
            return labels_cropped
        
        return labels_full
    
    @staticmethod
    def create_cluster_masks(labels_3d, n_clusters):
        """
        Create individual boolean masks for each cluster
        
        Args:
            labels_3d: 3D array of cluster labels
            n_clusters: Number of clusters
            
        Returns:
            List of 3D boolean masks, one per cluster
        """
        masks = []
        for i in range(n_clusters):
            mask = (labels_3d == i)
            masks.append(mask)
        
        return masks
    
    @staticmethod
    def extract_slice_from_labels(labels_3d, axis, slice_index):
        """
        Extract 2D slice from 3D labels
        
        Args:
            labels_3d: 3D array of cluster labels
            axis: 'z', 'y', or 'x'
            slice_index: Index of slice
            
        Returns:
            2D array of cluster labels
        """
        if axis == 'z':
            return labels_3d[slice_index, :, :]
        elif axis == 'y':
            return labels_3d[:, slice_index, :]
        else:  # x
            return labels_3d[:, :, slice_index]
    
    @staticmethod
    def get_cluster_statistics(labels_3d, cluster_id, neutron_vol, xray_vol):
        """
        Get statistics for a specific cluster
        
        Args:
            labels_3d: 3D cluster labels
            cluster_id: Which cluster
            neutron_vol: 3D neutron data
            xray_vol: 3D X-ray data
            
        Returns:
            dict: Statistics
        """
        mask = (labels_3d == cluster_id)
        
        if np.sum(mask) == 0:
            return None
        
        n_vals = neutron_vol[mask]
        x_vals = xray_vol[mask]
        
        stats = {
            'cluster_id': cluster_id,
            'voxels': np.sum(mask),
            'volume_fraction': np.sum(mask) / mask.size,
            'neutron_mean': np.mean(n_vals),
            'neutron_std': np.std(n_vals),
            'xray_mean': np.mean(x_vals),
            'xray_std': np.std(x_vals),
        }
        
        return stats
    
    @staticmethod
    def create_convex_hull_roi_3d(neutron_vals, xray_vals, percentile=98, 
                                  density_aware=True):
        """
        Create convex hull ROI from 3D cluster data
        
        Args:
            neutron_vals: 1D array of neutron values from cluster
            xray_vals: 1D array of X-ray values from cluster
            percentile: Percentile for outlier removal (higher = more inclusive)
            density_aware: If True, sample more from low-density regions
            
        Returns:
            ROI vertices array for histogram
        """
        from scipy.spatial import ConvexHull
        
        # For better coverage of low-density features, use higher percentile
        # and stratified sampling
        
        if density_aware and len(neutron_vals) > 10000:
            # Stratified sampling: oversample from edges of distribution
            # to ensure low-density features are represented
            
            # Compute 2D histogram to identify density
            hist_2d, n_edges, x_edges = np.histogram2d(
                neutron_vals, xray_vals, bins=50
            )
            
            # Normalize to get probability
            hist_2d = hist_2d / np.sum(hist_2d)
            
            # Compute which bin each point belongs to
            n_bins = np.digitize(neutron_vals, n_edges) - 1
            x_bins = np.digitize(xray_vals, x_edges) - 1
            
            # Clip to valid range
            n_bins = np.clip(n_bins, 0, len(n_edges) - 2)
            x_bins = np.clip(x_bins, 0, len(x_edges) - 2)
            
            # Get density for each point
            densities = hist_2d[n_bins, x_bins]
            
            # Inverse density weighting: sample more from low-density regions
            # Points with low density get higher weight
            weights = 1.0 / (densities + 1e-10)
            weights = weights / np.sum(weights)
            
            # Sample 5000 points with inverse density weighting
            n_samples = min(5000, len(neutron_vals))
            indices = np.random.choice(
                len(neutron_vals), 
                size=n_samples, 
                replace=False,
                p=weights
            )
            
            neutron_sampled = neutron_vals[indices]
            xray_sampled = xray_vals[indices]
            
        else:
            # Standard subsampling for small clusters
            if len(neutron_vals) > 5000:
                indices = np.random.choice(len(neutron_vals), 5000, replace=False)
                neutron_sampled = neutron_vals[indices]
                xray_sampled = xray_vals[indices]
            else:
                neutron_sampled = neutron_vals
                xray_sampled = xray_vals
        
        # Use wider percentile to include more features
        n_lower = np.percentile(neutron_sampled, 100 - percentile)
        n_upper = np.percentile(neutron_sampled, percentile)
        x_lower = np.percentile(xray_sampled, 100 - percentile)
        x_upper = np.percentile(xray_sampled, percentile)
        
        mask = ((neutron_sampled >= n_lower) & (neutron_sampled <= n_upper) &
                (xray_sampled >= x_lower) & (xray_sampled <= x_upper))
        
        points = np.column_stack([neutron_sampled[mask], xray_sampled[mask]])
        
        # Create convex hull
        try:
            if len(points) >= 3:
                hull = ConvexHull(points)
                vertices = points[hull.vertices]
                return vertices
            else:
                # Too few points, create rectangle
                n_min, n_max = np.min(neutron_sampled), np.max(neutron_sampled)
                x_min, x_max = np.min(xray_sampled), np.max(xray_sampled)
                return np.array([
                    [n_min, x_min], [n_max, x_min],
                    [n_max, x_max], [n_min, x_max]
                ])
        except (ValueError, RuntimeError, Exception) as e:
            # Fallback to rectangle if convex hull fails (degenerate points, etc.)
            n_min, n_max = np.min(neutron_sampled), np.max(neutron_sampled)
            x_min, x_max = np.min(xray_sampled), np.max(xray_sampled)
            return np.array([
                [n_min, x_min], [n_max, x_min],
                [n_max, x_max], [n_min, x_max]
            ])
