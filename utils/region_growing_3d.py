"""
region_growing_3d.py - 3D Region Growing

Extends region growing to work on full 3D volumes
"""

import numpy as np
from collections import deque
import sys


class RegionGrowing3D:
    """
    3D Region Growing for volumetric data
    """
    
    @staticmethod
    def grow_region_univariate_3d(volume, seed_z, seed_y, seed_x, tolerance):
        """
        Grow region in 3D using single channel
        
        Args:
            volume: 3D array (Z, Y, X)
            seed_z: Z coordinate of seed point
            seed_y: Y coordinate of seed point
            seed_x: X coordinate of seed point
            tolerance: Intensity tolerance
            
        Returns:
            3D boolean mask
        """
        print(f"3D univariate region growing from ({seed_z}, {seed_y}, {seed_x})", 
              file=sys.stderr)
        
        shape = volume.shape
        mask = np.zeros(shape, dtype=bool)
        visited = np.zeros(shape, dtype=bool)
        
        # Seed value
        seed_value = volume[seed_z, seed_y, seed_x]
        
        # Queue for BFS
        queue = deque([(seed_z, seed_y, seed_x)])
        visited[seed_z, seed_y, seed_x] = True
        
        # 6-connectivity (face neighbors only for efficiency)
        neighbors = [
            (-1, 0, 0), (1, 0, 0),  # Z
            (0, -1, 0), (0, 1, 0),  # Y
            (0, 0, -1), (0, 0, 1)   # X
        ]
        
        count = 0
        
        while queue:
            z, y, x = queue.popleft()
            
            # Check if within tolerance
            if abs(volume[z, y, x] - seed_value) <= tolerance:
                mask[z, y, x] = True
                count += 1
                
                # Add neighbors
                for dz, dy, dx in neighbors:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    
                    # Check bounds
                    if (0 <= nz < shape[0] and 
                        0 <= ny < shape[1] and 
                        0 <= nx < shape[2]):
                        
                        if not visited[nz, ny, nx]:
                            visited[nz, ny, nx] = True
                            queue.append((nz, ny, nx))
        
        print(f"  3D region: {count} voxels", file=sys.stderr)
        return mask
    
    @staticmethod
    def grow_region_bivariate_3d(neutron_vol, xray_vol, 
                                 seed_z, seed_y, seed_x,
                                 neutron_tol, xray_tol):
        """
        Grow region in 3D using both channels
        
        Args:
            neutron_vol: 3D neutron array
            xray_vol: 3D X-ray array
            seed_z, seed_y, seed_x: Seed coordinates
            neutron_tol: Neutron tolerance
            xray_tol: X-ray tolerance
            
        Returns:
            3D boolean mask
        """
        print(f"3D bivariate region growing from ({seed_z}, {seed_y}, {seed_x})", 
              file=sys.stderr)
        
        shape = neutron_vol.shape
        mask = np.zeros(shape, dtype=bool)
        visited = np.zeros(shape, dtype=bool)
        
        # Seed values
        seed_n = neutron_vol[seed_z, seed_y, seed_x]
        seed_x_val = xray_vol[seed_z, seed_y, seed_x]
        
        # Queue
        queue = deque([(seed_z, seed_y, seed_x)])
        visited[seed_z, seed_y, seed_x] = True
        
        # 6-connectivity
        neighbors = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)
        ]
        
        count = 0
        
        while queue:
            z, y, x = queue.popleft()
            
            # Check both channels
            n_val = neutron_vol[z, y, x]
            x_val = xray_vol[z, y, x]
            
            if (abs(n_val - seed_n) <= neutron_tol and 
                abs(x_val - seed_x_val) <= xray_tol):
                
                mask[z, y, x] = True
                count += 1
                
                # Add neighbors
                for dz, dy, dx in neighbors:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    
                    if (0 <= nz < shape[0] and 
                        0 <= ny < shape[1] and 
                        0 <= nx < shape[2]):
                        
                        if not visited[nz, ny, nx]:
                            visited[nz, ny, nx] = True
                            queue.append((nz, ny, nx))
        
        print(f"  3D bivariate region: {count} voxels", file=sys.stderr)
        return mask
    
    @staticmethod
    def bivariate_region_growing_3d(neutron_vol, xray_vol, seed_3d, 
                                    neutron_tol, xray_tol, connectivity=1):
        """
        Wrapper for bivariate 3D region growing with connectivity parameter
        
        Args:
            neutron_vol: 3D neutron array
            xray_vol: 3D X-ray array
            seed_3d: Tuple (z, y, x) seed coordinates
            neutron_tol: Neutron tolerance
            xray_tol: X-ray tolerance
            connectivity: Not used (always 6-connected), kept for compatibility
            
        Returns:
            3D boolean mask
        """
        seed_z, seed_y, seed_x = seed_3d
        return RegionGrowing3D.grow_region_bivariate_3d(
            neutron_vol, xray_vol,
            seed_z, seed_y, seed_x,
            neutron_tol, xray_tol
        )
    
    @staticmethod
    def univariate_region_growing_3d(volume, seed_3d, tolerance, connectivity=1):
        """
        Wrapper for univariate 3D region growing with connectivity parameter
        
        Args:
            volume: 3D array
            seed_3d: Tuple (z, y, x) seed coordinates
            tolerance: Intensity tolerance
            connectivity: Not used (always 6-connected), kept for compatibility
            
        Returns:
            3D boolean mask
        """
        seed_z, seed_y, seed_x = seed_3d
        return RegionGrowing3D.grow_region_univariate_3d(
            volume,
            seed_z, seed_y, seed_x,
            tolerance
        )
    
    @staticmethod
    def extract_slice_from_3d_mask(mask_3d, axis, slice_index):
        """
        Extract 2D slice from 3D mask for visualization
        
        Args:
            mask_3d: 3D boolean array
            axis: 'z', 'y', or 'x'
            slice_index: Index of slice to extract
            
        Returns:
            2D boolean mask
        """
        if axis == 'z':
            return mask_3d[slice_index, :, :]
        elif axis == 'y':
            return mask_3d[:, slice_index, :]
        else:  # x
            return mask_3d[:, :, slice_index]
    
    @staticmethod
    def get_3d_statistics(mask_3d, neutron_vol, xray_vol):
        """
        Compute statistics for 3D mask
        
        Args:
            mask_3d: 3D boolean mask
            neutron_vol: 3D neutron volume
            xray_vol: 3D X-ray volume
            
        Returns:
            dict: Statistics
        """
        if np.sum(mask_3d) == 0:
            return None
        
        n_vals = neutron_vol[mask_3d]
        x_vals = xray_vol[mask_3d]
        
        stats = {
            'voxels': np.sum(mask_3d),
            'volume_fraction': np.sum(mask_3d) / mask_3d.size,
            'neutron_mean': np.mean(n_vals),
            'neutron_std': np.std(n_vals),
            'neutron_min': np.min(n_vals),
            'neutron_max': np.max(n_vals),
            'xray_mean': np.mean(x_vals),
            'xray_std': np.std(x_vals),
            'xray_min': np.min(x_vals),
            'xray_max': np.max(x_vals),
        }
        
        return stats
