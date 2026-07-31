"""
selection_library.py - Selection Library Management

Handles saving/loading selections and exporting statistics
"""

import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime


class SelectionLibrary:
    """Manages selection persistence and export"""
    
    VERSION = "1.0"
    
    @staticmethod
    def save_selections(selections, filepath):
        """
        Save selections to .bits file
        
        Args:
            selections: List of Selection objects
            filepath: Path to save
            
        Returns:
            str: Path to saved file
        """
        print(f"Saving {len(selections)} selections...", file=sys.stderr)
        
        data = {
            'version': SelectionLibrary.VERSION,
            'created': datetime.now().isoformat(),
            'num_selections': len(selections),
            'selections': []
        }
        
        for sel in selections:
            sel_data = {
                'name': sel.name,
                'cluster_id': sel.cluster_id,
                'visible': sel.visible
            }
            
            # Spatial mask as nested list
            if sel.spatial_mask is not None:
                sel_data['spatial_mask'] = sel.spatial_mask.tolist()
            else:
                sel_data['spatial_mask'] = None
            
            # Histogram ROI
            if sel.histogram_roi is not None:
                sel_data['histogram_roi'] = sel.histogram_roi.tolist()
            else:
                sel_data['histogram_roi'] = None
            
            # Color
            if sel.color is not None:
                sel_data['color'] = list(sel.color) if isinstance(sel.color, tuple) else sel.color
            else:
                sel_data['color'] = None
            
            data['selections'].append(sel_data)
        
        # Ensure .bits extension
        filepath = Path(filepath)
        if filepath.suffix != '.bits':
            filepath = filepath.with_suffix('.bits')
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved to {filepath}", file=sys.stderr)
        return str(filepath)
    
    @staticmethod
    def load_selections(filepath):
        """
        Load selections from .bits file
        
        Args:
            filepath: Path to .bits file
            
        Returns:
            list: Selection data dictionaries
        """
        print(f"Loading selections from {filepath}...", file=sys.stderr)
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        selections_data = []
        for sel_data in data['selections']:
            # Reconstruct arrays
            spatial_mask = None
            if sel_data.get('spatial_mask') is not None:
                spatial_mask = np.array(sel_data['spatial_mask'], dtype=bool)
            
            histogram_roi = None
            if sel_data.get('histogram_roi') is not None:
                histogram_roi = np.array(sel_data['histogram_roi'])
            
            color = None
            if sel_data.get('color') is not None:
                color = tuple(sel_data['color'])
            
            selections_data.append({
                'name': sel_data['name'],
                'spatial_mask': spatial_mask,
                'histogram_roi': histogram_roi,
                'cluster_id': sel_data.get('cluster_id'),
                'color': color,
                'visible': sel_data.get('visible', True)
            })
        
        print(f"Loaded {len(selections_data)} selections", file=sys.stderr)
        return selections_data
    
    @staticmethod
    def export_statistics_csv(selections, neutron_data, xray_data, filepath):
        """
        Export statistics to CSV
        
        Args:
            selections: List of Selection objects
            neutron_data: 2D neutron array
            xray_data: 2D xray array  
            filepath: Path to save CSV
        """
        import csv
        
        filepath = Path(filepath)
        if filepath.suffix != '.csv':
            filepath = filepath.with_suffix('.csv')
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Selection', 'Cluster_ID', 'Pixels', 'Volume_%',
                'Neutron_Mean', 'Neutron_Std', 'Neutron_Min', 'Neutron_Max',
                'Xray_Mean', 'Xray_Std', 'Xray_Min', 'Xray_Max'
            ])
            
            # Data
            total_pixels = neutron_data.size
            skipped = 0
            
            for sel in selections:
                if sel.spatial_mask is not None and np.sum(sel.spatial_mask) > 0:
                    mask = sel.spatial_mask
                    
                    # Check dimension match
                    if mask.shape != neutron_data.shape:
                        print(f"  Skipping '{sel.name}': dimension mismatch "
                              f"(mask {mask.shape} vs data {neutron_data.shape})", file=sys.stderr)
                        skipped += 1
                        continue
                    
                    n_pix = np.sum(mask)
                    
                    n_vals = neutron_data[mask]
                    x_vals = xray_data[mask]
                    
                    writer.writerow([
                        sel.name,
                        sel.cluster_id if sel.cluster_id is not None else '',
                        n_pix,
                        f'{100*n_pix/total_pixels:.2f}',
                        f'{np.mean(n_vals):.1f}',
                        f'{np.std(n_vals):.1f}',
                        f'{np.min(n_vals):.1f}',
                        f'{np.max(n_vals):.1f}',
                        f'{np.mean(x_vals):.1f}',
                        f'{np.std(x_vals):.1f}',
                        f'{np.min(x_vals):.1f}',
                        f'{np.max(x_vals):.1f}'
                    ])
        
        if skipped > 0:
            print(f"Exported statistics to {filepath} ({skipped} selections skipped due to dimension mismatch)", file=sys.stderr)
        else:
            print(f"Exported statistics to {filepath}", file=sys.stderr)
        
        return str(filepath), skipped
    
    @staticmethod
    def export_statistics_excel(selections, neutron_data, xray_data, filepath):
        """Export statistics to Excel (requires pandas)"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required. Install: pip install pandas openpyxl")
        
        filepath = Path(filepath)
        if filepath.suffix not in ['.xlsx', '.xls']:
            filepath = filepath.with_suffix('.xlsx')
        
        # Compute stats
        stats = []
        total_pixels = neutron_data.size
        skipped = 0
        
        for sel in selections:
            if sel.spatial_mask is not None and np.sum(sel.spatial_mask) > 0:
                mask = sel.spatial_mask
                
                # Check dimension match
                if mask.shape != neutron_data.shape:
                    print(f"  Skipping '{sel.name}': dimension mismatch "
                          f"(mask {mask.shape} vs data {neutron_data.shape})", file=sys.stderr)
                    skipped += 1
                    continue
                
                n_pix = np.sum(mask)
                
                n_vals = neutron_data[mask]
                x_vals = xray_data[mask]
                
                stats.append({
                    'Selection': sel.name,
                    'Cluster_ID': sel.cluster_id if sel.cluster_id is not None else '',
                    'Pixels': n_pix,
                    'Volume_%': f'{100*n_pix/total_pixels:.2f}',
                    'Neutron_Mean': f'{np.mean(n_vals):.1f}',
                    'Neutron_Std': f'{np.std(n_vals):.1f}',
                    'Neutron_Min': f'{np.min(n_vals):.1f}',
                    'Neutron_Max': f'{np.max(n_vals):.1f}',
                    'Xray_Mean': f'{np.mean(x_vals):.1f}',
                    'Xray_Std': f'{np.std(x_vals):.1f}',
                    'Xray_Min': f'{np.min(x_vals):.1f}',
                    'Xray_Max': f'{np.max(x_vals):.1f}'
                })
        
        df = pd.DataFrame(stats)
        df.to_excel(filepath, index=False, engine='openpyxl')
        
        if skipped > 0:
            print(f"Exported to {filepath} ({skipped} selections skipped due to dimension mismatch)", file=sys.stderr)
        else:
            print(f"Exported to {filepath}", file=sys.stderr)
        
        return str(filepath), skipped
