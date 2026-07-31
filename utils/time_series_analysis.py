"""
time_series_analysis.py - 4D Time Series Analysis

Track selections across timepoints
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys


class TimeSeriesAnalyzer:
    """
    Analyzes temporal evolution of selections in 4D data
    """
    
    def __init__(self):
        self.time_series_data = {}  # {selection_name: {timepoint: metrics}}
    
    def add_timepoint(self, timepoint, selections, neutron_vol, xray_vol):
        """
        Add data for a timepoint
        
        Args:
            timepoint: Timepoint index
            selections: List of Selection objects
            neutron_vol: 3D neutron volume
            xray_vol: 3D X-ray volume
        """
        print(f"Recording time series data for T={timepoint}", file=sys.stderr)
        
        for sel in selections:
            if sel.spatial_mask is None:
                continue
            
            # Initialize if needed
            if sel.name not in self.time_series_data:
                self.time_series_data[sel.name] = {}
            
            # Compute metrics for this timepoint
            # For simplicity, using current slice metrics
            # In production, would integrate over volume
            
            mask = sel.spatial_mask
            
            # Extract slice (Z-axis middle slice for consistency)
            slice_idx = neutron_vol.shape[0] // 2
            neutron_slice = neutron_vol[slice_idx, :, :]
            xray_slice = xray_vol[slice_idx, :, :]
            
            if mask.shape != neutron_slice.shape:
                continue
            
            n_pixels = np.sum(mask)
            
            if n_pixels > 0:
                n_vals = neutron_slice[mask]
                x_vals = xray_slice[mask]
                
                metrics = {
                    'pixels': n_pixels,
                    'neutron_mean': np.mean(n_vals),
                    'neutron_std': np.std(n_vals),
                    'xray_mean': np.mean(x_vals),
                    'xray_std': np.std(x_vals),
                    'ratio': np.mean(n_vals) / np.mean(x_vals) if np.mean(x_vals) > 0 else 0
                }
                
                self.time_series_data[sel.name][timepoint] = metrics
    
    def get_time_series(self, selection_name, metric='neutron_mean'):
        """
        Get time series for a selection
        
        Args:
            selection_name: Name of selection
            metric: Metric to extract
            
        Returns:
            (timepoints, values) tuples as numpy arrays
        """
        if selection_name not in self.time_series_data:
            return np.array([]), np.array([])
        
        data = self.time_series_data[selection_name]
        
        timepoints = sorted(data.keys())
        values = [data[t][metric] for t in timepoints if metric in data[t]]
        
        return np.array(timepoints), np.array(values)
    
    def plot_time_series(self, selection_names, metric='neutron_mean', 
                        title='Time Series', ylabel='Value'):
        """
        Plot time series for multiple selections
        
        Args:
            selection_names: List of selection names to plot
            metric: Metric to plot
            title: Plot title
            ylabel: Y-axis label
            
        Returns:
            fig, ax: Matplotlib figure and axis
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name in selection_names:
            timepoints, values = self.get_time_series(name, metric)
            
            if len(timepoints) > 0:
                ax.plot(timepoints, values, 'o-', label=name, linewidth=2, markersize=6)
        
        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def compute_trends(self, selection_name, metric='neutron_mean'):
        """
        Compute trends and statistics for a time series
        
        Args:
            selection_name: Name of selection
            metric: Metric to analyze
            
        Returns:
            dict: Trend statistics
        """
        timepoints, values = self.get_time_series(selection_name, metric)
        
        if len(values) < 2:
            return None
        
        trends = {}
        
        # Linear fit
        if len(timepoints) > 1:
            coeffs = np.polyfit(timepoints, values, 1)
            trends['slope'] = coeffs[0]
            trends['intercept'] = coeffs[1]
            trends['trend'] = 'increasing' if coeffs[0] > 0 else 'decreasing'
        
        # Statistics
        trends['mean'] = np.mean(values)
        trends['std'] = np.std(values)
        trends['min'] = np.min(values)
        trends['max'] = np.max(values)
        trends['range'] = np.max(values) - np.min(values)
        
        # Rate of change
        if len(values) > 1:
            changes = np.diff(values)
            trends['mean_change'] = np.mean(changes)
            trends['std_change'] = np.std(changes)
        
        return trends
    
    def detect_events(self, selection_name, metric='neutron_mean', 
                     threshold_pct=10):
        """
        Detect significant events (changes) in time series
        
        Args:
            selection_name: Name of selection
            metric: Metric to analyze
            threshold_pct: Percent change threshold for event detection
            
        Returns:
            list: List of detected events
        """
        timepoints, values = self.get_time_series(selection_name, metric)
        
        if len(values) < 2:
            return []
        
        events = []
        
        for i in range(1, len(values)):
            change = values[i] - values[i-1]
            pct_change = 100 * abs(change) / values[i-1] if values[i-1] != 0 else 0
            
            if pct_change > threshold_pct:
                events.append({
                    'timepoint': int(timepoints[i]),
                    'change': float(change),
                    'pct_change': float(pct_change),
                    'type': 'increase' if change > 0 else 'decrease'
                })
        
        return events
    
    def export_time_series_csv(self, filepath, selection_names=None):
        """
        Export time series data to CSV
        
        Args:
            filepath: Path to save CSV
            selection_names: List of selections to export (None = all)
        """
        import csv
        
        if selection_names is None:
            selection_names = list(self.time_series_data.keys())
        
        # Collect all timepoints
        all_timepoints = set()
        for name in selection_names:
            if name in self.time_series_data:
                all_timepoints.update(self.time_series_data[name].keys())
        
        timepoints = sorted(all_timepoints)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Timepoint']
            for name in selection_names:
                header.extend([
                    f'{name}_Pixels',
                    f'{name}_N_Mean',
                    f'{name}_N_Std',
                    f'{name}_X_Mean',
                    f'{name}_X_Std',
                    f'{name}_Ratio'
                ])
            writer.writerow(header)
            
            # Data rows
            for tp in timepoints:
                row = [tp]
                
                for name in selection_names:
                    if name in self.time_series_data and tp in self.time_series_data[name]:
                        data = self.time_series_data[name][tp]
                        row.extend([
                            data.get('pixels', ''),
                            f"{data.get('neutron_mean', 0):.1f}",
                            f"{data.get('neutron_std', 0):.1f}",
                            f"{data.get('xray_mean', 0):.1f}",
                            f"{data.get('xray_std', 0):.1f}",
                            f"{data.get('ratio', 0):.3f}"
                        ])
                    else:
                        row.extend([''] * 6)
                
                writer.writerow(row)
        
        print(f"Exported time series to {filepath}", file=sys.stderr)
    
    def clear(self):
        """Clear all time series data"""
        self.time_series_data = {}
