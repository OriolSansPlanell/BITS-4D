"""
statistics_panel.py - Real-time Statistics Panel

Displays live statistics for selections and comparisons
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np
import sys


class StatisticsPanel(QWidget):
    """
    Panel showing real-time statistics for selections
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("📊 Statistics")
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(title)
        
        # Tabs for different views
        self.tabs = QTabWidget()
        
        # Tab 1: Quick Stats
        self.quick_stats_widget = self._create_quick_stats_widget()
        self.tabs.addTab(self.quick_stats_widget, "Quick Stats")
        
        # Tab 2: Detailed Table
        self.detail_table = self._create_detail_table()
        self.tabs.addTab(self.detail_table, "Detailed")
        
        # Tab 3: Comparison
        self.comparison_widget = self._create_comparison_widget()
        self.tabs.addTab(self.comparison_widget, "Compare")
        
        layout.addWidget(self.tabs)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Statistics")
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)
        
        self.setLayout(layout)
    
    def _create_quick_stats_widget(self):
        """Create quick statistics widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Labels for quick display
        self.total_label = QLabel("Total Selections: 0")
        self.visible_label = QLabel("Visible: 0")
        self.pixels_label = QLabel("Total Pixels: 0")
        self.coverage_label = QLabel("Coverage: 0.0%")
        
        font = QFont()
        font.setPointSize(10)
        
        for label in [self.total_label, self.visible_label, 
                     self.pixels_label, self.coverage_label]:
            label.setFont(font)
            layout.addWidget(label)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _create_detail_table(self):
        """Create detailed statistics table"""
        table = QTableWidget()
        table.setColumnCount(8)
        table.setHorizontalHeaderLabels([
            'Selection', 'Pixels', 'Vol%', 
            'N Mean', 'N Std', 'X Mean', 'X Std', 'N/X Ratio'
        ])
        
        # Resize columns
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 8):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        return table
    
    def _create_comparison_widget(self):
        """Create comparison widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        info = QLabel("Select 2 selections to compare")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info)
        
        # Comparison table
        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(3)
        self.compare_table.setHorizontalHeaderLabels(['Metric', 'Selection 1', 'Selection 2'])
        layout.addWidget(self.compare_table)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def update_statistics(self, selections, neutron_data, xray_data):
        """
        Update all statistics displays
        
        Args:
            selections: List of Selection objects
            neutron_data: 2D neutron array
            xray_data: 2D X-ray array
        """
        print(f"Updating statistics for {len(selections)} selections", file=sys.stderr)
        
        # Quick stats
        total = len(selections)
        visible = len([s for s in selections if s.visible])
        
        total_pixels = 0
        total_image_pixels = neutron_data.size
        
        # Detailed stats
        self.detail_table.setRowCount(0)
        
        valid_selections = []
        
        for sel in selections:
            if sel.spatial_mask is not None and sel.spatial_mask.shape == neutron_data.shape:
                mask = sel.spatial_mask
                n_pixels = np.sum(mask)
                
                if n_pixels > 0:
                    total_pixels += n_pixels
                    
                    # Compute statistics
                    n_vals = neutron_data[mask]
                    x_vals = xray_data[mask]
                    
                    stats = {
                        'name': sel.name,
                        'pixels': n_pixels,
                        'vol_pct': 100 * n_pixels / total_image_pixels,
                        'n_mean': np.mean(n_vals),
                        'n_std': np.std(n_vals),
                        'x_mean': np.mean(x_vals),
                        'x_std': np.std(x_vals),
                        'ratio': np.mean(n_vals) / np.mean(x_vals) if np.mean(x_vals) > 0 else 0
                    }
                    
                    valid_selections.append((sel, stats))
                    
                    # Add to table
                    row = self.detail_table.rowCount()
                    self.detail_table.insertRow(row)
                    
                    self.detail_table.setItem(row, 0, QTableWidgetItem(stats['name']))
                    self.detail_table.setItem(row, 1, QTableWidgetItem(f"{stats['pixels']:,}"))
                    self.detail_table.setItem(row, 2, QTableWidgetItem(f"{stats['vol_pct']:.2f}"))
                    self.detail_table.setItem(row, 3, QTableWidgetItem(f"{stats['n_mean']:.1f}"))
                    self.detail_table.setItem(row, 4, QTableWidgetItem(f"{stats['n_std']:.1f}"))
                    self.detail_table.setItem(row, 5, QTableWidgetItem(f"{stats['x_mean']:.1f}"))
                    self.detail_table.setItem(row, 6, QTableWidgetItem(f"{stats['x_std']:.1f}"))
                    self.detail_table.setItem(row, 7, QTableWidgetItem(f"{stats['ratio']:.3f}"))
        
        # Update quick stats
        coverage = 100 * total_pixels / total_image_pixels if total_image_pixels > 0 else 0
        
        self.total_label.setText(f"Total Selections: {total}")
        self.visible_label.setText(f"Visible: {visible}")
        self.pixels_label.setText(f"Total Pixels: {total_pixels:,}")
        self.coverage_label.setText(f"Coverage: {coverage:.1f}%")
        
        # Store for comparison
        self.valid_selections = valid_selections
        
        print(f"  Updated {len(valid_selections)} valid selections", file=sys.stderr)
    
    def compare_selections(self, sel1_name, sel2_name):
        """
        Compare two selections
        
        Args:
            sel1_name: Name of first selection
            sel2_name: Name of second selection
        """
        # Find selections
        sel1_stats = None
        sel2_stats = None
        
        for sel, stats in self.valid_selections:
            if sel.name == sel1_name:
                sel1_stats = stats
            if sel.name == sel2_name:
                sel2_stats = stats
        
        if sel1_stats is None or sel2_stats is None:
            return
        
        # Populate comparison table
        self.compare_table.setRowCount(0)
        
        metrics = [
            ('Pixels', 'pixels', '{:,}'),
            ('Volume %', 'vol_pct', '{:.2f}'),
            ('Neutron Mean', 'n_mean', '{:.1f}'),
            ('Neutron Std', 'n_std', '{:.1f}'),
            ('X-ray Mean', 'x_mean', '{:.1f}'),
            ('X-ray Std', 'x_std', '{:.1f}'),
            ('N/X Ratio', 'ratio', '{:.3f}')
        ]
        
        for label, key, fmt in metrics:
            row = self.compare_table.rowCount()
            self.compare_table.insertRow(row)
            
            self.compare_table.setItem(row, 0, QTableWidgetItem(label))
            self.compare_table.setItem(row, 1, QTableWidgetItem(fmt.format(sel1_stats[key])))
            self.compare_table.setItem(row, 2, QTableWidgetItem(fmt.format(sel2_stats[key])))
    
    def refresh(self):
        """Refresh signal (to be connected to main window)"""
        print("Statistics refresh requested", file=sys.stderr)
