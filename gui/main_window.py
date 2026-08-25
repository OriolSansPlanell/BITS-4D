"""
main_window.py - Main Application Window for BiTS 4D

Integrates all components with progress feedback for long operations
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QFileDialog, QMessageBox, QStatusBar, QProgressBar, QAction,
    QMenu, QMenuBar, QToolBar, QSplitter, QApplication, QCheckBox, QDoubleSpinBox,
    QDialog, QDialogButtonBox, QScrollArea, QFrame, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import Dataset4D, TIFF4DLoader
from histograms import HistogramEngine4D
from segmentation import SegmentationEngine4D
from utils import (
    run_with_progress, show_loading_message, ProgressCallback,
    config
)
from gui.time_navigation_widget import TimeNavigationWidget
from gui.dual_histogram_widget import DualHistogramWidget
from gui.flow_layout import FlowLayout
from gui.material_panel import MaterialPanel, describe_strength
import matplotlib.colors as _mcolors
from utils.roi_manager import CLASS_COLORS

#: Slice-viewer highlights are the class palette at a fixed transparency.
#: Derived, not a second list — see BiTS4DMainWindow._colour_for_layer.
#: How small the slice canvas may get. See gui/dual_histogram_widget.py —
#: these minimums add up to the window's own minimum width.
SLICE_CANVAS_MINIMUM = (240, 200)

OVERLAY_ALPHA = 0.50
OVERLAY_COLORS = [
    _mcolors.to_rgba(hex_colour, OVERLAY_ALPHA) for hex_colour in CLASS_COLORS
]


class SliceViewerWidget(QWidget):
    """
    Slice viewer widget with segmentation overlay and axis selection
    """
    
    # Signal emitted when user wants to create histogram ROI from spatial selection
    # Arguments: (spatial_coords, axis, slice_index)
    spatial_roi_to_histogram = pyqtSignal(tuple, str, int)
    
    # Signal emitted when clusters are detected (for selection manager)
    # Arguments: (list of (name, spatial_mask, histogram_roi, cluster_id, color))
    clusters_detected = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_slice_data = None
        self.segmentation_mask = None
        self.current_axis = 'z'  # 'z', 'y', or 'x'
        self.current_slice_index = None
        self.volume_shape = None
        self.view_mode = 'neutron'  # 'neutron' or 'xray'
        self.vmin = None  # Dynamic range
        self.vmax = None
        # >1 when the displayed volumes are median-binned copies of the data
        self.display_bin_factor = 1

        # Debounce slice-slider updates: while dragging, only the label
        # follows instantly; the full redraw runs once the slider settles.
        from PyQt5.QtCore import QTimer
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(30)
        self._redraw_timer.timeout.connect(self._update_display)

        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls for axis selection. Flow layouts throughout: these rows
        # are long, and a non-wrapping row makes its width the window's
        # minimum width.
        controls_layout = FlowLayout()

        controls_layout.addWidget(QLabel("View Axis:"))
        
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QSlider
        from PyQt5.QtCore import Qt
        
        self.axis_group = QButtonGroup()
        
        self.z_axis_btn = QRadioButton("XY (Z-slice)")
        self.z_axis_btn.setChecked(True)
        self.z_axis_btn.toggled.connect(lambda: self._on_axis_changed('z'))
        self.axis_group.addButton(self.z_axis_btn)
        controls_layout.addWidget(self.z_axis_btn)
        
        self.y_axis_btn = QRadioButton("XZ (Y-slice)")
        self.y_axis_btn.toggled.connect(lambda: self._on_axis_changed('y'))
        self.axis_group.addButton(self.y_axis_btn)
        controls_layout.addWidget(self.y_axis_btn)
        
        self.x_axis_btn = QRadioButton("YZ (X-slice)")
        self.x_axis_btn.toggled.connect(lambda: self._on_axis_changed('x'))
        self.axis_group.addButton(self.x_axis_btn)
        controls_layout.addWidget(self.x_axis_btn)
        
        controls_layout.addStretch()
        
        # Slice index slider
        controls_layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        self.slice_slider.setEnabled(False)
        controls_layout.addWidget(self.slice_slider)
        
        self.slice_label = QLabel("0")
        controls_layout.addWidget(self.slice_label)
        
        controls_layout.addStretch()
        
        # View mode selection
        controls_layout.addWidget(QLabel("Data:"))
        
        self.view_group = QButtonGroup()
        
        self.neutron_view_btn = QRadioButton("Neutron")
        self.neutron_view_btn.setChecked(True)
        self.neutron_view_btn.toggled.connect(lambda: self._on_view_mode_changed('neutron'))
        self.view_group.addButton(self.neutron_view_btn)
        controls_layout.addWidget(self.neutron_view_btn)
        
        self.xray_view_btn = QRadioButton("X-ray")
        self.xray_view_btn.toggled.connect(lambda: self._on_view_mode_changed('xray'))
        self.view_group.addButton(self.xray_view_btn)
        controls_layout.addWidget(self.xray_view_btn)
        
        controls_layout.addStretch()
        
        # Dynamic range controls
        from PyQt5.QtWidgets import QDoubleSpinBox
        
        controls_layout.addWidget(QLabel("Range:"))
        self.slice_vmin = QDoubleSpinBox()
        self.slice_vmin.setRange(0, 1e10)
        self.slice_vmin.setValue(0)
        self.slice_vmin.setPrefix("Min: ")
        self.slice_vmin.valueChanged.connect(self._on_range_changed)
        controls_layout.addWidget(self.slice_vmin)
        
        self.slice_vmax = QDoubleSpinBox()
        self.slice_vmax.setRange(0, 1e10)
        self.slice_vmax.setValue(65535)
        self.slice_vmax.setPrefix("Max: ")
        self.slice_vmax.valueChanged.connect(self._on_range_changed)
        controls_layout.addWidget(self.slice_vmax)
        
        slice_auto_btn = QPushButton("Auto Range")
        slice_auto_btn.clicked.connect(self._auto_range_slice)
        controls_layout.addWidget(slice_auto_btn)
        
        layout.addLayout(controls_layout)

        # Spatial ROI tools
        spatial_roi_layout = FlowLayout()

        spatial_roi_layout.addWidget(QLabel("Spatial Selection:"))
        
        self.rect_tool_btn = QPushButton("□ Rectangle")
        self.rect_tool_btn.setCheckable(True)
        self.rect_tool_btn.setToolTip("Draw rectangle on slice to select spatial region")
        self.rect_tool_btn.clicked.connect(self._on_rect_tool_clicked)
        spatial_roi_layout.addWidget(self.rect_tool_btn)
        
        self.region_grow_btn = QPushButton("🪄 Region Grow")
        self.region_grow_btn.setCheckable(True)
        self.region_grow_btn.setToolTip("Click a seed point to grow connected region")
        self.region_grow_btn.clicked.connect(self._on_region_grow_clicked)
        spatial_roi_layout.addWidget(self.region_grow_btn)
        
        # Bivariate mode checkbox
        self.bivariate_cb = QCheckBox("Bivariate")
        self.bivariate_cb.setToolTip("Use both neutron AND X-ray values (more selective)")
        self.bivariate_cb.setChecked(False)
        self.bivariate_cb.stateChanged.connect(self._on_bivariate_changed)
        self.bivariate_cb.setEnabled(False)
        spatial_roi_layout.addWidget(self.bivariate_cb)
        
        # 3D mode checkbox (NEW for v15.0)
        self.mode_3d_cb = QCheckBox("3D Volume")
        self.mode_3d_cb.setToolTip("Apply to entire 3D volume instead of current slice")
        self.mode_3d_cb.setChecked(False)
        self.mode_3d_cb.setEnabled(False)
        spatial_roi_layout.addWidget(self.mode_3d_cb)
        
        # Tolerance control for region growing
        spatial_roi_layout.addWidget(QLabel("Tol:"))
        self.tolerance_spinbox = QDoubleSpinBox()
        self.tolerance_spinbox.setRange(1, 10000)
        self.tolerance_spinbox.setValue(1000)
        self.tolerance_spinbox.setSingleStep(100)
        self.tolerance_spinbox.setToolTip("Intensity tolerance for region growing")
        self.tolerance_spinbox.setEnabled(False)
        self.tolerance_spinbox.setMaximumWidth(80)
        spatial_roi_layout.addWidget(self.tolerance_spinbox)
        
        # Second tolerance (for bivariate mode)
        self.tolerance2_label = QLabel("Tol2:")
        self.tolerance2_label.setToolTip("Tolerance for other modality (bivariate mode)")
        self.tolerance2_label.setVisible(False)
        spatial_roi_layout.addWidget(self.tolerance2_label)
        
        self.tolerance2_spinbox = QDoubleSpinBox()
        self.tolerance2_spinbox.setRange(1, 10000)
        self.tolerance2_spinbox.setValue(1000)
        self.tolerance2_spinbox.setSingleStep(100)
        self.tolerance2_spinbox.setToolTip("Tolerance for other modality (bivariate mode)")
        self.tolerance2_spinbox.setEnabled(False)
        self.tolerance2_spinbox.setVisible(False)
        self.tolerance2_spinbox.setMaximumWidth(80)
        spatial_roi_layout.addWidget(self.tolerance2_spinbox)
        
        # Show/hide mask toggle
        self.show_mask_cb = QCheckBox("Show Mask")
        self.show_mask_cb.setChecked(True)
        self.show_mask_cb.setToolTip("Toggle region growing mask overlay")
        self.show_mask_cb.stateChanged.connect(self._on_show_mask_changed)
        self.show_mask_cb.setEnabled(False)
        spatial_roi_layout.addWidget(self.show_mask_cb)
        
        # Auto-detect features button
        self.auto_detect_btn = QPushButton("🔍 Auto-Detect")
        self.auto_detect_btn.setToolTip("Automatically detect features in slice")
        self.auto_detect_btn.clicked.connect(self._on_auto_detect)
        spatial_roi_layout.addWidget(self.auto_detect_btn)
        
        self.clear_spatial_roi_btn = QPushButton("✕ Clear")
        self.clear_spatial_roi_btn.setToolTip("Clear spatial ROI")
        self.clear_spatial_roi_btn.clicked.connect(self._clear_spatial_roi)
        self.clear_spatial_roi_btn.setEnabled(False)
        spatial_roi_layout.addWidget(self.clear_spatial_roi_btn)

        self.clear_highlight_btn = QPushButton("🧹 Clear Highlight")
        self.clear_highlight_btn.setToolTip(
            "Remove all coloured overlays from the slice view\n"
            "(region-grow mask and histogram selection highlights).\n"
            "Useful when switching between manual ROI selections."
        )
        self.clear_highlight_btn.setEnabled(False)
        self.clear_highlight_btn.clicked.connect(self._clear_highlight)
        spatial_roi_layout.addWidget(self.clear_highlight_btn)
        
        spatial_roi_layout.addStretch()
        
        self.create_hist_roi_btn = QPushButton("→ Create Histogram ROI from Selection")
        self.create_hist_roi_btn.setToolTip("Extract values from spatial ROI and create histogram ROI")
        self.create_hist_roi_btn.clicked.connect(self._create_histogram_roi_from_spatial)
        self.create_hist_roi_btn.setEnabled(False)
        spatial_roi_layout.addWidget(self.create_hist_roi_btn)
        
        layout.addLayout(spatial_roi_layout)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(*SLICE_CANVAS_MINIMUM)
        layout.addWidget(self.canvas, stretch=1)
        
        # Initialize spatial ROI state
        self.spatial_roi_selector = None
        self.spatial_roi_coords = None
        
        # Region growing state
        self.region_grow_mode = False
        self.region_grow_mask = None
        self.region_grow_mask_3d = None  # For 3D region growing (v15.0)
        self.mask_overlay = None  # Matplotlib artist for mask display
        
        # Multiple mask overlays (segmentation layers + selection manager).
        # Masks may be 3-D (whole-volume layers, re-sliced on every redraw so
        # the highlight follows the plane/slice) or 2-D (single-slice
        # selections, shown only on a matching slice).
        self.mask_overlays = []  # List of (name, mask, color) tuples
        self.overlay_artists = []  # Matplotlib artists for overlays
        self._visible_mask_pixels = 0  # Highlighted pixels on the current slice
        
        # Auto-detected features
        self.detected_features = []  # List of (y, x) coordinates
        self.feature_markers = []  # List of matplotlib artists for feature markers
        self.cluster_map = None  # 2D cluster labels (from k-means)
        self.cluster_map_3d = None  # 3D cluster labels (v15.0)
        self.cluster_centers = None  # Cluster centers
        self.num_clusters = 0  # Number of clusters
        
        # Zoom state
        self.zoom_level = 1.0
        self.zoom_center = None  # (x, y) in data coordinates
        
        # Info label
        self.info_label = QLabel("Load dataset to view slices")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Pixel value indicator
        self.pixel_value_label = QLabel("Pixel value: -- | Position: (---, ---)")
        self.pixel_value_label.setAlignment(Qt.AlignCenter)
        self.pixel_value_label.setStyleSheet("QLabel { color: #666; font-size: 10pt; }")
        layout.addWidget(self.pixel_value_label)
        
        self.setLayout(layout)
        self._setup_plot()
    
    def _setup_plot(self):
        """Setup the plot appearance"""
        self.ax.set_title("Volume Slice")
        self.ax.axis('off')
        
        # Connect motion event to show pixel values
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        # Connect scroll event for zoom
        self.cid_scroll = self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)
    
    def _on_mouse_move(self, event):
        """Display pixel value under cursor"""
        if event.inaxes != self.ax:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        if not hasattr(self, 'current_slice') or self.current_slice is None:
            return
        
        # Get pixel coordinates
        x_pixel = int(round(event.xdata))
        y_pixel = int(round(event.ydata))
        
        # Check bounds
        if (x_pixel >= 0 and x_pixel < self.current_slice.shape[1] and
            y_pixel >= 0 and y_pixel < self.current_slice.shape[0]):
            
            # Get pixel value for displayed modality
            pixel_value = self.current_slice[y_pixel, x_pixel]
            
            # Check if bivariate mode is enabled
            if hasattr(self, 'bivariate_cb') and self.bivariate_cb.isChecked():
                # Show BOTH neutron and X-ray values
                if self.current_slice_data is not None:
                    neutron_vol, xray_vol = self.current_slice_data
                    
                    # Extract both slices
                    if self.current_axis == 'z':
                        neutron_val = neutron_vol[self.current_slice_index, y_pixel, x_pixel]
                        xray_val = xray_vol[self.current_slice_index, y_pixel, x_pixel]
                    elif self.current_axis == 'y':
                        neutron_val = neutron_vol[y_pixel, self.current_slice_index, x_pixel]
                        xray_val = xray_vol[y_pixel, self.current_slice_index, x_pixel]
                    else:  # 'x'
                        neutron_val = neutron_vol[y_pixel, x_pixel, self.current_slice_index]
                        xray_val = xray_vol[y_pixel, x_pixel, self.current_slice_index]
                    
                    # Display both values
                    self.pixel_value_label.setText(
                        f"Neutron: {neutron_val:.0f} | X-ray: {xray_val:.0f} | Pos: ({x_pixel}, {y_pixel})"
                    )
                else:
                    # Fallback if slice data not available
                    self.pixel_value_label.setText(
                        f"Pixel value: {pixel_value:.0f} | Position: ({x_pixel}, {y_pixel})"
                    )
            else:
                # Show only current modality
                modality = "Neutron" if self.view_mode == 'neutron' else "X-ray"
                self.pixel_value_label.setText(
                    f"{modality}: {pixel_value:.0f} | Position: ({x_pixel}, {y_pixel})"
                )
    
    def _on_scroll_zoom(self, event):
        """Handle mouse wheel zoom"""
        if event.inaxes != self.ax:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Get cursor position
        xdata = event.xdata
        ydata = event.ydata
        
        # Zoom factor
        zoom_factor = 1.2 if event.button == 'up' else 1/1.2
        
        # Calculate new limits centered on cursor
        x_range = (xlim[1] - xlim[0]) / zoom_factor
        y_range = (ylim[1] - ylim[0]) / zoom_factor
        
        # Center on cursor position
        x_center_ratio = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        y_center_ratio = (ydata - ylim[0]) / (ylim[1] - ylim[0])
        
        new_xlim = [xdata - x_range * x_center_ratio, 
                    xdata + x_range * (1 - x_center_ratio)]
        new_ylim = [ydata - y_range * y_center_ratio,
                    ydata + y_range * (1 - y_center_ratio)]
        
        # Apply limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        # Update zoom level
        if hasattr(self, 'current_slice') and self.current_slice is not None:
            full_width = self.current_slice.shape[1]
            current_width = new_xlim[1] - new_xlim[0]
            self.zoom_level = full_width / current_width
        
        # Redraw
        self.canvas.draw_idle()
    
    def _on_auto_detect(self):
        """Automatically detect features using histogram clustering + spatial density"""
        import sys
        from PyQt5.QtWidgets import QInputDialog, QMessageBox, QDialog, QVBoxLayout, QLabel, QRadioButton, QDialogButtonBox
        
        print("Auto-detecting features using histogram clustering...", file=sys.stderr)
        
        if self.current_slice is None or self.current_slice_data is None:
            QMessageBox.warning(self, "No Data", "No slice data available")
            return
        
        # Check if 3D mode is possible
        is_3d_available = self.mode_3d_cb.isChecked()
        
        # If 3D mode, offer choice
        if is_3d_available:
            dialog = QDialog(self)
            dialog.setWindowTitle("Clustering Mode")
            layout = QVBoxLayout()
            
            layout.addWidget(QLabel("Choose clustering mode:"))
            
            mode_3d = QRadioButton("3D Volume Clustering (Comprehensive)")
            mode_3d.setToolTip("Clusters entire 3D volume - best for connected features")
            
            mode_2d = QRadioButton("2D Slice Clustering (Current slice only)")
            mode_2d.setToolTip("Clusters current slice - better for low-density features")
            
            mode_hybrid = QRadioButton("Hybrid Mode (Recommended)")
            mode_hybrid.setToolTip("Runs both 2D and 3D clustering, keeps unique ROIs from each")
            mode_hybrid.setChecked(True)  # Default to hybrid
            
            layout.addWidget(mode_3d)
            layout.addWidget(mode_2d)
            layout.addWidget(mode_hybrid)
            
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() != QDialog.Accepted:
                return
            
            # Determine mode
            if mode_hybrid.isChecked():
                self._auto_detect_hybrid()
            elif mode_3d.isChecked():
                self._auto_detect_3d()
            else:
                self._auto_detect_2d()
        else:
            # 2D mode only
            self._auto_detect_2d()
    
    def _auto_detect_3d(self):
        """3D k-means clustering on entire volume"""
        import sys
        from utils.clustering_3d import KMeans3D
        from utils.region_growing_3d import RegionGrowing3D
        from PyQt5.QtWidgets import QInputDialog
        import numpy as np
        import matplotlib.cm as cm
        
        print("Starting 3D k-means clustering on volume...", file=sys.stderr)
        
        # Get volumes
        neutron_vol, xray_vol = self.current_slice_data
        
        # Ask user for number of clusters
        num_clusters, ok = QInputDialog.getInt(
            self, "3D Histogram Clustering", 
            "Number of clusters/regions to detect (3D):",
            5, 2, 20, 1
        )
        
        if not ok:
            return
        
        try:
            # Show progress
            from utils.progress_dialog import ProgressDialog
            progress = ProgressDialog(
                "3D Clustering",
                "Computing k-means on 3D volume...",
                0,  # Indeterminate
                self
            )
            progress.show()
            QApplication.processEvents()
            
            # Perform 3D k-means
            labels_3d, centers, cluster_stats = KMeans3D.cluster_volume(
                neutron_vol,
                xray_vol,
                n_clusters=num_clusters
            )
            
            progress.close()
            
            # Store 3D cluster information
            self.cluster_map_3d = labels_3d
            self.cluster_centers = centers
            self.num_clusters = num_clusters
            
            # Extract 2D slice for display
            self.cluster_map = RegionGrowing3D.extract_slice_from_3d_mask(
                labels_3d,
                self.current_axis,
                self.current_slice_index
            )
            
            # Create cluster selections
            cluster_selections = []
            cmap = cm.get_cmap('tab10')
            
            for cluster_id in range(num_clusters):
                # Get 3D mask
                mask_3d = (labels_3d == cluster_id)
                
                # Get 2D slice mask
                mask_2d = RegionGrowing3D.extract_slice_from_3d_mask(
                    mask_3d,
                    self.current_axis,
                    self.current_slice_index
                )
                
                # Get ROI for histogram
                neutron_vals = neutron_vol[mask_3d]
                xray_vals = xray_vol[mask_3d]
                
                roi_vertices = KMeans3D.create_convex_hull_roi_3d(
                    neutron_vals,
                    xray_vals
                )
                
                color = cmap(cluster_id / num_clusters)
                
                cluster_selections.append((
                    f"3D Cluster {cluster_id}",
                    mask_2d,  # 2D for display
                    roi_vertices,
                    cluster_id,
                    color,
                    mask_3d  # Store 3D mask too
                ))
            
            # Emit signal with clusters
            self.clusters_detected.emit(cluster_selections)
            
            QMessageBox.information(
                self,
                "3D Clustering Complete",
                f"Detected {num_clusters} clusters in 3D volume.\n\n"
                f"Clusters saved to Selection Manager.\n"
                f"Each selection represents the entire 3D cluster.\n\n"
                f"💡 Tip: Use histogram polygon tool to manually add\n"
                f"ROIs for low-density features that were missed."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"3D clustering failed:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _auto_detect_hybrid(self):
        """Hybrid mode: Run both 2D and 3D clustering, merge unique ROIs"""
        import sys
        from PyQt5.QtWidgets import QInputDialog
        
        print("Starting HYBRID clustering (2D + 3D)...", file=sys.stderr)
        
        # Ask for number of clusters
        num_clusters, ok = QInputDialog.getInt(
            self, "Hybrid Clustering", 
            "Number of clusters for each method:",
            5, 2, 10, 1
        )
        
        if not ok:
            return
        
        try:
            from utils.progress_dialog import ProgressDialog
            progress = ProgressDialog(
                "Hybrid Clustering",
                "Running 2D clustering on current slice...",
                0,
                self
            )
            progress.show()
            QApplication.processEvents()
            
            # Run 2D clustering first
            print("Phase 1: 2D clustering...", file=sys.stderr)
            
            # Get current slice for both neutron and X-ray
            neutron_vol, xray_vol = self.current_slice_data
            
            # Extract current slice based on axis
            if self.current_axis == 'z':
                neutron_slice = neutron_vol[self.current_slice_index, :, :]
                xray_slice = xray_vol[self.current_slice_index, :, :]
            elif self.current_axis == 'y':
                neutron_slice = neutron_vol[:, self.current_slice_index, :]
                xray_slice = xray_vol[:, self.current_slice_index, :]
            else:  # x
                neutron_slice = neutron_vol[:, :, self.current_slice_index]
                xray_slice = xray_vol[:, :, self.current_slice_index]
            
            # Flatten and cluster
            neutron_flat = neutron_slice.flatten()
            xray_flat = xray_slice.flatten()
            points = np.column_stack([neutron_flat, xray_flat])
            
            from sklearn.cluster import KMeans
            kmeans_2d = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels_2d = kmeans_2d.fit_predict(points)
            labels_2d = labels_2d.reshape(neutron_slice.shape)
            
            # Store 2D clusters
            clusters_2d = []
            for i in range(num_clusters):
                mask = (labels_2d == i)
                neutron_vals = neutron_slice[mask]
                xray_vals = xray_slice[mask]
                clusters_2d.append((mask, neutron_vals, xray_vals))
            
            print(f"  2D: {num_clusters} clusters found", file=sys.stderr)
            
            # Run 3D clustering
            progress.setLabelText("Running 3D clustering on volume...")
            QApplication.processEvents()
            
            print("Phase 2: 3D clustering...", file=sys.stderr)
            # neutron_vol and xray_vol already extracted above
            
            from utils.clustering_3d import KMeans3D
            from utils.region_growing_3d import RegionGrowing3D
            
            labels_3d, centers_3d, stats_3d = KMeans3D.cluster_volume(
                neutron_vol, xray_vol,
                n_clusters=num_clusters
            )
            
            # Store 3D clusters
            clusters_3d = []
            for i in range(num_clusters):
                mask_3d = (labels_3d == i)
                mask_2d = RegionGrowing3D.extract_slice_from_3d_mask(
                    mask_3d, self.current_axis, self.current_slice_index
                )
                neutron_vals = neutron_vol[mask_3d]
                xray_vals = xray_vol[mask_3d]
                clusters_3d.append((mask_2d, neutron_vals, xray_vals, mask_3d))
            
            print(f"  3D: {num_clusters} clusters found", file=sys.stderr)
            
            # Merge: Keep all 2D clusters + non-overlapping 3D clusters
            progress.setLabelText("Merging results...")
            QApplication.processEvents()
            
            print("Phase 3: Merging unique ROIs...", file=sys.stderr)
            
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab20')  # More colors for hybrid
            
            all_clusters = []
            color_idx = 0
            
            # Add all 2D clusters
            for i, (mask_2d, n_vals, x_vals) in enumerate(clusters_2d):
                roi_vertices = self._create_roi_from_cluster(n_vals, x_vals)
                color = cmap(color_idx / (num_clusters * 2))
                all_clusters.append((
                    f"2D Cluster {i}",
                    mask_2d,
                    roi_vertices,
                    i,
                    color
                ))
                color_idx += 1
            
            # Add 3D clusters
            for i, (mask_2d, n_vals, x_vals, mask_3d) in enumerate(clusters_3d):
                roi_vertices = KMeans3D.create_convex_hull_roi_3d(
                    n_vals, x_vals,
                    percentile=98,
                    density_aware=True
                )
                color = cmap(color_idx / (num_clusters * 2))
                all_clusters.append((
                    f"3D Cluster {i}",
                    mask_2d,
                    roi_vertices,
                    i + num_clusters,  # Offset cluster IDs
                    color,
                    mask_3d
                ))
                color_idx += 1
            
            progress.close()
            
            # Emit all clusters
            self.clusters_detected.emit(all_clusters)
            
            QMessageBox.information(
                self,
                "Hybrid Clustering Complete",
                f"Combined results:\n"
                f"  • {num_clusters} clusters from 2D (current slice)\n"
                f"  • {num_clusters} clusters from 3D (full volume)\n"
                f"  • Total: {len(all_clusters)} selections\n\n"
                f"2D clusters capture low-density features\n"
                f"3D clusters capture volumetric coherence\n\n"
                f"All saved to Selection Manager."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Hybrid clustering failed:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _create_roi_from_cluster(self, neutron_vals, xray_vals):
        """Helper to create ROI from cluster data (for 2D clusters)"""
        from scipy.spatial import ConvexHull
        
        # Subsample if needed
        if len(neutron_vals) > 5000:
            indices = np.random.choice(len(neutron_vals), 5000, replace=False)
            neutron_vals = neutron_vals[indices]
            xray_vals = xray_vals[indices]
        
        # Create convex hull
        points = np.column_stack([neutron_vals, xray_vals])
        
        try:
            if len(points) >= 3:
                hull = ConvexHull(points)
                return points[hull.vertices]
            else:
                # Rectangle fallback
                n_min, n_max = np.min(neutron_vals), np.max(neutron_vals)
                x_min, x_max = np.min(xray_vals), np.max(xray_vals)
                return np.array([
                    [n_min, x_min], [n_max, x_min],
                    [n_max, x_max], [n_min, x_max]
                ])
        except:
            n_min, n_max = np.min(neutron_vals), np.max(neutron_vals)
            x_min, x_max = np.min(xray_vals), np.max(xray_vals)
            return np.array([
                [n_min, x_min], [n_max, x_min],
                [n_max, x_max], [n_min, x_max]
            ])
    
    def _auto_detect_2d(self):
        """2D k-means clustering on current slice (original behavior)"""
        import sys
        from utils.feature_detection import FeatureDetector
        from utils.region_growing import RegionGrowing
        from PyQt5.QtWidgets import QInputDialog
        from sklearn.cluster import KMeans
        import numpy as np
        import matplotlib.cm as cm
        
        print("Auto-detecting features using histogram clustering...", file=sys.stderr)
        
        if self.current_slice is None or self.current_slice_data is None:
            QMessageBox.warning(self, "No Data", "No slice data available")
            return
        
        # Get both slices for bivariate analysis
        neutron_vol, xray_vol = self.current_slice_data
        
        # Extract both slices
        if self.current_axis == 'z':
            neutron_slice = neutron_vol[self.current_slice_index, :, :]
            xray_slice = xray_vol[self.current_slice_index, :, :]
        elif self.current_axis == 'y':
            neutron_slice = neutron_vol[:, self.current_slice_index, :]
            xray_slice = xray_vol[:, self.current_slice_index, :]
        else:  # 'x'
            neutron_slice = neutron_vol[:, :, self.current_slice_index]
            xray_slice = xray_vol[:, :, self.current_slice_index]
        
        # Ask user for number of clusters
        num_clusters, ok = QInputDialog.getInt(
            self, "Histogram Clustering", 
            "Number of clusters/regions to detect:",
            5, 2, 20, 1
        )
        
        if not ok:
            return
        
        try:
            print(f"  Performing k-means clustering with {num_clusters} clusters", file=sys.stderr)
            
            # Create 2D point cloud from histogram
            neutron_flat = neutron_slice.flatten()
            xray_flat = xray_slice.flatten()
            
            # Stack into (N, 2) array for clustering
            points = np.column_stack([neutron_flat, xray_flat])
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(points)
            cluster_centers = kmeans.cluster_centers_
            
            print(f"  Cluster centers:", file=sys.stderr)
            for i, center in enumerate(cluster_centers):
                print(f"    Cluster {i}: Neutron={center[0]:.0f}, X-ray={center[1]:.0f}", file=sys.stderr)
            
            # Reshape labels back to image shape
            label_map = labels.reshape(neutron_slice.shape)
            
            # Store cluster information
            self.cluster_map = label_map
            self.cluster_centers = cluster_centers
            self.num_clusters = num_clusters
            
            # Calculate spatial density for each cluster
            print("  Computing spatial density for each cluster...", file=sys.stderr)
            
            # Divide slice into grid for density calculation
            grid_rows, grid_cols = 10, 10  # 10x10 grid
            height, width = neutron_slice.shape
            cell_height = height // grid_rows
            cell_width = width // grid_cols
            
            # For each cluster, find where it's concentrated
            cluster_density = np.zeros((num_clusters, grid_rows, grid_cols))
            
            for cluster_id in range(num_clusters):
                cluster_mask = (label_map == cluster_id)
                
                # Count pixels in each grid cell
                for i in range(grid_rows):
                    for j in range(grid_cols):
                        y_start = i * cell_height
                        y_end = (i + 1) * cell_height if i < grid_rows - 1 else height
                        x_start = j * cell_width
                        x_end = (j + 1) * cell_width if j < grid_cols - 1 else width
                        
                        cell_mask = cluster_mask[y_start:y_end, x_start:x_end]
                        cluster_density[cluster_id, i, j] = np.sum(cell_mask)
            
            # Find representative points for each cluster (density peaks)
            features = []
            for cluster_id in range(num_clusters):
                density = cluster_density[cluster_id]
                
                # Find cell with highest density
                max_i, max_j = np.unravel_index(np.argmax(density), density.shape)
                
                # Get center of that cell
                y_center = int(max_i * cell_height + cell_height / 2)
                x_center = int(max_j * cell_width + cell_width / 2)
                
                # Find exact peak within that cell
                y_start = max_i * cell_height
                y_end = (max_i + 1) * cell_height if max_i < grid_rows - 1 else height
                x_start = max_j * cell_width
                x_end = (max_j + 1) * cell_width if max_j < grid_cols - 1 else width
                
                # Get cluster pixels in this cell
                cell_cluster_mask = label_map[y_start:y_end, x_start:x_end] == cluster_id
                
                if np.any(cell_cluster_mask):
                    # Find centroid of cluster pixels in this cell
                    y_coords, x_coords = np.where(cell_cluster_mask)
                    y_peak = int(np.mean(y_coords)) + y_start
                    x_peak = int(np.mean(x_coords)) + x_start
                    
                    features.append((y_peak, x_peak, cluster_id))
                    
                    print(f"  Cluster {cluster_id}: Peak at ({y_peak}, {x_peak}), "
                          f"Density={density[max_i, max_j]:.0f}", file=sys.stderr)
            
            if len(features) == 0:
                QMessageBox.information(self, "No Features", "No cluster peaks detected.")
                return
            
            # Store detected features (without cluster_id for display)
            self.detected_features = [(y, x) for y, x, _ in features]
            self.cluster_assignments = {(y, x): cid for y, x, cid in features}
            
            # Display cluster map overlay
            self._display_cluster_map()
            
            # Display feature markers with cluster colors
            self._display_cluster_markers()
            
            # Update info
            self.info_label.setText(
                f"Detected {num_clusters} clusters - Click on marker to select region"
            )
            
            # Enable region grow mode
            if not self.region_grow_btn.isChecked():
                self.region_grow_btn.setChecked(True)
            
            # Prepare cluster selections for emission
            cmap = cm.get_cmap('tab10')
            
            cluster_selections = []
            
            for cluster_id in range(num_clusters):
                # Create mask for this cluster
                cluster_mask = (label_map == cluster_id)
                
                # Get cluster color
                color = cmap(cluster_id / num_clusters)
                
                # Extract values for this cluster
                neutron_vals = neutron_slice[cluster_mask]
                xray_vals = xray_slice[cluster_mask]
                
                # Create ROI vertices (convex hull)
                roi_vertices = RegionGrowing.create_convex_hull_roi(
                    neutron_vals, xray_vals, margin=0.05
                )
                
                # Add to list
                cluster_selections.append((
                    f"Cluster {cluster_id}",  # name
                    cluster_mask,              # spatial_mask
                    roi_vertices,              # histogram_roi
                    cluster_id,                # cluster_id
                    color                      # color
                ))
            
            # Emit signal for main window to save to selection manager
            self.clusters_detected.emit(cluster_selections)
            
            print(f"  Emitted {num_clusters} clusters for saving", file=sys.stderr)
            
        except Exception as e:
            print(f"Error detecting features: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Detection Error", f"Error detecting features:\n{e}")
    
    def _display_cluster_map(self):
        """Display cluster map as colored overlay"""
        if not hasattr(self, 'cluster_map'):
            return
        
        # Remove old cluster overlay
        if hasattr(self, 'cluster_overlay') and self.cluster_overlay is not None:
            try:
                self.cluster_overlay.remove()
            except:
                pass
            self.cluster_overlay = None
        
        # Create colored overlay from cluster map
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Use a colormap
        cmap = cm.get_cmap('tab10')
        
        # Create RGBA image
        cluster_rgba = np.zeros((*self.cluster_map.shape, 4))
        
        for cluster_id in range(self.num_clusters):
            mask = (self.cluster_map == cluster_id)
            color = cmap(cluster_id / self.num_clusters)
            cluster_rgba[mask] = [color[0], color[1], color[2], 0.3]  # Semi-transparent
        
        # Display overlay
        self.cluster_overlay = self.ax.imshow(
            cluster_rgba,
            extent=self.ax.images[0].get_extent() if self.ax.images else None,
            zorder=9,
            interpolation='nearest'
        )
        
        self.canvas.draw_idle()
    
    def _display_cluster_markers(self):
        """Display markers for cluster centers with cluster-specific colors"""
        import matplotlib.cm as cm
        
        # Remove old markers
        for marker in self.feature_markers:
            try:
                marker.remove()
            except:
                pass
        self.feature_markers = []
        
        # Get colormap
        cmap = cm.get_cmap('tab10')
        
        # Add new markers with cluster colors
        for y, x in self.detected_features:
            cluster_id = self.cluster_assignments[(y, x)]
            color = cmap(cluster_id / self.num_clusters)
            
            marker = self.ax.plot(x, y, 'o', 
                                markersize=15, 
                                markeredgewidth=3,
                                markerfacecolor='none',
                                markeredgecolor=color,
                                zorder=15)[0]
            self.feature_markers.append(marker)
        
        self.canvas.draw_idle()
    
    def set_volume_shape(self, shape):
        """Set the volume shape for slider configuration"""
        self.volume_shape = shape
        self._configure_slider()
        
    def _configure_slider(self):
        """Configure slider based on current axis and volume shape"""
        if self.volume_shape is None:
            return
        
        axis_map = {'z': 0, 'y': 1, 'x': 2}
        max_val = self.volume_shape[axis_map[self.current_axis]] - 1
        
        self.slice_slider.setMaximum(max_val)
        self.slice_slider.setValue(max_val // 2)
        self.slice_slider.setEnabled(True)
        self.current_slice_index = max_val // 2
        self.slice_label.setText(str(self.current_slice_index))
    
    def _on_axis_changed(self, axis):
        """Handle axis selection change"""
        print(f"Axis changed to: {axis}", file=sys.stderr)
        self.current_axis = axis
        self._configure_slider()
        self._update_display()
    
    def _on_slice_changed(self, value):
        """Handle slice index change (debounced while dragging)"""
        self.current_slice_index = value
        self.slice_label.setText(str(value))
        self._redraw_timer.start()
    
    def _on_view_mode_changed(self, mode):
        """Handle view mode change (neutron vs X-ray)"""
        print(f"View mode changed to: {mode}", file=sys.stderr)
        self.view_mode = mode
        self._update_display()
    
    def _on_range_changed(self):
        """Handle manual range change"""
        self.vmin = self.slice_vmin.value()
        self.vmax = self.slice_vmax.value()
        self._update_display()
    
    def _auto_range_slice(self):
        """Auto-set range based on current slice"""
        if self.current_slice_data is None:
            return
        
        # Get the current volume based on view mode
        neutron_vol, xray_vol = self.current_slice_data
        vol = neutron_vol if self.view_mode == 'neutron' else xray_vol
        
        # Set range to volume min/max
        vmin = float(vol.min())
        vmax = float(vol.max())
        
        self.slice_vmin.setValue(vmin)
        self.slice_vmax.setValue(vmax)
        self.vmin = vmin
        self.vmax = vmax
        self._update_display()
    
    def _on_rect_tool_clicked(self, checked):
        """Handle rectangle tool activation"""
        import sys
        print(f"Rectangle tool clicked: {checked}", file=sys.stderr)
        
        if checked:
            # Activate rectangle selector
            from matplotlib.widgets import RectangleSelector
            
            # Disconnect previous selector if exists
            if self.spatial_roi_selector is not None:
                try:
                    self.spatial_roi_selector.set_active(False)
                except:
                    pass
            
            # Create new rectangle selector
            # Note: Compatible with matplotlib 3.5+
            try:
                self.spatial_roi_selector = RectangleSelector(
                    self.ax,
                    self._on_rectangle_selected,
                    useblit=True,
                    button=[1],  # Left mouse button
                    minspanx=5,
                    minspany=5,
                    interactive=True,
                    props=dict(facecolor='green', edgecolor='green', 
                              alpha=0.3, linewidth=2)
                )
                print("  Rectangle selector activated (modern API)", file=sys.stderr)
            except TypeError as e:
                # Fallback for older matplotlib versions
                print(f"  Warning: {e}", file=sys.stderr)
                print("  Trying fallback initialization...", file=sys.stderr)
                self.spatial_roi_selector = RectangleSelector(
                    self.ax,
                    self._on_rectangle_selected,
                    interactive=True
                )
                print("  Rectangle selector activated (fallback)", file=sys.stderr)
            
            self.info_label.setText("Draw rectangle on slice to select spatial region")
            print("  Rectangle selector ready", file=sys.stderr)
        else:
            # Deactivate
            if self.spatial_roi_selector is not None:
                self.spatial_roi_selector.set_active(False)
            self.info_label.setText("Rectangle tool deactivated")
    
    def _on_rectangle_selected(self, eclick, erelease):
        """Callback when rectangle is drawn"""
        import sys
        print(f"Rectangle selected:", file=sys.stderr)
        print(f"  eclick: ({eclick.xdata}, {eclick.ydata})", file=sys.stderr)
        print(f"  erelease: ({erelease.xdata}, {erelease.ydata})", file=sys.stderr)
        
        # Store rectangle coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        self.spatial_roi_coords = (x1, y1, x2, y2)
        
        # Enable buttons
        self.clear_spatial_roi_btn.setEnabled(True)
        self.create_hist_roi_btn.setEnabled(True)
        
        # Update info
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        self.info_label.setText(
            f"Spatial ROI: {width:.0f} × {height:.0f} pixels selected"
        )
        
        print(f"  Spatial ROI stored: {self.spatial_roi_coords}", file=sys.stderr)
    
    def _clear_spatial_roi(self):
        """Clear spatial ROI (rectangle or region growing)"""
        import sys
        print("Clearing spatial ROI", file=sys.stderr)
        
        # Clear rectangle coordinates
        self.spatial_roi_coords = None
        
        # Clear region growing state
        self.region_grow_mask = None
        if self.mask_overlay is not None:
            try:
                self.mask_overlay.remove()
            except:
                pass
            self.mask_overlay = None
        
        # Clear detected features
        self.detected_features = []
        for marker in self.feature_markers:
            try:
                marker.remove()
            except:
                pass
        self.feature_markers = []
        
        # Clear cluster overlay
        if hasattr(self, 'cluster_overlay') and self.cluster_overlay is not None:
            try:
                self.cluster_overlay.remove()
            except:
                pass
            self.cluster_overlay = None
        
        # Deactivate and clear selector
        if self.spatial_roi_selector is not None:
            try:
                # Set inactive
                self.spatial_roi_selector.set_active(False)
                # Clear the selection (removes visual rectangle)
                self.spatial_roi_selector.clear()
                # Force canvas redraw
                self.canvas.draw_idle()
                print("  Rectangle selector cleared and canvas redrawn", file=sys.stderr)
            except Exception as e:
                print(f"  Warning clearing selector: {e}", file=sys.stderr)
                # Fallback: just redraw the display
                self._update_display()
        
        # Uncheck tool buttons
        self.rect_tool_btn.setChecked(False)
        self.region_grow_btn.setChecked(False)
        
        # Disable buttons
        self.clear_spatial_roi_btn.setEnabled(False)
        self.create_hist_roi_btn.setEnabled(False)
        self.tolerance_spinbox.setEnabled(False)
        self.show_mask_cb.setEnabled(False)

        # Redraw to remove mask
        self._update_display()

        self.info_label.setText("Spatial ROI cleared")

    def _clear_highlight(self):
        """Remove all coloured overlays from the slice view without affecting the ROI."""
        # Clear single region-grow overlay
        self.region_grow_mask = None
        if self.mask_overlay is not None:
            try:
                self.mask_overlay.remove()
            except Exception:
                pass
            self.mask_overlay = None

        # Clear multi-ROI selection overlays
        self.mask_overlays = []
        self._clear_overlay_artists()

        self.clear_highlight_btn.setEnabled(False)
        self.canvas.draw_idle()
        self.info_label.setText("Highlight cleared")
    
    def _on_region_grow_clicked(self, checked):
        """Handle region grow tool button"""
        import sys
        print(f"Region grow tool: {checked}", file=sys.stderr)
        
        if checked:
            # Deactivate rectangle tool if active
            if self.rect_tool_btn.isChecked():
                self.rect_tool_btn.setChecked(False)
            
            # Activate region grow mode
            self.region_grow_mode = True
            self.tolerance_spinbox.setEnabled(True)
            self.bivariate_cb.setEnabled(True)
            self.mode_3d_cb.setEnabled(True)  # Enable 3D mode option
            self.info_label.setText("Click on slice to select seed point")
            
            # Connect click event
            self.cid_click = self.canvas.mpl_connect('button_press_event', self._on_seed_click)
            print("  Region grow mode activated", file=sys.stderr)
        else:
            # Deactivate
            self.region_grow_mode = False
            self.tolerance_spinbox.setEnabled(False)
            self.bivariate_cb.setEnabled(False)
            self.tolerance2_spinbox.setEnabled(False)
            
            # Disconnect click event
            if hasattr(self, 'cid_click'):
                self.canvas.mpl_disconnect(self.cid_click)
            
            self.info_label.setText("Region grow tool deactivated")
    
    def _on_bivariate_changed(self, state):
        """Handle bivariate mode toggle"""
        import sys
        
        is_bivariate = (state == 2)  # Qt.Checked
        print(f"Bivariate mode: {is_bivariate}", file=sys.stderr)
        
        # Show/hide second tolerance control
        self.tolerance2_label.setVisible(is_bivariate)
        self.tolerance2_spinbox.setVisible(is_bivariate)
        self.tolerance2_spinbox.setEnabled(is_bivariate and self.region_grow_mode)
        
        # Update tooltip
        if is_bivariate:
            self.tolerance_spinbox.setToolTip("Tolerance for displayed modality")
            self.info_label.setText("Bivariate mode: Both neutron AND X-ray checked")
        else:
            self.tolerance_spinbox.setToolTip("Intensity tolerance for region growing")
            if self.region_grow_mode:
                self.info_label.setText("Click on slice to select seed point")
    
    def _on_seed_click(self, event):
        """Handle seed point click for region growing"""
        import sys
        
        if not self.region_grow_mode:
            return
        
        if event.inaxes != self.ax or event.button != 1:  # Left click only
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        print(f"Seed click at: ({event.xdata}, {event.ydata})", file=sys.stderr)
        
        # Get current slice
        if self.current_slice is None:
            print("  No slice data", file=sys.stderr)
            return
        
        # Convert to pixel coordinates
        x_pixel = int(round(event.xdata))
        y_pixel = int(round(event.ydata))
        
        print(f"  Pixel coordinates: ({x_pixel}, {y_pixel})", file=sys.stderr)
        
        # Check bounds
        if x_pixel < 0 or x_pixel >= self.current_slice.shape[1]:
            print(f"  X out of bounds", file=sys.stderr)
            return
        if y_pixel < 0 or y_pixel >= self.current_slice.shape[0]:
            print(f"  Y out of bounds", file=sys.stderr)
            return
        
        # Perform region growing
        self._grow_region_from_seed(y_pixel, x_pixel)
    
    def _grow_region_from_seed(self, y_pixel, x_pixel):
        """Grow region from seed point (univariate or bivariate, 2D or 3D)"""
        import sys
        from utils.region_growing import RegionGrowing
        
        print(f"Growing region from seed: ({y_pixel}, {x_pixel})", file=sys.stderr)
        
        # Check if 3D mode
        is_3d_mode = self.mode_3d_cb.isChecked()
        is_bivariate = self.bivariate_cb.isChecked()
        
        if is_3d_mode:
            # 3D VOLUME MODE
            from utils.region_growing_3d import RegionGrowing3D
            
            print("  Using 3D VOLUME mode", file=sys.stderr)
            
            if self.current_slice_data is None:
                print("  ERROR: No volume data", file=sys.stderr)
                return
            
            # Get volumes
            neutron_vol, xray_vol = self.current_slice_data
            
            # Convert 2D seed to 3D seed
            if self.current_axis == 'z':
                seed_3d = (self.current_slice_index, y_pixel, x_pixel)
            elif self.current_axis == 'y':
                seed_3d = (y_pixel, self.current_slice_index, x_pixel)
            else:  # 'x'
                seed_3d = (y_pixel, x_pixel, self.current_slice_index)
            
            print(f"  3D seed point: {seed_3d}", file=sys.stderr)
            
            if is_bivariate:
                # 3D Bivariate
                neutron_tolerance = self.tolerance_spinbox.value()
                xray_tolerance = self.tolerance2_spinbox.value()
                
                print(f"  3D Bivariate: N_tol={neutron_tolerance}, X_tol={xray_tolerance}", file=sys.stderr)
                
                mask_3d = RegionGrowing3D.bivariate_region_growing_3d(
                    neutron_vol,
                    xray_vol,
                    seed_3d,
                    neutron_tolerance,
                    xray_tolerance,
                    connectivity=1  # 6-connected for speed
                )
            else:
                # 3D Univariate
                tolerance = self.tolerance_spinbox.value()
                
                # Use displayed modality
                if self.current_view_mode == 'neutron':
                    volume = neutron_vol
                else:
                    volume = xray_vol
                
                print(f"  3D Univariate: tolerance={tolerance}", file=sys.stderr)
                
                mask_3d = RegionGrowing3D.univariate_region_growing_3d(
                    volume,
                    seed_3d,
                    tolerance,
                    connectivity=1
                )
            
            # Store 3D mask
            self.region_grow_mask_3d = mask_3d
            
            # Extract 2D slice for display
            self.region_grow_mask = RegionGrowing3D.extract_slice_from_3d_mask(
                mask_3d,
                self.current_axis,
                self.current_slice_index
            )
            
            print(f"  3D result: {np.sum(mask_3d):,} voxels total", file=sys.stderr)
            print(f"  2D slice: {np.sum(self.region_grow_mask):,} pixels", file=sys.stderr)
            
        else:
            # 2D SLICE MODE (original behavior)
            if is_bivariate:
                # Bivariate mode - need both neutron and X-ray slices
                print("  Using BIVARIATE mode", file=sys.stderr)
                
                if self.current_slice_data is None:
                    print("  ERROR: No slice data", file=sys.stderr)
                    return
                
                # Get both volumes
                neutron_vol, xray_vol = self.current_slice_data
                
                # Extract both slices
                if self.current_axis == 'z':
                    neutron_slice = neutron_vol[self.current_slice_index, :, :]
                    xray_slice = xray_vol[self.current_slice_index, :, :]
                elif self.current_axis == 'y':
                    neutron_slice = neutron_vol[:, self.current_slice_index, :]
                    xray_slice = xray_vol[:, self.current_slice_index, :]
                else:  # 'x'
                    neutron_slice = neutron_vol[:, :, self.current_slice_index]
                    xray_slice = xray_vol[:, :, self.current_slice_index]
                
                # Get tolerances
                neutron_tolerance = self.tolerance_spinbox.value()
                xray_tolerance = self.tolerance2_spinbox.value()
                
                print(f"  Neutron tolerance: {neutron_tolerance}", file=sys.stderr)
                print(f"  X-ray tolerance: {xray_tolerance}", file=sys.stderr)
                
                # Grow region using BOTH modalities
                self.region_grow_mask = RegionGrowing.flood_fill_bivariate(
                    neutron_slice,
                    xray_slice,
                    (y_pixel, x_pixel),
                    neutron_tolerance,
                    xray_tolerance,
                    connectivity=2  # 8-connected
                )
            else:
                # Univariate mode - use only displayed modality
                print("  Using UNIVARIATE mode", file=sys.stderr)
                
                # Get tolerance
                tolerance = self.tolerance_spinbox.value()
                
                # Grow region
                self.region_grow_mask = RegionGrowing.flood_fill_tolerance(
                    self.current_slice,
                    (y_pixel, x_pixel),
                    tolerance,
                    connectivity=2  # 8-connected
                )
            
            # No 3D mask in 2D mode
            self.region_grow_mask_3d = None
        
        num_pixels = np.sum(self.region_grow_mask)
        print(f"  Region grown: {num_pixels} pixels", file=sys.stderr)
        
        if num_pixels == 0:
            mode_str = "bivariate" if is_bivariate else "univariate"
            QMessageBox.warning(
                self,
                "No Region Found",
                f"No connected region found at seed point ({mode_str} mode).\n"
                f"Try adjusting the tolerance(s)."
            )
            return
        
        # Display mask overlay
        self._display_mask_overlay()
        
        # Enable buttons
        self.clear_spatial_roi_btn.setEnabled(True)
        self.create_hist_roi_btn.setEnabled(True)
        self.show_mask_cb.setEnabled(True)
        
        # Update info
        mode_str = "bivariate" if is_bivariate else "univariate"
        self.info_label.setText(f"Region selected: {num_pixels:,} pixels ({mode_str})")
        
        # Store that we have a region (for histogram ROI creation)
        # We'll use region_grow_mask instead of spatial_roi_coords
    
    def _display_mask_overlay(self):
        """Display region growing mask overlay on slice"""
        if self.region_grow_mask is None:
            return

        # Remove old overlay
        if self.mask_overlay is not None:
            try:
                self.mask_overlay.remove()
            except Exception:
                pass
            self.mask_overlay = None

        # Show mask if checkbox is checked
        if self.show_mask_cb.isChecked():
            # Create colored overlay
            mask_rgba = np.zeros((*self.region_grow_mask.shape, 4))
            mask_rgba[self.region_grow_mask] = [0, 1, 0, 0.4]  # Green with alpha

            # Display overlay
            self.mask_overlay = self.ax.imshow(
                mask_rgba,
                extent=self.ax.images[0].get_extent() if self.ax.images else None,
                zorder=10,
                interpolation='nearest'
            )

        self.clear_highlight_btn.setEnabled(True)
        self.canvas.draw_idle()

    def _on_show_mask_changed(self, state):
        """Toggle mask overlay visibility"""
        if self.region_grow_mask is not None:
            self._display_mask_overlay()

    def set_mask_overlays(self, overlays, redraw=True):
        """
        Set multiple mask overlays for display

        Args:
            overlays: List of (name, mask, color) tuples
                     mask: 3-D volume mask (re-sliced for the current view)
                           or a 2-D single-slice mask
                     color: matplotlib color (tuple or string)
            redraw:  Draw the overlays now. Pass False when a full
                     _update_display() follows, so they render only once.
        """
        self.mask_overlays = overlays
        # Enable clear-highlight whenever overlays are non-empty
        self.clear_highlight_btn.setEnabled(bool(overlays))
        if redraw:
            self._display_mask_overlays()

    def clear_mask_overlays(self):
        """Clear all mask overlays"""
        self.mask_overlays = []
        self._clear_overlay_artists()
        self.clear_highlight_btn.setEnabled(False)
        self.canvas.draw_idle()
    
    def _clear_overlay_artists(self):
        """Remove overlay artists from plot"""
        for artist in self.overlay_artists:
            try:
                artist.remove()
            except:
                pass
        self.overlay_artists = []
    
    def _slice_mask_for_display(self, mask, plane=None):
        """Return the 2-D view of *mask* matching the current axis and slice.

        3-D masks (whole-volume layers) are sliced on demand so the highlight
        follows slice-index and viewing-plane changes.

        2-D masks belong to the single slice they were created on. When
        *plane* is given as ``(axis, slice_index)`` the mask is shown only
        there; without it, only the shape is checked — which is not enough on
        an isotropic volume, where a stale mask would be drawn on the wrong
        plane. Returns None when the mask cannot be shown in the current view.
        """
        if mask is None or self.current_slice is None:
            return None

        mask = np.asarray(mask)

        if mask.ndim == 2 and plane is not None:
            source_axis, source_index = plane
            if source_axis != self.current_axis:
                return None
            if source_index is not None and source_index != self.current_slice_index:
                return None

        if mask.ndim == 3:
            index = self.current_slice_index
            if index is None:
                return None
            axis_position = {'z': 0, 'y': 1, 'x': 2}.get(self.current_axis, 0)
            if not 0 <= index < mask.shape[axis_position]:
                return None
            if self.current_axis == 'z':
                slice_2d = mask[index, :, :]
            elif self.current_axis == 'y':
                slice_2d = mask[:, index, :]
            else:
                slice_2d = mask[:, :, index]
        elif mask.ndim == 2:
            slice_2d = mask
        else:
            return None

        if slice_2d.shape != self.current_slice.shape:
            return None
        return slice_2d

    def _display_mask_overlays(self):
        """Display multiple mask overlays with different colours.

        Masks are re-sliced from their stored form on every redraw, so the
        highlight stays correct while scrolling slices or switching planes.
        """
        self._clear_overlay_artists()
        self._visible_mask_pixels = 0

        if not self.ax.images or len(self.ax.images) == 0:
            return

        if self.current_slice is None:
            return

        # Use the same extent / origin / aspect as the base image
        extent = self.ax.images[0].get_extent()

        import matplotlib.colors as mcolors

        for entry in self.mask_overlays:
            # Entries are (name, mask, color) or (name, mask, color, plane),
            # where plane is (axis, slice_index) for single-slice masks.
            name, mask, color = entry[0], entry[1], entry[2]
            plane = entry[3] if len(entry) > 3 else None
            slice_2d = self._slice_mask_for_display(mask, plane)
            if slice_2d is None:
                continue
            visible = int(np.count_nonzero(slice_2d))
            if visible == 0:
                # Nothing of this layer intersects the current slice
                continue
            self._visible_mask_pixels += visible

            # Parse colour to RGBA float
            try:
                r, g, b, _ = mcolors.to_rgba(color, alpha=None)
                a = color[3] if (isinstance(color, (tuple, list)) and len(color) > 3) else 0.5
            except Exception:
                r, g, b, a = 1.0, 0.0, 0.0, 0.5

            overlay_rgba = np.zeros((*slice_2d.shape, 4), dtype=np.float32)
            overlay_rgba[slice_2d] = [r, g, b, a]

            artist = self.ax.imshow(
                overlay_rgba,
                extent=extent,
                origin='upper',
                aspect='equal',
                zorder=11,
                interpolation='nearest'
            )
            self.overlay_artists.append(artist)

        self.canvas.draw_idle()
    
    def _create_histogram_roi_from_spatial(self):
        """Extract values from spatial ROI and create histogram ROI"""
        import sys
        
        print("=" * 60, file=sys.stderr)
        print("Creating histogram ROI from spatial selection", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # Check if we have rectangle or region growing selection
        has_rectangle = (self.spatial_roi_coords is not None)
        has_region = (self.region_grow_mask is not None)
        
        if not has_rectangle and not has_region:
            print("  ERROR: No spatial ROI defined", file=sys.stderr)
            return
        
        if self.current_slice_data is None:
            print("  ERROR: No slice data loaded", file=sys.stderr)
            return
        
        if has_region:
            # Region growing mode - emit the mask instead of coords.
            # Prefer the 3-D mask when the region was grown through the
            # volume: building the histogram ROI from the current slice
            # alone would describe only the few voxels visible here, giving
            # an ROI far narrower than the region actually selected.
            mask = (
                self.region_grow_mask_3d
                if self.region_grow_mask_3d is not None
                else self.region_grow_mask
            )
            print(f"  Using region growing mask "
                  f"({np.count_nonzero(mask):,} voxels, {mask.ndim}-D)",
                  file=sys.stderr)

            self.spatial_roi_to_histogram.emit(
                ('mask', mask),
                self.current_axis,
                self.current_slice_index
            )
        else:
            # Rectangle mode - emit coords as before
            print(f"  Using rectangle coords: {self.spatial_roi_coords}", file=sys.stderr)
            print(f"  Emitting signal with: coords={self.spatial_roi_coords}, axis={self.current_axis}, slice={self.current_slice_index}", file=sys.stderr)
            self.spatial_roi_to_histogram.emit(
                self.spatial_roi_coords,
                self.current_axis,
                self.current_slice_index
            )
        
    def set_slice_data(self, neutron_vol, xray_vol, segmentation_vol=None):
        """Update slice data - now receives full 3D volumes"""
        self.current_slice_data = (neutron_vol, xray_vol)
        self.segmentation_mask = segmentation_vol
        
        if self.volume_shape is None or self.volume_shape != neutron_vol.shape:
            self.set_volume_shape(neutron_vol.shape)
        
        self._update_display()
    
    def _update_display(self):
        """Update the display"""
        if self.current_slice_data is None:
            return

        neutron_vol, xray_vol = self.current_slice_data
        
        # Select volume based on view mode
        data_vol = neutron_vol if self.view_mode == 'neutron' else xray_vol
        view_label = "Neutron" if self.view_mode == 'neutron' else "X-ray"
        
        # Extract the appropriate slice
        if self.current_slice_index is None:
            self.current_slice_index = data_vol.shape[0] // 2 if self.current_axis == 'z' else data_vol.shape[1] // 2
        
        # Clamp the index so switching to a shorter axis cannot go out of range
        axis_position = {'z': 0, 'y': 1, 'x': 2}[self.current_axis]
        self.current_slice_index = int(
            min(max(self.current_slice_index, 0),
                data_vol.shape[axis_position] - 1)
        )

        if self.current_axis == 'z':
            data_slice = data_vol[self.current_slice_index, :, :]
            title = f"XY Slice (Z={self.current_slice_index}, {view_label})"
        elif self.current_axis == 'y':
            data_slice = data_vol[:, self.current_slice_index, :]
            title = f"XZ Slice (Y={self.current_slice_index}, {view_label})"
        else:  # 'x'
            data_slice = data_vol[:, :, self.current_slice_index]
            title = f"YZ Slice (X={self.current_slice_index}, {view_label})"


        # Store current slice for region growing
        self.current_slice = data_slice
        
        self.ax.clear()
        self.ax.set_title(title)
        self.ax.axis('off')
        
        # Determine vmin/vmax for display
        if self.vmin is not None and self.vmax is not None and self.vmax > self.vmin:
            vmin_display = self.vmin
            vmax_display = self.vmax
        else:
            vmin_display = None
            vmax_display = None
        
        # Define explicit extent to ensure perfect alignment
        # extent = [left, right, bottom, top] for origin='upper'
        extent = [0, data_slice.shape[1], data_slice.shape[0], 0]
        
        im = self.ax.imshow(
            data_slice, 
            cmap='gray', 
            interpolation='nearest',
            vmin=vmin_display,
            vmax=vmax_display,
            extent=extent,
            origin='upper',
            aspect='equal'  # Preserve aspect ratio - no distortion
        )
        
        # Draw multi-colour mask overlays; 3-D masks are re-sliced here so the
        # highlight tracks the current plane and slice index.
        self._display_mask_overlays()

        # Report what is actually highlighted on this slice
        if self._visible_mask_pixels > 0:
            pct = 100.0 * self._visible_mask_pixels / data_slice.size
            info = (
                f"{view_label} | Segmented here: "
                f"{self._visible_mask_pixels:,} pixels ({pct:.1f}%)"
            )
        else:
            info = (
                f"{view_label} | Shape: {data_slice.shape} | "
                f"Range: [{data_slice.min():.0f}, {data_slice.max():.0f}]"
            )

        if self.display_bin_factor > 1:
            info += (
                f" | display binned x{self.display_bin_factor} (median) — "
                "segmentation runs at full resolution"
            )

        self.info_label.setText(info)

        self.fig.tight_layout()
        self.canvas.draw_idle()


def _scrollable(widget):
    """Wrap a tool panel so a short screen scrolls it instead of growing.

    Without this, a tab's tallest content becomes the window's minimum
    height, and one long panel makes the whole application unusable on a
    laptop.
    """
    from PyQt5.QtWidgets import QScrollArea

    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return area


def _plain_validation_summary(validation) -> str:
    """Held-out scores, said without the vocabulary of the method."""
    return (
        f"Tested on {validation.n_folds} region(s) the classifier never saw "
        f"during training: {100 * validation.accuracy:.1f}% of voxels "
        f"correct, {100 * validation.mean_iou:.1f}% average overlap with the "
        f"materials you drew."
    )


class AnchorSelectionDialog(QDialog):
    """Pick the materials that cannot change during the experiment.

    An anchor is a phase that cannot really change during the experiment, so
    any movement of its histogram centroid must be instrumental. Picking a
    reactive class here would fit the physics away as if it were drift, which
    is why the choice is the user's and not a default.
    """

    def __init__(self, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Materials")
        self.anchor_classes = []
        self.estimate_scale = False

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Choose the materials that cannot change during the experiment "
            "— a container, a support, a structural metal. If one of these "
            "appears to move, it is the instrument that moved, not the "
            "sample.\n\n"
            "Do not choose a material that reacts: its real change would be "
            "subtracted from every other material as though it were an "
            "instrument effect."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self._checkboxes = []
        for name in class_names:
            box = QCheckBox(name)
            layout.addWidget(box)
            self._checkboxes.append((name, box))

        self.scale_box = QCheckBox(
            "Also correct for a change in scale (needs two or more "
            "separated control materials)"
        )
        self.scale_box.setToolTip(
            "Some instrument changes stretch the histogram as well as\n"
            "shifting it. With a single control material this cannot be\n"
            "separated from a plain shift, so it is left alone."
        )
        layout.addWidget(self.scale_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self):
        self.anchor_classes = [
            name for name, box in self._checkboxes if box.isChecked()
        ]
        self.estimate_scale = self.scale_box.isChecked()
        if not self.anchor_classes:
            QMessageBox.warning(
                self, "No Control Materials",
                "Select at least one material to use as a control."
            )
            return
        self.accept()


class ExportOptionsDialog(QDialog):
    """
    Dialog that lets the user choose which segmentation layers and which
    output modalities to include in an export.

    Parameters
    ----------
    layers : list of (mask_3d, color, name) tuples
        The available segmentation layers.
    parent : QWidget, optional

    After exec_() returns Accepted, read:
        dialog.selected_layers  → list of (mask_3d, color, name) for chosen layers
        dialog.export_mask      → bool – write binary mask TIFF
        dialog.export_neutron   → bool – write masked neutron TIFF
        dialog.export_xray      → bool – write masked X-ray TIFF
        dialog.export_labels    → bool – write per-layer integer label TIFF
    """

    def __init__(self, layers, parent=None):
        super().__init__(parent)
        self.layers = layers
        self.setWindowTitle("Export Options")
        self.setMinimumWidth(400)
        self._build_ui()

    def _build_ui(self):
        from PyQt5.QtWidgets import (
            QScrollArea, QDialogButtonBox, QFrame
        )
        from PyQt5.QtGui import QColor, QPalette
        from PyQt5.QtCore import Qt

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # ── Layers ─────────────────────────────────────────────────────────────
        layers_group = QGroupBox("Layers to export")
        layers_vbox = QVBoxLayout()
        layers_vbox.setSpacing(4)

        # "Select all" convenience checkbox
        self._all_layers_cb = QCheckBox("Select / deselect all")
        self._all_layers_cb.setChecked(True)
        self._all_layers_cb.stateChanged.connect(self._toggle_all_layers)
        layers_vbox.addWidget(self._all_layers_cb)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        layers_vbox.addWidget(sep)

        self._layer_cbs = []
        scroll_widget = QWidget()
        scroll_vbox = QVBoxLayout(scroll_widget)
        scroll_vbox.setSpacing(3)
        scroll_vbox.setContentsMargins(2, 2, 2, 2)

        for mask_3d, color, name in self.layers:
            row = QHBoxLayout()

            # Colour swatch
            swatch = QLabel("  ")
            try:
                import matplotlib.colors as mcolors
                r, g, b, _ = mcolors.to_rgba(color)
                hex_col = "#{:02x}{:02x}{:02x}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
                swatch.setStyleSheet(
                    f"background-color: {hex_col}; border: 1px solid #888;"
                )
            except Exception:
                pass
            swatch.setFixedSize(16, 16)
            row.addWidget(swatch)

            n_vox = int(np.sum(mask_3d))
            cb = QCheckBox(f"{name}  ({n_vox:,} voxels)")
            cb.setChecked(True)
            self._layer_cbs.append((cb, mask_3d, color, name))
            row.addWidget(cb)
            row.addStretch()
            scroll_vbox.addLayout(row)

        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(160)
        scroll_area.setFrameShape(QFrame.NoFrame)
        layers_vbox.addWidget(scroll_area)
        layers_group.setLayout(layers_vbox)
        main_layout.addWidget(layers_group)

        # ── Modalities ─────────────────────────────────────────────────────────
        mod_group = QGroupBox("Output files (per selected layer)")
        mod_vbox = QVBoxLayout()
        mod_vbox.setSpacing(4)

        self._mask_cb    = QCheckBox("Binary mask  (0/255 TIFF)")
        self._neutron_cb = QCheckBox("Neutron volume  (masked intensity)")
        self._xray_cb    = QCheckBox("X-ray volume  (masked intensity)")
        self._labels_cb  = QCheckBox("Integer label volume  (all selected layers combined)")
        self._report_cb = QCheckBox(
            "Text report  (class names, label values, voxels per timepoint)"
        )
        self._report_cb.setToolTip(
            "Write a segmentation_report.txt describing each class: its name,\n"
            "the integer value it takes in the label volumes, its voxel count\n"
            "at every timepoint, and how the segmentation was produced."
        )
        self._histogram_cb = QCheckBox(
            "Bimodal histogram of the class  (.npy + .png)"
        )
        self._histogram_cb.setToolTip(
            "For every selected class and timepoint, compute the 2-D\n"
            "neutron/X-ray histogram of that class's segmented voxels.\n"
            "The bins and limits are the same as the main histogram, so the\n"
            "files can be compared bin-for-bin across classes and time."
        )

        self._mask_cb.setChecked(True)
        self._neutron_cb.setChecked(True)
        self._xray_cb.setChecked(True)
        self._labels_cb.setChecked(False)
        self._histogram_cb.setChecked(False)
        self._report_cb.setChecked(True)

        mod_vbox.addWidget(self._mask_cb)
        mod_vbox.addWidget(self._neutron_cb)
        mod_vbox.addWidget(self._xray_cb)
        mod_vbox.addWidget(self._labels_cb)
        mod_vbox.addWidget(self._histogram_cb)
        mod_vbox.addWidget(self._report_cb)

        mod_group.setLayout(mod_vbox)
        main_layout.addWidget(mod_group)

        # ── File-count preview ────────────────────────────────────────────────
        self._preview_label = QLabel()
        self._preview_label.setStyleSheet("color: #555; font-style: italic; font-size: 9pt;")
        main_layout.addWidget(self._preview_label)

        # Connect all checkboxes to the preview update
        for cb, *_ in self._layer_cbs:
            cb.stateChanged.connect(self._update_preview)
        for cb in (self._mask_cb, self._neutron_cb, self._xray_cb,
                   self._labels_cb, self._histogram_cb, self._report_cb):
            cb.stateChanged.connect(self._update_preview)
        self._update_preview()

        # ── Buttons ────────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def _toggle_all_layers(self, state):
        checked = (state == 2)
        for cb, *_ in self._layer_cbs:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._update_preview()

    def _update_preview(self):
        n_layers = sum(1 for cb, *_ in self._layer_cbs if cb.isChecked())
        n_mods   = sum([self._mask_cb.isChecked(),
                        self._neutron_cb.isChecked(),
                        self._xray_cb.isChecked()])
        # A histogram export writes a counts file and an image per layer
        if self._histogram_cb.isChecked():
            n_mods += 2
        n_label  = 1 if self._labels_cb.isChecked() else 0
        per_tp   = n_layers * n_mods + n_label
        self._preview_label.setText(
            f"→  {n_layers} layer(s) × {n_mods} file(s) each"
            + (f" + 1 label file" if n_label else "")
            + f"  =  {per_tp} file(s) per timepoint"
        )

    def _on_accept(self):
        self.selected_layers = [
            (mask_3d, color, name)
            for cb, mask_3d, color, name in self._layer_cbs
            if cb.isChecked()
        ]
        self.export_mask    = self._mask_cb.isChecked()
        self.export_neutron = self._neutron_cb.isChecked()
        self.export_xray    = self._xray_cb.isChecked()
        self.export_labels  = self._labels_cb.isChecked()
        self.export_histogram = self._histogram_cb.isChecked()
        self.export_report = self._report_cb.isChecked()
        self.accept()


class BiTS4DMainWindow(QMainWindow):
    """
    Main window for BiTS 4D application
    """
    
    def __init__(self):
        super().__init__()
        
        # Data components
        self.dataset = None
        self.histogram_engine = None
        self.segmentation_engine = SegmentationEngine4D()

        # Latest 3-D K-means result, available to copy in as materials
        self._last_kmeans_cluster_selections = []
        self._cluster_timepoint = None

        # Current state
        self.global_histogram = None
        # Result of the last model-based run, kept for inspection/export
        self.model_result = None
        self._manual_window = None
        # segmentation_masks: {timepoint -> [(mask_3d, color, name), ...]}
        # Each entry is one coloured segmentation layer for that timepoint.
        self.segmentation_masks = {}
        # segmentation_layer_shapes: {(timepoint, layer_name) -> Nx2 vertices}
        # Exact histogram-space outline of layers created from an ROI, so the
        # histogram overlay can show the true selected shape instead of a
        # convex hull re-derived from the segmented voxel intensities.
        self.segmentation_layer_shapes = {}

        # Display pyramid for large datasets: median-binned copies of every
        # timepoint used only for visualization (segmentation stays at full
        # resolution). None when the data is small enough to show directly.
        self.display_data = None            # ([neutron_3d...], [xray_3d...])
        self.display_bin_factor = 1
        self._display_mask_cache = {}       # {(t, name) -> (mask_ref, binned)}
        # Convex hulls derived from layers that have no drawn ROI shape
        self._derived_outline_cache = {}    # {(t, name) -> (mask_ref, verts)}
        
        # Mode setting
        self.mode = '4D'  # '3D' or '4D' - determines if temporal dimension is used
        
        # Processing settings
        self.force_cpu = False  # Force CPU processing (for GPU memory issues)
        self.gpu_device = 0  # GPU device ID (default to first GPU)
        
        # Detect available GPUs
        self.available_gpus = self._detect_gpus()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface - 3-panel splitter layout."""
        self.setWindowTitle(f"BiTS 3D/4D v{config.APP_VERSION}")
        # Geometry is set once the layout exists — see
        # _apply_layout_for_screen, which fits the real screen.

        # Import extra widgets needed here
        from PyQt5.QtWidgets import (
            QTabWidget, QSplitter, QComboBox, QSpinBox, QFormLayout
        )
        from PyQt5.QtCore import Qt

        self._create_menu_bar()
        self._create_toolbar()

        # ── Top-level splitter: LEFT | CENTRE | RIGHT ─────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # ══════════════════════════════════════════════════════════════════════
        # LEFT PANEL — 2-D histograms + time navigation (always visible)
        # ══════════════════════════════════════════════════════════════════════
        left_widget = QWidget()
        left_vbox = QVBoxLayout()
        left_vbox.setContentsMargins(4, 4, 4, 4)
        left_vbox.setSpacing(4)

        # --- Histograms ---
        hist_group = QGroupBox("Bivariate histogram")
        hist_layout = QVBoxLayout()
        hist_layout.setContentsMargins(4, 4, 4, 4)
        hist_layout.setSpacing(3)
        self.dual_histogram = DualHistogramWidget()
        self.dual_histogram.roi_updated.connect(self._on_roi_updated)
        # The panel owns the ROIs, this window owns the segmentation layers,
        # so removing a class has to ask here what it would throw away.
        self.dual_histogram.layer_count_provider = self._count_layers_for_class
        self.dual_histogram.class_removed.connect(self._on_class_removed)
        hist_layout.addWidget(self.dual_histogram)

        seg_btns = QHBoxLayout()
        seg_btns.addStretch()
        self.segment_current_btn = QPushButton("✂ Segment Current")
        self.segment_current_btn.setEnabled(False)
        self.segment_current_btn.setToolTip("Apply current ROI as a segmentation layer on this timepoint")
        self.segment_current_btn.clicked.connect(self._segment_current_volume)
        seg_btns.addWidget(self.segment_current_btn)
        self.segment_all_btn = QPushButton("✂✂ Segment All")
        self.segment_all_btn.setEnabled(False)
        self.segment_all_btn.setToolTip("Apply current ROI as a segmentation layer on every timepoint")
        self.segment_all_btn.clicked.connect(self._segment_all_volumes)
        seg_btns.addWidget(self.segment_all_btn)
        hist_layout.addLayout(seg_btns)
        hist_group.setLayout(hist_layout)
        left_vbox.addWidget(hist_group, stretch=4)

        # Time navigation is built here but lives in a full-width strip along
        # the bottom (see the final assembly). A slider stacked inside a
        # narrow column wastes the column's height and gives the slider no
        # room to be precise with.
        self.time_navigation = TimeNavigationWidget(num_timepoints=1)
        self.time_navigation.setEnabled(False)
        self.time_navigation.timepoint_changed.connect(self._on_timepoint_changed)

        left_widget.setLayout(left_vbox)
        splitter.addWidget(left_widget)

        # ══════════════════════════════════════════════════════════════════════
        # CENTRE PANEL — slice viewer (always visible)
        # ══════════════════════════════════════════════════════════════════════
        centre_widget = QWidget()
        centre_vbox = QVBoxLayout()
        centre_vbox.setContentsMargins(2, 2, 2, 2)
        centre_vbox.setSpacing(2)

        viewer_group = QGroupBox("Volume viewer")
        viewer_layout = QVBoxLayout()
        viewer_layout.setContentsMargins(4, 4, 4, 4)
        viewer_layout.setSpacing(3)
        self.slice_viewer = SliceViewerWidget()
        self.slice_viewer.spatial_roi_to_histogram.connect(
            self._on_create_histogram_roi_from_spatial)
        self.slice_viewer.clusters_detected.connect(self._on_clusters_detected)
        viewer_layout.addWidget(self.slice_viewer)
        viewer_group.setLayout(viewer_layout)
        centre_vbox.addWidget(viewer_group)
        centre_widget.setLayout(centre_vbox)
        splitter.addWidget(centre_widget)

        # ══════════════════════════════════════════════════════════════════════
        # RIGHT PANEL — tabbed workflow tools
        # ══════════════════════════════════════════════════════════════════════
        right_tabs = QTabWidget()
        self.right_tabs = right_tabs
        right_tabs.setTabPosition(QTabWidget.North)
        right_tabs.setMinimumWidth(250)
        right_tabs.setMaximumWidth(460)

        # ── Tab 1 : Manual / ROI segmentation ─────────────────────────────────
        tab_manual = QWidget()
        tm_layout = QVBoxLayout()
        tm_layout.setSpacing(6)

        roi_info = QLabel(
            "<b>Manual ROI Segmentation</b><br>"
            "Draw ROIs on the histogram, save each as a named class, "
            "then segment to create coloured overlays in the viewer."
        )
        roi_info.setWordWrap(True)
        roi_info.setStyleSheet("color: #555; font-size: 9pt;")
        tm_layout.addWidget(roi_info)

        sel_group = QGroupBox("Selection Manager")
        sel_layout = QVBoxLayout()
        from gui.selection_manager import SelectionManagerWidget
        self.selection_manager = SelectionManagerWidget()
        self.selection_manager.selection_recalled.connect(self._on_selection_recalled)
        self.selection_manager.selections_changed.connect(self._on_selections_changed)
        self.selection_manager.save_btn.clicked.disconnect()
        self.selection_manager.save_btn.clicked.connect(self._on_save_selection_clicked)
        sel_layout.addWidget(self.selection_manager)
        sel_group.setLayout(sel_layout)
        tm_layout.addWidget(sel_group)

        stat_group = QGroupBox("Statistics")
        stat_layout = QVBoxLayout()
        from gui.statistics_panel import StatisticsPanel
        self.statistics_panel = StatisticsPanel()
        stat_layout.addWidget(self.statistics_panel)
        stat_group.setLayout(stat_layout)
        tm_layout.addWidget(stat_group)

        tab_manual.setLayout(tm_layout)
        right_tabs.addTab(_scrollable(tab_manual), "🖊 Manual ROI")

        # ── Tab 2 : Automated segmentation (Otsu / K-means) ───────────────────
        tab_auto = QWidget()
        ta_layout = QVBoxLayout()
        ta_layout.setSpacing(6)

        auto_info = QLabel(
            "<b>Automated Segmentation</b><br>"
            "Run Otsu thresholding or K-means to generate initial class masks "
            "on the current timepoint, which are then ready to train the RF."
        )
        auto_info.setWordWrap(True)
        auto_info.setStyleSheet("color: #555; font-size: 9pt;")
        ta_layout.addWidget(auto_info)

        # ── Otsu ──────────────────────────────────────────────────────────────
        otsu_group = QGroupBox("Multi-level Otsu Thresholding")
        otsu_form_layout = QVBoxLayout()
        otsu_params = QFormLayout()

        self.otsu_classes_spin = QSpinBox()
        self.otsu_classes_spin.setRange(2, 8)
        self.otsu_classes_spin.setValue(3)
        self.otsu_classes_spin.setToolTip(
            "Total number of output classes including background (class 0).\n"
            "Otsu finds (n-1) optimal thresholds."
        )
        otsu_params.addRow("Classes (incl. bg):", self.otsu_classes_spin)

        self.otsu_channel_combo = QComboBox()
        self.otsu_channel_combo.addItems(["Neutron", "X-ray", "Combined (average)"])
        self.otsu_channel_combo.setCurrentIndex(2)
        self.otsu_channel_combo.setToolTip("Which channel to compute thresholds on")
        otsu_params.addRow("Channel:", self.otsu_channel_combo)

        otsu_form_layout.addLayout(otsu_params)

        self.otsu_run_btn = QPushButton("▶ Run Otsu & Segment")
        self.otsu_run_btn.setEnabled(False)
        self.otsu_run_btn.setToolTip(
            "Run multi-level Otsu on the reference timepoint,\n"
            "create 3-D class masks, and display them in the viewer.\n"
            "The masks are immediately available to train the RF."
        )
        self.otsu_run_btn.clicked.connect(self._run_otsu_segment)
        otsu_form_layout.addWidget(self.otsu_run_btn)

        self.otsu_status_label = QLabel("Status: ready")
        self.otsu_status_label.setStyleSheet("color: gray; font-style: italic; font-size: 9pt;")
        otsu_form_layout.addWidget(self.otsu_status_label)

        otsu_group.setLayout(otsu_form_layout)
        ta_layout.addWidget(otsu_group)

        # ── K-means ───────────────────────────────────────────────────────────
        kmeans_group = QGroupBox("K-means Clustering")
        km_layout = QVBoxLayout()
        km_info = QLabel(
            "Use the <b>🔍 Auto-Detect</b> button in the Volume Viewer to run\n"
            "3-D k-means clustering. Then convert every detected cluster into\n"
            "a saved RF training class with one click."
        )
        km_info.setWordWrap(True)
        km_info.setStyleSheet("color: #555; font-size: 9pt;")
        km_layout.addWidget(km_info)

        self.copy_clusters_btn = QPushButton(
            "⚡ Copy K-means Clusters to Materials"
        )
        self.copy_clusters_btn.setEnabled(False)
        self.copy_clusters_btn.setToolTip(
            "Turn each 3-D K-means cluster into a material.\n"
            "Existing drawn and Otsu materials are kept.\n"
            "Copying again replaces the earlier cluster-derived ones.\n"
            "They appear on the Materials tab, where each one can be set\n"
            "to change or to stay unchanged like any other material."
        )
        self.copy_clusters_btn.clicked.connect(
            self._convert_kmeans_clusters_to_materials
        )
        km_layout.addWidget(self.copy_clusters_btn)

        self.kmeans_status_label = QLabel(
            "Status: run 3-D Auto-Detect first"
        )
        self.kmeans_status_label.setWordWrap(True)
        self.kmeans_status_label.setStyleSheet(
            "color: gray; font-style: italic; font-size: 9pt;"
        )
        km_layout.addWidget(self.kmeans_status_label)

        kmeans_group.setLayout(km_layout)
        ta_layout.addWidget(kmeans_group)

        ta_layout.addStretch()
        tab_auto.setLayout(ta_layout)
        right_tabs.addTab(_scrollable(tab_auto), "⚙ Auto Seg")

        # ── Tab 3 : Materials ─────────────────────────────────────────────
        self.material_panel = MaterialPanel()
        self.material_panel.refresh_requested.connect(
            self._refresh_material_panel
        )
        self.material_panel.copy_clusters_requested.connect(
            self._convert_kmeans_clusters_to_materials
        )
        self.material_panel.preview_requested.connect(
            lambda: self._run_material_tracking(preview=True)
        )
        self.material_panel.run_requested.connect(
            lambda: self._run_material_tracking(preview=False)
        )
        tab_materials = QWidget()
        tr_layout = QVBoxLayout()
        tr_layout.setContentsMargins(0, 0, 0, 0)
        tr_layout.addWidget(self.material_panel)
        tab_materials.setLayout(tr_layout)
        self._materials_tab_index = right_tabs.addTab(
            _scrollable(tab_materials), "🧱 Materials"
        )

        # ── Tab 4 : Export ────────────────────────────────────────────────────
        tab_export = QWidget()
        te_layout = QVBoxLayout()
        te_layout.setSpacing(6)

        exp_info = QLabel(
            "<b>Export Segmentations</b><br>"
            "Save segmented volumes, binary masks and per-layer label TIFFs."
        )
        exp_info.setWordWrap(True)
        exp_info.setStyleSheet("color: #555; font-size: 9pt;")
        te_layout.addWidget(exp_info)

        # Manual segmentation export
        man_exp_group = QGroupBox("Manual Segmentation")
        man_exp_layout = QVBoxLayout()
        self.export_current_btn = QPushButton("💾 Export Current Timepoint")
        self.export_current_btn.setEnabled(False)
        self.export_current_btn.clicked.connect(self._export_current_timepoint)
        man_exp_layout.addWidget(self.export_current_btn)
        self.export_all_btn = QPushButton("💾 Export All Timepoints")
        self.export_all_btn.setEnabled(False)
        self.export_all_btn.clicked.connect(self._export_all_timepoints)
        man_exp_layout.addWidget(self.export_all_btn)
        man_exp_group.setLayout(man_exp_layout)
        te_layout.addWidget(man_exp_group)

        # ROI save/load
        roi_exp_group = QGroupBox("ROI Settings")
        roi_exp_layout = QVBoxLayout()
        save_roi_btn = QPushButton("💾 Save ROI to File")
        save_roi_btn.clicked.connect(self._save_roi)
        roi_exp_layout.addWidget(save_roi_btn)
        load_roi_btn = QPushButton("📂 Load ROI from File")
        load_roi_btn.clicked.connect(self._load_roi)
        roi_exp_layout.addWidget(load_roi_btn)
        roi_exp_group.setLayout(roi_exp_layout)
        te_layout.addWidget(roi_exp_group)

        te_layout.addStretch()
        tab_export.setLayout(te_layout)
        right_tabs.addTab(_scrollable(tab_export), "💾 Export")

        splitter.addWidget(right_tabs)

        self.main_splitter = splitter
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)

        # ── Final assembly ─────────────────────────────────────────────────────
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(4, 4, 4, 2)
        container_layout.setSpacing(3)
        container_layout.addWidget(splitter, stretch=1)

        self.time_group = QGroupBox("Time")
        time_strip = QHBoxLayout(self.time_group)
        time_strip.setContentsMargins(6, 2, 6, 4)
        time_strip.addWidget(self.time_navigation)
        container_layout.addWidget(self.time_group)

        container.setLayout(container_layout)
        self.setCentralWidget(container)
        self._apply_layout_for_screen()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Ready — load a dataset to begin  (Settings → Data Mode to switch 3D/4D)"
        )
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _create_menu_bar(self):
        """Create the menu bar"""
        from PyQt5.QtWidgets import QActionGroup

        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_4d_action = QAction("Load 4D Dataset...", self)
        load_4d_action.setShortcut("Ctrl+O")
        load_4d_action.triggered.connect(self.load_dataset)
        file_menu.addAction(load_4d_action)
        
        file_menu.addSeparator()
        
        save_roi_action = QAction("Save ROI Settings...", self)
        save_roi_action.setShortcut("Ctrl+S")
        save_roi_action.triggered.connect(self._save_roi)
        file_menu.addAction(save_roi_action)
        
        load_roi_action = QAction("Load ROI Settings...", self)
        load_roi_action.triggered.connect(self._load_roi)
        file_menu.addAction(load_roi_action)
        
        file_menu.addSeparator()
        
        # Export actions
        export_current_action = QAction("Export Current Timepoint...", self)
        export_current_action.setShortcut("Ctrl+E")
        export_current_action.triggered.connect(self._export_current_timepoint)
        file_menu.addAction(export_current_action)
        
        export_all_action = QAction("Export All Timepoints...", self)
        export_all_action.setShortcut("Ctrl+Shift+E")
        export_all_action.triggered.connect(self._export_all_timepoints)
        file_menu.addAction(export_all_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        # CPU/GPU toggle
        self.force_cpu_action = QAction("Force CPU Processing", self)
        self.force_cpu_action.setCheckable(True)
        self.force_cpu_action.setChecked(False)
        self.force_cpu_action.toggled.connect(self._on_cpu_gpu_toggled)
        settings_menu.addAction(self.force_cpu_action)
        
        # GPU device selection (if multiple GPUs available)
        if len(self.available_gpus) > 1:
            settings_menu.addSeparator()
            
            # Create submenu for GPU selection
            gpu_menu = settings_menu.addMenu("🖥️ Select GPU Device")

            # Create action group for radio button behavior
            gpu_action_group = QActionGroup(self)
            gpu_action_group.setExclusive(True)
            
            # Add action for each GPU
            for gpu_info in self.available_gpus:
                gpu_id = gpu_info['id']
                gpu_name = gpu_info['name']
                gpu_mem = gpu_info['memory']
                
                action = QAction(f"GPU {gpu_id}: {gpu_name} ({gpu_mem:.1f} GB)", self)
                action.setCheckable(True)
                action.setChecked(gpu_id == 0)  # Default to GPU 0
                action.triggered.connect(lambda checked, gid=gpu_id: self._on_gpu_device_changed(gid))
                
                gpu_action_group.addAction(action)
                gpu_menu.addAction(action)
        
        settings_menu.addSeparator()
        
        # 3D/4D Mode selection
        mode_menu = settings_menu.addMenu("📊 Data Mode")
        
        # Create action group for radio button behavior
        mode_action_group = QActionGroup(self)
        mode_action_group.setExclusive(True)
        
        # 3D mode action
        self.mode_3d_action = QAction("3D Mode (Single Timepoint)", self)
        self.mode_3d_action.setCheckable(True)
        self.mode_3d_action.setToolTip("Load single 3D volume (neutron + X-ray)")
        self.mode_3d_action.triggered.connect(lambda: self._on_mode_changed('3D'))
        mode_action_group.addAction(self.mode_3d_action)
        mode_menu.addAction(self.mode_3d_action)
        
        # 4D mode action
        self.mode_4d_action = QAction("4D Mode (Time Series)", self)
        self.mode_4d_action.setCheckable(True)
        self.mode_4d_action.setChecked(True)  # Default to 4D
        self.mode_4d_action.setToolTip("Load multiple 3D volumes as time series")
        self.mode_4d_action.triggered.connect(lambda: self._on_mode_changed('4D'))
        mode_action_group.addAction(self.mode_4d_action)
        mode_menu.addAction(self.mode_4d_action)
        
        # Export menu (NEW for v14.0)
        export_menu = menubar.addMenu("📊 Export")
        
        # Selections submenu
        selections_submenu = export_menu.addMenu("Selections")
        
        save_selections_action = QAction("Save Selection Library...", self)
        save_selections_action.setShortcut("Ctrl+Shift+S")
        save_selections_action.triggered.connect(self._on_save_selection_library)
        selections_submenu.addAction(save_selections_action)
        
        load_selections_action = QAction("Load Selection Library...", self)
        load_selections_action.setShortcut("Ctrl+Shift+L")
        load_selections_action.triggered.connect(self._on_load_selection_library)
        selections_submenu.addAction(load_selections_action)
        
        export_menu.addSeparator()
        
        # Statistics submenu
        stats_submenu = export_menu.addMenu("Statistics")
        
        export_csv_action = QAction("Export to CSV...", self)
        export_csv_action.triggered.connect(self._on_export_statistics_csv)
        stats_submenu.addAction(export_csv_action)
        
        export_excel_action = QAction("Export to Excel...", self)
        export_excel_action.triggered.connect(self._on_export_statistics_excel)
        stats_submenu.addAction(export_excel_action)
        
        export_menu.addSeparator()
        
        # PDF Report
        generate_report_action = QAction("📄 Generate PDF Report...", self)
        generate_report_action.setShortcut("Ctrl+R")
        generate_report_action.triggered.connect(self._on_generate_pdf_report)
        export_menu.addAction(generate_report_action)
        
        # Analytics menu (NEW for v14.1)
        analytics_menu = menubar.addMenu("🔬 Analytics")
        
        # Morphological analysis
        morphology_action = QAction("Shape Analysis...", self)
        morphology_action.triggered.connect(self._on_morphological_analysis)
        analytics_menu.addAction(morphology_action)
        
        # Compare selections
        compare_action = QAction("Compare Selections...", self)
        compare_action.triggered.connect(self._on_compare_selections)
        analytics_menu.addAction(compare_action)

        # Temporal histogram analyses
        hist_menu = analytics_menu.addMenu("Histogram Time Analysis")

        hist_evolution_action = QAction(
            "Evolution vs First Timepoint...", self
        )
        hist_evolution_action.setToolTip(
            "Save an image comparing every timepoint's histogram to the\n"
            "first one on a log scale — shows total drift from the start.\n"
            "Red/blue areas gained/lost voxels."
        )
        hist_evolution_action.triggered.connect(
            self._on_export_histogram_evolution
        )
        hist_menu.addAction(hist_evolution_action)

        hist_increment_action = QAction(
            "Change vs Previous Timepoint (incremental)...", self
        )
        hist_increment_action.setToolTip(
            "Save an image comparing each timepoint's histogram to the one\n"
            "before it — shows *when* changes happen, so a sudden event\n"
            "stands out instead of being buried in cumulative drift."
        )
        hist_increment_action.triggered.connect(
            self._on_export_histogram_increment
        )
        hist_menu.addAction(hist_increment_action)

        hist_marginal_action = QAction(
            "Marginal Evolution vs First Timepoint...", self
        )
        hist_marginal_action.setToolTip(
            "Save marginal kymographs: each modality's 1-D histogram against\n"
            "time, compared with T0. Separates a shift in neutron from a\n"
            "shift in X-ray, which the joint histogram can hide."
        )
        hist_marginal_action.triggered.connect(
            self._on_export_marginal_evolution
        )
        hist_menu.addAction(hist_marginal_action)

        hist_marginal_increment_action = QAction(
            "Marginal Change vs Previous Timepoint...", self
        )
        hist_marginal_increment_action.setToolTip(
            "Same marginal kymographs, but each timepoint is compared with\n"
            "the one before it — shows the steps where an intensity band\n"
            "actually moves rather than the drift accumulated since T0."
        )
        hist_marginal_increment_action.triggered.connect(
            self._on_export_marginal_increment
        )
        hist_menu.addAction(hist_marginal_increment_action)

        metrics_action = QAction("Histogram & Segmentation Metrics...", self)
        metrics_action.setToolTip(
            "Compute the ground-truth-free histogram metrics (S_h, S_v, S_d,\n"
            "A_x, Delta_n) and class metrics (DB, spreads, elongation, drift)\n"
            "for the global histogram and for every timepoint.\n"
            "Writes a CSV of all values plus a plot of each metric over time."
        )
        metrics_action.triggered.connect(self._on_export_histogram_metrics)
        hist_menu.addAction(metrics_action)

        spatial_metrics_action = QAction("Spatial Metrics...", self)
        spatial_metrics_action.setToolTip(
            "Metrics computed in the volume rather than the histogram plane:\n"
            "centre of mass and its drift, radius of gyration, connected\n"
            "components, surface-to-volume, and class interface areas.\n"
            "A speckled or displaced segmentation is invisible to the\n"
            "histogram metrics but obvious here."
        )
        spatial_metrics_action.triggered.connect(self._on_export_spatial_metrics)
        hist_menu.addAction(spatial_metrics_action)

        analytics_menu.addSeparator()

        # ── Time-series segmentation ────────────────────────────────────
        model_menu = analytics_menu.addMenu("Time Series Segmentation")

        check_action = QAction("Check Data...", self)
        check_action.setToolTip(
            "How much of the volume was actually measured, whether both\n"
            "instruments cover the same region, and whether that changes\n"
            "part-way through the series."
        )
        check_action.triggered.connect(self._on_check_data)
        model_menu.addAction(check_action)

        model_run_action = QAction("Track Materials Across Time...", self)
        model_run_action.setToolTip(
            "Follow the materials you defined through every timepoint.\n"
            "The definitions stay fixed and voxels move between them, so a\n"
            "change in a volume is a change in the sample.\n"
            "Cleans up noisy assignments using neighbouring voxels, and\n"
            "checks the result before showing it."
        )
        model_run_action.triggered.connect(self._on_model_segmentation)
        model_menu.addAction(model_run_action)

        drift_action = QAction("Check Instrument Stability...", self)
        drift_action.setToolTip(
            "Measure how far the histogram moved at each timepoint, using\n"
            "materials that cannot change. Any movement of those is the\n"
            "instrument, not the sample."
        )
        drift_action.triggered.connect(self._on_estimate_drift)
        model_menu.addAction(drift_action)


        analytics_menu.addSeparator()
        
        # Time series
        time_series_menu = analytics_menu.addMenu("Time Series")
        
        track_action = QAction("Track Current Timepoint", self)
        track_action.triggered.connect(self._on_track_timepoint)
        time_series_menu.addAction(track_action)
        
        track_all_action = QAction("Track All Timepoints...", self)
        track_all_action.triggered.connect(self._on_track_all_timepoints)
        time_series_menu.addAction(track_all_action)
        
        time_series_menu.addSeparator()
        
        plot_action = QAction("Plot Time Series...", self)
        plot_action.triggered.connect(self._on_plot_time_series)
        time_series_menu.addAction(plot_action)
        
        export_ts_action = QAction("Export Time Series CSV...", self)
        export_ts_action.triggered.connect(self._on_export_time_series)
        time_series_menu.addAction(export_ts_action)
        
        clear_ts_action = QAction("Clear Time Series Data", self)
        clear_ts_action.triggered.connect(self._on_clear_time_series)
        time_series_menu.addAction(clear_ts_action)
        
        settings_menu.addSeparator()

        # GPU availability status (informational). _detect_gpus already probed
        # PyTorch and CuPy; don't fire the toggle signal here — the status bar
        # does not exist yet during menu construction.
        if self.available_gpus:
            gpu_status = f"✅ {len(self.available_gpus)} GPU(s) Available"
        else:
            gpu_status = "⚠️ GPU Not Available (using CPU)"
            self.force_cpu = True
            self.force_cpu_action.blockSignals(True)
            self.force_cpu_action.setChecked(True)
            self.force_cpu_action.blockSignals(False)
            self.force_cpu_action.setEnabled(False)

        gpu_status_action = QAction(gpu_status, self)
        gpu_status_action.setEnabled(False)  # Just informational
        settings_menu.addAction(gpu_status_action)

        # Help menu
        # ── View ────────────────────────────────────────────────────────
        view_menu = menubar.addMenu("View")

        self.compact_action = QAction("Compact layout (laptop screens)", self)
        self.compact_action.setCheckable(True)
        self.compact_action.setToolTip(
            "Folds the current-timepoint histogram away and narrows the tool\n"
            "column, so the whole window fits a 14-inch screen.\n"
            "Chosen automatically from your screen size at start-up."
        )
        self.compact_action.toggled.connect(self.set_compact_layout)
        view_menu.addAction(self.compact_action)

        view_menu.addSeparator()

        self.local_hist_action = QAction("Current-timepoint histogram", self)
        self.local_hist_action.setCheckable(True)
        self.local_hist_action.setChecked(True)
        self.local_hist_action.setToolTip(
            "The second histogram, showing only the timepoint on screen.\n"
            "Folding it away leaves more room for the one you draw on; the\n"
            "splitter handle brings it back."
        )
        self.local_hist_action.toggled.connect(self.show_local_histogram)
        view_menu.addAction(self.local_hist_action)

        tool_panel_action = QAction("Tool panel", self)
        tool_panel_action.setCheckable(True)
        tool_panel_action.setChecked(True)
        tool_panel_action.toggled.connect(self.show_tool_panel)
        view_menu.addAction(tool_panel_action)

        time_strip_action = QAction("Time strip", self)
        time_strip_action.setCheckable(True)
        time_strip_action.setChecked(True)
        time_strip_action.toggled.connect(self.show_time_strip)
        view_menu.addAction(time_strip_action)

        view_menu.addSeparator()

        fit_action = QAction("Fit window to screen", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.setToolTip(
            "Resize and re-balance the panels for the screen you are on."
        )
        fit_action.triggered.connect(self._apply_layout_for_screen)
        view_menu.addAction(fit_action)

        help_menu = menubar.addMenu("Help")

        manual_action = QAction("Manual...", self)
        manual_action.setShortcut("F1")
        manual_action.setToolTip(
            "How to do each operation, and the mathematics behind it."
        )
        manual_action.triggered.connect(self._show_manual)
        help_menu.addAction(manual_action)

        for label, section in (
            ("Getting Started", "start"),
            ("Defining Materials", "define"),
            ("Control Materials", "controls"),
            ("The Health Check", "health"),
            ("If Something Looks Wrong", "trouble"),
        ):
            action = QAction(label, self)
            action.triggered.connect(
                lambda _checked=False, target=section: self._show_manual(target)
            )
            help_menu.addAction(action)

        help_menu.addSeparator()

        maths_menu = help_menu.addMenu("Mathematics")
        for label, section in (
            ("The Bivariate Histogram", "m_hist"),
            ("Region Containment", "m_contain"),
            ("Material Definitions and the Match Score", "m_material"),
            ("Spatial Smoothing", "m_smooth"),
            ("Choosing the Smoothing Strength", "m_auto"),
            ("Which Voxels Count", "m_valid"),
            ("Instrument Drift", "m_drift"),
            ("Mixed Boundaries", "m_partial"),
            ("The Metrics", "m_metrics"),
            ("Why There Is No Classifier", "m_why"),
            ("References", "refs"),
        ):
            action = QAction(label, self)
            action.triggered.connect(
                lambda _checked=False, target=section: self._show_manual(target)
            )
            maths_menu.addAction(action)

        help_menu.addSeparator()

        about_action = QAction("About BiTS 4D", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)


    def _create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        load_action = QAction("📁 Load Dataset", self)
        load_action.triggered.connect(self.load_dataset)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        segment_action = QAction("✂ Segment Current", self)
        segment_action.triggered.connect(self._segment_current_volume)
        toolbar.addAction(segment_action)

        toolbar.addSeparator()

        # export buttons — also registered as instance attributes for
        # setEnabled calls elsewhere in the code
        export_current_action = QAction("💾 Export Current", self)
        export_current_action.triggered.connect(self._export_current_timepoint)
        toolbar.addAction(export_current_action)

        export_all_action = QAction("💾💾 Export All", self)
        export_all_action.triggered.connect(self._export_all_timepoints)
        toolbar.addAction(export_all_action)


    @pyqtSlot()
    def load_dataset(self):
        """Load dataset (3D or 4D) with progress feedback"""
        # Update dialog based on mode
        if self.mode == '3D':
            neutron_title = "Select Neutron 3D Volume (TIFF)"
            xray_title = "Select X-ray 3D Volume (TIFF)"
        else:
            neutron_title = "Select Neutron TIFF Stack (4D)"
            xray_title = "Select X-ray TIFF Stack (4D)"
        
        # Get neutron file
        neutron_path, _ = QFileDialog.getOpenFileName(
            self,
            neutron_title,
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        
        if not neutron_path:
            return
        
        # Get X-ray file
        xray_path, _ = QFileDialog.getOpenFileName(
            self,
            xray_title,
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        
        if not xray_path:
            return
        
        # Validate files
        is_valid, message = TIFF4DLoader.validate_files(neutron_path, xray_path)
        if not is_valid:
            QMessageBox.critical(self, "Validation Error", message)
            return
        
        # Estimate memory
        try:
            mem_est = TIFF4DLoader.estimate_memory(neutron_path, xray_path)
            
            # Check if data matches mode
            shape = mem_est['neutron_shape']
            num_timepoints = shape[0] if len(shape) == 4 else 1
            
            if self.mode == '3D' and num_timepoints > 1:
                reply = QMessageBox.question(
                    self,
                    "Multi-timepoint Data Detected",
                    f"This appears to be 4D data with {num_timepoints} timepoints.\n"
                    f"In 3D mode, only the first timepoint will be loaded.\n\n"
                    f"Switch to 4D mode to load all timepoints?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.mode_4d_action.setChecked(True)
                    # Will reload after mode change
                    return
            
            # Build confirmation message
            if self.mode == '3D':
                mode_info = f"Mode: 3D (single volume)"
                shape_info = f"Volume shape: {shape[-3:]} (Z×Y×X)" if len(shape) == 4 else f"Volume shape: {shape} (Z×Y×X)"
            else:
                mode_info = f"Mode: 4D (time series)"
                shape_info = f"Shape: {shape} (T×Z×Y×X)" if len(shape) == 4 else f"Shape: {shape}"
            
            msg = (
                f"Dataset information:\n"
                f"{mode_info}\n"
                f"{shape_info}\n"
                f"Memory required: {mem_est['total_gb']:.2f} GB\n\n"
                f"Load dataset?"
            )
            
            reply = QMessageBox.question(
                self, "Confirm Load", msg,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not estimate memory: {e}")
        
        # Load with progress dialog
        def load_operation(progress_callback):
            result = TIFF4DLoader.load(
                neutron_path, xray_path,
                use_memmap=False,
                progress_callback=progress_callback
            )

            # If in 3D mode and data is 4D, keep only the first timepoint
            # (with a singleton time dimension for a consistent 4-D layout).
            if self.mode == '3D' and result is not None:
                if result.neutron_data.shape[0] > 1:
                    result = Dataset4D(
                        result.neutron_data[:1],
                        result.xray_data[:1],
                        result.metadata,
                    )
            return result

        dataset = run_with_progress(
            self,
            "Loading Dataset",
            f"Loading {self.mode} bivariate tomography data...",
            load_operation
        )

        if dataset is None:
            # Cancelled by the user or failed (already reported by the dialog)
            self.status_bar.showMessage("Dataset load cancelled")
            return
        
        self.dataset = dataset
        self.segmentation_masks.clear()  # Clear any previous segmentation masks
        self._clear_layer_shapes()
        self.model_result = None
        self._last_kmeans_cluster_selections = []
        self._cluster_timepoint = None
        self.copy_clusters_btn.setEnabled(False)
        self.material_panel.set_clusters_available(False)
        self.material_panel.clear_result()
        self.material_panel.set_materials([])
        self.kmeans_status_label.setText("Status: run 3-D Auto-Detect first")
        self.kmeans_status_label.setStyleSheet(
            "color: gray; font-style: italic; font-size: 9pt;"
        )
        self.otsu_run_btn.setEnabled(True)
        self.otsu_status_label.setText("Status: ready")
        self.status_bar.showMessage(
            f"Dataset loaded: {dataset.shape} | "
            f"{dataset.get_memory_usage()['total']:.1f} MB"
        )
        
        # Initialize histogram engine. The local-histogram cache is sized to
        # hold every timepoint so histograms are computed once and then
        # served from memory while scrolling through time.
        use_gpu = config.USE_GPU_DEFAULT and not self.force_cpu
        self.histogram_engine = HistogramEngine4D(
            bins=config.DEFAULT_BINS,
            cache_size=max(
                config.DEFAULT_HISTOGRAM_CACHE_SIZE, dataset.num_timepoints
            ),
            use_gpu=use_gpu
        )

        # Compute global histogram
        try:
            self._compute_global_histogram()
        except Exception:
            import traceback
            traceback.print_exc()

        # Big-dataset preparation: cache every local histogram and build the
        # median-binned display pyramid. Both are optional accelerations —
        # cancelling either leaves the application fully functional.
        self._precompute_local_histograms()
        self._prepare_display_volumes()


        # Enable controls based on mode
        if self.mode == '3D':
            # In 3D mode, hide time navigation
            self.time_navigation.setVisible(False)
            self.time_navigation.setEnabled(False)
            self.status_bar.showMessage(
                f"3D dataset loaded: {dataset.shape[-3:]} | "
                f"{dataset.get_memory_usage()['total']:.1f} MB"
            )
        else:
            # In 4D mode, show and enable time navigation
            self.time_navigation.setVisible(True)
            self.time_navigation.setEnabled(True)
            self.time_navigation.set_num_timepoints(dataset.num_timepoints)
            self.status_bar.showMessage(
                f"4D dataset loaded: {dataset.shape} | {dataset.num_timepoints} timepoints | "
                f"{dataset.get_memory_usage()['total']:.1f} MB"
            )
        
        # Load first (or only) timepoint
        self._update_current_timepoint(0)
    
    def _compute_global_histogram(self):
        """Compute the global histogram with progress feedback.

        Note: gui.runtime_fixes replaces this method at import time with a
        variant that also records cancellation state for load_dataset; both
        implementations share this behaviour.
        """
        if not self.dataset or not self.histogram_engine:
            return None

        def compute_operation(progress_callback):
            return self.histogram_engine.compute_global_histogram(
                self.dataset.neutron_data,
                self.dataset.xray_data,
                progress_callback=progress_callback
            )

        global_hist = run_with_progress(
            self,
            "Computing Global Histogram",
            "Analyzing entire dataset...",
            compute_operation
        )

        if global_hist is not None:
            self.global_histogram = global_hist
            self.dual_histogram.set_global_histogram(global_hist)
            self.status_bar.showMessage("Global histogram computed")
        return global_hist

    # ── Big-dataset acceleration helpers ─────────────────────────────────────

    def _precompute_local_histograms(self):
        """Compute every timepoint's local histogram once and keep it cached.

        Scrolling through time then serves histograms from memory instead of
        re-reading the volumes. Cancelling is safe: missing histograms are
        computed lazily on first visit as before.
        """
        if (self.dataset is None or self.histogram_engine is None
                or self.global_histogram is None):
            return
        num_timepoints = self.dataset.num_timepoints
        if num_timepoints <= 1:
            return

        from utils.cancellation import OperationCancelled, OperationFailed

        def operation(progress_callback=None, cancel_check=None):
            self.histogram_engine.precompute_all_local_histograms(
                self.dataset.neutron_data,
                self.dataset.xray_data,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            return True

        try:
            run_with_progress(
                self,
                "Caching Histograms",
                f"Pre-computing {num_timepoints} local histograms...",
                operation,
            )
        except (OperationCancelled, OperationFailed):
            pass

    def _prepare_display_volumes(self):
        """Build median-binned display copies of every timepoint.

        The bin factor is the smallest integer that brings one display
        volume under config.DISPLAY_MAX_VOLUME_BYTES. Factor 1 (small data)
        displays the original volumes directly. Cancelling falls back to
        full-resolution display.
        """
        from utils.cancellation import OperationCancelled, OperationFailed
        from utils.display_downsampler import DisplayDownsampler

        self.display_data = None
        self.display_bin_factor = 1
        self._display_mask_cache = {}
        self.slice_viewer.display_bin_factor = 1
        if self.dataset is None:
            return

        sample = self.dataset.neutron_data[0]
        factor = DisplayDownsampler.choose_bin_factor(
            sample.shape,
            self.dataset.neutron_data.dtype.itemsize,
            config.DISPLAY_MAX_VOLUME_BYTES,
        )
        if factor == 1:
            return

        def operation(progress_callback=None, cancel_check=None):
            return DisplayDownsampler.bin_dataset(
                self.dataset.neutron_data,
                self.dataset.xray_data,
                factor,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

        try:
            result = run_with_progress(
                self,
                "Preparing Display Volumes",
                f"Median-binning volumes x{factor} for smooth display...",
                operation,
            )
        except (OperationCancelled, OperationFailed):
            result = None

        if result is not None:
            self.display_data = result
            self.display_bin_factor = factor
            self.slice_viewer.display_bin_factor = factor
            self.status_bar.showMessage(
                f"Display volumes binned x{factor} (median); segmentation "
                "still runs at full resolution"
            )

    def _display_volumes_at(self, timepoint):
        """Display (possibly binned) volumes for one timepoint."""
        if self.display_data is not None:
            neutron_list, xray_list = self.display_data
            return neutron_list[timepoint], xray_list[timepoint]
        return self.dataset.get_volume_at_time(timepoint)

    def _current_display_volumes(self):
        """Display volumes for the current timepoint."""
        return self._display_volumes_at(self.dataset.current_timepoint)

    def _binned_display_mask(self, timepoint, name, mask):
        """Full-resolution layer mask scaled to the display grid (cached)."""
        if self.display_bin_factor <= 1:
            return mask
        key = (int(timepoint), name)
        cached = self._display_mask_cache.get(key)
        if cached is not None and cached[0] is mask:
            return cached[1]
        from utils.display_downsampler import DisplayDownsampler
        binned = DisplayDownsampler.bin_mask(mask, self.display_bin_factor)
        self._display_mask_cache[key] = (mask, binned)
        return binned

    @pyqtSlot(int)
    def _on_timepoint_changed(self, timepoint):
        """Handle timepoint change"""
        self._update_current_timepoint(timepoint)

    def _update_current_timepoint(self, timepoint):
        """Update displays for current timepoint"""
        if not self.dataset or not self.histogram_engine:
            return

        self.dataset.set_timepoint(timepoint)

        # Serve the local histogram from the in-memory cache; only compute
        # when this timepoint has not been visited/precomputed yet.
        try:
            local_hist = self.histogram_engine.get_cached_local_histogram(
                timepoint
            )
            if local_hist is None:
                neutron_vol, xray_vol = self.dataset.get_current_volume()
                local_hist = self.histogram_engine.compute_local_histogram(
                    neutron_vol, xray_vol, timepoint
                )
            self.dual_histogram.set_local_histogram(local_hist)
        except RuntimeError as e:
            # Global histogram not computed yet — leave local display empty
            print(f"Local histogram unavailable: {e}", file=sys.stderr)
        except Exception:
            import traceback
            traceback.print_exc()

        # Slice viewer: base image plus this timepoint's overlays in one pass
        # (the display volumes are the binned copies for large datasets)
        self._apply_segmentation_overlays(timepoint)

        # Update histogram overlays to show RF class distributions for this timepoint

        self.status_bar.showMessage(f"Viewing timepoint {timepoint}")
    
    @pyqtSlot()
    def _on_roi_updated(self):
        """Handle ROI update.

        Clearing the active ROI must not delete computed segmentation layers;
        it only removes the transient slice highlight when no ROI remains.
        """
        roi_manager = self.dual_histogram.get_roi_manager()
        has_roi = roi_manager.has_roi()

        if not has_roi and self.dataset is not None:
            self.slice_viewer._clear_highlight()

        self.segment_current_btn.setEnabled(has_roi)
        self.segment_all_btn.setEnabled(has_roi and self.mode == '4D')
        self.selection_manager.enable_save_button(has_roi)
        if has_roi:
            self.status_bar.showMessage("ROI defined - ready to segment")
        elif not any(self.segmentation_masks.values()):
            self.export_current_btn.setEnabled(False)
            self.export_all_btn.setEnabled(False)
    
    @pyqtSlot(tuple, str, int)
    def _on_create_histogram_roi_from_spatial(self, spatial_coords, axis, slice_index):
        """
        Create histogram ROI from spatial selection (rectangle or region growing)
        
        Args:
            spatial_coords: Either (x1, y1, x2, y2) for rectangle, or ('mask', mask_array) for region growing
            axis: 'z', 'y', or 'x'
            slice_index: Index of current slice
        """
        import sys
        from utils.value_extractor import ValueExtractor
        from utils.region_growing import RegionGrowing
        
        print("=" * 70, file=sys.stderr)
        print("MAIN WINDOW: Creating histogram ROI from spatial selection", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"  spatial_coords: {spatial_coords}", file=sys.stderr)
        print(f"  axis: {axis}, slice_index: {slice_index}", file=sys.stderr)
        
        if not self.dataset:
            QMessageBox.warning(
                self,
                "No Dataset",
                "Please load a dataset first"
            )
            return
        
        try:
            # Spatial coordinates come from the slice viewer, which shows the
            # (possibly binned) display volumes — extract values from those so
            # coordinates and mask shapes match what the user drew on.
            neutron_vol, xray_vol = self._current_display_volumes()

            # Check if this is a mask-based or rectangle-based selection
            is_mask = isinstance(spatial_coords, tuple) and len(spatial_coords) == 2 and spatial_coords[0] == 'mask'
            
            if is_mask:
                # Region growing mode - extract from mask
                mask = np.asarray(spatial_coords[1])

                if mask.ndim == 3:
                    # Grown through the volume: use every selected voxel, so
                    # the ROI covers the whole region's intensity spread
                    # rather than just the slice that happens to be shown.
                    print("  Mode: Region growing (3-D mask)", file=sys.stderr)
                    neutron_values, xray_values = (
                        RegionGrowing.extract_values_from_mask(
                            neutron_vol, xray_vol, mask
                        )
                    )
                else:
                    print("  Mode: Region growing (2-D mask)", file=sys.stderr)
                    if axis == 'z':
                        neutron_slice = neutron_vol[slice_index, :, :]
                        xray_slice = xray_vol[slice_index, :, :]
                    elif axis == 'y':
                        neutron_slice = neutron_vol[:, slice_index, :]
                        xray_slice = xray_vol[:, slice_index, :]
                    else:  # 'x'
                        neutron_slice = neutron_vol[:, :, slice_index]
                        xray_slice = xray_vol[:, :, slice_index]

                    neutron_values, xray_values = (
                        RegionGrowing.extract_values_from_mask(
                            neutron_slice, xray_slice, mask
                        )
                    )
            else:
                # Rectangle mode - extract from coords
                print("  Mode: Rectangle", file=sys.stderr)
                neutron_values, xray_values = ValueExtractor.extract_from_rectangle(
                    neutron_vol,
                    xray_vol,
                    spatial_coords,
                    axis,
                    slice_index
                )
            
            if len(neutron_values) == 0:
                QMessageBox.warning(
                    self,
                    "Empty Selection",
                    "No pixels found in spatial ROI.\n"
                    "Please draw a larger region or check the selection."
                )
                return
            
            print(f"  Extracted {len(neutron_values)} pixel values", file=sys.stderr)
            
            # Create histogram ROI from values
            if is_mask:
                # For region growing, create convex hull polygon ROI
                print("  Creating convex hull polygon ROI", file=sys.stderr)
                polygon_points = RegionGrowing.create_convex_hull_roi(
                    neutron_values,
                    xray_values,
                    margin=0.05
                )
                
                print(f"  Polygon vertices: {len(polygon_points)}", file=sys.stderr)
                
                # Update histogram with polygon ROI
                roi_manager = self.dual_histogram.get_roi_manager()
                roi_manager.set_polygon_roi(polygon_points)
            else:
                # For rectangle, create bounding box ROI (as before)
                print("  Creating bounding box ROI", file=sys.stderr)
                hist_roi_coords = ValueExtractor.create_histogram_roi_from_values(
                    neutron_values,
                    xray_values,
                    margin=0.05
                )
                
                print(f"  Created histogram ROI: {hist_roi_coords}", file=sys.stderr)
                
                # Update histogram ROI
                roi_manager = self.dual_histogram.get_roi_manager()
                roi_manager.set_rectangle_roi(*hist_roi_coords)
            
            # Update histogram display (trigger ROI update)
            self.dual_histogram._on_roi_updated()
            
            # Show success message
            if is_mask:
                # Polygon ROI from region growing
                n_min, n_max = np.min(neutron_values), np.max(neutron_values)
                x_min, x_max = np.min(xray_values), np.max(xray_values)
                unit = "voxels" if mask.ndim == 3 else "pixels"
                self.status_bar.showMessage(
                    f"✅ Created polygon ROI from {len(neutron_values):,} {unit} "
                    f"(Neutron: [{n_min:.1f}, {n_max:.1f}], X-ray: [{x_min:.1f}, {x_max:.1f}])"
                )
            else:
                # Rectangle ROI
                self.status_bar.showMessage(
                    f"✅ Created histogram ROI from {len(neutron_values):,} pixels "
                    f"(Neutron: [{hist_roi_coords[0]:.1f}, {hist_roi_coords[2]:.1f}], "
                    f"X-ray: [{hist_roi_coords[1]:.1f}, {hist_roi_coords[3]:.1f}])"
                )
            
            # Enable segmentation buttons
            self.segment_current_btn.setEnabled(True)
            self.segment_all_btn.setEnabled(True)
            
            print("  SUCCESS: Histogram ROI created from spatial selection", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to create histogram ROI:\n{str(e)}"
            )
    
    def _on_selection_recalled(self, selection):
        """Handle recalling a saved selection"""
        import sys
        print(f"Recalling selection: {selection.name}", file=sys.stderr)
        
        # Restore spatial mask if available
        if selection.spatial_mask is not None and hasattr(self.slice_viewer, 'region_grow_mask'):
            self.slice_viewer.region_grow_mask = selection.spatial_mask.copy()
            self.slice_viewer._display_mask_overlay()
            self.slice_viewer.info_label.setText(f"Recalled: {selection.name}")
        
        # Restore histogram ROI if available
        if selection.histogram_roi is not None:
            roi_manager = self.dual_histogram.get_roi_manager()
            roi_manager.set_polygon_roi(selection.histogram_roi)
            self.dual_histogram._on_roi_updated()
        
        self.status_bar.showMessage(f"✅ Recalled selection: {selection.name}")
    
    def _on_selections_changed(self):
        """Handle changes to selections list"""
        # Redraw histograms if needed to show multiple selections
        self._update_histogram_overlays()
        
        # Update statistics panel (NEW for v14.1)
        self._update_statistics_panel()
    
    def _update_statistics_panel(self):
        """Update statistics panel with current data"""
        if not hasattr(self, 'statistics_panel'):
            return
        
        if not self.dataset:
            return
        
        try:
            # Selections are drawn on the displayed slices, so statistics use
            # the same (possibly binned) display volumes for matching shapes.
            neutron_vol, xray_vol = self._current_display_volumes()

            # Extract current slice
            if self.slice_viewer.current_axis == 'z':
                neutron = neutron_vol[self.slice_viewer.current_slice_index, :, :]
                xray = xray_vol[self.slice_viewer.current_slice_index, :, :]
            elif self.slice_viewer.current_axis == 'y':
                neutron = neutron_vol[:, self.slice_viewer.current_slice_index, :]
                xray = xray_vol[:, self.slice_viewer.current_slice_index, :]
            else:
                neutron = neutron_vol[:, :, self.slice_viewer.current_slice_index]
                xray = xray_vol[:, :, self.slice_viewer.current_slice_index]

            # Update panel
            self.statistics_panel.update_statistics(
                self.selection_manager.selections,
                neutron,
                xray
            )
        except Exception as e:
            print(f"Error updating statistics: {e}", file=sys.stderr)
    
    def _on_clusters_detected(self, cluster_selections):
        """
        Handle clusters detected signal from slice viewer
        
        Args:
            cluster_selections: List of (name, spatial_mask, histogram_roi, cluster_id, color) tuples
                               or (name, spatial_mask, histogram_roi, cluster_id, color, mask_3d) for 3D
        """
        import sys
        print(f"Main window: Received {len(cluster_selections)} clusters", file=sys.stderr)

        # Capture valid 3-D cluster payloads immediately. Enabling conversion
        # must not depend on the optional auxiliary cluster-map reconstruction.
        received_3d_clusters = [
            tuple(cluster_data)
            for cluster_data in cluster_selections
            if len(cluster_data) == 6
        ]
        if received_3d_clusters:
            self._last_kmeans_cluster_selections = received_3d_clusters
            self._cluster_timepoint = (
                self.dataset.current_timepoint
                if self.dataset is not None
                else 0
            )

            if hasattr(self, "copy_clusters_btn"):
                self.copy_clusters_btn.setEnabled(True)

            if hasattr(self, "kmeans_status_label"):
                self.kmeans_status_label.setText(
                    f"Status: {len(received_3d_clusters)} cluster(s) ready "
                    f"from T={self._cluster_timepoint}"
                )
                self.kmeans_status_label.setStyleSheet(
                    "color: green; font-style: italic; font-size: 9pt;"
                )

            print(
                f"  Captured {len(received_3d_clusters)} full 3-D cluster "
                f"payload(s) for RF conversion",
                file=sys.stderr,
            )

        # The plane the 2-D cluster slices were extracted on, so they are not
        # later drawn over a different plane.
        source_axis = self.slice_viewer.current_axis
        source_index = self.slice_viewer.current_slice_index

        # Add each cluster to selection manager
        for cluster_data in cluster_selections:
            # Handle both 5-tuple (2D) and 6-tuple (3D) formats
            mask_3d = None
            if len(cluster_data) == 6:
                name, spatial_mask, histogram_roi, cluster_id, color, mask_3d = cluster_data
            else:
                name, spatial_mask, histogram_roi, cluster_id, color = cluster_data

            self.selection_manager.add_selection(
                name=name,
                spatial_mask=spatial_mask,
                histogram_roi=histogram_roi,
                cluster_id=cluster_id,
                color=color,
                # 3-D clusters keep their volume mask so the highlight
                # follows slice and plane changes
                spatial_mask_3d=mask_3d,
                source_axis=source_axis,
                source_slice_index=source_index,
            )
        
        # Rebuild cluster_map_3d for RF training from 6-tuple cluster selections.
        # The masks live on whatever grid the clustering ran on (the display
        # grid for large datasets), so the map uses the masks' own shape.
        try:
            three_d_clusters = [
                cd for cd in cluster_selections if len(cd) == 6
            ]
            if three_d_clusters and self.dataset is not None:
                mask_shape = np.asarray(three_d_clusters[0][5]).shape
                cluster_map = np.full(mask_shape, -1, dtype=np.int32)
                for cd in three_d_clusters:
                    _, _, _, cid, _, mask_3d = cd
                    cluster_map[mask_3d.astype(bool)] = int(cid)
                # Replace any uncovered voxel with the modal cluster id (shouldn't happen)
                if np.any(cluster_map < 0):
                    cluster_map[cluster_map < 0] = 0
                self._last_kmeans_cluster_selections = list(three_d_clusters)
                self._cluster_timepoint = self.dataset.current_timepoint
                self.copy_clusters_btn.setEnabled(True)
                self.kmeans_status_label.setText(
                    f"Status: {len(three_d_clusters)} cluster(s) ready "
                    f"from T={self._cluster_timepoint}"
                )
                self.kmeans_status_label.setStyleSheet(
                    "color: green; font-style: italic; font-size: 9pt;"
                )
                print(
                    f"  Stored 3-D cluster map for RF training "
                    f"({len(three_d_clusters)} clusters)",
                    file=sys.stderr,
                )
        except Exception as _e:
            print(f"  Note: could not build RF cluster map: {_e}", file=sys.stderr)

        print(f"  All clusters saved to selection manager", file=sys.stderr)
    
    @pyqtSlot()
    def _convert_kmeans_clusters_to_materials(self):
        """
        Convert the most recent 3-D K-means result into RF training classes.

        Every cluster becomes one 3-D segmentation layer. Existing manual and
        Otsu layers are preserved; earlier K-means-derived layers for the same
        cluster ids are replaced so the operation is idempotent.
        """
        if self.dataset is None:
            QMessageBox.warning(
                self, "No Dataset", "Please load a dataset first."
            )
            return

        payloads = list(self._last_kmeans_cluster_selections)
        source_t = self._cluster_timepoint

        if not payloads or source_t is None:
            QMessageBox.warning(
                self,
                "No 3-D K-means Result",
                "Run Auto-Detect in 3-D K-means mode first.",
            )
            return

        try:
            from segmentation.kmeans_class_conversion import (
                build_material_layers_from_cluster_selections,
            )
            from utils.display_downsampler import DisplayDownsampler

            neutron_vol, xray_vol = self.dataset.get_volume_at_time(source_t)

            # Clustering runs on the display grid for large datasets; RF
            # trains at full resolution, so upscale the cluster masks first.
            if self.display_bin_factor > 1:
                upscaled = []
                for payload in payloads:
                    payload = list(payload)
                    payload[5] = DisplayDownsampler.upscale_mask(
                        payload[5], self.display_bin_factor, neutron_vol.shape
                    )
                    upscaled.append(tuple(payload))
                payloads = upscaled

            new_layers, summary = build_material_layers_from_cluster_selections(
                payloads,
                neutron_vol.shape,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "K-means Conversion Failed",
                f"Could not create RF classes:\n{exc}",
            )
            return

        generated_names = {
            f"K-means cluster {cluster_id}"
            for cluster_id in summary.cluster_ids
        }
        legacy_names = {
            f"3D Cluster {cluster_id}"
            for cluster_id in summary.cluster_ids
        }

        existing_layers = self.segmentation_masks.get(source_t, [])
        preserved_layers = [
            layer
            for layer in existing_layers
            if str(layer[2]) not in generated_names | legacy_names
        ]
        self.segmentation_masks[source_t] = preserved_layers + new_layers

        # The cluster handler normally saves histogram selections immediately.
        # Restore any missing selection without duplicating existing entries.
        existing_selection_keys = {
            (
                getattr(selection, "name", None),
                getattr(selection, "cluster_id", None),
            )
            for selection in self.selection_manager.selections
        }
        added_selections = 0
        for payload in sorted(payloads, key=lambda item: int(item[3])):
            (
                name,
                spatial_mask,
                histogram_roi,
                cluster_id,
                color,
                payload_mask_3d,
            ) = payload
            key = (name, cluster_id)
            if key in existing_selection_keys:
                continue
            self.selection_manager.add_selection(
                name=name,
                spatial_mask=spatial_mask,
                histogram_roi=histogram_roi,
                cluster_id=cluster_id,
                color=color,
                spatial_mask_3d=payload_mask_3d,
                source_axis=self.slice_viewer.current_axis,
                source_slice_index=self.slice_viewer.current_slice_index,
            )
            existing_selection_keys.add(key)
            added_selections += 1

        self.export_current_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)

        if self.dataset.current_timepoint == source_t:
            self._apply_segmentation_overlays(
                source_t, neutron_vol, xray_vol
            )
        self._update_class_histogram_overlays(source_t, neutron_vol, xray_vol)
        self._refresh_material_panel()

        self.kmeans_status_label.setText(
            f"Status: {len(new_layers)} material(s) copied from T={source_t}"
        )
        self.kmeans_status_label.setStyleSheet(
            "color: green; font-style: italic; font-size: 9pt;"
        )
        self.status_bar.showMessage(
            f"Converted {len(new_layers)} K-means clusters to RF classes "
            f"at T={source_t}"
        )

        coverage_percent = (
            100.0 * summary.covered_voxels / summary.total_voxels
        )
        details = (
            f"Created {len(new_layers)} RF training classes from T={source_t}.\n\n"
            f"Covered voxels: {summary.covered_voxels:,} "
            f"({coverage_percent:.2f}%)\n"
            f"Uncovered/background voxels: {summary.uncovered_voxels:,}\n"
            f"Overlapping voxels: {summary.overlapping_voxels:,}\n"
            f"New saved histogram selections: {added_selections}\n\n"
            "The RF reference timepoint has been set automatically. "
            "You can now click 'Train RF on Current Segmentation'."
        )

        if summary.overlapping_voxels:
            QMessageBox.warning(
                self,
                "K-means Classes Created with Overlap",
                details
                + "\n\nOverlapping masks are resolved by layer order during "
                  "RF label construction.",
            )
        else:
            QMessageBox.information(
                self,
                "K-means Classes Ready for RF",
                details,
            )

    def _on_save_selection_clicked(self):
        """Handle save selection button click"""
        from PyQt5.QtWidgets import QInputDialog
        
        # Get name from user
        name, ok = QInputDialog.getText(
            self, "Save Selection", "Enter name for this selection:"
        )
        
        if ok and name:
            self._save_current_selection(name)
    
    def _save_current_selection(self, name):
        """Save current selection (mask + ROI)"""
        # Get current mask from slice viewer
        spatial_mask = None
        if getattr(self.slice_viewer, 'region_grow_mask', None) is not None:
            spatial_mask = self.slice_viewer.region_grow_mask.copy()

        # Store the active histogram ROI as polygon vertices; rectangles are
        # converted to a 4-vertex polygon for a uniform representation.
        histogram_roi = None
        roi_manager = self.dual_histogram.get_roi_manager()
        if roi_manager.roi_type == 'polygon':
            histogram_roi = np.array(roi_manager.polygon_points, dtype=float)
        elif roi_manager.roi_type == 'rectangle':
            x1, y1, x2, y2 = roi_manager.rectangle
            histogram_roi = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=float)

        if spatial_mask is None and histogram_roi is None:
            QMessageBox.warning(
                self, "Nothing to Save",
                "Draw a histogram ROI or a spatial selection first."
            )
            return

        # A 3-D region-grow result keeps its volume mask so the highlight
        # follows plane changes; a 2-D one is pinned to the plane it was
        # grown on.
        self.selection_manager.add_selection(
            name=name,
            spatial_mask=spatial_mask,
            histogram_roi=histogram_roi,
            spatial_mask_3d=getattr(
                self.slice_viewer, 'region_grow_mask_3d', None
            ),
            source_axis=self.slice_viewer.current_axis,
            source_slice_index=self.slice_viewer.current_slice_index,
        )

        self.status_bar.showMessage(f"✅ Saved selection: {name}")
    
    def _update_histogram_overlays(self):
        """Show visible saved selections on the histograms and slice viewer.

        The slice viewer's highlights are re-composed (rather than replaced)
        so changing selections never erases the segmentation layers.
        """
        visible_selections = self.selection_manager.get_visible_selections()
        show_all = self.selection_manager.show_all_cb.isChecked()

        if show_all and len(visible_selections) > 0:
            histogram_overlays = []
            for selection in visible_selections:
                if selection.histogram_roi is not None:
                    color = selection.color if selection.color is not None else (1, 0, 0, 0.8)
                    histogram_overlays.append((
                        selection.name,
                        selection.histogram_roi,
                        color
                    ))

            # Update histograms
            for canvas in [self.dual_histogram.global_canvas, self.dual_histogram.local_canvas]:
                canvas.set_roi_overlays(histogram_overlays)

            # Slice viewer: segmentation layers + the visible selections
            self._refresh_slice_overlays()
        else:
            # Clear the selection overlays, keeping segmentation highlights
            for canvas in [self.dual_histogram.global_canvas, self.dual_histogram.local_canvas]:
                canvas.clear_roi_overlays()
            self._refresh_slice_overlays()

    # ── RF histogram overlay ──────────────────────────────────────────────────

    def _layer_outline(self, timepoint, name, mask_3d, neutron_vol, xray_vol):
        """Histogram outline for one segmentation layer.

        Layers created from a histogram ROI use the exact recorded ROI shape
        so the overlay is identical to what the user drew. Layers without a
        recorded shape (RF predictions, Otsu, K-means) fall back to a convex
        hull of the segmented voxel intensities — an approximation of where
        that class lives in histogram space, not a drawn selection.

        Derived hulls are cached per layer: recomputing them means gathering
        every segmented voxel's intensities from the full-resolution volume,
        which would otherwise run on each timepoint switch.
        """
        key = (int(timepoint), name)
        recorded = self.segmentation_layer_shapes.get(key)
        if recorded is not None:
            return recorded

        cached = self._derived_outline_cache.get(key)
        if cached is not None and cached[0] is mask_3d:
            return cached[1]

        from utils.clustering_3d import KMeans3D
        mask = np.asarray(mask_3d, dtype=bool)
        n_vals = np.asarray(neutron_vol)[mask].astype(np.float64)
        x_vals = np.asarray(xray_vol)[mask].astype(np.float64)
        if len(n_vals) < 4:
            return None
        vertices = KMeans3D.create_convex_hull_roi_3d(
            n_vals, x_vals, percentile=98, density_aware=True
        )
        self._derived_outline_cache[key] = (mask_3d, vertices)
        return vertices

    # ── Segmentation colour helpers ───────────────────────────────────────────

    #: The overlay palette is the class palette with an alpha for the slice
    #: viewer. It must stay derived from it: two palettes is exactly how one
    #: class ends up two colours.
    _OVERLAY_ALPHA = OVERLAY_ALPHA
    _OVERLAY_COLORS = OVERLAY_COLORS

    @staticmethod
    def _colour_for_layer(roi_manager, name, alpha=OVERLAY_ALPHA):
        """The RGBA a layer called *name* is drawn with, everywhere.

        Resolved from the class itself, never from where it sits in a list.
        Positions drift — hiding a class, removing one, or reloading regions
        from a file all reorder the layers relative to the saved classes —
        and a colour that drifts with them stops identifying anything.
        """
        return _mcolors.to_rgba(roi_manager.color_for(name), alpha)

    def _next_roi_color(self, roi_manager, layer_index: int):
        """Colour for a layer made from the unsaved active selection."""
        named = roi_manager.get_visible_named_rois()
        if named:
            return self._colour_for_layer(roi_manager, named[-1].get('name'))
        return self._OVERLAY_COLORS[layer_index % len(self._OVERLAY_COLORS)]

    def _current_roi_name(self, roi_manager) -> str:
        """Return a human-readable label for the current active ROI."""
        named = roi_manager.get_visible_named_rois()
        if named:
            return named[-1].get('name', 'ROI')
        if roi_manager.roi_type == 'polygon':
            return 'Polygon ROI'
        if roi_manager.roi_type == 'rectangle':
            return 'Rectangle ROI'
        return 'ROI'

    def _hidden_class_names(self):
        """Names of the classes currently unticked in the selection panel.

        A layer keeps the name of the ROI that produced it, so this is what
        links a hidden class to the segmentation it already created.
        """
        dual_histogram = getattr(self, "dual_histogram", None)
        if dual_histogram is None:
            return set()
        return {
            roi['name']
            for roi in dual_histogram.get_roi_manager().named_rois
            if not roi.get('visible', True)
        }

    def _count_layers_for_class(self, name):
        """How many stored segmentation layers came from class *name*."""
        return sum(
            1
            for layers in self.segmentation_masks.values()
            for layer in layers
            if layer[2] == name
        )

    @pyqtSlot(str, bool)
    def _on_class_removed(self, name, discard_segmentation):
        """A class was removed from the selection panel.

        Its segmentation layers are only deleted when the user chose to
        discard them; otherwise they stay as ordinary layers (now with no
        class controlling their visibility).
        """
        if not discard_segmentation:
            return

        removed = 0
        for timepoint in list(self.segmentation_masks):
            layers = self.segmentation_masks[timepoint]
            kept = [layer for layer in layers if layer[2] != name]
            if len(kept) == len(layers):
                continue
            removed += len(layers) - len(kept)
            self.segmentation_masks[timepoint] = kept
            for cache in (self.segmentation_layer_shapes,
                          self._derived_outline_cache,
                          self._display_mask_cache):
                cache.pop((int(timepoint), name), None)

        if removed:
            self.status_bar.showMessage(
                f"Removed '{name}' and discarded {removed} segmentation layer(s)"
            )

    def _visible_layers(self, timepoint):
        """Segmentation layers for *timepoint* whose class is not hidden.

        Unticking a class hides the segmentation it produced as well as its
        ROI; the mask stays stored, so ticking the class back on brings the
        layer straight back.
        """
        hidden = self._hidden_class_names()
        return [
            layer for layer in self.segmentation_masks.get(timepoint, [])
            if layer[2] not in hidden
        ]

    def _compose_slice_overlays(self, timepoint):
        """Build the slice-viewer overlay list for *timepoint*.

        Combines the two independent sources of highlights so neither can
        erase the other:

        * segmentation layers — whole 3-D masks on the display grid, which
          the viewer re-slices on every redraw so the highlight follows the
          slice index and the viewing plane;
        * visible saved selections — 2-D single-slice masks, shown only
          while "Show All on Histogram" is enabled.

        Layers belonging to an unticked class are left out, so hiding a
        class hides its segmentation too.
        """
        overlays = [
            (name, self._binned_display_mask(timepoint, name, mask_3d), color)
            for mask_3d, color, name in self._visible_layers(timepoint)
        ]

        manager = getattr(self, "selection_manager", None)
        if manager is not None and manager.show_all_cb.isChecked():
            for selection in manager.get_visible_selections():
                color = (
                    selection.color
                    if selection.color is not None
                    else (1, 0, 0, 0.5)
                )
                mask_3d = getattr(selection, "spatial_mask_3d", None)
                if mask_3d is not None:
                    # Whole-volume selection (e.g. 3-D k-means): re-sliced by
                    # the viewer, so it tracks slice and plane changes.
                    overlays.append((selection.name, mask_3d, color))
                elif selection.spatial_mask is not None:
                    # Single-slice selection: pinned to the plane it was made
                    # on so it is never drawn over a different one.
                    plane = (
                        getattr(selection, "source_axis", None),
                        getattr(selection, "source_slice_index", None),
                    )
                    overlays.append(
                        (selection.name, selection.spatial_mask, color,
                         plane if plane[0] is not None else None)
                    )
        return overlays

    def _refresh_slice_overlays(self):
        """Re-push the overlay set without re-rendering the base image."""
        if self.dataset is None:
            self.slice_viewer.clear_mask_overlays()
            return
        overlays = self._compose_slice_overlays(self.dataset.current_timepoint)
        if overlays:
            self.slice_viewer.set_mask_overlays(overlays)
        else:
            self.slice_viewer.clear_mask_overlays()

    def _update_class_histogram_overlays(self, timepoint, neutron_vol=None,
                                         xray_vol=None):
        """Redraw the class outlines on both histogram canvases.

        Two things are shown together:

        1. the regions you drew, exactly as drawn, and
        2. an outline around any segmentation layer that is *not* one of them
           — from Otsu, from K-means, or from a material-tracking run — so you
           can see where those boundaries fall relative to your own.

        A layer produced by a drawn region repeats that region's shape exactly,
        so it is skipped: drawing both would put two outlines on one class and
        make the second look like a disagreement.

        This method owns the overlay list while a dataset is loaded, which is
        why it re-adds the saved class regions every time — otherwise it would
        wipe the outlines the selection panel had just drawn.
        """
        if neutron_vol is None or xray_vol is None:
            try:
                neutron_vol, xray_vol = self.dataset.get_volume_at_time(timepoint)
            except Exception:
                return

        roi_manager = self.dual_histogram.get_roi_manager()
        overlays = list(roi_manager.get_named_roi_overlays())
        class_names = {
            roi['name'] for roi in roi_manager.get_visible_named_rois()
        }

        # Layers of unticked classes are left out, so hiding a class removes
        # its outline from the histogram as well as its highlight.
        for mask_3d, color, name in self._visible_layers(timepoint):
            if name in class_names:
                continue
            try:
                verts = self._layer_outline(
                    timepoint, name, mask_3d, neutron_vol, xray_vol
                )
                if verts is not None:
                    overlays.append((f"Seg: {name}", verts, color))
            except Exception:
                pass

        self.dual_histogram.global_canvas.set_roi_overlays(overlays)
        self.dual_histogram.local_canvas.set_roi_overlays(overlays)

    def _apply_segmentation_overlays(self, timepoint, neutron_vol=None, xray_vol=None):
        """Show *timepoint*'s display volumes with all of its highlights.

        The base image is always the *display* volume pair (median-binned for
        large datasets); full-resolution layer masks are scaled to the same
        grid so overlays line up with the displayed slices. The optional
        volume arguments are accepted for backward compatibility but the
        display copies are what gets shown.
        """
        display_neutron, display_xray = self._display_volumes_at(timepoint)
        overlays = self._compose_slice_overlays(timepoint)

        # Register the overlays before rendering so the base-image redraw
        # draws this timepoint's highlights (not the previous timepoint's)
        # and the whole update paints exactly once.
        self.slice_viewer.set_mask_overlays(overlays, redraw=False)
        self.slice_viewer.set_slice_data(
            display_neutron, display_xray, segmentation_vol=None
        )

    # ── Segmentation ──────────────────────────────────────────────────────────

    @staticmethod
    def _enumerate_roi_specs(roi_manager):
        """Return every ROI to segment as a list of uniform spec dicts.

        Includes every *visible* named class ROI plus the active (unsaved)
        ROI, so segmentation always covers exactly the selection displayed on
        the histogram canvases. Classes hidden in the selection panel are
        neither drawn nor segmented.
        """
        specs = []
        for roi in roi_manager.get_visible_named_rois():
            spec = {
                'name': roi['name'],
                'roi_type': roi['roi_type'],
                'color': roi.get('color', '#e6194b'),
            }
            # Copy the geometry: segmentation may run on a worker thread while
            # the user keeps editing, and these specs must describe the ROI as
            # it was when the action started.
            if roi['roi_type'] == 'polygon':
                spec['points'] = np.array(roi['points'], dtype=float)
            else:
                spec['rectangle'] = tuple(roi['rectangle'])
            specs.append(spec)

        if roi_manager.roi_type is not None:
            spec = {
                'name': 'Active ROI',
                'roi_type': roi_manager.roi_type,
                'color': config.ROI_COLOR,
            }
            if roi_manager.roi_type == 'polygon':
                spec['points'] = np.array(
                    roi_manager.polygon_points, dtype=float
                )
            else:
                spec['rectangle'] = tuple(roi_manager.rectangle)
            specs.append(spec)
        return specs

    @staticmethod
    def _roi_spec_vertices(spec):
        """Histogram-space outline (Nx2 vertices) for one ROI spec dict."""
        if spec['roi_type'] == 'polygon':
            return np.array(spec['points'], dtype=float)
        x1, y1, x2, y2 = spec['rectangle']
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)

    @staticmethod
    def _active_roi_vertices(roi_manager):
        """Outline of the active ROI, or None when no active ROI exists."""
        return roi_manager.get_active_vertices()

    def _record_layer_shape(self, timepoint, name, vertices):
        """Remember the exact histogram outline a layer was created from.

        Stores an independent copy: this is the record of what was actually
        segmented, so later edits to the ROI must never rewrite it.
        """
        if vertices is not None:
            self.segmentation_layer_shapes[(int(timepoint), name)] = (
                np.array(vertices, dtype=float)
            )

    def _clear_layer_shapes(self, timepoint=None):
        """Drop cached layer outlines and display masks.

        Clears everything, or just one timepoint's entries. Keeping the
        derived-hull and binned-mask caches in step with the layers stops
        them growing without bound over a long session.
        """
        caches = (
            self.segmentation_layer_shapes,
            self._derived_outline_cache,
            self._display_mask_cache,
        )
        if timepoint is None:
            for cache in caches:
                cache.clear()
            return
        timepoint = int(timepoint)
        for cache in caches:
            for key in [k for k in cache if k[0] == timepoint]:
                del cache[key]

    def _segment_current_volume(self):
        """Segment current volume with progress feedback.

        When multiple named class ROIs exist, each is segmented independently
        and stored as its own coloured layer so the slice viewer shows N
        distinct colours for N ROIs.
        """
        if not self.dataset:
            return

        roi_manager = self.dual_histogram.get_roi_manager()
        if not roi_manager.has_roi():
            QMessageBox.warning(self, "No ROI", "Please define an ROI first")
            return

        neutron_vol, xray_vol = self.dataset.get_current_volume()
        current_t = self.dataset.current_timepoint

        if current_t not in self.segmentation_masks:
            self.segmentation_masks[current_t] = []

        # ── Multi-class path: one mask per ROI shown on the histogram ─────────
        # This includes the active (unsaved) ROI so the segmented layers always
        # match the selection displayed on the histogram.
        if roi_manager.has_named_rois():
            roi_specs = self._enumerate_roi_specs(roi_manager)

            def multi_segment_op(progress_callback):
                from utils.roi_manager import ROIManager as _RM
                results = []
                for i, roi in enumerate(roi_specs):
                    pct = int(10 + 80 * i / len(roi_specs))
                    progress_callback(pct, f"Segmenting \'{roi['name']}\' ...")

                    # Build a temporary single-ROI manager for this one ROI
                    tmp_rm = _RM()
                    if roi['roi_type'] == 'polygon':
                        tmp_rm.set_polygon_roi(roi['points'])
                    else:
                        tmp_rm.set_rectangle_roi(*roi['rectangle'])

                    mask = self.segmentation_engine.segment_volume(
                        neutron_vol, xray_vol, tmp_rm
                    )
                    results.append((mask, roi['color'], roi['name']))
                return results

            layers = run_with_progress(
                self,
                "Segmenting Volume",
                f"Segmenting {len(roi_specs)} ROIs for T={current_t} ...",
                multi_segment_op,
            )

            if layers is None:
                return

            import matplotlib.colors as mcolors

            # Replace earlier layers with the same name so pressing the button
            # twice does not stack duplicate overlays.
            new_names = {name for _mask, _color, name in layers}
            kept = [
                layer for layer in self.segmentation_masks[current_t]
                if layer[2] not in new_names
            ]

            total_voxels = 0
            for mask, _color, name in layers:
                color_rgba = self._colour_for_layer(roi_manager, name)
                kept.append((mask, color_rgba, name))
                total_voxels += int(np.sum(mask))
            self.segmentation_masks[current_t] = kept
            for roi in roi_specs:
                self._record_layer_shape(
                    current_t, roi['name'], self._roi_spec_vertices(roi)
                )

            self.status_bar.showMessage(
                f"T={current_t}: {len(layers)} ROIs | {total_voxels:,} voxels total"
            )

        # ── Single active ROI path (original behaviour) ───────────────────────
        else:
            # Previous Segment-Current layers stay stored (clearing the active
            # ROI must not destroy them), so ask the user what to do with them
            # instead of silently stacking every past ROI in the slice viewer.
            existing = self.segmentation_masks.get(current_t, [])
            if existing:
                existing_names = ", ".join(layer[2] for layer in existing)
                reply = QMessageBox.question(
                    self,
                    "Previous Segmentation Layers",
                    f"Timepoint {current_t} already has {len(existing)} "
                    f"segmentation layer(s):\n{existing_names}\n\n"
                    "Keep them and add this ROI as an additional layer?\n"
                    "• Yes — keep the previous layers as well\n"
                    "• No — replace them with this ROI only",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Cancel:
                    return
                if reply == QMessageBox.No:
                    self.segmentation_masks[current_t] = []
                    self._clear_layer_shapes(current_t)
                existing = self.segmentation_masks[current_t]

            def segment_operation(progress_callback):
                return self.segmentation_engine.segment_volume(
                    neutron_vol, xray_vol, roi_manager,
                    progress_callback=progress_callback
                )

            mask = run_with_progress(
                self,
                "Segmenting Volume",
                f"Segmenting timepoint {current_t}...",
                segment_operation,
            )

            if mask is None:
                return

            color = self._next_roi_color(roi_manager, len(existing))
            # Unique layer name so kept layers remain distinguishable
            base_name = self._current_roi_name(roi_manager)
            existing_names = {layer[2] for layer in existing}
            roi_name = base_name
            suffix = 2
            while roi_name in existing_names:
                roi_name = f"{base_name} ({suffix})"
                suffix += 1

            self.segmentation_masks[current_t].append((mask, color, roi_name))
            self._record_layer_shape(
                current_t, roi_name, self._active_roi_vertices(roi_manager)
            )

            stats = self.segmentation_engine.get_segmentation_statistics(
                mask, neutron_vol, xray_vol
            )
            self.status_bar.showMessage(
                f"T={current_t}: {stats['num_voxels']:,} voxels "
                f"({stats['percentage']:.1f}%) | \'{roi_name}\'"
            )

        # ── Common tail ───────────────────────────────────────────────────────
        self.export_current_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)
        self._apply_segmentation_overlays(current_t, neutron_vol, xray_vol)
        self._update_class_histogram_overlays(current_t, neutron_vol, xray_vol)

    def _run_otsu_segment(self):
        """
        Run multi-level Otsu thresholding on the current timepoint, create a
        3-D segmentation layer per class, and display them in the viewer.

        The resulting masks are stored in segmentation_masks[current_t] and are
        immediately available to train the RF without any extra steps.
        """
        if not self.dataset:
            return

        try:
            from skimage.filters import threshold_multiotsu
        except ImportError:
            QMessageBox.critical(
                self, "Missing dependency",
                "scikit-image is required for Otsu thresholding.\n"
                "Install it with:  pip install scikit-image"
            )
            return

        current_t   = self.dataset.current_timepoint
        n_classes   = self.otsu_classes_spin.value()
        channel_idx = self.otsu_channel_combo.currentIndex()
        channel_map = {0: "neutron", 1: "xray", 2: "both"}
        channel     = channel_map[channel_idx]

        neutron_vol, xray_vol = self.dataset.get_current_volume()

        self.otsu_status_label.setText("Status: computing …")
        self.otsu_status_label.setStyleSheet("color: orange; font-style: italic;")
        self.otsu_run_btn.setEnabled(False)
        QApplication.processEvents()

        def _otsu_op(progress_callback):
            progress_callback(10, "Computing Otsu thresholds …")
            from segmentation.random_forest_4d import labels_from_otsu
            labels = labels_from_otsu(neutron_vol, xray_vol, n_classes, channel)
            progress_callback(80, "Building class masks …")
            return labels

        labels = run_with_progress(
            self, "Otsu Thresholding",
            f"Running {n_classes}-class Otsu on T={current_t} …",
            _otsu_op,
        )

        self.otsu_run_btn.setEnabled(True)

        if labels is None:
            self.otsu_status_label.setText("Status: failed")
            self.otsu_status_label.setStyleSheet("color: red; font-style: italic;")
            return

        # Build one layer per non-background class and store in segmentation_masks.
        # Earlier Otsu layers are replaced by name so re-running stays idempotent.
        classes = [c for c in np.unique(labels) if c != 0]
        otsu_names = {f"Otsu class {cls_id}" for cls_id in classes}
        kept_layers = [
            layer for layer in self.segmentation_masks.get(current_t, [])
            if layer[2] not in otsu_names
        ]

        roi_manager = self.dual_histogram.get_roi_manager()
        for cls_id in classes:
            mask_3d = (labels == cls_id)
            name = f"Otsu class {cls_id}"
            kept_layers.append(
                (mask_3d, self._colour_for_layer(roi_manager, name), name)
            )
        self.segmentation_masks[current_t] = kept_layers

        # Display in viewer and histogram
        self._apply_segmentation_overlays(current_t, neutron_vol, xray_vol)
        self._update_class_histogram_overlays(current_t, neutron_vol, xray_vol)

        self.export_current_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)

        n_voxels_total = int(np.sum(labels > 0))
        pct = n_voxels_total / labels.size * 100
        self.otsu_status_label.setText(
            f"Status: {len(classes)} classes | {n_voxels_total:,} voxels ({pct:.1f}%)"
        )
        self.otsu_status_label.setStyleSheet("color: green; font-style: italic;")
        self.status_bar.showMessage(
            f"Otsu T={current_t}: {len(classes)} classes, {n_voxels_total:,} voxels segmented"
        )

    # NOTE: _segment_all_volumes lives in gui/runtime_fixes.py, which
    # installs the canonical implementation onto this class at import
    # time (see gui/__init__.py). It segments every ROI shown on the
    # histogram across all timepoints and records their outlines.

    # ─────────────────────────────────────────────────────────
    #  Random Forest slots
    # ─────────────────────────────────────────────────────────

    @pyqtSlot()
    # ─────────────────────────────────────────────────────────
    #  End Random Forest slots
    # ─────────────────────────────────────────────────────────

    @pyqtSlot()
    def _save_roi(self):
        """Save ROI settings to file"""
        roi_manager = self.dual_histogram.get_roi_manager()
        
        if not roi_manager.has_roi():
            QMessageBox.warning(self, "No ROI", "No ROI to save")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI Settings",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filepath:
            try:
                roi_manager.save_to_file(filepath)
                self.status_bar.showMessage(f"ROI saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")
    
    @pyqtSlot()
    def _load_roi(self):
        """Load ROI settings from file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load ROI Settings",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filepath:
            try:
                roi_manager = self.dual_histogram.get_roi_manager()
                roi_manager.load_from_file(filepath)
                
                # Update displays
                self.dual_histogram.global_canvas.update_plot()
                self.dual_histogram.local_canvas.update_plot()
                
                self.segment_current_btn.setEnabled(True)
                self.segment_all_btn.setEnabled(True)
                self.status_bar.showMessage(f"ROI loaded from {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ROI: {e}")
    
    def _detect_gpus(self):
        """
        Detect available GPUs
        
        Returns:
            List of GPU info dictionaries with 'id', 'name', 'memory'
        """
        import sys
        gpus = []
        
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                print(f"Detected {num_gpus} CUDA GPU(s)", file=sys.stderr)
                
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        'id': i,
                        'name': props.name,
                        'memory': props.total_memory / (1024**3)  # GB
                    }
                    gpus.append(gpu_info)
                    print(f"  GPU {i}: {props.name} ({gpu_info['memory']:.1f} GB)", file=sys.stderr)
            else:
                print("No CUDA GPUs detected by PyTorch", file=sys.stderr)
        except ImportError:
            print("PyTorch not available, checking CuPy...", file=sys.stderr)
            
            try:
                import cupy as cp
                # CuPy detection
                num_gpus = cp.cuda.runtime.getDeviceCount()
                print(f"Detected {num_gpus} CUDA GPU(s) via CuPy", file=sys.stderr)
                
                for i in range(num_gpus):
                    cp.cuda.Device(i).use()
                    mem_info = cp.cuda.Device(i).mem_info
                    total_mem_gb = mem_info[1] / (1024**3)
                    
                    gpu_info = {
                        'id': i,
                        'name': f"GPU {i}",
                        'memory': total_mem_gb
                    }
                    gpus.append(gpu_info)
                    print(f"  GPU {i}: {total_mem_gb:.1f} GB", file=sys.stderr)
            except:
                print("Could not detect GPUs", file=sys.stderr)
        
        return gpus
    
    @pyqtSlot(int)
    def _on_gpu_device_changed(self, gpu_id):
        """Handle GPU device selection change"""
        import sys
        self.gpu_device = gpu_id
        
        print("=" * 60, file=sys.stderr)
        print(f"SWITCHING TO GPU DEVICE {gpu_id}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # Set PyTorch device
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                print(f"PyTorch device set to GPU {gpu_id}", file=sys.stderr)
        except ImportError:
            pass
        
        # Set CuPy device
        try:
            import cupy as cp
            cp.cuda.Device(gpu_id).use()
            print(f"CuPy device set to GPU {gpu_id}", file=sys.stderr)
        except:
            pass
        
        # Update status bar
        if self.available_gpus:
            gpu_name = self.available_gpus[gpu_id]['name']
            self.status_bar.showMessage(f"🖥️ Using GPU {gpu_id}: {gpu_name}")
        else:
            self.status_bar.showMessage(f"🖥️ Using GPU {gpu_id}")

        # Switch the backend in place. Recreating the engine here would drop
        # the global data range and cached histograms, breaking subsequent
        # local-histogram updates; CPU and GPU accumulation produce identical
        # counts, so the caches stay valid.
        if self.histogram_engine and not self.force_cpu:
            try:
                import torch
                self.histogram_engine.use_gpu = bool(torch.cuda.is_available())
            except ImportError:
                self.histogram_engine.use_gpu = False
    
    def _on_mode_changed(self, mode):
        """Handle 3D/4D mode change"""
        import sys
        
        if self.dataset is not None:
            reply = QMessageBox.question(
                self,
                "Mode Change",
                f"Switching to {mode} mode will clear the current dataset.\nContinue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                # Revert the selection
                if mode == '3D':
                    self.mode_4d_action.setChecked(True)
                else:
                    self.mode_3d_action.setChecked(True)
                return
        
        print("=" * 60, file=sys.stderr)
        print(f"SWITCHING TO {mode} MODE", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        self.mode = mode
        
        # Update UI elements
        if mode == '3D':
            # Hide time navigation in 3D mode
            self.time_navigation.setVisible(False)
            self.time_navigation.setEnabled(False)
            self.segment_all_btn.setEnabled(False)
            self.segment_all_btn.setToolTip("Not available in 3D mode")
            self.export_all_btn.setEnabled(False)
            self.export_all_btn.setToolTip("Not available in 3D mode")
            self.status_bar.showMessage("📊 Switched to 3D Mode - Single timepoint analysis")
            self.setWindowTitle(f"BiTS 3D/4D v{config.APP_VERSION} - 3D Mode")
        else:
            # Show time navigation in 4D mode
            self.time_navigation.setVisible(True)
            self.time_navigation.setEnabled(True)
            self.segment_all_btn.setEnabled(False)  # Will enable when ROI defined
            self.segment_all_btn.setToolTip("Segment all timepoints with current ROI")
            self.export_all_btn.setEnabled(False)  # Will enable when segmented
            self.export_all_btn.setToolTip("Export all segmented timepoints")
            self.status_bar.showMessage("📊 Switched to 4D Mode - Time series analysis")
            self.setWindowTitle(f"BiTS 3D/4D v{config.APP_VERSION} - 4D Mode")
        
        # Clear current dataset if loaded
        if self.dataset is not None:
            self.dataset = None
            self.global_histogram = None
            self.segmentation_masks.clear()
            self._clear_layer_shapes()
            self.display_data = None
            self.display_bin_factor = 1
            self._display_mask_cache = {}
            self.slice_viewer.display_bin_factor = 1
            self.dual_histogram.clear_roi()
            self.slice_viewer.set_slice_data(
                np.zeros((10, 10, 10)),
                np.zeros((10, 10, 10))
            )
        
        print(f"Mode set to: {mode}", file=sys.stderr)
    
    @pyqtSlot()
    @pyqtSlot(bool)
    def _on_cpu_gpu_toggled(self, force_cpu):
        """Handle CPU/GPU processing toggle"""
        import sys
        self.force_cpu = force_cpu
        
        if force_cpu:
            self.status_bar.showMessage("⚠️ CPU Processing Mode (GPU disabled)")
        else:
            self.status_bar.showMessage("✅ GPU Processing Mode (if available)")

        # Switch the backend in place (see _on_gpu_device_changed): recreating
        # the engine would lose the global data range and cached histograms.
        if self.histogram_engine:
            if force_cpu:
                self.histogram_engine.use_gpu = False
            else:
                try:
                    import torch
                    self.histogram_engine.use_gpu = bool(torch.cuda.is_available())
                except ImportError:
                    self.histogram_engine.use_gpu = False
    
    # ── layout ───────────────────────────────────────────────────────────
    #: Below this width the three-column layout stops being comfortable and
    #: the second histogram is folded away by default. A 14" laptop is
    #: typically 1366 or 1440 wide, which lands under it.
    COMPACT_WIDTH = 1500

    def _screen_size(self):
        """Usable screen area, or a conservative guess when there is none."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return 1366, 768
        available = screen.availableGeometry()
        return available.width(), available.height()

    def _apply_layout_for_screen(self):
        """Size the window and the panels to the screen actually present.

        The application used to open at a fixed 1800×960 with a minimum width
        of nearly 3900, which simply cannot be used on a laptop. Everything
        here is chosen from the real screen instead, and every choice remains
        adjustable afterwards — the splitters stay draggable and the View
        menu switches between the two arrangements.
        """
        width, height = self._screen_size()
        window_width = min(1780, max(1000, int(width * 0.94)))
        window_height = min(1000, max(620, int(height * 0.92)))
        self.resize(window_width, window_height)
        self.move(
            max(0, (width - window_width) // 2),
            max(0, (height - window_height) // 2),
        )
        self.set_compact_layout(width < self.COMPACT_WIDTH)

    def set_compact_layout(self, compact: bool):
        """Switch between the laptop and wide-monitor arrangements.

        Compact folds the local histogram away and gives the tool column the
        narrower of its two widths. Nothing is removed — the local histogram
        is a splitter handle away, and the View menu toggles it back.
        """
        self._compact_layout = bool(compact)
        self.show_local_histogram(not compact)
        total = max(self.width(), 1000)
        if compact:
            tools = 260
            histogram = int((total - tools) * 0.46)
        else:
            tools = 360
            histogram = int((total - tools) * 0.44)
        viewer = max(total - tools - histogram, 260)
        self.main_splitter.setSizes([histogram, viewer, tools])
        if hasattr(self, "compact_action"):
            self.compact_action.blockSignals(True)
            self.compact_action.setChecked(self._compact_layout)
            self.compact_action.blockSignals(False)

    def show_local_histogram(self, visible: bool):
        """Show or fold away the current-timepoint histogram."""
        self.dual_histogram.set_local_visible(visible)
        if hasattr(self, "local_hist_action"):
            self.local_hist_action.blockSignals(True)
            self.local_hist_action.setChecked(bool(visible))
            self.local_hist_action.blockSignals(False)

    def show_tool_panel(self, visible: bool):
        self.right_tabs.setVisible(bool(visible))

    def show_time_strip(self, visible: bool):
        self.time_group.setVisible(bool(visible))

    def _show_manual(self, section=None):
        """Open the manual, optionally at a particular section.

        Kept non-modal and reused across calls, so it can stay open beside
        the application while you work through it.
        """
        from gui.manual import ManualWindow

        if getattr(self, "_manual_window", None) is None:
            self._manual_window = ManualWindow(self)
        if isinstance(section, str):
            self._manual_window.show_section(section)
        self._manual_window.show()
        self._manual_window.raise_()
        self._manual_window.activateWindow()
        return self._manual_window

    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About BiTS 4D",
            f"<h2>{config.APP_TITLE}</h2>"
            f"<p>Version {config.APP_VERSION}</p>"
            f"<p>Bivariate Tomography Segmentation tool for 4D datasets</p>"
            f"<p>Supports time-resolved neutron and X-ray tomography data</p>"
            f"<p><b>Features:</b></p>"
            f"<ul>"
            f"<li>Dual histogram display (Global + Local)</li>"
            f"<li>Time navigation with playback</li>"
            f"<li>ROI-based segmentation</li>"
            f"<li>Data export (TIFF format)</li>"
            f"<li>CPU/GPU processing options</li>"
            f"<li>GPU acceleration support (optional)</li>"
            f"</ul>"
        )
    
    @pyqtSlot()
    def _write_segmentation_report(self, output_dir, class_names, timepoints,
                                   layers_by_timepoint):
        """Write the text report describing an exported segmentation.

        *layers_by_timepoint* maps a timepoint to its list of
        ``(mask, colour, name)`` layers, in the order they were exported —
        which is also the order that fixes their label values.
        """
        import os
        from utils.segmentation_report import write_segmentation_report

        voxels_per_timepoint = {
            timepoint: {
                name: int(np.count_nonzero(mask))
                for mask, _color, name in layers_by_timepoint.get(timepoint, [])
            }
            for timepoint in timepoints
        }
        # Label volumes number the selected layers 1..N in export order
        label_values = {name: index for index, name in enumerate(class_names, 1)}

        dataset_info = {}
        metadata = getattr(self.dataset, "metadata", None) or {}
        for key in ("neutron_file", "xray_file"):
            if metadata.get(key):
                dataset_info[key] = str(metadata[key])
        dataset_info["dataset shape (T,Z,Y,X)"] = str(tuple(self.dataset.shape))
        dataset_info["mode"] = self.mode

        roi_manager = self.dual_histogram.get_roi_manager()
        roi_info = {}
        for roi in roi_manager.named_rois:
            roi_info[roi['name']] = (
                f"{roi['roi_type']} ROI, class id {roi['class_id']}"
                + ("" if roi.get('visible', True) else "  (hidden)")
            )
        if roi_manager.roi_type is not None:
            roi_info["Active ROI"] = f"{roi_manager.roi_type} (unsaved)"

        settings = {
            "histogram bins": str(self.histogram_engine.bins)
            if self.histogram_engine else "n/a",
        }
        if self.global_histogram is not None:
            neutron_range, xray_range = self.global_histogram.data_range
            settings["neutron range"] = (
                f"[{neutron_range[0]:.6g}, {neutron_range[1]:.6g}]"
            )
            settings["X-ray range"] = (
                f"[{xray_range[0]:.6g}, {xray_range[1]:.6g}]"
            )
        if getattr(self, "model_result", None) is not None:
            settings["spatial smoothing"] = describe_strength(
                getattr(self.model_result, "smoothing", None)
            )
            library = getattr(self.model_result, "library", None)
            if library is not None and library.inert_names:
                settings["control materials"] = ", ".join(library.inert_names)
        if self.display_bin_factor > 1:
            settings["display binning"] = (
                f"x{self.display_bin_factor} (median) — display only; "
                "segmentation and export are full resolution"
            )

        notes = [
            "Voxel counts are for the full-resolution segmentation.",
            "Label values apply to the *_labels.tif volumes; individual "
            "class masks are written as 0/255.",
        ]

        volume_shape = self.dataset.shape[-3:]
        return write_segmentation_report(
            os.path.join(output_dir, "segmentation_report.txt"),
            class_names=class_names,
            label_values=label_values,
            voxels_per_timepoint=voxels_per_timepoint,
            volume_shape=volume_shape,
            dataset_info=dataset_info,
            roi_info=roi_info,
            settings=settings,
            notes=notes,
        )

    def _export_class_histogram(self, timepoint, name, mask_3d, output_dir,
                                path_prefix):
        """Write the bimodal histogram of one segmented class.

        Computed on the full-resolution volumes and on the global
        histogram's bin grid, so every exported class shares identical edges
        and can be compared bin-for-bin.
        """
        from utils.histogram_export import save_class_histogram

        if self.histogram_engine is None or self.global_histogram is None:
            return []

        neutron_vol, xray_vol = self.dataset.get_volume_at_time(timepoint)
        class_hist = self.histogram_engine.compute_masked_histogram(
            neutron_vol, xray_vol, mask_3d
        )
        return save_class_histogram(
            class_hist,
            f"{path_prefix}_hist",
            title=f"{name} — T={timepoint}  ({class_hist.num_voxels:,} voxels)",
        )

    def _export_current_timepoint(self):
        """Export segmented data for current timepoint with user-chosen options."""
        if not self.dataset:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first")
            return

        current_t = self.dataset.current_timepoint

        if not self.segmentation_masks.get(current_t):
            QMessageBox.warning(
                self, "No Segmentation",
                f"Timepoint {current_t} has not been segmented yet.\n"
                "Please segment it first."
            )
            return

        layers = self.segmentation_masks[current_t]

        # ── Export options dialog ────────────────────────────────────────────
        dlg = ExportOptionsDialog(layers, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return

        sel_layers   = dlg.selected_layers
        do_mask      = dlg.export_mask
        do_neutron   = dlg.export_neutron
        do_xray      = dlg.export_xray
        do_labels    = dlg.export_labels
        do_histogram = dlg.export_histogram
        do_report    = dlg.export_report

        if not sel_layers:
            QMessageBox.warning(self, "Nothing selected", "No layers were selected for export.")
            return
        if not (do_mask or do_neutron or do_xray or do_labels or do_histogram
                or do_report):
            QMessageBox.warning(self, "Nothing selected", "No output modalities were selected.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "", QFileDialog.ShowDirsOnly
        )
        if not output_dir:
            return

        try:
            import os, tifffile
            from utils.histogram_export import sanitize_name, save_bin_edges
            neutron_vol, xray_vol = self.dataset.get_volume_at_time(current_t)
            base = f"timepoint_{current_t:03d}"
            files_written = []

            if do_histogram and self.global_histogram is not None:
                files_written += save_bin_edges(self.global_histogram, output_dir)

            for mask_3d, color, name in sel_layers:
                # Use the class's own name (e.g. "Lithium") in the file name
                safe_name = sanitize_name(name)
                mask_bool = mask_3d.astype(bool)
                pfx = os.path.join(output_dir, f"{base}_{safe_name}")

                if do_mask:
                    p = f"{pfx}_mask.tif"
                    tifffile.imwrite(p, mask_bool.astype(np.uint8) * 255)
                    files_written.append(os.path.basename(p))
                if do_neutron:
                    vol = neutron_vol.copy(); vol[~mask_bool] = 0
                    p = f"{pfx}_neutron.tif"
                    tifffile.imwrite(p, vol)
                    files_written.append(os.path.basename(p))
                if do_xray:
                    vol = xray_vol.copy(); vol[~mask_bool] = 0
                    p = f"{pfx}_xray.tif"
                    tifffile.imwrite(p, vol)
                    files_written.append(os.path.basename(p))
                if do_histogram:
                    files_written += self._export_class_histogram(
                        current_t, name, mask_bool, output_dir, pfx
                    )

            if do_labels:
                label_vol = np.zeros(neutron_vol.shape, dtype=np.uint8)
                for idx, (mask_3d, _, _) in enumerate(sel_layers, start=1):
                    label_vol[mask_3d.astype(bool)] = idx
                p = os.path.join(output_dir, f"{base}_labels.tif")
                tifffile.imwrite(p, label_vol)
                files_written.append(os.path.basename(p))

            if do_report:
                report = self._write_segmentation_report(
                    output_dir,
                    [name for _m, _c, name in sel_layers],
                    [current_t],
                    {current_t: sel_layers},
                )
                files_written.append(os.path.basename(report))

            preview = "\n".join(f"• {f}" for f in files_written[:10])
            if len(files_written) > 10:
                preview += f"\n  … and {len(files_written)-10} more"
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(files_written)} file(s) for timepoint {current_t}:\n\n"
                f"{preview}\n\nSaved to: {output_dir}"
            )
            self.status_bar.showMessage(
                f"Exported {len(files_written)} files for T={current_t}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")
            import traceback; traceback.print_exc()
    
    @pyqtSlot()
    def _export_all_timepoints(self):
        """Export segmented data for all timepoints with user-chosen options."""
        if not self.dataset:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first")
            return

        if not any(self.segmentation_masks.values()):
            QMessageBox.warning(
                self, "No Segmentation",
                "No timepoints have been segmented yet.\n"
                "Please segment at least one timepoint first."
            )
            return

        # Collect the union of all unique layer names across timepoints so the
        # user can pick which ones to include across the whole batch.
        # Use the layers from the first non-empty timepoint as representative.
        representative_layers = next(
            layers for layers in self.segmentation_masks.values() if layers
        )

        # ── Export options dialog ────────────────────────────────────────────
        dlg = ExportOptionsDialog(representative_layers, parent=self)
        dlg.setWindowTitle(
            "Export Options — applies to all segmented timepoints"
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        # Match selected layers by name across every timepoint
        sel_names    = {name for _, _, name in dlg.selected_layers}
        do_mask      = dlg.export_mask
        do_neutron   = dlg.export_neutron
        do_xray      = dlg.export_xray
        do_labels    = dlg.export_labels
        do_histogram = dlg.export_histogram
        do_report    = dlg.export_report

        if not sel_names:
            QMessageBox.warning(self, "Nothing selected", "No layers were selected.")
            return
        if not (do_mask or do_neutron or do_xray or do_labels or do_histogram
                or do_report):
            QMessageBox.warning(self, "Nothing selected", "No output modalities were selected.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "", QFileDialog.ShowDirsOnly
        )
        if not output_dir:
            return

        num_timepoints = sum(1 for v in self.segmentation_masks.values() if v)

        try:
            import os, tifffile
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            from utils.histogram_export import sanitize_name, save_bin_edges

            progress = QProgressDialog(
                "Exporting…", "Cancel", 0, num_timepoints, self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            total_files = 0

            # The bin grid is shared by every class and timepoint, so it is
            # written once for the whole export.
            if do_histogram and self.global_histogram is not None:
                total_files += len(
                    save_bin_edges(self.global_histogram, output_dir)
                )

            # One fixed class order for the whole batch, so label values are
            # stable across timepoints and the report legend applies to all.
            class_order = [
                name for _m, _c, name in dlg.selected_layers if name in sel_names
            ]
            label_values = {
                name: index for index, name in enumerate(class_order, start=1)
            }
            exported_layers = {}

            for i, (t, layers) in enumerate(sorted(self.segmentation_masks.items())):
                if not layers:
                    continue
                if progress.wasCanceled():
                    break

                progress.setLabelText(f"Exporting timepoint {t}…  ({i+1}/{num_timepoints})")
                progress.setValue(i)
                QApplication.processEvents()

                neutron_vol, xray_vol = self.dataset.get_volume_at_time(t)
                base = f"timepoint_{t:03d}"

                # Filter this timepoint's layers by the selected names
                t_layers = [(m, c, n) for m, c, n in layers if n in sel_names]
                if not t_layers:
                    continue

                for mask_3d, color, name in t_layers:
                    # Use the class's own name (e.g. "Lithium") in the file name
                    safe = sanitize_name(name)
                    pfx  = os.path.join(output_dir, f"{base}_{safe}")
                    mask_bool = mask_3d.astype(bool)

                    if do_mask:
                        tifffile.imwrite(f"{pfx}_mask.tif",
                                         mask_bool.astype(np.uint8) * 255)
                        total_files += 1
                    if do_neutron:
                        vol = neutron_vol.copy(); vol[~mask_bool] = 0
                        tifffile.imwrite(f"{pfx}_neutron.tif", vol)
                        total_files += 1
                    if do_xray:
                        vol = xray_vol.copy(); vol[~mask_bool] = 0
                        tifffile.imwrite(f"{pfx}_xray.tif", vol)
                        total_files += 1
                    if do_histogram:
                        total_files += len(self._export_class_histogram(
                            t, name, mask_bool, output_dir, pfx
                        ))

                exported_layers[t] = t_layers

                if do_labels:
                    label_vol = np.zeros(neutron_vol.shape, dtype=np.uint8)
                    # Label values come from the fixed class order, not this
                    # timepoint's list, so a value means the same class in
                    # every exported volume even if a class is missing here.
                    for mask_3d, _color, layer_name in t_layers:
                        label_vol[mask_3d.astype(bool)] = label_values[layer_name]
                    tifffile.imwrite(
                        os.path.join(output_dir, f"{base}_labels.tif"), label_vol
                    )
                    total_files += 1

            progress.setValue(num_timepoints)

            if do_report and exported_layers:
                self._write_segmentation_report(
                    output_dir, class_order,
                    sorted(exported_layers), exported_layers,
                )
                total_files += 1

            if total_files > 0:
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {num_timepoints} timepoint(s).\n"
                    f"Total files written: {total_files}\n\n"
                    f"Saved to: {output_dir}"
                )
                self.status_bar.showMessage(
                    f"Exported {total_files} files to {output_dir}"
                )
            else:
                QMessageBox.information(self, "Export Cancelled",
                                         "Export was cancelled or nothing matched.")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")
            import traceback; traceback.print_exc()

    # ========== v14.0: Selection Library & Reporting Methods ==========

    def _on_save_selection_library(self):
        """Save current selections to .bits file"""
        from PyQt5.QtWidgets import QFileDialog
        from utils.selection_library import SelectionLibrary
        
        if len(self.selection_manager.selections) == 0:
            QMessageBox.information(self, "No Selections", 
                                   "No selections to save. Create selections first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Selection Library", "", "BiTS Files (*.bits)"
        )
        
        if filepath:
            try:
                saved_path = SelectionLibrary.save_selections(
                    self.selection_manager.selections, filepath
                )
                QMessageBox.information(self, "Saved", 
                                       f"Saved {len(self.selection_manager.selections)} selections to:\n{saved_path}")
                self.status_bar.showMessage(f"Selection library saved: {saved_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
    
    def _on_load_selection_library(self):
        """Load selections from .bits file"""
        from PyQt5.QtWidgets import QFileDialog
        from utils.selection_library import SelectionLibrary
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Selection Library", "", "BiTS Files (*.bits)"
        )
        
        if filepath:
            try:
                selections_data = SelectionLibrary.load_selections(filepath)
                
                # Clear existing selections
                self.selection_manager.clear_all()
                
                # Add loaded selections
                for sel_data in selections_data:
                    self.selection_manager.add_selection(
                        name=sel_data['name'],
                        spatial_mask=sel_data['spatial_mask'],
                        histogram_roi=sel_data['histogram_roi'],
                        cluster_id=sel_data.get('cluster_id'),
                        color=sel_data.get('color')
                    )
                
                QMessageBox.information(self, "Loaded", 
                                       f"Loaded {len(selections_data)} selections from:\n{filepath}")
                self.status_bar.showMessage(f"Selection library loaded: {len(selections_data)} selections")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")
                import traceback
                traceback.print_exc()
    
    def _on_export_statistics_csv(self):
        """Export statistics to CSV"""
        from PyQt5.QtWidgets import QFileDialog
        from utils.selection_library import SelectionLibrary
        
        if len(self.selection_manager.selections) == 0:
            QMessageBox.information(self, "No Selections", 
                                   "No selections to export.")
            return
        
        if not self.dataset:
            QMessageBox.information(self, "No Data", "Load dataset first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics to CSV", "", "CSV Files (*.csv)"
        )
        
        if filepath:
            try:
                # Selection masks live on the displayed (possibly binned) grid
                neutron_vol, xray_vol = self._current_display_volumes()

                # Extract current slice
                if self.slice_viewer.current_axis == 'z':
                    neutron = neutron_vol[self.slice_viewer.current_slice_index, :, :]
                    xray = xray_vol[self.slice_viewer.current_slice_index, :, :]
                elif self.slice_viewer.current_axis == 'y':
                    neutron = neutron_vol[:, self.slice_viewer.current_slice_index, :]
                    xray = xray_vol[:, self.slice_viewer.current_slice_index, :]
                else:
                    neutron = neutron_vol[:, :, self.slice_viewer.current_slice_index]
                    xray = xray_vol[:, :, self.slice_viewer.current_slice_index]

                saved_path, skipped = SelectionLibrary.export_statistics_csv(
                    self.selection_manager.selections, neutron, xray, filepath
                )
                
                msg = f"Statistics exported to:\n{saved_path}"
                if skipped > 0:
                    msg += f"\n\n⚠ Warning: {skipped} selection(s) skipped due to dimension mismatch.\n"
                    msg += "Selections were created on a different slice/axis.\n"
                    msg += "Navigate to the original slice to export all selections."
                
                QMessageBox.information(self, "Exported", msg)
                self.status_bar.showMessage(f"Statistics exported: {saved_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")
                import traceback
                traceback.print_exc()
    
    def _on_export_statistics_excel(self):
        """Export statistics to Excel"""
        from PyQt5.QtWidgets import QFileDialog
        from utils.selection_library import SelectionLibrary
        
        if len(self.selection_manager.selections) == 0:
            QMessageBox.information(self, "No Selections", 
                                   "No selections to export.")
            return
        
        if not self.dataset:
            QMessageBox.information(self, "No Data", "Load dataset first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics to Excel", "", "Excel Files (*.xlsx)"
        )
        
        if filepath:
            try:
                # Selection masks live on the displayed (possibly binned) grid
                neutron_vol, xray_vol = self._current_display_volumes()

                # Extract current slice
                if self.slice_viewer.current_axis == 'z':
                    neutron = neutron_vol[self.slice_viewer.current_slice_index, :, :]
                    xray = xray_vol[self.slice_viewer.current_slice_index, :, :]
                elif self.slice_viewer.current_axis == 'y':
                    neutron = neutron_vol[:, self.slice_viewer.current_slice_index, :]
                    xray = xray_vol[:, self.slice_viewer.current_slice_index, :]
                else:
                    neutron = neutron_vol[:, :, self.slice_viewer.current_slice_index]
                    xray = xray_vol[:, :, self.slice_viewer.current_slice_index]

                saved_path, skipped = SelectionLibrary.export_statistics_excel(
                    self.selection_manager.selections, neutron, xray, filepath
                )
                
                msg = f"Statistics exported to:\n{saved_path}"
                if skipped > 0:
                    msg += f"\n\n⚠ Warning: {skipped} selection(s) skipped due to dimension mismatch.\n"
                    msg += "Selections were created on a different slice/axis.\n"
                    msg += "Navigate to the original slice to export all selections."
                
                QMessageBox.information(self, "Exported", msg)
                self.status_bar.showMessage(f"Statistics exported: {saved_path}")
            except ImportError:
                QMessageBox.warning(self, "Missing Package", 
                                   "pandas and openpyxl required for Excel export.\n\n"
                                   "Install with: pip install pandas openpyxl")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")
                import traceback
                traceback.print_exc()
    
    def _on_generate_pdf_report(self):
        """Generate PDF report with figures and statistics"""
        from PyQt5.QtWidgets import QFileDialog
        from utils.pdf_reporter import PDFReporter
        
        if len(self.selection_manager.selections) == 0:
            QMessageBox.information(self, "No Selections", 
                                   "No selections to report.")
            return
        
        if not self.dataset:
            QMessageBox.information(self, "No Data", "Load dataset first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Generate PDF Report", "", "PDF Files (*.pdf)"
        )
        
        if filepath:
            try:
                self.status_bar.showMessage("Generating PDF report...")

                # Selection masks live on the displayed (possibly binned) grid
                neutron_vol, xray_vol = self._current_display_volumes()

                # Extract current slice
                if self.slice_viewer.current_axis == 'z':
                    neutron = neutron_vol[self.slice_viewer.current_slice_index, :, :]
                    xray = xray_vol[self.slice_viewer.current_slice_index, :, :]
                elif self.slice_viewer.current_axis == 'y':
                    neutron = neutron_vol[:, self.slice_viewer.current_slice_index, :]
                    xray = xray_vol[:, self.slice_viewer.current_slice_index, :]
                else:
                    neutron = neutron_vol[:, :, self.slice_viewer.current_slice_index]
                    xray = xray_vol[:, :, self.slice_viewer.current_slice_index]
                
                # Get histogram data if available
                histogram_data = None
                if self.global_histogram is not None:
                    histogram_data = (
                        self.global_histogram.histogram,
                        self.global_histogram.x_edges,  # Fixed: was neutron_edges
                        self.global_histogram.y_edges   # Fixed: was xray_edges
                    )
                
                saved_path = PDFReporter.generate_report(
                    filepath,
                    self.selection_manager.selections,
                    neutron,
                    xray,
                    histogram_data
                )
                
                QMessageBox.information(self, "Report Generated", 
                                       f"PDF report created:\n{saved_path}")
                self.status_bar.showMessage(f"PDF report generated: {saved_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to generate report:\n{e}")
                import traceback
                traceback.print_exc()
    
    # ========== v14.1: Advanced Analytics Methods ==========
    
    def _run_histogram_time_analysis(self, title, dialog_caption, render,
                                     explanation):
        """Shared driver for the temporal histogram analyses.

        Collects every timepoint's local histogram (from cache, computing any
        that are missing), then hands the list to *render*, which draws the
        figure and returns the saved path.
        """
        from PyQt5.QtWidgets import QFileDialog
        from utils.cancellation import OperationCancelled, OperationFailed

        if not self.dataset or not self.histogram_engine:
            QMessageBox.information(self, "No Data", "Load a dataset first.")
            return
        if self.global_histogram is None:
            QMessageBox.information(
                self, "No Histograms", "Compute the global histogram first."
            )
            return
        num_timepoints = self.dataset.num_timepoints
        if num_timepoints < 2:
            QMessageBox.information(
                self, "Single Timepoint",
                f"{title} needs at least two timepoints."
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, dialog_caption, "", "PNG Files (*.png);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith((".png", ".jpg", ".pdf", ".svg")):
            filepath += ".png"

        def operation(progress_callback=None, cancel_check=None):
            histograms = []
            for timepoint in range(num_timepoints):
                if cancel_check:
                    cancel_check()
                hist = self.histogram_engine.get_cached_local_histogram(
                    timepoint
                )
                if hist is None:
                    hist = self.histogram_engine.compute_local_histogram(
                        self.dataset.neutron_data[timepoint],
                        self.dataset.xray_data[timepoint],
                        timepoint,
                    )
                histograms.append(hist)
                if progress_callback:
                    progress_callback(
                        int(90 * (timepoint + 1) / num_timepoints),
                        f"Collecting histogram {timepoint + 1}/{num_timepoints}",
                    )
            if progress_callback:
                progress_callback(95, "Rendering figure...")
            return render(histograms, filepath)

        try:
            saved = run_with_progress(
                self, title, f"Building {title.lower()}...", operation
            )
        except (OperationCancelled, OperationFailed):
            return

        if saved:
            self.status_bar.showMessage(f"{title} saved: {saved}")
            QMessageBox.information(
                self, "Image Saved",
                f"{title} saved to:\n{saved}\n\n{explanation}"
            )

    def _collect_metrics_rows(self, progress_callback=None, cancel_check=None):
        """Metrics for the global histogram and for every timepoint.

        The global row analyses the whole-dataset histogram with the classes
        of the reference timepoint; each timepoint row analyses that
        timepoint's local histogram and its own classes. Delta_n and the
        drift metrics are measured against the first segmented timepoint.
        """
        from utils.histogram_metrics import (
            MetricsRow, compute_class_metrics, compute_shape_metrics,
        )

        num_timepoints = self.dataset.num_timepoints
        rows = []

        def class_masks_for(timepoint):
            return {
                name: mask for mask, _color, name in self._visible_layers(timepoint)
            }

        # Global scope: whole-dataset histogram, classes of the first
        # segmented timepoint (they are what the histogram ROIs describe).
        reference_t = next(
            (t for t in range(num_timepoints) if class_masks_for(t)), None
        )
        global_row = MetricsRow(scope="global")
        global_row.scalars.update(compute_shape_metrics(self.global_histogram))
        if reference_t is not None:
            neutron_vol, xray_vol = self.dataset.get_volume_at_time(reference_t)
            scalars, per_class, _ = compute_class_metrics(
                neutron_vol, xray_vol, class_masks_for(reference_t)
            )
            global_row.scalars.update(scalars)
            global_row.per_class = per_class
        rows.append(global_row)

        # Per-timepoint scope
        reference_hist = None
        reference_centroids = None
        for timepoint in range(num_timepoints):
            if cancel_check:
                cancel_check()

            local_hist = self.histogram_engine.get_cached_local_histogram(timepoint)
            if local_hist is None:
                local_hist = self.histogram_engine.compute_local_histogram(
                    self.dataset.neutron_data[timepoint],
                    self.dataset.xray_data[timepoint],
                    timepoint,
                )
            if reference_hist is None:
                reference_hist = local_hist

            row = MetricsRow(scope="timepoint", timepoint=timepoint)
            row.scalars.update(
                compute_shape_metrics(local_hist, reference=reference_hist)
            )

            masks = class_masks_for(timepoint)
            if masks:
                neutron_vol, xray_vol = self.dataset.get_volume_at_time(timepoint)
                scalars, per_class, centroids = compute_class_metrics(
                    neutron_vol, xray_vol, masks,
                    reference_centroids=reference_centroids,
                )
                row.scalars.update(scalars)
                row.per_class = per_class
                if reference_centroids is None:
                    reference_centroids = centroids
            rows.append(row)

            if progress_callback:
                progress_callback(
                    int(95 * (timepoint + 1) / num_timepoints),
                    f"Metrics for timepoint {timepoint + 1}/{num_timepoints}",
                )
        return rows

    def _on_export_histogram_metrics(self):
        """Compute the metrics and save them as a CSV plus evolution plots."""
        from PyQt5.QtWidgets import QFileDialog
        from utils.cancellation import OperationCancelled, OperationFailed
        from utils.histogram_metrics import (
            plot_metric_evolution, write_metrics_csv,
        )

        if not self.dataset or not self.histogram_engine:
            QMessageBox.information(self, "No Data", "Load a dataset first.")
            return
        if self.global_histogram is None:
            QMessageBox.information(
                self, "No Histograms", "Compute the global histogram first."
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".csv"):
            filepath += ".csv"
        plot_path = filepath[:-4] + "_evolution.png"

        def operation(progress_callback=None, cancel_check=None):
            rows = self._collect_metrics_rows(progress_callback, cancel_check)
            if progress_callback:
                progress_callback(97, "Writing CSV and plots...")
            write_metrics_csv(rows, filepath)
            return rows, plot_metric_evolution(rows, plot_path)

        try:
            result = run_with_progress(
                self, "Histogram Metrics",
                "Computing metrics for every timepoint...", operation,
            )
        except (OperationCancelled, OperationFailed):
            return
        if result is None:
            return

        rows, saved_plot = result
        segmented = sum(
            1 for row in rows if row.scope == "timepoint" and row.per_class.get("voxels_k")
        )
        message = (
            f"Metrics written to:\n{filepath}\n\n"
            f"{len(rows) - 1} timepoint(s) analysed, "
            f"{segmented} with segmentation classes."
        )
        if saved_plot:
            message += f"\n\nEvolution plot:\n{saved_plot}"
        else:
            message += "\n\n(No evolution plot — it needs at least 2 timepoints.)"
        if not segmented:
            message += (
                "\n\nClass metrics (DB, spreads, elongation, drift) need "
                "segmented classes; only the histogram shape metrics were "
                "computed."
            )
        self.status_bar.showMessage(f"Metrics saved: {filepath}")
        QMessageBox.information(self, "Metrics Saved", message)

    # ── model-based time-series segmentation ─────────────────────────────
    def _model_class_masks(self, timepoint):
        """Visible class layers at *timepoint* as ``{name: mask}``."""
        return {
            name: np.asarray(mask, dtype=bool)
            for mask, _color, name in self._visible_layers(timepoint)
        }

    def _on_check_data(self, silent=False):
        """Step 2 of the workflow: what is actually measurable here.

        Runs on load and from the menu. The fact this surfaces — that the two
        instruments may not cover the same region — is invisible otherwise,
        and quietly corrupts every paired quantity computed from it.
        """
        from model import channel_coverage, find_acquisition_steps, validity_report

        if not self.dataset:
            if not silent:
                QMessageBox.information(self, "No Data", "Load a dataset first.")
            return None

        reports = []
        for timepoint in range(self.dataset.num_timepoints):
            neutron, xray = self.dataset.get_volume_at_time(timepoint)
            reports.append(validity_report(neutron, xray))
        first = reports[0]
        steps = find_acquisition_steps(reports)

        lines = [
            f"Volume size: {tuple(int(v) for v in self.dataset.shape[-3:])} "
            f"(Z, Y, X), {self.dataset.num_timepoints} timepoint(s).",
            "",
            f"{100 * first['overlap_fraction']:.1f}% of the array has data "
            f"from both instruments and can be used.",
        ]
        one_sided = first["neutron_only"] + first["xray_only"]
        if one_sided:
            share = 100 * one_sided / max(first["total_voxels"], 1)
            which = []
            if first["neutron_only"]:
                which.append(
                    f"{100 * first['neutron_only_fraction']:.1f}% has neutron "
                    f"data but no X-ray data"
                )
            if first["xray_only"]:
                which.append(
                    f"{100 * first['xray_only_fraction']:.1f}% has X-ray data "
                    f"but no neutron data"
                )
            lines += [
                "",
                f"{share:.1f}% of the array was measured by only one "
                f"instrument — " + ", and ".join(which) + ".",
                "These voxels are excluded: a material can only be identified "
                "where both measurements exist.",
            ]
        if steps:
            timepoint, before, after = steps[0]
            lines += [
                "",
                f"The amount of usable data changes at timepoint {timepoint} "
                f"({100 * (1 - before):.0f}% to {100 * (1 - after):.0f}%).",
                "Check whether the acquisition changed there. Comparing "
                "volumes across that point may not be meaningful.",
            ]

        self.data_check_reports = reports
        message = "\n".join(lines)
        self.status_bar.showMessage(
            f"Usable data: {100 * first['overlap_fraction']:.1f}% of the array"
        )
        if not silent:
            QMessageBox.information(self, "Data Check", message)
        return reports

    _CLUSTER_PREFIXES = ("K-means cluster", "3D Cluster", "Cluster ")

    def _material_source(self, name: str) -> str:
        """Where a material came from, for the panel's second column."""
        if any(name.startswith(prefix) for prefix in self._CLUSTER_PREFIXES):
            return "K-means"
        if name.startswith("Otsu"):
            return "Otsu"
        return "drawn"

    def _material_reference_timepoint(self) -> int:
        """The timepoint the material definitions are taken from."""
        if not self.dataset:
            return 0
        current = self.dataset.current_timepoint
        if self._model_class_masks(current):
            return current
        for timepoint in range(self.dataset.num_timepoints):
            if self._model_class_masks(timepoint):
                return timepoint
        return current

    def _refresh_material_panel(self):
        """Re-read the materials from whatever is currently segmented."""
        if not self.dataset:
            self.material_panel.set_materials([])
            return []
        reference = self._material_reference_timepoint()
        materials = [
            {
                "name": name,
                "source": self._material_source(name),
                "voxels": int(np.count_nonzero(mask)),
            }
            for mask, _color, name in self._visible_layers(reference)
        ]
        self.material_panel.set_materials(materials)
        self.material_panel.set_clusters_available(
            bool(self._last_kmeans_cluster_selections)
        )
        return materials

    def _on_model_segmentation(self):
        """Menu route: show the materials, then run the series."""
        if hasattr(self, "right_tabs") and hasattr(self, "material_panel"):
            index = self._materials_tab_index
            if index is not None and index >= 0:
                self.right_tabs.setCurrentIndex(index)
        self._refresh_material_panel()
        self._run_material_tracking(preview=False)

    def _run_material_tracking(self, preview=False):
        """Measure every timepoint (or just this one) against the materials."""
        from utils.cancellation import OperationCancelled, OperationFailed

        if not self.dataset or self.global_histogram is None:
            QMessageBox.information(
                self, "No Data",
                "Load a dataset and compute the histogram first."
            )
            return

        reference = self._material_reference_timepoint()
        masks = self._model_class_masks(reference)
        if not masks:
            QMessageBox.information(
                self, "No Materials",
                "Draw and segment at least one material first. The materials "
                "you define are what the whole series is measured against."
            )
            return

        self._refresh_material_panel()
        settings = self.material_panel.settings()
        # Only materials still on screen can be controls
        settings["control_materials"] = [
            name for name in settings["control_materials"] if name in masks
        ]

        if not settings["lock_definitions"]:
            self._run_adaptive_tracking(settings, reference, masks)
            return

        from model import ClassLibrary, LockedSegmenter
        from model.spatial_prior import ROIDerivedMRF
        from model.validity import build_valid_mask

        timepoints = (
            [self.dataset.current_timepoint] if preview else None
        )
        neutron_reference, xray_reference = self.dataset.get_volume_at_time(reference)

        def operation(progress_callback=None, cancel_check=None):
            if progress_callback:
                progress_callback(2, "Reading the materials you defined...")
            valid = build_valid_mask(neutron_reference, xray_reference)
            library = ClassLibrary.from_masks(
                neutron_reference, xray_reference, masks,
                valid_mask=valid, inert=settings["control_materials"],
            )
            segmenter = LockedSegmenter(
                library, prior=ROIDerivedMRF(beta=1.0, n_sweeps=6),
                bins=self.histogram_engine.bins,
            )
            segmenter.set_grid(
                self.global_histogram.x_edges, self.global_histogram.y_edges
            )

            reference_labels = np.zeros(valid.shape, dtype=np.int32)
            for index, name in enumerate(library.names):
                reference_labels[masks[name]] = index
            segmenter.learn_boundaries(reference_labels, valid_mask=valid)

            sweep = None
            if settings["smoothing_mode"] == "auto":
                if progress_callback:
                    progress_callback(8, "Choosing the smoothing strength...")

                def sweep_progress(value, message):
                    if progress_callback:
                        progress_callback(8 + int(0.25 * value), message)

                strength, sweep = segmenter.auto_smoothing(
                    neutron_reference, xray_reference,
                    progress_callback=sweep_progress, cancel_check=cancel_check,
                )
            else:
                strength = settings["smoothing_strength"]

            def run_progress(value, message):
                if progress_callback:
                    progress_callback(35 + int(0.6 * value), message)

            outcome = segmenter.segment_series(
                self.dataset, timepoints=timepoints, beta=strength,
                progress_callback=run_progress, cancel_check=cancel_check,
                enforce_guards=False,
            )
            outcome.smoothing_sweep = sweep
            return segmenter, outcome, library

        title = "Preview" if preview else "Track Materials"
        try:
            result = run_with_progress(
                self, title,
                "Measuring the materials at this timepoint..." if preview
                else "Measuring the materials through the series...",
                operation,
            )
        except (OperationCancelled, OperationFailed):
            return
        if result is None:
            return

        segmenter, outcome, library = result
        self.material_panel.set_smoothing_result(
            outcome.smoothing, settings["smoothing_mode"] == "auto"
        )

        refusals = segmenter.check_guards(outcome)
        if refusals:
            self.material_panel.set_status(
                "These results are not reliable, so they have not been "
                "applied.", "fail",
            )
            self.material_panel.set_findings(refusals)
            QMessageBox.warning(
                self, "Segmentation Problem",
                "These results are not reliable, so they have not been "
                "applied:\n\n" + "\n\n".join(f"• {line}" for line in refusals),
            )
            return

        self._apply_locked_result(outcome)
        self.model_result = outcome
        self._show_health_check(outcome, library, settings, preview=preview)

    def _apply_locked_result(self, outcome):
        """Put the result into the slice viewer as ordinary layers."""
        for entry in outcome.timepoints:
            timepoint = entry.timepoint
            self.segmentation_masks[timepoint] = []
            self._clear_layer_shapes(timepoint)
            roi_manager = self.dual_histogram.get_roi_manager()
            for name in entry.class_names:
                mask = entry.mask_for(name)
                if not mask.any():
                    continue
                self.segmentation_masks[timepoint].append(
                    (mask, self._colour_for_layer(roi_manager, name), name)
                )

        current = self.dataset.current_timepoint
        neutron, xray = self.dataset.get_volume_at_time(current)
        self._apply_segmentation_overlays(current, neutron, xray)
        self._update_class_histogram_overlays(current, neutron, xray)

    def _show_health_check(self, outcome, library, settings=None,
                           preview=False):
        """Step 8: check the run before its numbers are used."""
        from model import Status, run_health_check

        mixing_report = None
        if settings and settings.get("find_mixed_boundaries"):
            mixing_report = self._suggest_mixed_boundaries(library)

        report = run_health_check(
            outcome,
            control_materials=library.inert_names,
            mixing_report=mixing_report,
        )
        self.health_report = report

        self.material_panel.show_health_report(report)

        lines = [report.headline(), ""]
        for finding in report.findings:
            lines.append(str(finding))
        if outcome.smoothing_sweep is not None:
            lines += [
                "",
                "Smoothing: " + describe_strength(outcome.smoothing)
                + " (chosen automatically).",
            ]
        if preview:
            lines += [
                "",
                "This was one timepoint only. Some checks — control "
                "materials, and whether the usable data changes — need the "
                "whole series to mean anything.",
            ]

        show = (
            QMessageBox.warning if report.status is Status.FAIL
            else QMessageBox.information
        )
        show(self, "Health Check", "\n".join(lines))
        self.status_bar.showMessage(report.headline())

    def _suggest_mixed_boundaries(self, library):
        """Which materials look like a boundary between two others."""
        from model.partial_volume import MixelComponent, verify_mixels

        class _Fitted:
            names = library.names
            means = np.array([material.mu for material in library])
            covariances = np.array([material.sigma for material in library])
            n_components = len(library)

        from model import detect_mixing_lines

        suggestions = detect_mixing_lines(_Fitted())
        if not suggestions:
            return None
        return verify_mixels(_Fitted(), suggestions)

    def _run_adaptive_tracking(self, settings, reference, masks):
        """The advanced path: definitions allowed to move.

        Only appropriate when the instrument is known to drift. The dialog
        warns before this is reached; this keeps it available rather than
        removing the capability.
        """
        from utils.cancellation import OperationCancelled, OperationFailed
        from model import (
            DriftTracker, ROIAnchoredMixture, ROIDerivedMRF, SequentialSegmenter,
        )
        from model.temporal import DriftTransition, StaticTransition

        neutron_reference, xray_reference = self.dataset.get_volume_at_time(reference)
        tracker = (
            DriftTracker(anchor_classes=settings["control_materials"])
            if settings["control_materials"] else None
        )
        strength = (
            1.0 if settings["smoothing_strength"] is None
            else settings["smoothing_strength"]
        )

        def operation(progress_callback=None, cancel_check=None):
            segmenter = SequentialSegmenter(
                mixture=ROIAnchoredMixture(outlier_component=True),
                mrf=(
                    ROIDerivedMRF(beta=strength, n_sweeps=5)
                    if strength > 0 else None
                ),
                temporal=DriftTransition(memory=0.5) if tracker else StaticTransition(),
                drift_tracker=tracker,
                bins=self.histogram_engine.bins,
            )
            segmenter.prepare(
                neutron_reference, xray_reference, masks,
                self.global_histogram.x_edges, self.global_histogram.y_edges,
                anchor_strength=0.5,
            )
            return segmenter.run(
                self.dataset,
                progress_callback=progress_callback, cancel_check=cancel_check,
            )

        try:
            outcome = run_with_progress(
                self, "Track Materials",
                "Following your materials, definitions allowed to move...",
                operation,
            )
        except (OperationCancelled, OperationFailed):
            return
        if outcome is None:
            return

        self._install_model_layers(outcome)
        self.model_result = outcome
        moved = []
        for entry in outcome.timepoints:
            if entry.fit is None:
                continue
            for name, distance in entry.fit.moved_sigma().items():
                if distance > 0.5 and name in settings["control_materials"]:
                    moved.append(name)
        message = [
            f"{len(outcome)} timepoint(s) segmented with the material "
            f"definitions allowed to move."
        ]
        if moved:
            message += [
                "",
                "These control materials moved noticeably: "
                + ", ".join(sorted(set(moved)))
                + ". A control material that moves is picking up something "
                "else — check the result before using it.",
            ]
        QMessageBox.information(self, "Done", "\n".join(message))


    def _on_estimate_drift(self):
        """Measure the histogram's instrumental drift across the series."""
        from PyQt5.QtWidgets import QFileDialog
        from utils.cancellation import OperationCancelled, OperationFailed

        if not self.dataset or self.global_histogram is None:
            QMessageBox.information(
                self, "No Data",
                "Load a dataset and compute the global histogram first."
            )
            return
        reference = self.dataset.current_timepoint
        masks = self._model_class_masks(reference)
        if not masks:
            QMessageBox.information(
                self, "No Materials",
                "Segment a timepoint first, then choose which of its "
                "materials should not change during the experiment."
            )
            return

        dialog = AnchorSelectionDialog(sorted(masks), self)
        if dialog.exec_() != QDialog.Accepted or not dialog.anchor_classes:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Drift Estimates", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".csv"):
            filepath += ".csv"

        neutron_reference, xray_reference = self.dataset.get_volume_at_time(reference)

        def operation(progress_callback=None, cancel_check=None):
            import csv

            from model import (
                DriftTracker, build_histogram_cache, build_valid_mask,
                estimate_process_noise, moments_from_mask,
            )

            valid = build_valid_mask(neutron_reference, xray_reference)
            moments = {}
            for name, mask in masks.items():
                entry = moments_from_mask(
                    neutron_reference, xray_reference, mask & valid
                )
                if entry is not None:
                    moments[name] = entry

            tracker = DriftTracker(
                anchor_classes=dialog.anchor_classes,
                estimate_scale=dialog.estimate_scale,
            )
            tracker.fit_reference(moments)

            estimates = []
            previous = None
            total = self.dataset.num_timepoints
            for timepoint in range(total):
                if cancel_check:
                    cancel_check()
                neutron, xray = self.dataset.get_volume_at_time(timepoint)
                cache = build_histogram_cache(
                    neutron, xray,
                    self.global_histogram.x_edges,
                    self.global_histogram.y_edges,
                    valid_mask=build_valid_mask(neutron, xray),
                )
                estimate = tracker.estimate(cache, timepoint, previous=previous)
                estimates.append(estimate)
                previous = estimate
                if progress_callback:
                    progress_callback(
                        int(95 * (timepoint + 1) / total),
                        f"Timepoint {timepoint + 1}/{total}",
                    )

            with open(filepath, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "timepoint", "shift_neutron", "shift_xray", "magnitude",
                    "scale_neutron", "scale_xray", "residual", "rejected_anchors",
                ])
                for estimate in estimates:
                    writer.writerow([
                        estimate.timepoint,
                        estimate.shift[0], estimate.shift[1], estimate.magnitude,
                        estimate.scale[0], estimate.scale[1], estimate.residual,
                        "|".join(estimate.rejected_anchors),
                    ])
            return estimates, estimate_process_noise(estimates)

        try:
            result = run_with_progress(
                self, "Instrumental Drift",
                "Locating the anchor classes at each timepoint...", operation,
            )
        except (OperationCancelled, OperationFailed):
            return
        if result is None:
            return

        estimates, noise = result
        largest = max((e.magnitude for e in estimates), default=0.0)
        rejected = sum(1 for e in estimates if e.rejected_anchors)
        message = (
            f"Drift estimates written to:\n{filepath}\n\n"
            f"Largest shift since T0: {largest:.4g} intensity units.\n"
            f"Instrumental noise floor (per-axis variance): "
            f"{noise[0]:.4g}, {noise[1]:.4g}."
        )
        if rejected:
            message += (
                f"\n\n{rejected} timepoint(s) had an anchor rejected as "
                f"implausibly far from where it was last seen; those carry the "
                f"previous estimate forward."
            )
        self.status_bar.showMessage(f"Drift estimates saved: {filepath}")
        QMessageBox.information(self, "Drift Estimated", message)

    def _on_export_spatial_metrics(self):
        """Spatial descriptors of the current segmentation, over time."""
        from PyQt5.QtWidgets import QFileDialog
        from utils.cancellation import OperationCancelled, OperationFailed
        from utils.histogram_metrics import plot_metric_evolution, write_metrics_csv
        from utils.metrics_spatial import combined_registry, spatial_metrics_rows

        if not self.dataset:
            QMessageBox.information(self, "No Data", "Load a dataset first.")
            return
        masks_by_timepoint = {
            timepoint: self._model_class_masks(timepoint)
            for timepoint in range(self.dataset.num_timepoints)
        }
        masks_by_timepoint = {
            timepoint: masks for timepoint, masks in masks_by_timepoint.items()
            if masks
        }
        if not masks_by_timepoint:
            QMessageBox.information(
                self, "No Segmentation",
                "Segment at least one timepoint first — these metrics describe "
                "the shape and position of the segmented classes."
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Spatial Metrics", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".csv"):
            filepath += ".csv"
        plot_path = filepath[:-4] + "_evolution.png"

        def operation(progress_callback=None, cancel_check=None):
            if progress_callback:
                progress_callback(10, "Measuring class shapes...")
            rows = spatial_metrics_rows(masks_by_timepoint)
            info, scalars, per_class = combined_registry()
            if progress_callback:
                progress_callback(85, "Writing CSV and plot...")
            write_metrics_csv(
                rows, filepath, metric_info=info,
                scalar_metrics=scalars, per_class_metrics=per_class,
            )
            saved = plot_metric_evolution(
                rows, plot_path, metric_info=info,
                scalar_metrics=scalars, per_class_metrics=per_class,
            )
            return rows, saved

        try:
            result = run_with_progress(
                self, "Spatial Metrics",
                "Measuring the segmentation in the volume...", operation,
            )
        except (OperationCancelled, OperationFailed):
            return
        if result is None:
            return

        rows, saved_plot = result
        message = f"Spatial metrics written to:\n{filepath}\n\n"
        message += f"{len(rows)} timepoint(s) measured."
        if saved_plot:
            message += f"\n\nEvolution plot:\n{saved_plot}"
        else:
            message += "\n\n(No evolution plot — it needs at least 2 timepoints.)"
        self.status_bar.showMessage(f"Spatial metrics saved: {filepath}")
        QMessageBox.information(self, "Spatial Metrics Saved", message)

    def _on_export_histogram_evolution(self):
        """Save each timepoint's log-histogram change against T0."""
        from utils.histogram_evolution import (
            REFERENCE_FIRST, save_histogram_evolution_image,
        )

        self._run_histogram_time_analysis(
            "Histogram Evolution",
            "Save Histogram Evolution Image",
            lambda histograms, path: save_histogram_evolution_image(
                histograms, path, reference_mode=REFERENCE_FIRST
            ),
            "Red areas gained voxels relative to T0, blue areas lost them "
            "(log scale). This is the cumulative drift from the start.",
        )

    def _on_export_histogram_increment(self):
        """Save each timepoint's log-histogram change against the previous one."""
        from utils.histogram_evolution import (
            REFERENCE_PREVIOUS, save_histogram_evolution_image,
        )

        self._run_histogram_time_analysis(
            "Incremental Histogram Change",
            "Save Incremental Histogram Change Image",
            lambda histograms, path: save_histogram_evolution_image(
                histograms, path, reference_mode=REFERENCE_PREVIOUS
            ),
            "Each panel compares a timepoint with the one before it, so the "
            "steps where change actually happens stand out instead of being "
            "buried in cumulative drift.",
        )

    def _on_export_marginal_evolution(self):
        """Save marginal kymographs of each modality against T0."""
        from utils.histogram_evolution import (
            REFERENCE_FIRST, save_marginal_evolution_image,
        )

        self._run_histogram_time_analysis(
            "Marginal Evolution",
            "Save Marginal Evolution Image",
            lambda histograms, path: save_marginal_evolution_image(
                histograms, path, reference_mode=REFERENCE_FIRST
            ),
            "Each panel stacks one modality's 1-D histogram against time "
            "(log2 vs T0). Red intensity bands grew, blue bands shrank — "
            "this separates a shift in neutron from a shift in X-ray.",
        )

    def _on_export_marginal_increment(self):
        """Save marginal kymographs comparing each timepoint with the previous."""
        from utils.histogram_evolution import (
            REFERENCE_PREVIOUS, save_marginal_evolution_image,
        )

        self._run_histogram_time_analysis(
            "Incremental Marginal Change",
            "Save Incremental Marginal Change Image",
            lambda histograms, path: save_marginal_evolution_image(
                histograms, path, reference_mode=REFERENCE_PREVIOUS
            ),
            "Each column compares a timepoint with the one before it, so the "
            "steps where an intensity band actually moves stand out. T0 is "
            "blank because it has no predecessor.",
        )

    def _on_morphological_analysis(self):
        """Perform morphological analysis on selected selections"""
        from PyQt5.QtWidgets import QDialog, QTextEdit, QVBoxLayout
        from utils.morphological_analysis import MorphologicalAnalyzer
        
        if len(self.selection_manager.selections) == 0:
            QMessageBox.information(self, "No Selections", 
                                   "No selections to analyze.")
            return
        
        # Dialog to show results
        dialog = QDialog(self)
        dialog.setWindowTitle("Morphological Analysis")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        dialog.setLayout(layout)
        
        # Analyze each selection
        results_text = "Morphological Analysis Results\n"
        results_text += "=" * 60 + "\n\n"
        
        for sel in self.selection_manager.selections:
            if sel.spatial_mask is not None:
                results_text += f"{sel.name}:\n"
                results_text += "-" * 40 + "\n"
                
                features = MorphologicalAnalyzer.analyze_selection(sel.spatial_mask)
                
                if features:
                    # Basic properties
                    results_text += f"  Area: {features['area']:,} pixels\n"
                    results_text += f"  Perimeter: {features['perimeter']:,} pixels\n"
                    results_text += f"  Circularity: {features['circularity']:.3f}\n"
                    results_text += f"  Compactness: {features['compactness']:.1f}\n"
                    
                    # Convex hull
                    if 'solidity' in features:
                        results_text += f"  Solidity: {features['solidity']:.3f}\n"
                    
                    # Shape
                    if 'eccentricity' in features:
                        results_text += f"  Eccentricity: {features['eccentricity']:.3f}\n"
                    if 'aspect_ratio' in features:
                        results_text += f"  Aspect Ratio: {features['aspect_ratio']:.2f}\n"
                    
                    # Connectivity
                    if 'num_components' in features:
                        results_text += f"  Components: {features['num_components']}\n"
                    
                    # Bounding box
                    if features['bbox']:
                        bbox = features['bbox']
                        results_text += f"  Bounding Box: {bbox['width']}×{bbox['height']} pixels\n"
                
                results_text += "\n"
        
        text_edit.setPlainText(results_text)
        dialog.exec_()
    
    def _on_compare_selections(self):
        """Compare two selections quantitatively"""
        from PyQt5.QtWidgets import QDialog, QLabel, QComboBox, QPushButton, QTextEdit
        
        if len(self.selection_manager.selections) < 2:
            QMessageBox.information(self, "Not Enough Selections", 
                                   "Need at least 2 selections to compare.")
            return
        
        # Dialog for selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Compare Selections")
        dialog.resize(600, 500)
        
        layout = QVBoxLayout()
        
        # Selection dropdowns
        sel1_label = QLabel("Selection 1:")
        layout.addWidget(sel1_label)
        sel1_combo = QComboBox()
        for sel in self.selection_manager.selections:
            sel1_combo.addItem(sel.name)
        layout.addWidget(sel1_combo)
        
        sel2_label = QLabel("Selection 2:")
        layout.addWidget(sel2_label)
        sel2_combo = QComboBox()
        for sel in self.selection_manager.selections:
            sel2_combo.addItem(sel.name)
        if len(self.selection_manager.selections) > 1:
            sel2_combo.setCurrentIndex(1)
        layout.addWidget(sel2_combo)
        
        # Compare button
        compare_btn = QPushButton("Compare")
        layout.addWidget(compare_btn)
        
        # Results
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        layout.addWidget(results_text)
        
        def do_comparison():
            from utils.morphological_analysis import MorphologicalAnalyzer
            
            sel1_name = sel1_combo.currentText()
            sel2_name = sel2_combo.currentText()
            
            # Find selections
            sel1 = None
            sel2 = None
            for sel in self.selection_manager.selections:
                if sel.name == sel1_name:
                    sel1 = sel
                if sel.name == sel2_name:
                    sel2 = sel
            
            if not sel1 or not sel2:
                return
            
            # Analyze
            features1 = MorphologicalAnalyzer.analyze_selection(sel1.spatial_mask)
            features2 = MorphologicalAnalyzer.analyze_selection(sel2.spatial_mask)
            
            if not features1 or not features2:
                return
            
            # Statistics panel comparison
            self.statistics_panel.compare_selections(sel1_name, sel2_name)
            
            # Morphology comparison
            comparison = MorphologicalAnalyzer.compare_morphologies(features1, features2)
            
            text = f"Comparison: {sel1_name} vs {sel2_name}\n"
            text += "=" * 60 + "\n\n"
            
            text += "Morphological Differences:\n"
            text += "-" * 40 + "\n"
            
            for key, value in comparison.items():
                if 'rel_diff' in key:
                    metric = key.replace('_rel_diff', '')
                    text += f"  {metric}: {value:.1f}% difference\n"
            
            results_text.setPlainText(text)
        
        compare_btn.clicked.connect(do_comparison)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def _on_track_timepoint(self):
        """Track current timepoint in time series"""
        if not hasattr(self, 'time_series_analyzer'):
            from utils.time_series_analysis import TimeSeriesAnalyzer
            self.time_series_analyzer = TimeSeriesAnalyzer()
        
        if not self.dataset:
            QMessageBox.information(self, "No Data", "Load dataset first.")
            return
        
        # Record current timepoint (display grid matches the selection masks)
        neutron_vol, xray_vol = self._current_display_volumes()
        current_tp = self.dataset.current_timepoint

        self.time_series_analyzer.add_timepoint(
            current_tp,
            self.selection_manager.selections,
            neutron_vol,
            xray_vol
        )
        
        self.status_bar.showMessage(f"Tracked timepoint {current_tp}")
        QMessageBox.information(self, "Tracked", 
                               f"Recorded data for timepoint {current_tp}")
    
    def _on_track_all_timepoints(self):
        """Track all timepoints automatically"""
        if not hasattr(self, 'time_series_analyzer'):
            from utils.time_series_analysis import TimeSeriesAnalyzer
            self.time_series_analyzer = TimeSeriesAnalyzer()
        
        if not self.dataset:
            QMessageBox.information(self, "No Data", "Load dataset first.")
            return
        
        if len(self.selection_manager.selections) == 0:
            QMessageBox.information(self, "No Selections", 
                                   "Create selections first before tracking.")
            return
        
        # Check if 4D mode
        if self.mode != '4D':
            QMessageBox.information(self, "3D Mode", 
                                   "Time series tracking is for 4D datasets.\n\n"
                                   "Switch to 4D mode to track multiple timepoints.")
            return
        
        # Confirm with user
        num_timepoints = getattr(self.dataset, 'num_timepoints', 1)
        
        if num_timepoints <= 1:
            QMessageBox.information(self, "Single Timepoint", 
                                   "Dataset has only one timepoint.\n\n"
                                   "Load a 4D dataset with multiple timepoints to use this feature.")
            return
        
        reply = QMessageBox.question(
            self,
            "Track All Timepoints",
            f"This will track {len(self.selection_manager.selections)} selection(s) "
            f"across all {num_timepoints} timepoints.\n\n"
            f"This may take a few minutes.\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Save current timepoint to restore later
        original_timepoint = self.dataset.current_timepoint
        
        # Create progress dialog
        from utils.progress_dialog import ProgressDialog
        progress = ProgressDialog(
            "Tracking Time Series",
            f"Processing timepoint 0 of {num_timepoints}...",
            num_timepoints,
            self
        )
        progress.show()
        
        try:
            tracked_count = 0
            
            for tp in range(num_timepoints):
                # Update progress
                progress.setValue(tp)
                progress.setLabelText(f"Processing timepoint {tp} of {num_timepoints}...")
                QApplication.processEvents()
                
                # Check if cancelled
                if progress.wasCanceled():
                    break
                
                # Navigate to timepoint
                self.dataset.set_timepoint(tp)
                self._on_timepoint_changed(tp)

                # Get data (display grid matches the selection masks)
                neutron_vol, xray_vol = self._display_volumes_at(tp)

                # Record
                self.time_series_analyzer.add_timepoint(
                    tp,
                    self.selection_manager.selections,
                    neutron_vol,
                    xray_vol
                )
                
                tracked_count += 1
            
            progress.setValue(num_timepoints)
            
            # Restore original timepoint
            self.dataset.set_timepoint(original_timepoint)
            self._on_timepoint_changed(original_timepoint)
            
            # Show summary
            if tracked_count > 0:
                QMessageBox.information(
                    self,
                    "Tracking Complete",
                    f"Successfully tracked {tracked_count} timepoint(s)\n"
                    f"for {len(self.selection_manager.selections)} selection(s).\n\n"
                    f"Use Analytics → Time Series → Plot Time Series\n"
                    f"to visualize the data."
                )
                self.status_bar.showMessage(
                    f"Tracked {tracked_count} timepoints for {len(self.selection_manager.selections)} selections"
                )
            else:
                QMessageBox.information(self, "Cancelled", "Tracking was cancelled.")
        
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to track timepoints:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
        finally:
            # Ensure progress dialog is closed
            progress.close()
    
    def _on_plot_time_series(self):
        """Plot time series for selections"""
        if not hasattr(self, 'time_series_analyzer'):
            QMessageBox.information(self, "No Data", 
                                   "No time series data recorded yet.\n\n"
                                   "Use Analytics → Time Series → Track Current Timepoint")
            return
        
        from PyQt5.QtWidgets import QDialog, QCheckBox
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        
        # Dialog for selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Plot Time Series")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Selection checkboxes
        layout.addWidget(QLabel("Select data to plot:"))
        
        checkboxes = []
        for sel in self.selection_manager.selections:
            cb = QCheckBox(sel.name)
            cb.setChecked(True)
            layout.addWidget(cb)
            checkboxes.append((cb, sel.name))
        
        # Plot button
        plot_btn = QPushButton("📈 Plot")
        layout.addWidget(plot_btn)
        
        # Canvas for plot
        fig, ax = plt.subplots(figsize=(8, 5))
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        def do_plot():
            selected_names = [name for cb, name in checkboxes if cb.isChecked()]
            
            if not selected_names:
                return
            
            ax.clear()
            
            for name in selected_names:
                timepoints, values = self.time_series_analyzer.get_time_series(
                    name, 'neutron_mean'
                )
                
                if len(timepoints) > 0:
                    ax.plot(timepoints, values, 'o-', label=name, linewidth=2)
            
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Neutron Mean Intensity')
            ax.set_title('Time Series Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            canvas.draw()
        
        plot_btn.clicked.connect(do_plot)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def _on_export_time_series(self):
        """Export time series data to CSV"""
        if not hasattr(self, 'time_series_analyzer'):
            QMessageBox.information(self, "No Data", 
                                   "No time series data to export.")
            return
        
        from PyQt5.QtWidgets import QFileDialog
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Time Series", "", "CSV Files (*.csv)"
        )
        
        if filepath:
            try:
                self.time_series_analyzer.export_time_series_csv(filepath)
                QMessageBox.information(self, "Exported", 
                                       f"Time series exported to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")
    
    def _on_clear_time_series(self):
        """Clear all time series data"""
        if not hasattr(self, 'time_series_analyzer'):
            QMessageBox.information(self, "No Data", 
                                   "No time series data to clear.")
            return
        
        reply = QMessageBox.question(
            self,
            "Clear Time Series",
            "This will delete all recorded time series data.\n\n"
            "Are you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.time_series_analyzer.clear()
            QMessageBox.information(self, "Cleared", 
                                   "Time series data has been cleared.")
            self.status_bar.showMessage("Time series data cleared")


# End of main_window.py class definition

