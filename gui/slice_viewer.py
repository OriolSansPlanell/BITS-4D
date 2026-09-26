"""The slice viewer: one slice of the volume with its highlights.

Split out of ``main_window.py``; it is re-exported there.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QCheckBox
)
from PyQt5.QtCore import pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import sys

from gui.responsive import flow_row


def _group_labelled(widgets, max_follow=2):
    """Join each bare QLabel to (at most *max_follow*) controls after it.

    The result goes into a wrapping row: a label and the control it names
    then move to the next line together instead of being split, while long
    runs of buttons after a label can still wrap.
    """
    from gui.responsive import group

    grouped, current = [], []

    def flush():
        if current:
            grouped.append(group(*current) if len(current) > 1 else current[0])
            current.clear()

    for widget in widgets:
        if isinstance(widget, QLabel):
            flush()
            current.append(widget)
        elif current and len(current) <= max_follow:
            current.append(widget)
        else:
            flush()
            grouped.append(widget)
    flush()
    return grouped


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

    # The Auto-Detect button: K-means is configured and run by the window
    kmeans_requested = pyqtSignal()

    # The Clear Highlight button: the window hides the layers behind it
    highlight_cleared = pyqtSignal()
    
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
        
        # Controls for axis selection
        controls = []
        
        controls.append(QLabel("View Axis:"))
        
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QSlider
        from PyQt5.QtCore import Qt
        
        self.axis_group = QButtonGroup()
        
        self.z_axis_btn = QRadioButton("XY (Z-slice)")
        self.z_axis_btn.setChecked(True)
        self.z_axis_btn.toggled.connect(lambda: self._on_axis_changed('z'))
        self.axis_group.addButton(self.z_axis_btn)
        controls.append(self.z_axis_btn)
        
        self.y_axis_btn = QRadioButton("XZ (Y-slice)")
        self.y_axis_btn.toggled.connect(lambda: self._on_axis_changed('y'))
        self.axis_group.addButton(self.y_axis_btn)
        controls.append(self.y_axis_btn)
        
        self.x_axis_btn = QRadioButton("YZ (X-slice)")
        self.x_axis_btn.toggled.connect(lambda: self._on_axis_changed('x'))
        self.axis_group.addButton(self.x_axis_btn)
        controls.append(self.x_axis_btn)
        
        
        # Slice index slider
        controls.append(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        self.slice_slider.setEnabled(False)
        self.slice_slider.setMinimumWidth(120)
        controls.append(self.slice_slider)
        
        self.slice_label = QLabel("0")
        controls.append(self.slice_label)
        
        
        # View mode selection
        controls.append(QLabel("Data:"))
        
        self.view_group = QButtonGroup()
        
        self.neutron_view_btn = QRadioButton("Neutron")
        self.neutron_view_btn.setChecked(True)
        self.neutron_view_btn.toggled.connect(lambda: self._on_view_mode_changed('neutron'))
        self.view_group.addButton(self.neutron_view_btn)
        controls.append(self.neutron_view_btn)
        
        self.xray_view_btn = QRadioButton("X-ray")
        self.xray_view_btn.toggled.connect(lambda: self._on_view_mode_changed('xray'))
        self.view_group.addButton(self.xray_view_btn)
        controls.append(self.xray_view_btn)
        
        
        # Dynamic range controls
        from PyQt5.QtWidgets import QDoubleSpinBox
        
        controls.append(QLabel("Range:"))
        self.slice_vmin = QDoubleSpinBox()
        self.slice_vmin.setRange(0, 1e10)
        self.slice_vmin.setValue(0)
        self.slice_vmin.setPrefix("Min: ")
        self.slice_vmin.setMaximumWidth(130)
        self.slice_vmin.valueChanged.connect(self._on_range_changed)
        controls.append(self.slice_vmin)
        
        self.slice_vmax = QDoubleSpinBox()
        self.slice_vmax.setRange(0, 1e10)
        self.slice_vmax.setValue(65535)
        self.slice_vmax.setPrefix("Max: ")
        self.slice_vmax.setMaximumWidth(130)
        self.slice_vmax.valueChanged.connect(self._on_range_changed)
        controls.append(self.slice_vmax)
        
        slice_auto_btn = QPushButton("Auto Range")
        slice_auto_btn.clicked.connect(self._auto_range_slice)
        controls.append(slice_auto_btn)
        
        # Show/hide the spatial-selection row: on a short screen the slice
        # needs that height more than tools that are not always in use.
        from PyQt5.QtWidgets import QToolButton
        self.spatial_tools_btn = QToolButton()
        self.spatial_tools_btn.setText("🛠 Spatial tools")
        self.spatial_tools_btn.setCheckable(True)
        self.spatial_tools_btn.setChecked(True)
        self.spatial_tools_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.spatial_tools_btn.setToolTip(
            "Show or hide the spatial selection tools (rectangle, region\n"
            "growing, auto-detect) to give the slice more room."
        )
        controls.append(self.spatial_tools_btn)

        # Group each label with what it names, so they wrap as a unit
        controls = _group_labelled(controls)
        layout.addWidget(flow_row(*controls))
        
        # Spatial ROI tools
        spatial = []
        
        spatial.append(QLabel("Spatial Selection:"))
        
        self.rect_tool_btn = QPushButton("□ Rectangle")
        self.rect_tool_btn.setCheckable(True)
        self.rect_tool_btn.setToolTip("Draw rectangle on slice to select spatial region")
        self.rect_tool_btn.clicked.connect(self._on_rect_tool_clicked)
        spatial.append(self.rect_tool_btn)
        
        self.region_grow_btn = QPushButton("🪄 Region Grow")
        self.region_grow_btn.setCheckable(True)
        self.region_grow_btn.setToolTip("Click a seed point to grow connected region")
        self.region_grow_btn.clicked.connect(self._on_region_grow_clicked)
        spatial.append(self.region_grow_btn)
        
        # Bivariate mode checkbox
        self.bivariate_cb = QCheckBox("Bivariate")
        self.bivariate_cb.setToolTip("Use both neutron AND X-ray values (more selective)")
        self.bivariate_cb.setChecked(False)
        self.bivariate_cb.stateChanged.connect(self._on_bivariate_changed)
        self.bivariate_cb.setEnabled(False)
        spatial.append(self.bivariate_cb)
        
        # 3D mode checkbox (NEW for v15.0)
        self.mode_3d_cb = QCheckBox("3D Volume")
        self.mode_3d_cb.setToolTip("Apply to entire 3D volume instead of current slice")
        self.mode_3d_cb.setChecked(False)
        self.mode_3d_cb.setEnabled(False)
        spatial.append(self.mode_3d_cb)
        
        # Tolerance control for region growing
        spatial.append(QLabel("Tol:"))
        self.tolerance_spinbox = QDoubleSpinBox()
        self.tolerance_spinbox.setRange(1, 10000)
        self.tolerance_spinbox.setValue(1000)
        self.tolerance_spinbox.setSingleStep(100)
        self.tolerance_spinbox.setToolTip("Intensity tolerance for region growing")
        self.tolerance_spinbox.setEnabled(False)
        self.tolerance_spinbox.setMaximumWidth(80)
        spatial.append(self.tolerance_spinbox)
        
        # Second tolerance (for bivariate mode)
        self.tolerance2_label = QLabel("Tol2:")
        self.tolerance2_label.setToolTip("Tolerance for other modality (bivariate mode)")
        self.tolerance2_label.setVisible(False)
        spatial.append(self.tolerance2_label)
        
        self.tolerance2_spinbox = QDoubleSpinBox()
        self.tolerance2_spinbox.setRange(1, 10000)
        self.tolerance2_spinbox.setValue(1000)
        self.tolerance2_spinbox.setSingleStep(100)
        self.tolerance2_spinbox.setToolTip("Tolerance for other modality (bivariate mode)")
        self.tolerance2_spinbox.setEnabled(False)
        self.tolerance2_spinbox.setVisible(False)
        self.tolerance2_spinbox.setMaximumWidth(80)
        spatial.append(self.tolerance2_spinbox)
        
        # Show/hide mask toggle
        self.show_mask_cb = QCheckBox("Show Mask")
        self.show_mask_cb.setChecked(True)
        self.show_mask_cb.setToolTip("Toggle region growing mask overlay")
        self.show_mask_cb.stateChanged.connect(self._on_show_mask_changed)
        self.show_mask_cb.setEnabled(False)
        spatial.append(self.show_mask_cb)
        
        # Auto-detect features button
        self.auto_detect_btn = QPushButton("🔍 Auto-Detect")
        self.auto_detect_btn.setToolTip(
            "K-means clustering of the (neutron, X-ray) values: on this slice,\n"
            "on this timepoint's volume, or on the whole time series.\n"
            "Opens the K-means settings on the Auto Seg tab."
        )
        self.auto_detect_btn.clicked.connect(self._on_auto_detect)
        spatial.append(self.auto_detect_btn)
        
        self.clear_spatial_roi_btn = QPushButton("✕ Clear")
        self.clear_spatial_roi_btn.setToolTip("Clear spatial ROI")
        self.clear_spatial_roi_btn.clicked.connect(self._clear_spatial_roi)
        self.clear_spatial_roi_btn.setEnabled(False)
        spatial.append(self.clear_spatial_roi_btn)

        self.clear_highlight_btn = QPushButton("🧹 Clear Highlight")
        self.clear_highlight_btn.setToolTip(
            "Hide every coloured overlay: segmentation layers, saved\n"
            "selections and the region-grow mask. Classes are unticked, so\n"
            "the next segmentation shows only what you select next.\n"
            "Tick a class to bring its layers back; nothing is deleted."
        )
        self.clear_highlight_btn.setEnabled(False)
        self.clear_highlight_btn.clicked.connect(self._on_clear_highlight_clicked)
        spatial.append(self.clear_highlight_btn)
        
        
        self.create_hist_roi_btn = QPushButton("→ Histogram ROI")
        self.create_hist_roi_btn.setToolTip(
            "Create Histogram ROI from Selection: extract the values inside\n"
            "the spatial selection and turn them into a histogram ROI."
        )
        self.create_hist_roi_btn.clicked.connect(self._create_histogram_roi_from_spatial)
        self.create_hist_roi_btn.setEnabled(False)
        spatial.append(self.create_hist_roi_btn)
        
        spatial = _group_labelled(spatial)
        self.spatial_tools_row = flow_row(*spatial)
        layout.addWidget(self.spatial_tools_row)
        self.spatial_tools_btn.toggled.connect(self.set_spatial_tools_visible)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(240, 200)
        layout.addWidget(self.canvas, stretch=1)
        
        # Initialize spatial ROI state
        self.spatial_roi_selector = None
        self.spatial_roi_coords = None
        
        # Region growing state
        self.region_grow_mode = False
        self.region_grow_mask = None
        self.region_grow_mask_3d = None  # For 3D region growing (v15.0)
        # (axis, slice_index) a 2-D grown mask belongs to, so it is only
        # redrawn on that slice
        self.region_grow_plane = None
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
    
    def set_spatial_tools_visible(self, visible):
        """Show or hide the spatial-selection tool row."""
        visible = bool(visible)
        self.spatial_tools_row.setVisible(visible)
        if self.spatial_tools_btn.isChecked() != visible:
            self.spatial_tools_btn.setChecked(visible)

    def _setup_plot(self):
        """Setup the plot appearance"""
        self.ax.set_title("Volume Slice")
        self.ax.axis('off')
        
        # Connect motion event to show pixel values
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        # Connect scroll event for zoom
        self.cid_scroll = self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)

        # Keep the title inside the canvas when it is resized
        self.canvas.mpl_connect('resize_event', self._on_canvas_resize)

    def _on_canvas_resize(self, _event=None):
        try:
            self.fig.tight_layout()
        except Exception:
            pass
    
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
        """Ask for K-means clustering (slice, volume or time series).

        The clustering itself lives in the main window, which has the whole
        dataset and the histogram engine — the time-series level needs both.
        """
        self.kmeans_requested.emit()

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
        self.region_grow_mask_3d = None
        self.region_grow_plane = None
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

    def _on_clear_highlight_clicked(self):
        """The Clear Highlight button: clear now, and let the window hide the
        layers so the next redraw does not bring them straight back."""
        self._clear_highlight()
        self.highlight_cleared.emit()

    def _clear_highlight(self):
        """Remove all coloured overlays from the slice view without affecting the ROI.

        Only what is drawn right now: the window re-composes the overlays on
        the next redraw. The button goes through _on_clear_highlight_clicked,
        which also asks the window to keep them hidden.
        """
        # Clear single region-grow overlay
        self.region_grow_mask = None
        self.region_grow_mask_3d = None
        self.region_grow_plane = None
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
        
        self.region_grow_plane = (self.current_axis, self.current_slice_index)

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
    
    def _region_grow_display_slice(self):
        """The grown region's 2-D mask for the current view, or None.

        A 3-D grown region is re-sliced for the current plane; a 2-D one is
        shown only on the slice it was grown on.
        """
        if self.region_grow_mask_3d is not None:
            return self._slice_mask_for_display(self.region_grow_mask_3d)
        if self.region_grow_mask is None:
            return None
        return self._slice_mask_for_display(
            self.region_grow_mask, self.region_grow_plane
        )

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
        mask_2d = self._region_grow_display_slice()
        if (self.show_mask_cb.isChecked() and mask_2d is not None
                and self.ax.images):
            # Create colored overlay
            mask_rgba = np.zeros((*mask_2d.shape, 4))
            mask_rgba[mask_2d.astype(bool)] = [0, 1, 0, 0.4]  # Green with alpha

            # Display overlay
            self.mask_overlay = self.ax.imshow(
                mask_rgba,
                extent=self.ax.images[0].get_extent(),
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
        # ax.clear() already detached the region-grow artist
        self.mask_overlay = None
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
        # The grown region survives redraws (contrast, slice, plane changes)
        if self.region_grow_mask is not None:
            self._display_mask_overlay()

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
