"""
dual_histogram_widget.py - Dual Histogram Display

Displays global and local histograms side-by-side with ROI drawing capabilities
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton,
    QButtonGroup, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QInputDialog, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRectangle
import numpy as np

from histograms import HistogramData
from utils.roi_manager import ROIManager


class HistogramCanvas(FigureCanvas):
    """
    Canvas for displaying a single histogram with ROI drawing
    """
    
    roi_updated = pyqtSignal()
    
    def __init__(self, title="Histogram", parent=None):
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.title = title
        self.histogram_data = None
        self.use_log_scale = False
        self.show_roi = True
        self._colorbar = None  # Colorbar reference
        self.vmin = None  # Dynamic range minimum
        self.vmax = None  # Dynamic range maximum
        
        # ROI drawing
        self.roi_manager = None
        self.drawing_mode = None  # 'polygon', 'rectangle', or None
        self.polygon_points = []
        self.rect_start = None
        self.temp_artists = []
        
        # Multiple ROI overlays (for selection manager)
        self.roi_overlays = []  # List of (name, vertices, color) tuples
        self.overlay_artists = []  # Matplotlib artists for overlays
        
        # Connect mouse events
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Setup the plot appearance"""
        self.ax.set_title(self.title)
        self.ax.set_xlabel('Neutron Intensity')
        self.ax.set_ylabel('X-ray Intensity')
        self.ax.grid(True, alpha=0.3)
    
    def set_histogram_data(self, hist_data: HistogramData):
        """Update histogram data and redraw"""
        self.histogram_data = hist_data
        self.update_plot()

    def update_plot(self):
        """Redraw the histogram, the active ROI, and any named-ROI overlays."""
        if self.histogram_data is None:
            return

        self.ax.clear()
        self._setup_plot()

        if self.use_log_scale:
            data = self.histogram_data.to_log_scale()
        else:
            data = self.histogram_data.histogram

        # The stored histogram is oriented [xray_bin, neutron_bin]; with
        # origin='lower' this puts neutron on the x-axis and X-ray on the
        # y-axis — the same data coordinates the ROI manager tests against.
        extent = [
            self.histogram_data.x_edges[0],
            self.histogram_data.x_edges[-1],
            self.histogram_data.y_edges[0],
            self.histogram_data.y_edges[-1]
        ]

        im = self.ax.imshow(
            data,
            extent=extent,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            vmin=self.vmin,
            vmax=self.vmax
        )

        if self._colorbar is None:
            self._colorbar = self.fig.colorbar(im, ax=self.ax)
        else:
            self._colorbar.update_normal(im)

        if self.show_roi:
            if self.roi_manager and self.roi_manager.has_roi():
                self._draw_roi()
            if self.roi_overlays:
                self._draw_roi_overlays()

        self.fig.tight_layout()
        self.draw_idle()

    def _draw_roi(self):
        """Draw the current ROI on the plot"""
        if not self.roi_manager:
            return

        if self.roi_manager.roi_type == 'polygon':
            polygon = MplPolygon(
                self.roi_manager.polygon_points,
                closed=True,
                fill=False,
                edgecolor='lime',
                linewidth=2
            )
            self.ax.add_patch(polygon)
        
        elif self.roi_manager.roi_type == 'rectangle':
            x_min, y_min, x_max, y_max = self.roi_manager.rectangle
            rect = MplRectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor='lime',
                linewidth=2
            )
            self.ax.add_patch(rect)
    
    def set_drawing_mode(self, mode: str):
        """Set ROI drawing mode"""
        self.drawing_mode = mode
        self.polygon_points = []
        self.rect_start = None
        self._clear_temp_artists()
    
    def clear_drawing_mode(self):
        """Clear drawing mode"""
        self.drawing_mode = None
        self.polygon_points = []
        self.rect_start = None
        self._clear_temp_artists()
    
    def _clear_temp_artists(self):
        """Remove temporary drawing artists"""
        for artist in self.temp_artists:
            try:
                artist.remove()
            except (NotImplementedError, ValueError):
                # Artist was already detached (e.g. by ax.clear() in update_plot)
                pass
        self.temp_artists = []

    def set_roi_overlays(self, overlays):
        """
        Set multiple ROI overlays for display

        Args:
            overlays: List of (name, vertices, color) tuples
                     vertices: Nx2 array of (x, y) points
                     color: matplotlib color (tuple or string)
        """
        self.roi_overlays = overlays
        self.update_plot()

    def clear_roi_overlays(self):
        """Clear all ROI overlays"""
        self.roi_overlays = []
        self._clear_overlay_artists()
        self.update_plot()

    def _clear_overlay_artists(self):
        """Remove overlay artists from plot"""
        for artist in self.overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.overlay_artists = []

    def _draw_roi_overlays(self):
        """Draw multiple ROI overlays"""
        self._clear_overlay_artists()

        for name, vertices, color in self.roi_overlays:
            if vertices is not None and len(vertices) > 2:
                edge_color = color
                polygon = MplPolygon(
                    vertices,
                    closed=True,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=2,
                    linestyle='--',
                    alpha=0.8,
                    label=name
                )
                self.ax.add_patch(polygon)
                self.overlay_artists.append(polygon)

    def set_range(self, vmin, vmax):
        """Set display range for histogram"""
        self.vmin = vmin
        self.vmax = vmax
        self.update_plot()
    
    def on_mouse_press(self, event):
        """Handle mouse press for ROI drawing"""
        if event.inaxes != self.ax or not self.drawing_mode:
            return
        
        if self.drawing_mode == 'polygon':
            # Save current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Add point to polygon
            self.polygon_points.append((event.xdata, event.ydata))
            
            # Draw point
            point, = self.ax.plot(event.xdata, event.ydata, 'ro', markersize=5)
            self.temp_artists.append(point)
            
            # Draw line to previous point
            if len(self.polygon_points) > 1:
                x_coords = [p[0] for p in self.polygon_points]
                y_coords = [p[1] for p in self.polygon_points]
                line, = self.ax.plot(x_coords, y_coords, 'r-', linewidth=1.5)
                self.temp_artists.append(line)
            
            # Restore axis limits
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            
            self.draw()
        
        elif self.drawing_mode == 'rectangle':
            self.rect_start = (event.xdata, event.ydata)
    
    def on_mouse_move(self, event):
        """Handle mouse move for rectangle preview"""
        if event.inaxes != self.ax or not self.drawing_mode:
            return
        
        if self.drawing_mode == 'rectangle' and self.rect_start:
            # Save current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Clear previous rectangle
            self._clear_temp_artists()
            
            # Draw preview rectangle
            x1, y1 = self.rect_start
            x2, y2 = event.xdata, event.ydata
            rect = MplRectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1),
                abs(y2 - y1),
                fill=False,
                edgecolor='red',
                linewidth=1.5,
                linestyle='--'
            )
            self.ax.add_patch(rect)
            self.temp_artists.append(rect)
            
            # Restore axis limits
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            
            self.draw()
    
    def on_mouse_release(self, event):
        """Handle mouse release for ROI finalisation"""
        if event.inaxes != self.ax or not self.drawing_mode:
            return

        if self.drawing_mode == 'rectangle' and self.rect_start:
            x1, y1 = self.rect_start
            x2, y2 = event.xdata, event.ydata

            if self.roi_manager:
                self.roi_manager.set_rectangle_roi(
                    min(x1, x2), min(y1, y2),
                    max(x1, x2), max(y1, y2)
                )
                self._clear_temp_artists()
                self.clear_drawing_mode()
                # roi_updated listeners redraw both canvases
                self.roi_updated.emit()

    def finalize_polygon(self):
        """Finalise polygon ROI"""
        if self.drawing_mode == 'polygon' and len(self.polygon_points) >= 3:
            if self.roi_manager:
                self.roi_manager.set_polygon_roi(np.array(self.polygon_points))
                self._clear_temp_artists()
                self.clear_drawing_mode()
                # roi_updated listeners redraw both canvases
                self.roi_updated.emit()
    
    def set_log_scale(self, use_log: bool):
        """Toggle log scale"""
        self.use_log_scale = use_log
        self.update_plot()
    
    def set_show_roi(self, show: bool):
        """Toggle ROI visibility"""
        self.show_roi = show
        self.update_plot()


class DualHistogramWidget(QWidget):
    """
    Widget displaying both global and local histograms side-by-side
    """
    
    roi_updated = pyqtSignal()
    # Emitted with the class name when a saved class is pulled back for editing
    editing_class_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_manager = ROIManager()

        # Editable ROI handlers (will be created after canvases)
        self.global_editable_roi = None
        self.local_editable_roi = None

        # Identity of the class currently pulled back into the active ROI, so
        # saving it again restores its name, class id and colour.
        self._editing_class = None

        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Histogram displays
        hist_layout = QHBoxLayout()
        
        # Global histogram
        global_layout = QVBoxLayout()
        global_label = QLabel("<b>Global Histogram (All Timepoints)</b>")
        global_layout.addWidget(global_label)
        
        self.global_canvas = HistogramCanvas("Global Histogram")
        self.global_canvas.roi_manager = self.roi_manager
        self.global_canvas.roi_updated.connect(self._on_roi_updated)
        self.global_canvas.setMinimumSize(400, 300)
        global_layout.addWidget(self.global_canvas)
        
        self.global_toolbar = NavigationToolbar(self.global_canvas, self)
        global_layout.addWidget(self.global_toolbar)
        
        hist_layout.addLayout(global_layout)
        
        # Local histogram
        local_layout = QVBoxLayout()
        local_label = QLabel("<b>Local Histogram (Current Timepoint)</b>")
        local_layout.addWidget(local_label)
        
        self.local_canvas = HistogramCanvas("Local Histogram")
        self.local_canvas.roi_manager = self.roi_manager
        self.local_canvas.setMinimumSize(400, 300)
        local_layout.addWidget(self.local_canvas)
        
        self.local_toolbar = NavigationToolbar(self.local_canvas, self)
        local_layout.addWidget(self.local_toolbar)
        
        hist_layout.addLayout(local_layout)
        
        layout.addLayout(hist_layout)
        
        # Initialize editable ROI handlers
        from utils.editable_roi_handler import EditableROIHandler
        
        self.global_editable_roi = EditableROIHandler(
            self.global_canvas.ax,
            self.roi_manager,
            update_callback=self._on_editable_roi_updated
        )
        
        self.local_editable_roi = EditableROIHandler(
            self.local_canvas.ax,
            self.roi_manager,
            update_callback=self._on_editable_roi_updated
        )
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # ROI tools
        controls_layout.addWidget(QLabel("<b>ROI Tool:</b>"))
        
        self.roi_button_group = QButtonGroup()
        
        self.polygon_btn = QPushButton("🔺 Polygon")
        self.polygon_btn.setCheckable(True)
        self.polygon_btn.clicked.connect(self._on_polygon_clicked)
        self.roi_button_group.addButton(self.polygon_btn)
        controls_layout.addWidget(self.polygon_btn)
        
        self.rectangle_btn = QPushButton("▭ Rectangle")
        self.rectangle_btn.setCheckable(True)
        self.rectangle_btn.clicked.connect(self._on_rectangle_clicked)
        self.roi_button_group.addButton(self.rectangle_btn)
        controls_layout.addWidget(self.rectangle_btn)
        
        self.finalize_btn = QPushButton("✓ Finish Polygon")
        self.finalize_btn.setEnabled(False)
        self.finalize_btn.clicked.connect(self._finalize_polygon)
        controls_layout.addWidget(self.finalize_btn)

        self.save_class_btn = QPushButton("➕ Save as Class")
        self.save_class_btn.setEnabled(False)
        self.save_class_btn.setToolTip(
            "Save the current ROI as a named class for multi-class segmentation.\n"
            "You can then draw another ROI for the next class.\n"
            "All saved classes are used together for Random Forest training."
        )
        self.save_class_btn.clicked.connect(self._save_current_as_class)
        controls_layout.addWidget(self.save_class_btn)

        self.clear_roi_btn = QPushButton("Clear Active ROI")
        self.clear_roi_btn.setToolTip("Clear the current (unsaved) active ROI only.")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        controls_layout.addWidget(self.clear_roi_btn)

        controls_layout.addStretch()

        # Display options
        self.log_scale_cb = QCheckBox("Log scale")
        self.log_scale_cb.stateChanged.connect(self._on_log_scale_changed)
        controls_layout.addWidget(self.log_scale_cb)

        self.show_roi_cb = QCheckBox("Show ROI")
        self.show_roi_cb.setChecked(True)
        self.show_roi_cb.stateChanged.connect(self._on_show_roi_changed)
        controls_layout.addWidget(self.show_roi_cb)

        self.editable_roi_cb = QCheckBox("✏️ Editable ROI")
        self.editable_roi_cb.setChecked(False)
        self.editable_roi_cb.setToolTip("Enable dragging ROI vertices to fine-tune selection")
        self.editable_roi_cb.stateChanged.connect(self._on_editable_roi_changed)
        controls_layout.addWidget(self.editable_roi_cb)

        # Add dynamic range controls
        controls_layout.addWidget(QLabel("Range:"))
        self.vmin_spinbox = QDoubleSpinBox()
        self.vmin_spinbox.setRange(0, 1e10)
        self.vmin_spinbox.setValue(0)
        self.vmin_spinbox.setPrefix("Min: ")
        self.vmin_spinbox.valueChanged.connect(self._update_all_histograms)
        controls_layout.addWidget(self.vmin_spinbox)

        self.vmax_spinbox = QDoubleSpinBox()
        self.vmax_spinbox.setRange(0, 1e10)
        self.vmax_spinbox.setValue(0)
        self.vmax_spinbox.setPrefix("Max: ")
        self.vmax_spinbox.valueChanged.connect(self._update_all_histograms)
        controls_layout.addWidget(self.vmax_spinbox)

        auto_range_btn = QPushButton("Auto Range")
        auto_range_btn.clicked.connect(self._auto_range)
        controls_layout.addWidget(auto_range_btn)

        layout.addLayout(controls_layout)

        # ── Selection panel: manage the histogram selections ──────────────────
        roi_list_group = QGroupBox(
            "Histogram Selections  (classes for multi-class RF training)"
        )
        roi_list_layout = QVBoxLayout()
        roi_list_layout.setContentsMargins(4, 4, 4, 4)
        roi_list_layout.setSpacing(3)

        self.roi_list_widget = QListWidget()
        self.roi_list_widget.setMinimumHeight(110)
        self.roi_list_widget.setToolTip(
            "Every saved selection (class) drawn on the histogram.\n\n"
            "• Untick a row to hide it — hidden selections are not drawn\n"
            "  and not segmented.\n"
            "• 'Edit' moves a selection back to the active ROI so you can\n"
            "  reshape it, then save it as a class again.\n"
            "• Double-click a row to rename it.\n"
            "• 'Remove' deletes the highlighted selection."
        )
        self.roi_list_widget.itemDoubleClicked.connect(self._rename_roi_item)
        self.roi_list_widget.itemChanged.connect(self._on_roi_item_changed)
        self.roi_list_widget.currentRowChanged.connect(
            self._update_selection_buttons
        )
        roi_list_layout.addWidget(self.roi_list_widget)

        roi_list_btns = QHBoxLayout()
        self.edit_class_btn = QPushButton("✏️ Edit")
        self.edit_class_btn.setEnabled(False)
        self.edit_class_btn.setToolTip(
            "Work with this selection: it becomes the active ROI so you can\n"
            "reshape or re-segment it. Save it as a class again when done."
        )
        self.edit_class_btn.clicked.connect(self._edit_selected_class)
        roi_list_btns.addWidget(self.edit_class_btn)

        self.remove_class_btn = QPushButton("🗑 Remove")
        self.remove_class_btn.setEnabled(False)
        self.remove_class_btn.setToolTip("Delete the highlighted selection.")
        self.remove_class_btn.clicked.connect(self._remove_selected_class)
        roi_list_btns.addWidget(self.remove_class_btn)

        self.clear_all_classes_btn = QPushButton("🗑 Clear All")
        self.clear_all_classes_btn.setEnabled(False)
        self.clear_all_classes_btn.setToolTip("Delete every saved selection.")
        self.clear_all_classes_btn.clicked.connect(self._clear_all_classes)
        roi_list_btns.addWidget(self.clear_all_classes_btn)
        roi_list_layout.addLayout(roi_list_btns)

        vis_btns = QHBoxLayout()
        self.show_all_classes_btn = QPushButton("👁 Show All")
        self.show_all_classes_btn.setEnabled(False)
        self.show_all_classes_btn.clicked.connect(
            lambda: self._set_all_classes_visible(True)
        )
        vis_btns.addWidget(self.show_all_classes_btn)

        self.hide_all_classes_btn = QPushButton("🚫 Hide All")
        self.hide_all_classes_btn.setEnabled(False)
        self.hide_all_classes_btn.setToolTip(
            "Hide every selection. Hidden selections are excluded from\n"
            "segmentation, so you can work with one class at a time."
        )
        self.hide_all_classes_btn.clicked.connect(
            lambda: self._set_all_classes_visible(False)
        )
        vis_btns.addWidget(self.hide_all_classes_btn)

        self.only_selected_btn = QPushButton("🎯 Only This")
        self.only_selected_btn.setEnabled(False)
        self.only_selected_btn.setToolTip(
            "Show only the highlighted selection and hide the rest, so\n"
            "segmentation works with that one alone."
        )
        self.only_selected_btn.clicked.connect(self._isolate_selected_class)
        vis_btns.addWidget(self.only_selected_btn)
        roi_list_layout.addLayout(vis_btns)

        roi_list_group.setLayout(roi_list_layout)
        layout.addWidget(roi_list_group)
        # ── End selection panel ───────────────────────────────────────────────

        self.setLayout(layout)
    
    def set_global_histogram(self, hist_data: HistogramData):
        """Set global histogram data"""
        self.global_canvas.set_histogram_data(hist_data)

    def set_local_histogram(self, hist_data: HistogramData):
        """Set local histogram data"""
        self.local_canvas.set_histogram_data(hist_data)

    def _on_polygon_clicked(self):
        if self.polygon_btn.isChecked():
            self.global_canvas.set_drawing_mode('polygon')
            self.finalize_btn.setEnabled(True)
            self.save_class_btn.setEnabled(False)
        else:
            self.global_canvas.clear_drawing_mode()
            self.finalize_btn.setEnabled(False)

    def _on_rectangle_clicked(self):
        if self.rectangle_btn.isChecked():
            self.global_canvas.set_drawing_mode('rectangle')
            self.finalize_btn.setEnabled(False)
            self.save_class_btn.setEnabled(False)
        else:
            self.global_canvas.clear_drawing_mode()

    def _finalize_polygon(self):
        self.global_canvas.finalize_polygon()
        self.polygon_btn.setChecked(False)
        self.finalize_btn.setEnabled(False)
        # Enable saving as a named class now that the polygon is closed
        self.save_class_btn.setEnabled(self.roi_manager.roi_type is not None)

    def clear_roi(self):
        """Clear the current active (unsaved) ROI only. Named class ROIs are kept."""
        if self.editable_roi_cb.isChecked():
            self.editable_roi_cb.setChecked(False)
        self.roi_manager.clear_roi()
        self.save_class_btn.setEnabled(False)
        self._update_roi_list()
        self._apply_roi_change()

    # ── Named / multi-class ROI management ───────────────────────────────────

    def _save_current_as_class(self):
        """Prompt the user for a name and save the active ROI to the named list."""
        if not self.roi_manager.roi_type:
            QMessageBox.warning(self, "No ROI",
                                "Draw and finish an ROI before saving it as a class.")
            return

        # A class pulled back with 'Edit' keeps its identity by default
        editing = self._editing_class
        if editing is not None:
            default_name = editing['name']
        else:
            default_name = f"Class {len(self.roi_manager.named_rois) + 1}"

        name, ok = QInputDialog.getText(
            self, "Name this class",
            "Enter a label for this ROI (e.g. 'Lithium', 'Electrolyte'):",
            text=default_name
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        if editing is not None:
            self.roi_manager.add_named_roi(
                name, class_id=editing['class_id'], color=editing['color']
            )
            self._editing_class = None
        else:
            self.roi_manager.add_named_roi(name)

        # Clear the active ROI so the canvas is ready for the next one
        self.roi_manager.clear_roi()
        self.save_class_btn.setEnabled(False)

        self._update_roi_list()
        self._apply_roi_change()

    def _update_roi_list(self):
        """Rebuild the QListWidget from roi_manager.named_rois."""
        # Repopulating fires itemChanged for every row; ignore those so the
        # rebuild cannot be mistaken for the user toggling visibility.
        previous_row = self.roi_list_widget.currentRow()

        self.roi_list_widget.blockSignals(True)
        self.roi_list_widget.clear()
        for roi in self.roi_manager.named_rois:
            label = (f"Class {roi['class_id']}  |  {roi['name']}  "
                     f"[{roi['roi_type']}]")
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            self._style_roi_item(item, roi)
            self.roi_list_widget.addItem(item)

        # The ROI currently drawn but not yet saved is listed too, so every
        # selection is visible in one place. It has no class id or visibility
        # toggle until it is saved.
        if self.roi_manager.roi_type is not None:
            item = QListWidgetItem(
                f"✎ (unsaved selection)  [{self.roi_manager.roi_type}]"
            )
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            item.setToolTip(
                "The selection you are drawing or editing.\n"
                "Click 'Save as Class' (or double-click this row) to keep it\n"
                "as a named class for Random Forest training."
            )
            self.roi_list_widget.addItem(item)

        self.roi_list_widget.blockSignals(False)

        if 0 <= previous_row < self.roi_list_widget.count():
            self.roi_list_widget.setCurrentRow(previous_row)
        self._update_selection_buttons()

    def _named_count(self) -> int:
        """Number of saved class rows (the unsaved row, if any, comes after)."""
        return len(self.roi_manager.named_rois)

    def _is_active_row(self, row: int) -> bool:
        """True when *row* is the unsaved active-selection row."""
        return (self.roi_manager.roi_type is not None
                and row == self._named_count())

    @staticmethod
    def _style_roi_item(item, roi):
        """Set one row's check state and colour from its ROI entry."""
        visible = roi.get('visible', True)
        item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        try:
            color = QColor(roi['color'])
            color.setAlpha(60 if visible else 20)
            item.setBackground(color)
        except Exception:
            pass

    def _update_selection_buttons(self, *_args):
        """Enable the panel's buttons according to the current selection."""
        named = self._named_count()
        row = self.roi_list_widget.currentRow()
        on_named = 0 <= row < named
        on_active = self._is_active_row(row)

        self.clear_all_classes_btn.setEnabled(named > 0)
        self.show_all_classes_btn.setEnabled(named > 0)
        self.hide_all_classes_btn.setEnabled(named > 0)
        # Remove works on a saved class or on the unsaved selection
        self.remove_class_btn.setEnabled(on_named or on_active)
        # The unsaved selection is already the one being edited
        self.edit_class_btn.setEnabled(on_named)
        self.only_selected_btn.setEnabled(on_named or on_active)
        self.save_class_btn.setEnabled(self.roi_manager.roi_type is not None)

    def _refresh_named_roi_overlays(self):
        """Push the visible class ROIs to both canvases as coloured overlays."""
        overlays = self.roi_manager.get_named_roi_overlays()
        self.global_canvas.set_roi_overlays(overlays)
        self.local_canvas.set_roi_overlays(overlays)

    def _apply_roi_change(self):
        """Refresh overlays, canvases and listeners after an ROI edit."""
        self._refresh_named_roi_overlays()
        self.global_canvas.update_plot()
        self.local_canvas.update_plot()
        self.roi_updated.emit()

    def _on_roi_item_changed(self, item):
        """Checkbox toggled: show/hide that class.

        Restyles the row in place. Rebuilding the list here would delete the
        very item whose signal is being handled, which crashes Qt.
        """
        row = self.roi_list_widget.row(item)
        # The unsaved-selection row has no checkbox
        if not 0 <= row < self._named_count():
            return
        visible = item.checkState() == Qt.Checked
        if self.roi_manager.named_rois[row].get('visible', True) == visible:
            return

        self.roi_manager.set_named_roi_visible(row, visible)
        self.roi_list_widget.blockSignals(True)
        self._style_roi_item(item, self.roi_manager.named_rois[row])
        self.roi_list_widget.blockSignals(False)
        self._apply_roi_change()

    def _set_all_classes_visible(self, visible: bool):
        """Show or hide every saved selection at once."""
        for index in range(len(self.roi_manager.named_rois)):
            self.roi_manager.set_named_roi_visible(index, visible)
        self._update_roi_list()
        self._apply_roi_change()

    def _isolate_selected_class(self):
        """Show only the highlighted selection, hiding the others.

        On the unsaved row this hides every saved class, leaving the ROI
        being drawn as the only thing displayed and segmented.
        """
        row = self.roi_list_widget.currentRow()
        if self._is_active_row(row):
            self._set_all_classes_visible(False)
            self.roi_list_widget.setCurrentRow(row)
            return
        if not 0 <= row < self._named_count():
            return
        for index in range(self._named_count()):
            self.roi_manager.set_named_roi_visible(index, index == row)
        self._update_roi_list()
        self.roi_list_widget.setCurrentRow(row)
        self._apply_roi_change()

    def _edit_selected_class(self):
        """Move the highlighted class back into the active ROI for editing."""
        row = self.roi_list_widget.currentRow()
        if not 0 <= row < self._named_count():
            return

        if self.roi_manager.roi_type is not None:
            reply = QMessageBox.question(
                self, "Replace Active ROI",
                "There is already an active ROI being drawn.\n\n"
                "Editing this class will discard it. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        entry = self.roi_manager.take_named_roi(row)
        # Remember its identity so saving it again restores name/class/colour
        self._editing_class = {
            'name': entry['name'],
            'class_id': entry['class_id'],
            'color': entry.get('color'),
        }

        if self.editable_roi_cb.isChecked():
            self.global_editable_roi.draw_editable_roi()
            self.local_editable_roi.draw_editable_roi()

        self.save_class_btn.setEnabled(True)
        self._update_roi_list()
        self._apply_roi_change()
        self.editing_class_changed.emit(entry['name'])

    def _remove_selected_class(self):
        """Remove the highlighted selection — a saved class or the unsaved one."""
        row = self.roi_list_widget.currentRow()

        if self._is_active_row(row):
            # Discard the ROI being drawn; saved classes are untouched
            self.clear_roi()
            return

        if not 0 <= row < self._named_count():
            return
        name = self.roi_manager.named_rois[row]['name']
        reply = QMessageBox.question(
            self, "Remove Selection",
            f"Remove '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        self.roi_manager.remove_named_roi(row)
        self._update_roi_list()
        self._apply_roi_change()

    def _clear_all_classes(self):
        """Remove all named class ROIs after confirmation."""
        reply = QMessageBox.question(
            self, "Clear All Classes",
            "Remove all saved class ROIs?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.roi_manager.clear_named_rois()
        self._update_roi_list()
        self._apply_roi_change()

    def _rename_roi_item(self, item):
        """Double-click a row: rename a class, or save the unsaved selection."""
        row = self.roi_list_widget.row(item)

        if self._is_active_row(row):
            # Double-clicking the unsaved row is the natural way to keep it
            self._save_current_as_class()
            return

        if not 0 <= row < self._named_count():
            return
        current_name = self.roi_manager.named_rois[row]['name']
        new_name, ok = QInputDialog.getText(
            self, "Rename Class", "New name:", text=current_name
        )
        if ok and new_name.strip():
            self.roi_manager.named_rois[row]['name'] = new_name.strip()
            self._update_roi_list()
            self._apply_roi_change()

    # ── Sync / update ─────────────────────────────────────────────────────────

    def _on_roi_updated(self):
        """ROI was updated — refresh the panel, both canvases and overlays."""
        # Keeps the unsaved-selection row in step with what is being drawn
        self._update_roi_list()
        self._refresh_named_roi_overlays()
        self.global_canvas.update_plot()
        self.local_canvas.update_plot()
        # Enable 'Save as Class' whenever a complete active ROI exists
        self.save_class_btn.setEnabled(self.roi_manager.roi_type is not None)
        self.roi_updated.emit()
    
    def _on_log_scale_changed(self, state):
        use_log = state == 2  # Qt.Checked
        self.global_canvas.set_log_scale(use_log)
        self.local_canvas.set_log_scale(use_log)
    
    def _on_show_roi_changed(self, state):
        show = state == 2  # Qt.Checked
        self.global_canvas.set_show_roi(show)
        self.local_canvas.set_show_roi(show)
    
    def _on_editable_roi_changed(self, state):
        """Enable/disable editable ROI"""
        enable = state == 2  # Qt.Checked

        if enable:
            # Editing only makes sense with an active ROI to edit
            if self.roi_manager.roi_type is not None:
                self.global_editable_roi.enable()
                self.local_editable_roi.enable()
            else:
                self.editable_roi_cb.setChecked(False)
        else:
            self.global_editable_roi.disable()
            self.local_editable_roi.disable()
            # Redraw to remove vertex markers
            self.global_canvas.update_plot()
            self.local_canvas.update_plot()

    def _on_editable_roi_updated(self):
        """Called when ROI is modified by dragging vertices"""
        self.global_canvas.update_plot()
        self.local_canvas.update_plot()
        # Notify listeners so the segmentation state stays in sync
        self.roi_updated.emit()
    
    def get_roi_manager(self) -> ROIManager:
        """Get the ROI manager"""
        return self.roi_manager
    
    def _update_all_histograms(self):
        """Update both histograms with current range settings"""
        vmin = self.vmin_spinbox.value()
        vmax = self.vmax_spinbox.value()
        
        # Only apply if max > min
        if vmax > vmin:
            self.global_canvas.set_range(vmin, vmax)
            self.local_canvas.set_range(vmin, vmax)
    
    def _auto_range(self):
        """Auto-set range based on current histogram data"""
        # Use global histogram if available
        if self.global_canvas.histogram_data is not None:
            if self.global_canvas.use_log_scale:
                data = self.global_canvas.histogram_data.to_log_scale()
            else:
                data = self.global_canvas.histogram_data.histogram

            vmin = float(data.min())
            vmax = float(data.max())

            # Update the spinboxes without triggering a redraw per change,
            # then apply the new range once.
            for spinbox, value in ((self.vmin_spinbox, vmin),
                                   (self.vmax_spinbox, vmax)):
                spinbox.blockSignals(True)
                spinbox.setValue(value)
                spinbox.blockSignals(False)

            self.global_canvas.set_range(vmin, vmax)
            self.local_canvas.set_range(vmin, vmax)
