"""
dual_histogram_widget.py - Dual Histogram Display

Displays global and local histograms side-by-side with ROI drawing capabilities
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton,
    QButtonGroup, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QInputDialog, QMessageBox, QGroupBox
)
from PyQt5.QtCore import QSize, Qt, pyqtSignal
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
    # The selected ROI changed (clicked, or cleared) — redraw only
    selection_changed = pyqtSignal()
    # Backspace/Delete on the selected ROI; the panel decides what that means
    # (a saved class may have segmentation to ask about)
    remove_requested = pyqtSignal()

    #: Arrow-key step in histogram bins; Shift moves ten times as far
    KEY_STEP_BINS = 1
    KEY_STEP_FAST = 10

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
        self.mpl_connect('key_press_event', self.on_key_press)
        # Keep labels inside the canvas when it is resized (small screens)
        self.mpl_connect('resize_event', self._on_resize)

        # Arrow keys only reach the canvas when it can take keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

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
            if self.roi_overlays:
                self._draw_roi_overlays()
            if self.roi_manager and self.roi_manager.has_roi():
                self._draw_roi()
                self._draw_selection()

        self.fig.tight_layout()
        self.draw_idle()

    def _on_resize(self, _event=None):
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def _draw_roi(self):
        """Draw every unsaved ROI, each in the colour it will have as a class.

        ROIs are drawn *filled* as well as outlined. Matplotlib fills with the
        same winding rule that decides containment, so the shaded area is
        exactly the region that will be segmented. With an outline alone, a
        polygon whose edges cross itself looks like it encloses more than it
        actually selects.
        """
        if not self.roi_manager:
            return
        import matplotlib.colors as mcolors

        for roi in self.roi_manager.get_unsaved_rois():
            vertices = self.roi_manager.target_vertices(roi['target'])
            if vertices is None:
                continue
            red, green, blue, _ = mcolors.to_rgba(roi['color'])
            self.ax.add_patch(MplPolygon(
                vertices,
                closed=True,
                fill=True,
                facecolor=(red, green, blue, 0.18),
                edgecolor=(red, green, blue, 1.0),
                linewidth=2,
            ))

    def _draw_selection(self):
        """Mark the selected ROI with a heavy black-and-white outline."""
        if not self.roi_manager:
            return
        vertices = self.roi_manager.target_vertices(
            self.roi_manager.get_selected()
        )
        if vertices is None:
            return
        for color, width, style in (('white', 4.0, '-'), ('black', 1.6, '--')):
            self.ax.add_patch(MplPolygon(
                vertices, closed=True, fill=False,
                edgecolor=color, linewidth=width, linestyle=style, zorder=20,
            ))

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

        import matplotlib.colors as mcolors

        for name, vertices, color in self.roi_overlays:
            if vertices is not None and len(vertices) > 2:
                # Filled as well as outlined, so the shaded area is exactly
                # what this selection covers (see _draw_roi).
                try:
                    red, green, blue, _ = mcolors.to_rgba(color)
                except Exception:
                    red, green, blue = 1.0, 0.0, 0.0
                polygon = MplPolygon(
                    vertices,
                    closed=True,
                    fill=True,
                    facecolor=(red, green, blue, 0.15),
                    edgecolor=(red, green, blue, 0.9),
                    linewidth=2,
                    linestyle='--',
                    label=name
                )
                self.ax.add_patch(polygon)
                self.overlay_artists.append(polygon)

    def set_range(self, vmin, vmax):
        """Set display range for histogram"""
        self.vmin = vmin
        self.vmax = vmax
        self.update_plot()
    
    def _navigation_active(self) -> bool:
        """True while the toolbar's pan/zoom tool owns the mouse.

        Without this, zooming in to place a vertex precisely would *also*
        drop a vertex at the point where the zoom drag started, silently
        corrupting the polygon.
        """
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return True
        widgetlock = getattr(self.figure.canvas, "widgetlock", None)
        return bool(widgetlock is not None and widgetlock.locked())

    def on_mouse_press(self, event):
        """Place ROI vertices while drawing; otherwise select the ROI clicked."""
        if event.inaxes != self.ax:
            return
        if not self.drawing_mode:
            self._select_at(event)
            return
        # Only a plain left-click places/starts an ROI; ignore other buttons
        # and any click that belongs to the pan/zoom tools.
        if event.button != 1 or self._navigation_active():
            return
        if event.xdata is None or event.ydata is None:
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
        if event.xdata is None or event.ydata is None:
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
            if x2 is None or y2 is None or x1 == x2 or y1 == y2:
                # A click without a drag encloses nothing; keep waiting for
                # a real rectangle instead of raising.
                self.rect_start = None
                self._clear_temp_artists()
                self.draw_idle()
                return

            if self.roi_manager:
                # Drawing another ROI keeps the previous unsaved one
                self.roi_manager.stash_active()
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
                # Drawing another ROI keeps the previous unsaved one
                self.roi_manager.stash_active()
                self.roi_manager.set_polygon_roi(np.array(self.polygon_points))
                self._clear_temp_artists()
                self.clear_drawing_mode()
                # roi_updated listeners redraw both canvases
                self.roi_updated.emit()
    
    # ── selecting, moving and removing with the keyboard ─────────────────

    def _select_at(self, event):
        """Select the topmost ROI under a plain left-click (or none)."""
        if event.button != 1 or self._navigation_active():
            return
        if event.xdata is None or event.ydata is None or not self.roi_manager:
            return
        if not self.show_roi:
            return
        # Grab the keyboard so the arrow keys move what was just picked
        self.setFocus(Qt.MouseFocusReason)
        target = self.roi_manager.hit_test(event.xdata, event.ydata)
        if target == self.roi_manager.get_selected():
            return
        self.roi_manager.select(target)
        self.selection_changed.emit()

    def key_step(self, fast=False):
        """(dx, dy) moved by one arrow-key press: one histogram bin."""
        bins = self.KEY_STEP_FAST if fast else self.KEY_STEP_BINS
        data = self.histogram_data
        if data is not None and len(data.x_edges) > 1 and len(data.y_edges) > 1:
            return (bins * float(data.x_edges[1] - data.x_edges[0]),
                    bins * float(data.y_edges[1] - data.y_edges[0]))
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return bins * abs(x1 - x0) / 256.0, bins * abs(y1 - y0) / 256.0

    _KEY_DIRECTIONS = {
        'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1),
    }

    def on_key_press(self, event):
        """Arrow keys move the selected ROI; Backspace/Delete remove it."""
        if not self.roi_manager or not event.key:
            return
        target = self.roi_manager.get_selected()
        if target is None:
            return
        key = event.key
        fast = key.startswith('shift+')
        if fast:
            key = key[len('shift+'):]
        if key in self._KEY_DIRECTIONS:
            sx, sy = self._KEY_DIRECTIONS[key]
            dx, dy = self.key_step(fast)
            if self.roi_manager.translate(target, sx * dx, sy * dy):
                self.roi_updated.emit()
        elif key in ('backspace', 'delete'):
            self.remove_requested.emit()

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
    # Emitted as (class name, discard_segmentation) when a class is removed
    class_removed = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_manager = ROIManager()

        # Set by the main window: given a class name, returns how many
        # segmentation layers were computed from it. Without it the panel
        # simply does not offer to discard anything.
        self.layer_count_provider = None

        # Editable ROI handlers (will be created after canvases)
        self.global_editable_roi = None
        self.local_editable_roi = None

        # Identity of the class currently pulled back into the active ROI, so
        # saving it again restores its name, class id and colour.
        self._editing_class = None

        self.init_ui()
    
    def init_ui(self):
        from PyQt5.QtWidgets import QSplitter, QSizePolicy
        from gui.responsive import PlotPanes, flow_row, group

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Histogram displays: side by side when there is room, as tabs on a
        # narrow or short panel (a laptop screen), so neither plot is
        # squeezed flat.
        self.plot_panes = PlotPanes()

        def plot_column(title, canvas):
            column = QWidget()
            column_layout = QVBoxLayout(column)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(2)
            heading = QLabel(title)
            heading.setWordWrap(True)
            column_layout.addWidget(heading)
            column_layout.addWidget(canvas, stretch=1)
            toolbar = NavigationToolbar(canvas, self)
            toolbar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            toolbar.setIconSize(QSize(16, 16))
            column_layout.addWidget(toolbar)
            return column, toolbar, heading

        # Global histogram
        self.global_canvas = HistogramCanvas("Global Histogram")
        self.global_canvas.roi_manager = self.roi_manager
        self.global_canvas.roi_updated.connect(self._on_roi_updated)
        self.global_canvas.setMinimumSize(200, 200)
        global_column, self.global_toolbar, heading = plot_column(
            "<b>Global Histogram (All Timepoints)</b>", self.global_canvas
        )
        self.plot_panes.add_pane(global_column, "Global (all timepoints)",
                                 heading)

        # Local histogram
        self.local_canvas = HistogramCanvas("Local Histogram")
        self.local_canvas.roi_manager = self.roi_manager
        self.local_canvas.roi_updated.connect(self._on_roi_updated)
        self.local_canvas.setMinimumSize(200, 200)
        local_column, self.local_toolbar, heading = plot_column(
            "<b>Local Histogram (Current Timepoint)</b>", self.local_canvas
        )
        self.plot_panes.add_pane(local_column, "Local (this timepoint)",
                                 heading)

        for canvas in (self.global_canvas, self.local_canvas):
            canvas.selection_changed.connect(self._on_canvas_selection_changed)
            canvas.remove_requested.connect(self._remove_selected_roi)

        plots_and_controls = QWidget()
        plots_layout = QVBoxLayout(plots_and_controls)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.addWidget(self.plot_panes, stretch=1)
        
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
        
        # Controls — wrap onto several lines on a narrow panel
        controls = []

        # ROI tools
        controls.append(QLabel("<b>ROI Tool:</b>"))
        
        self.roi_button_group = QButtonGroup()
        
        self.polygon_btn = QPushButton("🔺 Polygon")
        self.polygon_btn.setCheckable(True)
        self.polygon_btn.clicked.connect(self._on_polygon_clicked)
        self.roi_button_group.addButton(self.polygon_btn)
        controls.append(self.polygon_btn)
        
        self.rectangle_btn = QPushButton("▭ Rectangle")
        self.rectangle_btn.setCheckable(True)
        self.rectangle_btn.clicked.connect(self._on_rectangle_clicked)
        self.roi_button_group.addButton(self.rectangle_btn)
        controls.append(self.rectangle_btn)
        
        self.finalize_btn = QPushButton("✓ Finish Polygon")
        self.finalize_btn.setEnabled(False)
        self.finalize_btn.clicked.connect(self._finalize_polygon)
        controls.append(self.finalize_btn)

        self.save_class_btn = QPushButton("➕ Save as Class")
        self.save_class_btn.setEnabled(False)
        self.save_class_btn.setToolTip(
            "Save every unsaved ROI as a named class — two ROIs drawn make\n"
            "two classes, each keeping the colour it was drawn in.\n"
            "You can then draw more ROIs for further classes."
        )
        self.save_class_btn.clicked.connect(self._save_current_as_class)
        controls.append(self.save_class_btn)

        self.clear_roi_btn = QPushButton("Clear Unsaved")
        self.clear_roi_btn.setToolTip(
            "Clear every ROI drawn but not saved. Saved classes are kept.\n"
            "To remove just one, click it on the histogram and press\n"
            "Backspace."
        )
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        controls.append(self.clear_roi_btn)

        # Display options
        self.log_scale_cb = QCheckBox("Log scale")
        self.log_scale_cb.stateChanged.connect(self._on_log_scale_changed)
        display = [self.log_scale_cb]

        self.show_roi_cb = QCheckBox("Show ROI")
        self.show_roi_cb.setChecked(True)
        self.show_roi_cb.stateChanged.connect(self._on_show_roi_changed)
        display.append(self.show_roi_cb)

        self.editable_roi_cb = QCheckBox("✏️ Editable ROI")
        self.editable_roi_cb.setChecked(False)
        self.editable_roi_cb.setToolTip("Enable dragging ROI vertices to fine-tune selection")
        self.editable_roi_cb.stateChanged.connect(self._on_editable_roi_changed)
        controls.append(self.editable_roi_cb)

        # Add dynamic range controls
        self.vmin_spinbox = QDoubleSpinBox()
        self.vmin_spinbox.setRange(0, 1e10)
        self.vmin_spinbox.setValue(0)
        self.vmin_spinbox.setPrefix("Min: ")
        self.vmin_spinbox.setMaximumWidth(130)
        self.vmin_spinbox.valueChanged.connect(self._update_all_histograms)

        self.vmax_spinbox = QDoubleSpinBox()
        self.vmax_spinbox.setRange(0, 1e10)
        self.vmax_spinbox.setValue(0)
        self.vmax_spinbox.setPrefix("Max: ")
        self.vmax_spinbox.setMaximumWidth(130)
        self.vmax_spinbox.valueChanged.connect(self._update_all_histograms)

        auto_range_btn = QPushButton("Auto Range")
        auto_range_btn.clicked.connect(self._auto_range)
        display.append(group(QLabel("Range:"), self.vmin_spinbox,
                             self.vmax_spinbox, auto_range_btn))

        # Display settings live in a drop-down, so on a short screen the
        # rows under the plot go to the tools you use while drawing.
        from PyQt5.QtWidgets import QMenu, QToolButton, QWidgetAction
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(8, 6, 8, 6)
        for widget in display:
            display_layout.addWidget(widget)
        display_menu = QMenu(self)
        display_action = QWidgetAction(display_menu)
        display_action.setDefaultWidget(display_panel)
        display_menu.addAction(display_action)
        self.display_btn = QToolButton()
        self.display_btn.setText("Display ▾")
        self.display_btn.setToolTip("Log scale, ROI visibility and colour range")
        self.display_btn.setPopupMode(QToolButton.InstantPopup)
        self.display_btn.setMenu(display_menu)
        controls.append(self.display_btn)

        plots_layout.addWidget(flow_row(*controls))

        # ── Selection panel: manage the histogram selections ──────────────────
        roi_list_group = QGroupBox(
            "Histogram Selections  (classes)"
        )
        roi_list_layout = QVBoxLayout()
        roi_list_layout.setContentsMargins(4, 4, 4, 4)
        roi_list_layout.setSpacing(3)

        self.roi_list_widget = QListWidget()
        self.roi_list_widget.setMinimumHeight(60)
        self.roi_list_widget.setToolTip(
            "Every saved selection (class) drawn on the histogram.\n\n"
            "• Untick a row to hide it — hidden selections are not drawn\n"
            "  and not segmented.\n"
            "• 'Edit' moves a selection back to the active ROI so you can\n"
            "  reshape it, then save it as a class again.\n"
            "• Double-click a row to rename it.\n"
            "• 'Remove' deletes the highlighted selection.\n\n"
            "On the histogram: click a selection to pick it, move it with\n"
            "the arrow keys (Shift for bigger steps), remove it with\n"
            "Backspace."
        )
        self.roi_list_widget.itemDoubleClicked.connect(self._rename_roi_item)
        self.roi_list_widget.itemChanged.connect(self._on_roi_item_changed)
        self.roi_list_widget.currentRowChanged.connect(
            self._update_selection_buttons
        )
        self.roi_list_widget.currentRowChanged.connect(
            self._on_list_row_changed
        )
        roi_list_layout.addWidget(self.roi_list_widget)

        roi_list_btns = []
        self.edit_class_btn = QPushButton("✏️ Edit")
        self.edit_class_btn.setEnabled(False)
        self.edit_class_btn.setToolTip(
            "Work with this selection: it becomes the active ROI so you can\n"
            "reshape or re-segment it. Save it as a class again when done."
        )
        self.edit_class_btn.clicked.connect(self._edit_selected_class)
        roi_list_btns.append(self.edit_class_btn)

        self.remove_class_btn = QPushButton("🗑 Remove")
        self.remove_class_btn.setEnabled(False)
        self.remove_class_btn.setToolTip("Delete the highlighted selection.")
        self.remove_class_btn.clicked.connect(self._remove_selected_class)
        roi_list_btns.append(self.remove_class_btn)

        self.clear_all_classes_btn = QPushButton("🗑 Clear All")
        self.clear_all_classes_btn.setEnabled(False)
        self.clear_all_classes_btn.setToolTip("Delete every saved selection.")
        self.clear_all_classes_btn.clicked.connect(self._clear_all_classes)
        roi_list_btns.append(self.clear_all_classes_btn)
        self.show_all_classes_btn = QPushButton("👁 Show All")
        self.show_all_classes_btn.setEnabled(False)
        self.show_all_classes_btn.clicked.connect(
            lambda: self._set_all_classes_visible(True)
        )
        roi_list_btns.append(self.show_all_classes_btn)

        self.hide_all_classes_btn = QPushButton("🚫 Hide All")
        self.hide_all_classes_btn.setEnabled(False)
        self.hide_all_classes_btn.setToolTip(
            "Hide every selection. Hidden selections are excluded from\n"
            "segmentation, so you can work with one class at a time."
        )
        self.hide_all_classes_btn.clicked.connect(
            lambda: self._set_all_classes_visible(False)
        )
        roi_list_btns.append(self.hide_all_classes_btn)

        self.only_selected_btn = QPushButton("🎯 Only This")
        self.only_selected_btn.setEnabled(False)
        self.only_selected_btn.setToolTip(
            "Show only the highlighted selection and hide the rest, so\n"
            "segmentation works with that one alone."
        )
        self.only_selected_btn.clicked.connect(self._isolate_selected_class)
        roi_list_btns.append(self.only_selected_btn)
        roi_list_layout.addWidget(flow_row(*roi_list_btns))

        roi_list_group.setLayout(roi_list_layout)
        # ── End selection panel ───────────────────────────────────────────────

        # Plots above, class list below; the divider can be dragged, which
        # matters on a short (laptop) screen.
        self.panel_splitter = QSplitter(Qt.Vertical)
        self.panel_splitter.setChildrenCollapsible(False)
        self.panel_splitter.addWidget(plots_and_controls)
        self.panel_splitter.addWidget(roi_list_group)
        self.panel_splitter.setStretchFactor(0, 4)
        self.panel_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.panel_splitter)

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
        self.save_class_btn.setEnabled(self.roi_manager.unsaved_count() > 0)

        # A crossing outline selects less than it appears to, so say so
        # rather than letting the segmentation quietly come out wrong.
        if self.roi_manager.roi_type == 'polygon':
            from utils.roi_manager import polygon_self_intersects
            if polygon_self_intersects(self.roi_manager.polygon_points):
                QMessageBox.warning(
                    self, "Polygon crosses itself",
                    "This polygon's outline crosses itself.\n\n"
                    "The shaded area shows what will actually be segmented — "
                    "a region enclosed by the crossing edges can be left out.\n\n"
                    "If that is not what you intended, clear the ROI and draw "
                    "it again without crossing edges."
                )

    def clear_roi(self):
        """Clear every unsaved ROI. Saved classes are kept."""
        if self.editable_roi_cb.isChecked():
            self.editable_roi_cb.setChecked(False)
        self.roi_manager.clear_roi()
        self.save_class_btn.setEnabled(False)
        self._update_roi_list()
        self._apply_roi_change()

    # ── Named / multi-class ROI management ───────────────────────────────────

    def _save_current_as_class(self):
        """Save every unsaved ROI as its own named class.

        With two ROIs drawn, two classes are made — one name is asked for
        each, and each keeps the colour it was drawn in.
        """
        unsaved = self.roi_manager.get_unsaved_rois()
        if not unsaved:
            QMessageBox.warning(self, "No ROI",
                                "Draw and finish an ROI before saving it as a class.")
            return

        # A class pulled back with 'Edit' is the active ROI (the last one)
        # and keeps its identity by default.
        editing = self._editing_class
        names, colors, class_ids = [], [], []
        next_number = len(self.roi_manager.named_rois) + 1
        for position, roi in enumerate(unsaved):
            is_edited = editing is not None and roi['target'] == ('active', 0)
            if is_edited:
                default_name = editing['name']
            else:
                default_name = f"Class {next_number}"
                next_number += 1
            prompt = "Enter a label for this ROI (e.g. 'Lithium', 'Electrolyte'):"
            if len(unsaved) > 1:
                prompt = (f"ROI {position + 1} of {len(unsaved)} "
                          f"(drawn in {roi['color']}).\n" + prompt)
            name, ok = QInputDialog.getText(
                self, "Name this class", prompt, text=default_name
            )
            if not ok or not name.strip():
                # Nothing is saved unless every ROI got a name
                return
            names.append(name.strip())
            colors.append(editing['color'] if is_edited else roi['color'])
            class_ids.append(editing['class_id'] if is_edited else None)

        self.roi_manager.save_unsaved_as_classes(names, colors, class_ids)
        self._editing_class = None
        self.save_class_btn.setEnabled(False)

        self._update_roi_list()
        self._apply_roi_change()

    def _update_roi_list(self):
        """Rebuild the QListWidget: saved classes, then every unsaved ROI."""
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

        # ROIs drawn but not yet saved are listed too, so every selection is
        # visible in one place. They have no class id or visibility toggle
        # until they are saved.
        for roi in self.roi_manager.get_unsaved_rois():
            item = QListWidgetItem(
                f"✎ (unsaved selection)  [{roi['roi_type']}]"
            )
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            color = QColor(roi['color'])
            color.setAlpha(45)
            item.setBackground(color)
            item.setToolTip(
                "A selection you have drawn but not saved.\n"
                "Click 'Save as Class' (or double-click this row) to keep\n"
                "every unsaved selection, each as its own class."
            )
            self.roi_list_widget.addItem(item)

        # Keep the row of the ROI selected on the histogram; rows shift when
        # ROIs are added or saved, so a bare row number could point elsewhere.
        row = self._row_for_target(self.roi_manager.get_selected())
        if row < 0 and 0 <= previous_row < self.roi_list_widget.count():
            row = previous_row
        self.roi_list_widget.setCurrentRow(row)
        self.roi_list_widget.blockSignals(False)
        self._update_selection_buttons()

    def _named_count(self) -> int:
        """Number of saved class rows (the unsaved rows come after)."""
        return len(self.roi_manager.named_rois)

    def _unsaved_target_for_row(self, row: int):
        """The unsaved ROI a list row shows, or None for a class row."""
        index = row - self._named_count()
        unsaved = self.roi_manager.get_unsaved_rois()
        if 0 <= index < len(unsaved):
            return unsaved[index]['target']
        return None

    def _row_for_target(self, target):
        if target is None:
            return -1
        if target[0] == 'named':
            return target[1]
        for offset, roi in enumerate(self.roi_manager.get_unsaved_rois()):
            if roi['target'] == target:
                return self._named_count() + offset
        return -1

    def _is_active_row(self, row: int) -> bool:
        """True when *row* shows an unsaved selection."""
        return self._unsaved_target_for_row(row) is not None

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
        # Remove works on a saved class or on an unsaved selection
        self.remove_class_btn.setEnabled(on_named or on_active)
        # An unsaved selection is already editable
        self.edit_class_btn.setEnabled(on_named)
        self.only_selected_btn.setEnabled(on_named or on_active)
        self.save_class_btn.setEnabled(self.roi_manager.unsaved_count() > 0)

    def _on_list_row_changed(self, row):
        """Picking a row selects that ROI on the histograms too."""
        target = (('named', row) if 0 <= row < self._named_count()
                  else self._unsaved_target_for_row(row))
        if target != self.roi_manager.get_selected():
            self.roi_manager.select(target)
            self.global_canvas.update_plot()
            self.local_canvas.update_plot()

    def _on_canvas_selection_changed(self):
        """An ROI was clicked on a histogram: show it on both, and in the list."""
        row = self._row_for_target(self.roi_manager.get_selected())
        self.roi_list_widget.blockSignals(True)
        self.roi_list_widget.setCurrentRow(row)
        self.roi_list_widget.blockSignals(False)
        self._update_selection_buttons()
        self.global_canvas.update_plot()
        self.local_canvas.update_plot()

    def _remove_selected_roi(self):
        """Backspace/Delete on a histogram: remove the selected ROI."""
        target = self.roi_manager.get_selected()
        if target is None:
            return
        row = self._row_for_target(target)
        if row < 0:
            return
        self.roi_list_widget.setCurrentRow(row)
        self._remove_selected_class()

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
        """Move the highlighted class back into the active ROI for editing.

        Anything already drawn stays as an unsaved selection, so nothing is
        lost; saving afterwards turns every unsaved selection into a class.
        """
        row = self.roi_list_widget.currentRow()
        if not 0 <= row < self._named_count():
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

        unsaved_target = self._unsaved_target_for_row(row)
        if unsaved_target is not None:
            # Discard that one unsaved selection; everything else stays
            if unsaved_target == ('active', 0) and self._editing_class:
                self._editing_class = None
            self.roi_manager.remove_unsaved(unsaved_target)
            self._update_roi_list()
            self._apply_roi_change()
            return

        if not 0 <= row < self._named_count():
            return
        name = self.roi_manager.named_rois[row]['name']

        layer_count = 0
        if callable(self.layer_count_provider):
            try:
                layer_count = int(self.layer_count_provider(name))
            except Exception:
                layer_count = 0

        if layer_count:
            # Removing the class would otherwise leave its already-computed
            # segmentation behind with nothing in the panel controlling it,
            # so ask rather than decide silently either way.
            reply = QMessageBox.question(
                self, "Remove Selection",
                f"Remove '{name}'?\n\n"
                f"{layer_count} segmentation layer(s) were computed from it.\n\n"
                "• Yes — remove the class and discard its segmentation\n"
                "• No — remove the class but keep its segmentation\n"
                "• Cancel — keep everything",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            discard = reply == QMessageBox.Yes
        else:
            reply = QMessageBox.question(
                self, "Remove Selection",
                f"Remove '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            discard = False

        self.roi_manager.remove_named_roi(row)
        # Announce before refreshing so the single redraw below already
        # reflects any discarded layers.
        self.class_removed.emit(name, discard)
        self._update_roi_list()
        self._apply_roi_change()

    def _clear_all_classes(self):
        """Remove all named class ROIs after confirmation."""
        names = [roi['name'] for roi in self.roi_manager.named_rois]
        if not names:
            return

        layer_count = 0
        if callable(self.layer_count_provider):
            try:
                layer_count = sum(
                    int(self.layer_count_provider(name)) for name in names
                )
            except Exception:
                layer_count = 0

        if layer_count:
            reply = QMessageBox.question(
                self, "Clear All Classes",
                f"Remove all {len(names)} saved class ROIs?\n\n"
                f"{layer_count} segmentation layer(s) were computed from them.\n\n"
                "• Yes — remove the classes and discard their segmentation\n"
                "• No — remove the classes but keep their segmentation\n"
                "• Cancel — keep everything",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            discard = reply == QMessageBox.Yes
        else:
            reply = QMessageBox.question(
                self, "Clear All Classes",
                "Remove all saved class ROIs?\nThis cannot be undone.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            discard = False

        self.roi_manager.clear_named_rois()
        for name in names:
            self.class_removed.emit(name, discard)
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
        # Enable 'Save as Class' whenever an unsaved ROI exists
        self.save_class_btn.setEnabled(self.roi_manager.unsaved_count() > 0)
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
