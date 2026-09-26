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
from gui.material_panel import MaterialPanel, describe_strength
from gui.responsive import fit_to_screen, flow_row, scrollable
from gui.figure_save_dialog import FigureSaveDialog, ask_figure_output

from gui.slice_viewer import SliceViewerWidget, _group_labelled  # noqa: F401
from gui.dialogs import (  # noqa: F401  (re-exported)
    AnchorSelectionDialog, ExportOptionsDialog, FigureExportDialog,
)


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
        # Latest time-series K-means result (utils.kmeans_levels)
        self.kmeans_series_result = None
        # Materials placed from attenuation coefficients:
        # {"references": {name: (μn, μx)}, "targets": {name: (μn, μx)}}
        self.physics_materials = None

        # Current state
        self.global_histogram = None
        # Result of the last model-based run, kept for inspection/export
        self.model_result = None
        self._manual_window = None
        # segmentation_masks: {timepoint -> [(mask_3d, color, name), ...]}
        # Each entry is one coloured segmentation layer for that timepoint.
        self.segmentation_masks = {}
        # Layers hidden by Clear Highlight: {(timepoint, name) -> mask}. Keyed
        # to the mask object, so re-segmenting (a new mask) shows it again.
        self._cleared_layers = {}
        # Classes unticked at the last ROI update, to notice re-ticking
        self._last_hidden_classes = set()
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
        # Open at a size that fits the screen (laptops included); the panels
        # scroll or wrap rather than forcing the window wider than that.
        window_size = fit_to_screen(self)

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
        hist_group = QGroupBox("2D Histograms")
        hist_layout = QVBoxLayout()
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
        left_vbox.addWidget(hist_group)

        # --- Time navigation: a full-width bar along the bottom of the
        # window (added in the final assembly), so the histogram column keeps
        # its height for the plots.
        time_group = QGroupBox()
        time_group.setToolTip("Time navigation")
        time_layout = QVBoxLayout()
        time_layout.setContentsMargins(4, 2, 4, 2)
        self.time_navigation = TimeNavigationWidget(num_timepoints=1)
        self.time_navigation.setEnabled(False)
        self.time_navigation.timepoint_changed.connect(self._on_timepoint_changed)
        time_layout.addWidget(self.time_navigation)
        time_group.setLayout(time_layout)

        left_widget.setLayout(left_vbox)
        # Width floor only: below the content's own minimum height the
        # column scrolls instead of squashing the plots.
        left_widget.setMinimumWidth(300)
        splitter.addWidget(scrollable(left_widget))

        # ══════════════════════════════════════════════════════════════════════
        # CENTRE PANEL — slice viewer (always visible)
        # ══════════════════════════════════════════════════════════════════════
        centre_widget = QWidget()
        centre_vbox = QVBoxLayout()
        centre_vbox.setContentsMargins(2, 2, 2, 2)
        centre_vbox.setSpacing(2)

        viewer_group = QGroupBox("Volume Viewer")
        viewer_layout = QVBoxLayout()
        self.slice_viewer = SliceViewerWidget()
        self.slice_viewer.spatial_roi_to_histogram.connect(
            self._on_create_histogram_roi_from_spatial)
        self.slice_viewer.clusters_detected.connect(self._on_clusters_detected)
        self.slice_viewer.kmeans_requested.connect(self._show_kmeans_controls)
        self.slice_viewer.highlight_cleared.connect(self._on_highlight_cleared)
        viewer_layout.addWidget(self.slice_viewer)
        viewer_group.setLayout(viewer_layout)
        centre_vbox.addWidget(viewer_group)
        centre_widget.setLayout(centre_vbox)
        centre_widget.setMinimumWidth(320)
        splitter.addWidget(scrollable(centre_widget))

        # ══════════════════════════════════════════════════════════════════════
        # RIGHT PANEL — tabbed workflow tools
        # ══════════════════════════════════════════════════════════════════════
        right_tabs = QTabWidget()
        self.right_tabs = right_tabs
        right_tabs.setTabPosition(QTabWidget.North)
        right_tabs.setMinimumWidth(240)
        right_tabs.setMaximumWidth(440)
        right_tabs.setUsesScrollButtons(True)

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
        right_tabs.addTab(scrollable(tab_manual, horizontal=False), "🖊 Manual ROI")

        # ── Tab 2 : Automated segmentation (Otsu / K-means) ───────────────────
        tab_auto = QWidget()
        ta_layout = QVBoxLayout()
        ta_layout.setSpacing(6)

        auto_info = QLabel(
            "<b>Automated Segmentation</b><br>"
            "Run Otsu thresholding or K-means clustering to generate "
            "segmentation layers without drawing anything."
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
            "Groups voxels by their (neutron, X-ray) values, at one of three "
            "scales. <b>Slice</b> is quick; <b>Volume</b> covers this "
            "timepoint; <b>Time series</b> covers every timepoint with one "
            "shared set of clusters and also finds phases present only in "
            "some timepoints."
        )
        km_info.setWordWrap(True)
        km_info.setStyleSheet("color: #555; font-size: 9pt;")
        km_layout.addWidget(km_info)

        km_form = QFormLayout()
        self.kmeans_scope_combo = QComboBox()
        self.kmeans_scope_combo.addItem("Slice — the slice on screen (fast)", "slice")
        self.kmeans_scope_combo.addItem("Volume — this timepoint", "volume")
        self.kmeans_scope_combo.addItem("Time series — every timepoint", "series")
        self.kmeans_scope_combo.setCurrentIndex(1)
        self.kmeans_scope_combo.setToolTip(
            "Slice: seconds; gives 2-D selections on this slice.\n"
            "Volume: every voxel of this timepoint; gives 3-D selections\n"
            "that can be copied to materials.\n"
            "Time series: one clustering for all timepoints, so a cluster\n"
            "keeps its colour through time; also finds phases that appear\n"
            "only in some timepoints. Writes segmentation layers for every\n"
            "timepoint."
        )
        km_form.addRow("Scope:", self.kmeans_scope_combo)

        self.kmeans_clusters_spin = QSpinBox()
        self.kmeans_clusters_spin.setRange(2, 20)
        self.kmeans_clusters_spin.setValue(4)
        self.kmeans_clusters_spin.setToolTip(
            "Number of clusters. For the time series, the number of phases\n"
            "present throughout; phases found only in some timepoints are\n"
            "added on top."
        )
        km_form.addRow("Clusters:", self.kmeans_clusters_spin)
        km_layout.addLayout(km_form)

        self.kmeans_transient_cb = QCheckBox(
            "Find phases present only in some timepoints"
        )
        self.kmeans_transient_cb.setChecked(True)
        self.kmeans_transient_cb.setToolTip(
            "Time series only. Looks for regions of the histogram that fill\n"
            "at some timepoints and are empty at others, and makes each one\n"
            "a cluster of its own, however small. Instrument drift can look\n"
            "like this too — check instrument stability first."
        )
        km_layout.addWidget(self.kmeans_transient_cb)

        self.kmeans_small_cb = QCheckBox("Give small phases more weight")
        self.kmeans_small_cb.setChecked(False)
        self.kmeans_small_cb.setToolTip(
            "Time series only. Lets a small phase that is present throughout\n"
            "win a cluster of its own, at the price of splitting a broad\n"
            "phase more readily."
        )
        km_layout.addWidget(self.kmeans_small_cb)

        def _scope_changed(_index=None):
            series = self.kmeans_scope_combo.currentData() == "series"
            self.kmeans_transient_cb.setEnabled(series)
            self.kmeans_small_cb.setEnabled(series)
        self.kmeans_scope_combo.currentIndexChanged.connect(_scope_changed)
        _scope_changed()

        self.kmeans_run_btn = QPushButton("▶ Run K-means")
        self.kmeans_run_btn.clicked.connect(self._run_kmeans)
        km_layout.addWidget(self.kmeans_run_btn)

        self.kmeans_timeline_btn = QPushButton("📈 Export Cluster Timeline...")
        self.kmeans_timeline_btn.setEnabled(False)
        self.kmeans_timeline_btn.setToolTip(
            "After a time-series run: a plot of every cluster's share of the\n"
            "sample over time (SVG, PDF, PNG or TIFF), and optionally the\n"
            "voxel counts and shares as CSV."
        )
        self.kmeans_timeline_btn.clicked.connect(self._export_kmeans_timeline)
        km_layout.addWidget(self.kmeans_timeline_btn)

        self.copy_clusters_btn = QPushButton(
            "⚡ Copy K-means Clusters to Materials"
        )
        self.copy_clusters_btn.setEnabled(False)
        self.copy_clusters_btn.setToolTip(
            "Turn each volume K-means cluster into a material.\n"
            "Existing drawn and Otsu materials are kept.\n"
            "Copying again replaces the earlier cluster-derived ones.\n"
            "They appear on the Materials tab, where each one can be set\n"
            "to change or to stay unchanged like any other material.\n"
            "(Time-series clusters are layers already.)"
        )
        self.copy_clusters_btn.clicked.connect(
            self._convert_kmeans_clusters_to_materials
        )
        # Every K-means run now puts its clusters in the selection panel
        # itself, so this button is no longer shown; the conversion stays
        # available to code that calls it.
        self.copy_clusters_btn.hide()

        self.kmeans_status_label = QLabel("Status: ready")
        self.kmeans_status_label.setWordWrap(True)
        self.kmeans_status_label.setStyleSheet(
            "color: gray; font-style: italic; font-size: 9pt;"
        )
        km_layout.addWidget(self.kmeans_status_label)

        kmeans_group.setLayout(km_layout)
        ta_layout.addWidget(kmeans_group)

        ta_layout.addStretch()
        tab_auto.setLayout(ta_layout)
        right_tabs.addTab(scrollable(tab_auto, horizontal=False), "⚙ Auto Seg")

        # ── Tab 3 : Materials ─────────────────────────────────────────────
        self.material_panel = MaterialPanel()
        self.material_panel.refresh_requested.connect(
            self._refresh_material_panel
        )
        self.material_panel.copy_clusters_requested.connect(
            self._convert_kmeans_clusters_to_materials
        )
        # K-means runs add their clusters to the selection panel directly
        self.material_panel.copy_clusters_btn.hide()
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
        right_tabs.addTab(scrollable(tab_materials, horizontal=False), "🧱 Materials")

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
        right_tabs.addTab(scrollable(tab_export, horizontal=False), "💾 Export")

        splitter.addWidget(right_tabs)

        # Splitter proportions: left=34%, centre=44%, right=22%
        total = max(window_size.width() - 20, 600)
        splitter.setSizes([int(total * 0.34), int(total * 0.44),
                           int(total * 0.22)])
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)

        # ── Final assembly ─────────────────────────────────────────────────────
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.addWidget(splitter, stretch=1)
        container_layout.addWidget(time_group)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Ready — load a dataset to begin  (Settings → Data Mode to switch 3D/4D)"
        )
        # On a laptop-sized screen start with the spatial tools folded away;
        # the "🛠 Spatial tools" button brings them back.
        if getattr(self, "_bits_small_screen", False):
            self.slice_viewer.set_spatial_tools_visible(False)

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

        export_figure_action = QAction("Export Histogram + Slice Figure...", self)
        export_figure_action.setShortcut("Ctrl+Shift+F")
        export_figure_action.setToolTip(
            "Save a two-panel SVG figure: the local histogram with every\n"
            "label's selection on top, and the slice on screen with the same\n"
            "labels highlighted. PDF, PNG and TIFF can be chosen instead."
        )
        export_figure_action.triggered.connect(
            self._on_export_histogram_slice_figure
        )
        file_menu.addAction(export_figure_action)
        
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

        figure_action = QAction("🖼 Histogram + Slice Figure...", self)
        figure_action.setToolTip(export_figure_action.toolTip())
        figure_action.triggered.connect(self._on_export_histogram_slice_figure)
        export_menu.addAction(figure_action)
        
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

        align_action = QAction("Check Alignment...", self)
        align_action.setToolTip(
            "Whether the X-ray volumes sit exactly on the neutron volumes.\n"
            "Even a one-voxel offset pairs voxels from different materials\n"
            "at every interface and smears the histogram. Measures the\n"
            "offset and offers to correct it."
        )
        align_action.triggered.connect(self._on_check_alignment)
        model_menu.addAction(align_action)

        physics_action = QAction("Add Materials from Attenuation Coefficients...", self)
        physics_action.setToolTip(
            "Place a material on the histogram from its neutron and X-ray\n"
            "attenuation coefficients, calibrated on two materials you have\n"
            "drawn — for a phase with no region to draw."
        )
        physics_action.triggered.connect(self._on_physics_materials)
        model_menu.addAction(physics_action)

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
            ("K-means at Three Scales", "m_kmeans"),
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
        self._cleared_layers = {}
        self._last_hidden_classes = set()
        self._clear_layer_shapes()
        self.model_result = None
        self._last_kmeans_cluster_selections = []
        self._cluster_timepoint = None
        self.physics_materials = None
        self.copy_clusters_btn.setEnabled(False)
        self.material_panel.set_clusters_available(False)
        self.material_panel.clear_result()
        self.material_panel.set_materials([])
        self.kmeans_series_result = None
        self.kmeans_timeline_btn.setEnabled(False)
        self.kmeans_status_label.setText("Status: ready")
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

        # Outlines of segmentation layers belong to this timepoint's layers,
        # so redraw them rather than leaving the previous timepoint's.
        self._update_class_histogram_overlays(timepoint)

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
            # Never keep a previous 3-D grow: "Create Histogram ROI" prefers
            # the 3-D mask and would otherwise use the old region.
            mask_3d = getattr(selection, "spatial_mask_3d", None)
            self.slice_viewer.region_grow_mask_3d = (
                None if mask_3d is None else np.array(mask_3d, dtype=bool)
            )
            source_axis = getattr(selection, "source_axis", None)
            self.slice_viewer.region_grow_plane = (
                (source_axis, getattr(selection, "source_slice_index", None))
                if source_axis is not None else None
            )
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
    
    # ── K-means at three scales ─────────────────────────────────────────

    #: Colours of time-series phases found only in some timepoints; kept
    #: apart from the overlay palette so they stand out.
    _TRANSIENT_COLORS = [
        (1.00, 0.00, 0.85, 0.60),   # magenta
        (0.00, 0.90, 1.00, 0.60),   # cyan
        (1.00, 0.84, 0.00, 0.60),   # gold
        (0.55, 1.00, 0.20, 0.60),   # lime
        (1.00, 0.40, 0.10, 0.60),   # orange
    ]

    def _show_kmeans_controls(self):
        """Auto-Detect in the viewer: bring the K-means settings forward."""
        for index in range(self.right_tabs.count()):
            if self.right_tabs.widget(index).isAncestorOf(self.kmeans_run_btn):
                self.right_tabs.setCurrentIndex(index)
                break
        self.kmeans_run_btn.setFocus()
        self.status_bar.showMessage(
            "Choose the K-means scope and number of clusters, then Run K-means"
        )

    @pyqtSlot()
    def _run_kmeans(self):
        """Run K-means at the scope chosen on the Auto Seg tab."""
        if self.dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first.")
            return
        scope = self.kmeans_scope_combo.currentData()
        n_clusters = self.kmeans_clusters_spin.value()
        from utils.cancellation import OperationCancelled, OperationFailed
        try:
            if scope == "slice":
                self._run_kmeans_slice(n_clusters)
            elif scope == "volume":
                self._run_kmeans_volume(n_clusters)
            else:
                self._run_kmeans_series(
                    n_clusters,
                    find_transient=self.kmeans_transient_cb.isChecked(),
                    emphasise_small=self.kmeans_small_cb.isChecked(),
                )
        except (OperationCancelled, OperationFailed):
            self.kmeans_status_label.setText("Status: cancelled or failed")
        except ValueError as exc:
            QMessageBox.warning(self, "K-means", str(exc))

    def _kmeans_color(self, index):
        return self._OVERLAY_COLORS[index % len(self._OVERLAY_COLORS)]

    def _kmeans_cell_bounds(self):
        """The histogram range, padded by one bin, that K-means cells are
        clipped to — padded so the extreme voxels sit inside, not on, it."""
        hist = self.global_histogram
        x_edges, y_edges = hist.x_edges, hist.y_edges
        dx = float(x_edges[1] - x_edges[0])
        dy = float(y_edges[1] - y_edges[0])
        return (float(x_edges[0]) - dx, float(y_edges[0]) - dy,
                float(x_edges[-1]) + dx, float(y_edges[-1]) + dy)

    def _replace_kmeans_results(self, prefixes, classes, layers_by_timepoint):
        """Put one K-means run into the selection panel and the viewer.

        *classes*: ``(name, outline, rgba, layer_only)`` — each becomes a
        class in the selection panel, so it can be ticked, hidden, renamed,
        edited and removed like a drawn one. *layers_by_timepoint*:
        ``{t: [(mask, rgba, name), ...]}``.

        Classes and layers of an earlier run of the same scope (names
        starting with one of *prefixes*) are replaced everywhere; nothing
        else is touched.
        """
        import matplotlib.colors as mcolors

        prefixes = tuple(prefixes)
        roi_manager = self.dual_histogram.get_roi_manager()
        roi_manager.remove_named_rois_by_name([
            roi['name'] for roi in roi_manager.named_rois
            if roi['name'].startswith(prefixes)
        ])
        for timepoint in list(self.segmentation_masks):
            layers = self.segmentation_masks[timepoint]
            kept = [layer for layer in layers
                    if not str(layer[2]).startswith(prefixes)]
            if len(kept) != len(layers):
                self.segmentation_masks[timepoint] = kept
                for cache in (self.segmentation_layer_shapes,
                              self._derived_outline_cache,
                              self._display_mask_cache):
                    for key in [k for k in cache
                                if k[0] == timepoint
                                and str(k[1]).startswith(prefixes)]:
                        del cache[key]

        outlines = {}
        for name, outline, rgba, layer_only in classes:
            if outline is None:
                continue
            roi_manager.add_named_polygon(
                name, outline, color=mcolors.to_hex(rgba[:3]),
                layer_only=layer_only,
            )
            outlines[name] = outline
        for timepoint, layers in layers_by_timepoint.items():
            self.segmentation_masks.setdefault(timepoint, []).extend(layers)
            for _mask, _rgba, name in layers:
                if name in outlines:
                    self._record_layer_shape(timepoint, name, outlines[name])

        self.dual_histogram._update_roi_list()
        self.dual_histogram._apply_roi_change()
        self.export_current_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)
        current = self.dataset.current_timepoint
        self._apply_segmentation_overlays(current)
        self._update_class_histogram_overlays(current)
        self._refresh_material_panel()

    def _full_resolution_mask(self, display_mask, timepoint):
        """A mask on the display grid, brought to the data's full grid."""
        if self.display_bin_factor <= 1:
            return display_mask
        from utils.display_downsampler import DisplayDownsampler
        shape = self.dataset.get_volume_at_time(timepoint)[0].shape
        return DisplayDownsampler.upscale_mask(
            display_mask, self.display_bin_factor, shape
        )

    def _run_kmeans_slice(self, n_clusters):
        """Level 1: the slice on screen. Fast.

        Each cluster becomes a class in the selection panel whose region is
        the cluster's exact K-means cell, and a layer on this slice only.
        Segment Current then extends a class to the whole volume.
        """
        from utils.clustering_3d import KMeans3D
        from utils.kmeans_levels import cluster_slice, kmeans_cell_polygon

        viewer = self.slice_viewer
        if viewer.current_slice_data is None or viewer.current_slice_index is None:
            raise ValueError("No slice is displayed")
        neutron_vol, xray_vol = viewer.current_slice_data
        axis, index = viewer.current_axis, viewer.current_slice_index
        neutron = KMeans3D.extract_slice_from_labels(neutron_vol, axis, index)
        xray = KMeans3D.extract_slice_from_labels(xray_vol, axis, index)

        result = cluster_slice(neutron, xray, n_clusters)
        bounds = self._kmeans_cell_bounds()
        timepoint = self.dataset.current_timepoint
        classes, layers = [], []
        for cluster in range(n_clusters):
            labels_2d = result.labels == cluster
            if not labels_2d.any():
                continue
            name = f"Slice cluster {cluster}"
            rgba = self._kmeans_color(cluster)
            classes.append((
                name,
                kmeans_cell_polygon(result.centers, result.scale, cluster, bounds),
                rgba, False,
            ))
            display_mask = np.zeros(neutron_vol.shape, dtype=bool)
            if axis == 'z':
                display_mask[index, :, :] = labels_2d
            elif axis == 'y':
                display_mask[:, index, :] = labels_2d
            else:
                display_mask[:, :, index] = labels_2d
            layers.append((self._full_resolution_mask(display_mask, timepoint),
                           rgba, name))

        self._replace_kmeans_results(("Slice cluster ",), classes,
                                     {timepoint: layers})
        self.kmeans_status_label.setText(
            f"Status: {len(classes)} cluster(s) on the {axis.upper()} slice "
            f"{index}, added to the selection panel — Segment Current "
            "extends them to the volume"
        )
        self.status_bar.showMessage(
            f"Slice K-means: {len(classes)} clusters on slice {index}"
        )

    def _run_kmeans_volume(self, n_clusters):
        """Level 2: every voxel of this timepoint.

        Each cluster becomes a class in the selection panel (its exact
        K-means cell, so Segment All reproduces it at every timepoint) and
        a layer at this timepoint.
        """
        from utils.clustering_3d import KMeans3D
        from utils.kmeans_levels import kmeans_cell_polygon

        timepoint = self.dataset.current_timepoint
        neutron_vol, xray_vol = self._display_volumes_at(timepoint)

        def operation(progress_callback=None, cancel_check=None):
            return KMeans3D.cluster_volume(
                neutron_vol, xray_vol, n_clusters=n_clusters,
                progress_callback=progress_callback, cancel_check=cancel_check,
            )

        outcome = run_with_progress(
            self, "K-means: Volume",
            f"Clustering timepoint {timepoint} into {n_clusters} groups...",
            operation,
        )
        if outcome is None:
            return
        labels, centers, stats = outcome

        bounds = self._kmeans_cell_bounds()
        classes, layers = [], []
        for cluster in range(n_clusters):
            mask = labels == cluster
            if not mask.any():
                continue
            name = f"K-means cluster {cluster}"
            rgba = self._kmeans_color(cluster)
            classes.append((
                name,
                kmeans_cell_polygon(centers, stats["feature_scale"], cluster,
                                    bounds),
                rgba, False,
            ))
            layers.append((self._full_resolution_mask(mask, timepoint),
                           rgba, name))

        self._replace_kmeans_results(("K-means cluster ", "3D Cluster "),
                                     classes, {timepoint: layers})
        self.kmeans_status_label.setText(
            f"Status: {len(classes)} cluster(s) at T={timepoint}, added to the "
            "selection panel — Segment All extends them to every timepoint"
        )
        self.status_bar.showMessage(
            f"Volume K-means: {len(classes)} clusters at T={timepoint}"
        )

    def _run_kmeans_series(self, n_clusters, find_transient=True,
                           emphasise_small=False):
        """Level 3: every timepoint, one shared clustering.

        Writes one segmentation layer per cluster at every timepoint where
        that cluster holds voxels, replacing earlier series layers.
        """
        from utils.kmeans_levels import describe_presence, run_series_clustering

        if self.histogram_engine is None or self.global_histogram is None:
            raise ValueError("Compute the histogram first (load a dataset).")

        def operation(progress_callback=None, cancel_check=None):
            return run_series_clustering(
                self.dataset, self.histogram_engine, n_clusters,
                emphasise_small=emphasise_small,
                find_transient=find_transient,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

        outcome = run_with_progress(
            self, "K-means: Time Series",
            f"Clustering all {self.dataset.num_timepoints} timepoints...",
            operation,
        )
        if outcome is None:
            return
        result, labels = outcome

        colors = [self._series_cluster_color(result, k)
                  for k in range(result.n_clusters)]
        names = [result.cluster_name(k) for k in range(result.n_clusters)]
        outlines = [result.outline(k) for k in range(result.n_clusters)]
        layers_by_timepoint = {}
        for t_index, timepoint in enumerate(result.timepoints):
            label_volume = labels.pop(timepoint)
            layers = []
            for cluster in range(result.n_clusters):
                if result.counts[t_index, cluster] <= 0:
                    continue
                mask = label_volume == cluster
                if mask.any():
                    layers.append((mask, colors[cluster], names[cluster]))
            layers_by_timepoint[timepoint] = layers

        # Listed as classes so each can be ticked on and off; layer-only,
        # because a transient phase cuts a hole in its neighbour's region
        # and no single outline describes that — the per-timepoint labels
        # are the segmentation.
        classes = [(names[k], outlines[k], colors[k], True)
                   for k in range(result.n_clusters)]
        self.kmeans_series_result = result
        self._kmeans_series_colors = colors
        self.kmeans_timeline_btn.setEnabled(True)
        self._replace_kmeans_results(("Series cluster ", "Transient phase "),
                                     classes, layers_by_timepoint)

        transient = result.transient_clusters()
        lines = [
            f"{result.n_persistent} phase(s) present through the series"
            + (f", and {result.n_clusters - result.n_persistent} found only "
               "in some timepoints." if result.n_clusters > result.n_persistent
               else "."),
            "",
        ]
        for cluster in range(result.n_clusters):
            peak = 100.0 * result.fractions[:, cluster].max()
            where = describe_presence(result.present_at(cluster),
                                      result.timepoints)
            flag = "  ← only some timepoints" if cluster in transient else ""
            lines.append(
                f"• {names[cluster]}: neutron {result.centers[cluster, 0]:.4g}, "
                f"X-ray {result.centers[cluster, 1]:.4g} — up to {peak:.2f}% "
                f"of the sample, present {where}{flag}"
            )
        lines += [
            "",
            "Each cluster is a segmentation layer at every timepoint where it "
            "has voxels. Export Cluster Timeline writes the share of each "
            "cluster over time.",
        ]
        self.kmeans_status_label.setText(
            f"Status: {result.n_clusters} series cluster(s), "
            f"{len(transient)} present only in some timepoints"
        )
        QMessageBox.information(self, "Time-Series K-means", "\n".join(lines))

    def _series_cluster_color(self, result, cluster):
        if result.is_transient_phase(cluster):
            offset = cluster - result.n_persistent
            return self._TRANSIENT_COLORS[offset % len(self._TRANSIENT_COLORS)]
        return self._kmeans_color(cluster)

    @pyqtSlot()
    def _export_kmeans_timeline(self):
        """CSV of every cluster's share over time, and a plot of it."""
        result = self.kmeans_series_result
        if result is None:
            QMessageBox.information(
                self, "No Time-Series Result",
                "Run K-means with the Time series scope first."
            )
            return
        output = ask_figure_output(
            self, "Export Cluster Timeline", "kmeans_timeline",
            csv_label="Also save each cluster's share over time as CSV",
        )
        if output is None:
            return
        from utils.kmeans_levels import plot_timeline
        written = []
        try:
            plot_timeline(result, str(output.figure_path),
                          colors=getattr(self, "_kmeans_series_colors", None),
                          dpi=output.dpi)
            written.append(str(output.figure_path))
            if output.save_csv:
                result.write_timeline_csv(output.data_path())
                written.append(str(output.data_path()))
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", f"Could not write:\n{exc}")
            return
        self.status_bar.showMessage("Timeline saved: " + ", ".join(written))

    @pyqtSlot()
    def _convert_kmeans_clusters_to_materials(self):
        """
        Convert the most recent volume K-means result into materials.

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
                "No Volume K-means Result",
                "Run K-means with the Volume scope first.",
            )
            return

        try:
            from segmentation.kmeans_class_conversion import (
                build_material_layers_from_cluster_selections,
            )
            from utils.display_downsampler import DisplayDownsampler

            neutron_vol, xray_vol = self.dataset.get_volume_at_time(source_t)

            # Clustering runs on the display grid for large datasets;
            # materials live at full resolution, so upscale the masks first.
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
                f"Could not create materials:\n{exc}",
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
            f"Converted {len(new_layers)} K-means clusters to materials "
            f"at T={source_t}"
        )

        coverage_percent = (
            100.0 * summary.covered_voxels / summary.total_voxels
        )
        details = (
            f"Created {len(new_layers)} materials from T={source_t}.\n\n"
            f"Covered voxels: {summary.covered_voxels:,} "
            f"({coverage_percent:.2f}%)\n"
            f"Uncovered/background voxels: {summary.uncovered_voxels:,}\n"
            f"Overlapping voxels: {summary.overlapping_voxels:,}\n"
            f"New saved histogram selections: {added_selections}\n\n"
            "They are listed on the Materials tab, ready to track through "
            "the series."
        )

        if summary.overlapping_voxels:
            QMessageBox.warning(
                self,
                "K-means Materials Created with Overlap",
                details
                + "\n\nOverlapping masks are resolved by layer order.",
            )
        else:
            QMessageBox.information(
                self,
                "K-means Materials Ready",
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
        # grown on — which may not be the slice on screen any more.
        plane = getattr(self.slice_viewer, 'region_grow_plane', None)
        if spatial_mask is None or plane is None:
            plane = (self.slice_viewer.current_axis,
                     self.slice_viewer.current_slice_index)
        self.selection_manager.add_selection(
            name=name,
            spatial_mask=spatial_mask,
            histogram_roi=histogram_roi,
            spatial_mask_3d=getattr(
                self.slice_viewer, 'region_grow_mask_3d', None
            ),
            source_axis=plane[0],
            source_slice_index=plane[1],
        )

        self.status_bar.showMessage(f"✅ Saved selection: {name}")
    
    def _update_histogram_overlays(self):
        """Show visible saved selections on the histograms and slice viewer.

        Both views are re-composed rather than replaced, so toggling the
        saved selections never erases the class outlines on the histograms
        or the segmentation layers on the slice.
        """
        if self.dataset is not None:
            self._update_class_histogram_overlays(self.dataset.current_timepoint)
        else:
            overlays = list(
                self.dual_histogram.get_roi_manager().get_named_roi_overlays()
            )
            overlays += self._selection_histogram_overlays()
            for canvas in (self.dual_histogram.global_canvas,
                           self.dual_histogram.local_canvas):
                canvas.set_roi_overlays(overlays)
        self._refresh_slice_overlays()

    def _selection_histogram_overlays(self):
        """Histogram outlines of the visible saved selections.

        Empty unless "Show All on Histogram" is ticked.
        """
        manager = getattr(self, "selection_manager", None)
        if manager is None or not manager.show_all_cb.isChecked():
            return []
        overlays = []
        for selection in manager.get_visible_selections():
            if selection.histogram_roi is None:
                continue
            color = (
                selection.color if selection.color is not None
                else (1, 0, 0, 0.8)
            )
            overlays.append((selection.name, selection.histogram_roi, color))
        return overlays

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

    _OVERLAY_COLORS = [
        (0.93, 0.11, 0.14, 0.50),   # red
        (0.13, 0.47, 0.71, 0.50),   # blue
        (0.17, 0.63, 0.17, 0.50),   # green
        (1.00, 0.50, 0.05, 0.50),   # orange
        (0.58, 0.40, 0.74, 0.50),   # purple
        (0.09, 0.75, 0.81, 0.50),   # cyan
        (0.89, 0.47, 0.76, 0.50),   # pink
        (0.74, 0.74, 0.13, 0.50),   # yellow-green
    ]

    def _next_roi_color(self, roi_manager, layer_index: int):
        """Return an RGBA colour for the next segmentation layer.

        Priority:
          1. The last visible named ROI's colour (if any exist).
          2. Cycle through the default palette.
        """
        named = roi_manager.get_visible_named_rois()
        if named:
            hex_color = named[-1].get('color', '#e6194b')
            try:
                import matplotlib.colors as mcolors
                r, g, b, _ = mcolors.to_rgba(hex_color)
                return (r, g, b, 0.50)
            except Exception:
                pass
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
        cleared = getattr(self, "_cleared_layers", {})
        return [
            layer for layer in self.segmentation_masks.get(timepoint, [])
            if layer[2] not in hidden
            and cleared.get((int(timepoint), layer[2])) is not layer[0]
        ]

    def _on_highlight_cleared(self):
        """Clear Highlight: hide every layer until it is asked for again.

        Before, the button only wiped the screen: the layers stayed visible,
        so the next redraw (unticking a class, drawing, segmenting) brought
        every one of them back. Now each layer is hidden until its class is
        ticked again or it is segmented again, and the classes are unticked
        — so the next segmentation covers, and shows, only what is selected
        next. Nothing is deleted.
        """
        self._cleared_layers = {
            (int(timepoint), layer[2]): layer[0]
            for timepoint, layers in self.segmentation_masks.items()
            for layer in layers
        }
        if self.selection_manager.show_all_cb.isChecked():
            self.selection_manager.show_all_cb.setChecked(False)
        roi_manager = self.dual_histogram.get_roi_manager()
        if roi_manager.get_visible_named_rois():
            self.dual_histogram._set_all_classes_visible(False)
        self._last_hidden_classes = self._hidden_class_names()
        if self.dataset is not None:
            self._refresh_slice_overlays()
            self._update_class_histogram_overlays(self.dataset.current_timepoint)
        self.status_bar.showMessage(
            "Highlights cleared — tick a class, or segment again, to show "
            "layers; nothing was deleted"
        )

    def _forget_cleared_for_reticked_classes(self):
        """A class ticked back on shows its layers again."""
        hidden_now = self._hidden_class_names()
        reticked = self._last_hidden_classes - hidden_now
        if reticked:
            self._cleared_layers = {
                key: mask for key, mask in self._cleared_layers.items()
                if key[1] not in reticked
            }
        self._last_hidden_classes = hidden_now

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

        Three things are shown together:

        1. the regions you drew, exactly as drawn, and
        2. an outline around any segmentation layer that is *not* one of them
           — from Otsu, from K-means, or from a material-tracking run — so you
           can see where those boundaries fall relative to your own, and
        3. the visible saved selections, while "Show All on Histogram" is on.

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

        overlays += self._selection_histogram_overlays()
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

        Includes every *visible* named class ROI plus every unsaved ROI, so
        segmentation always covers exactly the selection displayed on the
        histogram canvases. Classes hidden in the selection panel are
        neither drawn nor segmented. A single unsaved ROI is called
        "Active ROI"; several are numbered, each in the colour it is drawn in.
        """
        specs = []
        for roi in roi_manager.get_segmentable_named_rois():
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

        unsaved = roi_manager.get_unsaved_rois()
        for number, roi in enumerate(unsaved, start=1):
            spec = {
                'name': ('Active ROI' if len(unsaved) == 1
                         else f'Unsaved ROI {number}'),
                'roi_type': roi['roi_type'],
                'color': roi['color'],
            }
            if roi['roi_type'] == 'polygon':
                spec['points'] = np.array(roi['points'], dtype=float)
            else:
                spec['rectangle'] = tuple(roi['rectangle'])
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
        if (roi_manager.get_segmentable_named_rois()
                or roi_manager.unsaved_count() > 1):
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
            for mask, color, name in layers:
                try:
                    r, g, b, _ = mcolors.to_rgba(color)
                    color_rgba = (r, g, b, 0.50)
                except Exception:
                    idx = len(kept)
                    color_rgba = self._OVERLAY_COLORS[idx % len(self._OVERLAY_COLORS)]
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
            from segmentation.legacy.random_forest_4d import labels_from_otsu
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

        n_existing = len(kept_layers)
        for cls_id in classes:
            mask_3d = (labels == cls_id)
            color   = self._OVERLAY_COLORS[(n_existing + cls_id - 1) % len(self._OVERLAY_COLORS)]
            kept_layers.append((mask_3d, color, f"Otsu class {cls_id}"))
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
        unsaved = roi_manager.get_unsaved_rois()
        for number, roi in enumerate(unsaved, start=1):
            label = "Active ROI" if len(unsaved) == 1 else f"Unsaved ROI {number}"
            roi_info[label] = f"{roi['roi_type']} (unsaved)"

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
                                path_prefix, image_format="svg",
                                write_csv=False):
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
            image_format=image_format,
            write_csv=write_csv,
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
            import os
            from data.tiff_io import write_volume_tiff
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
                    write_volume_tiff(p, mask_bool.astype(np.uint8) * 255)
                    files_written.append(os.path.basename(p))
                if do_neutron:
                    vol = neutron_vol.copy(); vol[~mask_bool] = 0
                    p = f"{pfx}_neutron.tif"
                    write_volume_tiff(p, vol)
                    files_written.append(os.path.basename(p))
                if do_xray:
                    vol = xray_vol.copy(); vol[~mask_bool] = 0
                    p = f"{pfx}_xray.tif"
                    write_volume_tiff(p, vol)
                    files_written.append(os.path.basename(p))
                if do_histogram:
                    files_written += self._export_class_histogram(
                        current_t, name, mask_bool, output_dir, pfx,
                        image_format=dlg.histogram_format,
                        write_csv=dlg.histogram_csv,
                    )

            if do_labels:
                label_vol = np.zeros(
                    neutron_vol.shape,
                    dtype=np.uint8 if len(sel_layers) < 256 else np.uint16,
                )
                for idx, (mask_3d, _, _) in enumerate(sel_layers, start=1):
                    label_vol[mask_3d.astype(bool)] = idx
                p = os.path.join(output_dir, f"{base}_labels.tif")
                write_volume_tiff(p, label_vol)
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

        # Offer the union of layer names across every timepoint, so a class
        # segmented only at later timepoints can still be picked. Each name
        # is represented by its first occurrence (its swatch and voxel count).
        representative_layers = []
        seen_names = set()
        for _t, layers in sorted(self.segmentation_masks.items()):
            for layer in layers:
                if layer[2] not in seen_names:
                    seen_names.add(layer[2])
                    representative_layers.append(layer)

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
            import os
            from data.tiff_io import write_volume_tiff
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

            # Only non-empty timepoints count towards progress
            segmented = [
                (t, layers)
                for t, layers in sorted(self.segmentation_masks.items())
                if layers
            ]
            for i, (t, layers) in enumerate(segmented):
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
                        write_volume_tiff(f"{pfx}_mask.tif",
                                         mask_bool.astype(np.uint8) * 255)
                        total_files += 1
                    if do_neutron:
                        vol = neutron_vol.copy(); vol[~mask_bool] = 0
                        write_volume_tiff(f"{pfx}_neutron.tif", vol)
                        total_files += 1
                    if do_xray:
                        vol = xray_vol.copy(); vol[~mask_bool] = 0
                        write_volume_tiff(f"{pfx}_xray.tif", vol)
                        total_files += 1
                    if do_histogram:
                        total_files += len(self._export_class_histogram(
                            t, name, mask_bool, output_dir, pfx,
                            image_format=dlg.histogram_format,
                            write_csv=dlg.histogram_csv,
                        ))

                exported_layers[t] = t_layers

                if do_labels:
                    label_vol = np.zeros(
                        neutron_vol.shape,
                        dtype=np.uint8 if len(class_order) < 256 else np.uint16,
                    )
                    # Label values come from the fixed class order, not this
                    # timepoint's list, so a value means the same class in
                    # every exported volume even if a class is missing here.
                    for mask_3d, _color, layer_name in t_layers:
                        label_vol[mask_3d.astype(bool)] = label_values[layer_name]
                    write_volume_tiff(
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
                    f"Exported {len(exported_layers)} timepoint(s).\n"
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

    # ── Histogram + slice figure ─────────────────────────────────────────

    def _histogram_slice_figure_panels(self, include_active_roi=True):
        """What the figure export draws, gathered from what is on screen.

        Left panel: the local histogram exactly as displayed (scale, range)
        with every label outline the local canvas shows — saved classes,
        outlines of other segmentation layers and visible saved selections —
        plus, optionally, the unsaved ROI. Right panel: the slice on screen
        with the same labels' highlights, re-sliced for the current plane.

        Returns ``(HistogramPanel, SlicePanel)``, or None when there is no
        local histogram or slice to draw.
        """
        from utils.figure_export import HistogramPanel, SlicePanel

        canvas = self.dual_histogram.local_canvas
        viewer = self.slice_viewer
        if (self.dataset is None or canvas.histogram_data is None
                or getattr(viewer, "current_slice", None) is None):
            return None

        timepoint = self.dataset.current_timepoint
        overlays = [
            (name, np.asarray(vertices, dtype=float), color)
            for name, vertices, color in canvas.roi_overlays
            if vertices is not None
        ]
        roi_manager = self.dual_histogram.get_roi_manager()
        if include_active_roi:
            unsaved = roi_manager.get_unsaved_rois()
            for number, roi in enumerate(unsaved, start=1):
                label = ("Unsaved ROI" if len(unsaved) == 1
                         else f"Unsaved ROI {number}")
                overlays.append((
                    label, roi_manager.target_vertices(roi['target']),
                    roi['color'],
                ))

        slice_overlays = []
        for entry in viewer.mask_overlays:
            name, mask, color = entry[0], entry[1], entry[2]
            plane = entry[3] if len(entry) > 3 else None
            mask_2d = viewer._slice_mask_for_display(mask, plane)
            if mask_2d is not None:
                slice_overlays.append((name, np.asarray(mask_2d, bool), color))

        note = ""
        if viewer.display_bin_factor > 1:
            note = (
                f"Display binned x{viewer.display_bin_factor} (median); "
                "segmentation ran at full resolution"
            )

        histogram_panel = HistogramPanel(
            histogram_data=canvas.histogram_data,
            overlays=overlays,
            log_scale=canvas.use_log_scale,
            vmin=canvas.vmin,
            vmax=canvas.vmax,
            title=f"Local histogram (T={timepoint})",
        )
        slice_panel = SlicePanel(
            image=np.asarray(viewer.current_slice),
            overlays=slice_overlays,
            vmin=viewer.vmin,
            vmax=viewer.vmax,
            title=viewer.ax.get_title() or f"Slice (T={timepoint})",
            note=note,
        )
        return histogram_panel, slice_panel

    @pyqtSlot()
    def _on_export_histogram_slice_figure(self):
        """Save the local histogram and the current slice side by side."""
        if self.dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first")
            return
        if self.dual_histogram.local_canvas.histogram_data is None:
            QMessageBox.warning(
                self, "No Histogram",
                "The local histogram has not been computed yet."
            )
            return

        has_active = self.dual_histogram.get_roi_manager().unsaved_count() > 0
        dialog = FigureExportDialog(has_active_roi=has_active, parent=self)
        timepoint = self.dataset.current_timepoint
        axis = self.slice_viewer.current_axis
        index = self.slice_viewer.current_slice_index
        output = ask_figure_output(
            self, "Save Histogram + Slice Figure",
            f"histogram_slice_T{timepoint:03d}_{axis}{index}", dialog=dialog,
        )
        if output is None:
            return

        panels = self._histogram_slice_figure_panels(dialog.include_active_roi)
        if panels is None:
            QMessageBox.warning(
                self, "Nothing to Export",
                "There is no slice or local histogram on screen to export."
            )
            return

        from utils.figure_export import (
            save_histogram_slice_figure, write_histogram_slice_csv,
        )
        try:
            written = [str(save_histogram_slice_figure(
                output.figure_path, *panels,
                dpi=output.dpi,
                show_legend=dialog.show_legend,
                outline_highlights=dialog.outline_highlights,
            ))]
            if output.save_csv:
                written.append(write_histogram_slice_csv(
                    output.data_path(), *panels
                ))
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error", f"Could not save the figure:\n{exc}"
            )
            import traceback; traceback.print_exc()
            return

        self.status_bar.showMessage("Saved: " + ", ".join(written))

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
                                     explanation, default_name=None,
                                     csv_label="Also save the data as CSV"):
        """Shared driver for the temporal histogram analyses.

        Asks the figure format (and whether to save the data as CSV), then
        collects every timepoint's local histogram (from cache, computing any
        that are missing) and hands the list to *render(histograms, output)*,
        which saves the figure (and CSV) and returns the paths written.
        """
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

        output = ask_figure_output(
            self, dialog_caption,
            default_name or title.lower().replace(" ", "_"),
            csv_label=csv_label,
        )
        if output is None:
            return

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
            return render(histograms, output)

        try:
            saved = run_with_progress(
                self, title, f"Building {title.lower()}...", operation
            )
        except (OperationCancelled, OperationFailed):
            return

        if saved:
            files = "\n".join(str(path) for path in saved)
            self.status_bar.showMessage(f"{title} saved: {saved[0]}")
            QMessageBox.information(
                self, "Saved",
                f"{title} saved to:\n{files}\n\n{explanation}"
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

        output = ask_figure_output(
            self, "Save Histogram & Segmentation Metrics", "metrics",
            csv_label="Also save the metric values as CSV",
        )
        if output is None:
            return
        filepath = str(output.data_path()) if output.save_csv else None
        plot_path = str(output.figure_path)

        def operation(progress_callback=None, cancel_check=None):
            rows = self._collect_metrics_rows(progress_callback, cancel_check)
            if progress_callback:
                progress_callback(97, "Writing CSV and plots...")
            if filepath:
                write_metrics_csv(rows, filepath)
            return rows, plot_metric_evolution(rows, plot_path, dpi=output.dpi)

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
            (f"Metric values written to:\n{filepath}\n\n" if filepath else "")
            + f"{len(rows) - 1} timepoint(s) analysed, "
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
        self.status_bar.showMessage(
            f"Metrics saved: {saved_plot or filepath or 'nothing written'}"
        )
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

    def _on_check_alignment(self):
        """Measure the neutron/X-ray offset and offer to remove it."""
        from utils.cancellation import OperationCancelled, OperationFailed
        from model.registration import check_series_alignment

        if not self.dataset:
            QMessageBox.information(self, "No Data", "Load a dataset first.")
            return

        def operation(progress_callback=None, cancel_check=None):
            if progress_callback:
                progress_callback(10, "Comparing the two instruments' edges...")
            return check_series_alignment(self.dataset, cancel_check=cancel_check)

        try:
            results = run_with_progress(
                self, "Check Alignment",
                "Measuring the offset between the neutron and X-ray volumes...",
                operation,
            )
        except (OperationCancelled, OperationFailed):
            return
        if not results:
            return
        self.alignment_results = results

        lines = [f"T{t}: {a.describe()}" for t, a in results]
        shifts = np.array([a.correction for _t, a in results], dtype=float)
        needs = [a.worth_correcting for _t, a in results]
        if not any(needs):
            QMessageBox.information(
                self, "Alignment",
                "The two instruments are aligned (no offset of a whole voxel "
                "or more that mutual information confirms; a smaller offset is "
                "better left than interpolated away).\n\n"
                + "\n".join(lines),
            )
            self.status_bar.showMessage("Alignment: no correction needed")
            return

        spread = float(np.max(np.ptp(shifts, axis=0))) if len(shifts) > 1 else 0.0
        constant = spread < 0.5
        advice = (
            "The offset is the same throughout the series — a mounting "
            "offset. Correcting it shifts every X-ray volume by the same "
            "whole number of voxels."
            if constant else
            "The offset changes during the series, so something moved during "
            "the experiment. Correcting it measures and removes the offset "
            "at every timepoint separately."
        )
        reply = QMessageBox.question(
            self, "Alignment",
            "The X-ray volumes are offset from the neutron volumes:\n\n"
            + "\n".join(lines) + "\n\n" + advice
            + "\n\nCorrect the X-ray volumes now? The histograms are then "
            "recomputed; the files on disk are not changed.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        self._correct_alignment(
            [float(np.round(v)) for v in shifts.mean(axis=0)] if constant
            else None
        )

    def _correct_alignment(self, shift=None):
        """Shift the X-ray volumes onto the neutron volumes, in memory.

        *shift* applies one offset to every timepoint; None measures and
        applies each timepoint's own. Shifts are whole voxels (see
        :mod:`model.registration`), so no value is interpolated. Voxels shifted in from outside the
        field become NaN and are excluded like any unmeasured voxel.
        """
        from utils.cancellation import OperationCancelled, OperationFailed
        from model.registration import apply_shift, estimate_alignment

        dataset = self.dataset

        def operation(progress_callback=None, cancel_check=None):
            corrected = np.empty(dataset.xray_data.shape, dtype=np.float32)
            applied = []
            for t in range(dataset.num_timepoints):
                if cancel_check:
                    cancel_check()
                neutron, xray = dataset.get_volume_at_time(t)
                offset = shift
                if offset is None:
                    offset = estimate_alignment(neutron, xray).correction
                corrected[t] = apply_shift(xray, offset)
                applied.append(tuple(offset))
                if progress_callback:
                    progress_callback(int(100 * (t + 1) / dataset.num_timepoints),
                                      f"Aligned timepoint {t + 1}")
            return corrected, applied

        try:
            outcome = run_with_progress(self, "Correct Alignment",
                                        "Shifting the X-ray volumes...", operation)
        except (OperationCancelled, OperationFailed):
            return
        if outcome is None:
            return
        corrected, applied = outcome
        dataset.xray_data = corrected
        dataset.metadata["xray_alignment_shift"] = applied
        self.alignment_applied = applied

        # Everything computed from the X-ray intensities is now stale
        from histograms import HistogramEngine4D
        self.histogram_engine = HistogramEngine4D(
            bins=self.histogram_engine.bins,
            cache_size=self.histogram_engine.cache_size,
            use_gpu=self.histogram_engine.use_gpu,
        )
        self.global_histogram = None
        self._derived_outline_cache = {}
        self._compute_global_histogram()
        self._precompute_local_histograms()
        self._prepare_display_volumes()
        self._update_current_timepoint(dataset.current_timepoint)
        self.status_bar.showMessage(
            "X-ray volumes aligned to the neutron volumes; histograms "
            "recomputed"
        )

    _CLUSTER_PREFIXES = ("K-means cluster", "3D Cluster", "Cluster ")

    def _physics_library(self, sources):
        """Classes placed from coefficients, calibrated on *sources*.

        *sources* is ``{name: (neutron, xray, mask, valid, timepoint)}`` —
        the drawn materials, measured afresh on every run so a correction
        of the data (alignment) also moves the predictions. None when no
        material is placed from coefficients.
        """
        entry = self.physics_materials
        if not entry or not entry.get("targets"):
            return None
        from model.calibration import (
            fit_calibration, predicted_classes, reference_from_region,
        )
        references = []
        for name, coefficients in entry["references"].items():
            if name not in sources:
                continue
            neutron, xray, mask, valid, _t = sources[name]
            references.append(reference_from_region(
                name, coefficients, neutron, xray, mask & valid))
        if len(references) < 2:
            return None
        calibration = fit_calibration(references)
        self.physics_calibration = calibration
        return predicted_classes(entry["targets"], calibration)

    def _on_physics_materials(self):
        """Place materials on the histogram from attenuation coefficients."""
        if not self.dataset or self.global_histogram is None:
            QMessageBox.information(self, "No Data",
                                    "Load a dataset and compute the histogram first.")
            return
        reference = self._material_reference_timepoint()
        definitions = self._material_definitions(reference)
        if len(definitions) < 2:
            QMessageBox.information(
                self, "Materials from Coefficients",
                "Draw and segment at least two materials whose attenuation "
                "coefficients you know (air and aluminium are ideal). They "
                "calibrate the grey scale of each instrument.",
            )
            return
        from gui.physics_dialog import PhysicsMaterialsDialog
        dialog = PhysicsMaterialsDialog(list(definitions),
                                        self.physics_materials, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        references, targets, problems = dialog.values()
        if problems:
            return
        from model.validity import build_valid_mask

        sources = {}
        for name in references:
            timepoint, mask = definitions[name]
            neutron, xray = self.dataset.get_volume_at_time(timepoint)
            sources[name] = (neutron, xray, mask,
                             build_valid_mask(neutron, xray), timepoint)
        previous = self.physics_materials
        self.physics_materials = {"references": references, "targets": targets}
        try:
            library = self._physics_library(sources)
        except ValueError as error:
            self.physics_materials = previous
            QMessageBox.warning(self, "Materials from Coefficients", str(error))
            return
        calibration = self.physics_calibration
        lines = [calibration.describe(), "", "Predicted positions (neutron, X-ray):"]
        for material in library:
            spread = np.sqrt(np.diag(material.sigma))
            lines.append(f"  {material.name}: ({material.mu[0]:.4g}, "
                         f"{material.mu[1]:.4g}) ± ({spread[0]:.3g}, "
                         f"{spread[1]:.3g})")
        lines += ["", "They are tracked with the drawn materials from the next "
                  "run (locked definitions). A predicted material is expected "
                  "to be absent until it forms; the health check warns if it "
                  "is never found."]
        QMessageBox.information(self, "Materials from Coefficients",
                                "\n".join(lines))
        self._refresh_material_panel()
        self.status_bar.showMessage(
            f"{len(targets)} material(s) placed from attenuation coefficients")

    def _material_source(self, name: str) -> str:
        """Where a material came from, for the panel's second column."""
        if any(name.startswith(prefix) for prefix in self._CLUSTER_PREFIXES):
            return "K-means"
        if name.startswith("Otsu"):
            return "Otsu"
        return "drawn"

    def _material_definitions(self, reference):
        """Every material and the timepoint it is defined at.

        ``{name: (timepoint, mask)}``, in panel order. A material present at
        the reference timepoint is defined there. One that is not — a phase
        that only appears later, like a reaction product or a time-series
        transient phase — is defined at the first timepoint where it has
        voxels, instead of being impossible to define at all.
        """
        definitions = {}
        order = [reference] + [
            t for t in range(self.dataset.num_timepoints) if t != reference
        ]
        for timepoint in order:
            for mask, _color, name in self._visible_layers(timepoint):
                if name in definitions:
                    continue
                mask = np.asarray(mask, dtype=bool)
                if mask.any():
                    definitions[name] = (timepoint, mask)
        return definitions

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
                "source": self._material_source(name) + (
                    "" if timepoint == reference else f" (T{timepoint})"
                ),
                "voxels": int(np.count_nonzero(mask)),
            }
            for name, (timepoint, mask)
            in self._material_definitions(reference).items()
        ]
        drawn = {m["name"] for m in materials}
        for name in (self.physics_materials or {}).get("targets", {}):
            if name not in drawn:
                materials.append({"name": name, "source": "coefficients",
                                  "voxels": None})
        self.material_panel.set_materials(materials)
        self.material_panel.set_clusters_available(
            bool(self._last_kmeans_cluster_selections)
        )
        return materials

    def _on_model_segmentation(self):
        """Menu route: show the materials, then run the series."""
        if hasattr(self, "right_tabs") and hasattr(self, "material_panel"):
            # Tab pages are scroll areas, so look for the page holding it
            for index in range(self.right_tabs.count()):
                if self.right_tabs.widget(index).isAncestorOf(self.material_panel):
                    self.right_tabs.setCurrentIndex(index)
                    break
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
        definitions = self._material_definitions(reference)
        if not definitions:
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
            name for name in settings["control_materials"] if name in definitions
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
            # Each material from the timepoint it exists at, so a phase that
            # appears later can still be defined and tracked
            volumes = {reference: (neutron_reference, xray_reference, valid)}
            sources = {}
            for name, (timepoint, mask) in definitions.items():
                if timepoint not in volumes:
                    n_t, x_t = self.dataset.get_volume_at_time(timepoint)
                    volumes[timepoint] = (n_t, x_t, build_valid_mask(n_t, x_t))
                n_t, x_t, valid_t = volumes[timepoint]
                sources[name] = (n_t, x_t, mask, valid_t, timepoint)
            import warnings as _warnings
            with _warnings.catch_warnings():
                # Dropped classes are reported in the health check instead
                _warnings.simplefilter("ignore")
                library = ClassLibrary.from_sources(
                    sources, inert=settings["control_materials"],
                    max_components=settings.get("max_components", 1),
                )
            predicted = self._physics_library(sources)
            if predicted is not None:
                from model.calibration import merge_libraries
                library = merge_libraries(library, predicted)
                for material in library:
                    if material.name in settings["control_materials"]:
                        material.inert = True
            segmenter = LockedSegmenter(
                library, prior=ROIDerivedMRF(beta=1.0, n_sweeps=6),
                bins=self.histogram_engine.bins,
            )
            segmenter.set_grid(
                self.global_histogram.x_edges, self.global_histogram.y_edges
            )

            reference_labels = np.zeros(valid.shape, dtype=np.int32)
            for index, name in enumerate(library.names):
                if name in masks:
                    reference_labels[masks[name]] = index
            segmenter.learn_boundaries(reference_labels, valid_mask=valid)

            sweep = None
            if settings["smoothing_mode"] == "auto":
                if progress_callback:
                    progress_callback(8, "Choosing the smoothing strength...")

                def sweep_progress(value, message):
                    if progress_callback:
                        progress_callback(8 + int(0.25 * value), message)

                # Guards are also checked where a phase that appeared later
                # is still thin, and in the middle and at the end
                from model.locked import smoothing_check_timepoints
                checked = smoothing_check_timepoints(
                    self.dataset.num_timepoints, reference,
                    [t for t, _m in definitions.values()],
                )
                strength, sweep = segmenter.auto_smoothing(
                    neutron_reference, xray_reference,
                    progress_callback=sweep_progress, cancel_check=cancel_check,
                    extra_volumes=[
                        tuple(self.dataset.get_volume_at_time(t)) + (t,)
                        for t in checked[1:]
                    ],
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
            outcome.smoothing_report = (
                segmenter.last_smoothing_report if sweep is not None else None
            )
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
            for index, name in enumerate(entry.class_names):
                mask = entry.mask_for(name)
                if not mask.any():
                    continue
                color = self._OVERLAY_COLORS[index % len(self._OVERLAY_COLORS)]
                self.segmentation_masks[timepoint].append((mask, color, name))

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

        self._apply_locked_result(outcome)
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

        output = ask_figure_output(
            self, "Save Spatial Metrics", "spatial_metrics",
            csv_label="Also save the metric values as CSV",
        )
        if output is None:
            return
        filepath = str(output.data_path()) if output.save_csv else None
        plot_path = str(output.figure_path)

        def operation(progress_callback=None, cancel_check=None):
            if progress_callback:
                progress_callback(10, "Measuring class shapes...")
            rows = spatial_metrics_rows(masks_by_timepoint)
            info, scalars, per_class = combined_registry()
            if progress_callback:
                progress_callback(85, "Writing CSV and plot...")
            if filepath:
                write_metrics_csv(
                    rows, filepath, metric_info=info,
                    scalar_metrics=scalars, per_class_metrics=per_class,
                )
            saved = plot_metric_evolution(
                rows, plot_path, dpi=output.dpi, metric_info=info,
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
        message = (f"Spatial metric values written to:\n{filepath}\n\n"
                   if filepath else "")
        message += f"{len(rows)} timepoint(s) measured."
        if saved_plot:
            message += f"\n\nEvolution plot:\n{saved_plot}"
        else:
            message += "\n\n(No evolution plot — it needs at least 2 timepoints.)"
        self.status_bar.showMessage(
            f"Spatial metrics saved: {saved_plot or filepath or 'nothing written'}"
        )
        QMessageBox.information(self, "Spatial Metrics Saved", message)

    @staticmethod
    def _joint_evolution_renderer(reference_mode):
        """Save the joint-histogram panels, and optionally their data: a
        per-timepoint summary CSV and a per-bin CSV (``_bins``)."""
        def render(histograms, output):
            from utils.histogram_evolution import (
                save_histogram_evolution_image, write_evolution_csv,
            )
            written = [save_histogram_evolution_image(
                histograms, str(output.figure_path), dpi=output.dpi,
                reference_mode=reference_mode,
            )]
            if output.save_csv:
                written += write_evolution_csv(
                    histograms, output.data_path(), output.data_path("_bins"),
                    reference_mode=reference_mode,
                )
            return written
        return render

    @staticmethod
    def _marginal_evolution_renderer(reference_mode):
        """Save the marginal kymographs, and optionally their data."""
        def render(histograms, output):
            from utils.histogram_evolution import (
                save_marginal_evolution_image, write_marginal_csv,
            )
            written = [save_marginal_evolution_image(
                histograms, str(output.figure_path), dpi=output.dpi,
                reference_mode=reference_mode,
            )]
            if output.save_csv:
                written.append(write_marginal_csv(
                    histograms, output.data_path(), reference_mode=reference_mode,
                ))
            return written
        return render

    def _on_export_histogram_evolution(self):
        """Save each timepoint's log-histogram change against T0."""
        from utils.histogram_evolution import REFERENCE_FIRST

        self._run_histogram_time_analysis(
            "Histogram Evolution",
            "Save Histogram Evolution",
            self._joint_evolution_renderer(REFERENCE_FIRST),
            "Red areas gained voxels relative to T0, blue areas lost them "
            "(log scale). This is the cumulative drift from the start.",
        )

    def _on_export_histogram_increment(self):
        """Save each timepoint's log-histogram change against the previous one."""
        from utils.histogram_evolution import REFERENCE_PREVIOUS

        self._run_histogram_time_analysis(
            "Incremental Histogram Change",
            "Save Incremental Histogram Change",
            self._joint_evolution_renderer(REFERENCE_PREVIOUS),
            "Each panel compares a timepoint with the one before it, so the "
            "steps where change actually happens stand out instead of being "
            "buried in cumulative drift.",
        )

    def _on_export_marginal_evolution(self):
        """Save marginal kymographs of each modality against T0."""
        from utils.histogram_evolution import REFERENCE_FIRST

        self._run_histogram_time_analysis(
            "Marginal Evolution",
            "Save Marginal Evolution",
            self._marginal_evolution_renderer(REFERENCE_FIRST),
            "Each panel stacks one modality's 1-D histogram against time "
            "(log2 vs T0). Red intensity bands grew, blue bands shrank — "
            "this separates a shift in neutron from a shift in X-ray.",
        )

    def _on_export_marginal_increment(self):
        """Save marginal kymographs comparing each timepoint with the previous."""
        from utils.histogram_evolution import REFERENCE_PREVIOUS

        self._run_histogram_time_analysis(
            "Incremental Marginal Change",
            "Save Incremental Marginal Change",
            self._marginal_evolution_renderer(REFERENCE_PREVIOUS),
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
        
        plotted = []  # (name, timepoints, values) currently drawn

        def do_plot():
            selected_names = [name for cb, name in checkboxes if cb.isChecked()]
            
            if not selected_names:
                return
            
            ax.clear()
            plotted.clear()
            
            for name in selected_names:
                timepoints, values = self.time_series_analyzer.get_time_series(
                    name, 'neutron_mean'
                )
                
                if len(timepoints) > 0:
                    ax.plot(timepoints, values, 'o-', label=name, linewidth=2)
                    plotted.append((name, timepoints, values))
            
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Neutron Mean Intensity')
            ax.set_title('Time Series Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            canvas.draw()
        
        plot_btn.clicked.connect(do_plot)

        save_btn = QPushButton("💾 Save Figure...")
        save_btn.setToolTip(
            "Save the plot (SVG, PDF, PNG or TIFF) and, optionally, the\n"
            "plotted values as CSV."
        )

        def do_save():
            if not plotted:
                do_plot()
            if not plotted:
                return
            output = ask_figure_output(
                dialog, "Save Time Series Plot", "time_series",
                csv_label="Also save the plotted values as CSV",
            )
            if output is None:
                return
            from utils.figure_io import save_figure, write_csv
            written = [save_figure(fig, output.figure_path, output.dpi)]
            if output.save_csv:
                written.append(write_csv(
                    output.data_path(),
                    ("selection", "timepoint", "neutron_mean"),
                    ((name, int(t), float(v))
                     for name, times, values in plotted
                     for t, v in zip(times, values)),
                ))
            self.status_bar.showMessage("Saved: " + ", ".join(written))

        save_btn.clicked.connect(do_save)
        layout.addWidget(save_btn)
        
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

