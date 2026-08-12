"""
selection_manager.py - Selection Manager Widget

Manages saved spatial masks and histogram ROIs for quick recall
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QInputDialog, QMessageBox, QCheckBox
)
from PyQt5.QtCore import pyqtSignal, Qt
import numpy as np
import sys


class Selection:
    """
    Represents a saved selection (mask + histogram ROI)
    """
    def __init__(self, name, spatial_mask=None, histogram_roi=None,
                 cluster_id=None, color=None, spatial_mask_3d=None,
                 source_axis=None, source_slice_index=None):
        self.name = name
        self.spatial_mask = spatial_mask  # 2D boolean array (one slice)
        self.histogram_roi = histogram_roi  # Polygon vertices or None
        self.cluster_id = cluster_id  # For k-means clusters
        self.color = color  # For visualization
        self.visible = True  # Whether to show on histogram

        # Whole-volume mask when the selection came from a 3-D operation
        # (e.g. 3-D k-means). The slice viewer re-slices this so the
        # highlight follows the plane and slice being displayed.
        self.spatial_mask_3d = spatial_mask_3d

        # Plane the 2-D mask was created on. A 2-D mask is only meaningful
        # there — without this, an isotropic volume happily draws it on a
        # different plane, showing the wrong voxels.
        self.source_axis = source_axis
        self.source_slice_index = source_slice_index
    
    def __repr__(self):
        mask_info = f"{np.sum(self.spatial_mask)} px" if self.spatial_mask is not None else "No mask"
        roi_info = "Has ROI" if self.histogram_roi is not None else "No ROI"
        return f"{self.name} ({mask_info}, {roi_info})"


class SelectionManagerWidget(QWidget):
    """
    Widget for managing saved selections
    """
    
    # Signals
    selection_recalled = pyqtSignal(object)  # Emits Selection object
    selections_changed = pyqtSignal()  # Emitted when selections list changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.selections = []  # List of Selection objects
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Saved Selections")
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(title)
        
        # List widget
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("💾 Save Current")
        self.save_btn.setToolTip("Save current selection (mask + ROI)")
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        self.delete_btn = QPushButton("🗑 Delete")
        self.delete_btn.setToolTip("Delete selected saved selection")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        self.delete_btn.setEnabled(False)
        button_layout.addWidget(self.delete_btn)
        
        layout.addLayout(button_layout)
        
        # Info label
        self.info_label = QLabel("No selections saved")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.info_label)
        
        # Visualization options
        vis_layout = QHBoxLayout()
        self.show_all_cb = QCheckBox("Show All on Histogram")
        self.show_all_cb.setToolTip("Show all visible selections on histogram")
        self.show_all_cb.stateChanged.connect(self._on_show_all_changed)
        vis_layout.addWidget(self.show_all_cb)
        vis_layout.addStretch()
        layout.addLayout(vis_layout)
        
        self.setLayout(layout)
    
    def add_selection(self, name, spatial_mask, histogram_roi, cluster_id=None,
                      color=None, spatial_mask_3d=None, source_axis=None,
                      source_slice_index=None):
        """
        Add a new selection

        Args:
            name: Name of selection
            spatial_mask: 2D boolean array (the slice it was created on)
            histogram_roi: Polygon vertices (Nx2 array) or None
            cluster_id: Optional cluster ID
            color: Optional color for visualization
            spatial_mask_3d: Optional whole-volume mask, so the highlight can
                follow slice and plane changes
            source_axis / source_slice_index: the plane the 2-D mask belongs
                to, so it is not drawn on a different one
        """
        # Create selection object
        selection = Selection(
            name, spatial_mask, histogram_roi, cluster_id, color,
            spatial_mask_3d=spatial_mask_3d,
            source_axis=source_axis,
            source_slice_index=source_slice_index,
        )
        self.selections.append(selection)
        
        # Add to list widget
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)  # Visible by default
        item.setData(Qt.UserRole, len(self.selections) - 1)  # Store index
        self.list_widget.addItem(item)
        
        # Update info
        self._update_info()
        
        # Enable buttons
        self.delete_btn.setEnabled(True)
        
        # Emit signal
        self.selections_changed.emit()
    
    def get_current_selection(self):
        """Get currently selected selection"""
        current_item = self.list_widget.currentItem()
        if current_item is None:
            return None
        
        index = current_item.data(Qt.UserRole)
        if 0 <= index < len(self.selections):
            return self.selections[index]
        return None
    
    def get_visible_selections(self):
        """Get all selections marked as visible"""
        visible = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                index = item.data(Qt.UserRole)
                if 0 <= index < len(self.selections):
                    visible.append(self.selections[index])
        return visible
    
    def clear_all(self):
        """Clear all selections"""
        self.selections = []
        self.list_widget.clear()
        self._update_info()
        self.delete_btn.setEnabled(False)
        self.selections_changed.emit()
    
    def enable_save_button(self, enabled):
        """Enable/disable save button"""
        self.save_btn.setEnabled(enabled)
    
    def _on_save_clicked(self):
        """Handle save button click"""
        # Get name from user
        name, ok = QInputDialog.getText(
            self, "Save Selection", "Enter name for this selection:"
        )
        
        if ok and name:
            # Emit signal to main window to save current state
            # Main window will call add_selection with the data
            print(f"SelectionManager: Save requested with name '{name}'", file=sys.stderr)
            # Signal handled by main window
    
    def _on_delete_clicked(self):
        """Handle delete button click"""
        current_item = self.list_widget.currentItem()
        if current_item is None:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Selection",
            f"Delete selection '{current_item.text()}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            index = current_item.data(Qt.UserRole)
            
            # Remove from list
            self.list_widget.takeItem(self.list_widget.currentRow())
            
            # Remove from selections
            if 0 <= index < len(self.selections):
                del self.selections[index]
            
            # Update indices for remaining items
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                old_index = item.data(Qt.UserRole)
                if old_index > index:
                    item.setData(Qt.UserRole, old_index - 1)
            
            self._update_info()
            
            if len(self.selections) == 0:
                self.delete_btn.setEnabled(False)
            
            self.selections_changed.emit()
    
    def _on_item_double_clicked(self, item):
        """Handle double-click on selection"""
        index = item.data(Qt.UserRole)
        if 0 <= index < len(self.selections):
            selection = self.selections[index]
            print(f"SelectionManager: Recalling selection '{selection.name}'", file=sys.stderr)
            self.selection_recalled.emit(selection)
    
    def _on_item_changed(self, item):
        """Handle checkbox state change"""
        index = item.data(Qt.UserRole)
        if 0 <= index < len(self.selections):
            self.selections[index].visible = (item.checkState() == Qt.Checked)
            self.selections_changed.emit()
    
    def _on_show_all_changed(self, state):
        """Handle show all checkbox"""
        self.selections_changed.emit()
    
    def _update_info(self):
        """Update info label"""
        n = len(self.selections)
        if n == 0:
            self.info_label.setText("No selections saved")
        elif n == 1:
            self.info_label.setText("1 selection saved")
        else:
            self.info_label.setText(f"{n} selections saved")
