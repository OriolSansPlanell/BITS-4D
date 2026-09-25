"""Dialogs of the main window (anchors, export options, figure export).

Split out of ``main_window.py``; they are re-exported there.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QMessageBox,
    QCheckBox, QDialog, QDialogButtonBox, QComboBox,
)
import numpy as np

from gui.figure_save_dialog import FigureSaveDialog


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
            "Bimodal histogram of the class  (.npy + figure)"
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

        # Format of the histogram figures, and their counts as CSV
        from utils.figure_io import FIGURE_FORMATS
        hist_row = QHBoxLayout()
        hist_row.setContentsMargins(24, 0, 0, 0)
        hist_row.addWidget(QLabel("Figure format:"))
        self._histogram_format = QComboBox()
        for ext, label in FIGURE_FORMATS:
            self._histogram_format.addItem(label.split(" — ")[0], ext)
        index = self._histogram_format.findData(FigureSaveDialog.last_format)
        self._histogram_format.setCurrentIndex(max(index, 0))
        hist_row.addWidget(self._histogram_format)
        self._histogram_csv_cb = QCheckBox("+ counts as CSV")
        self._histogram_csv_cb.setToolTip(
            "Also write each class histogram as a CSV: one row per non-empty\n"
            "bin, with its neutron and X-ray centre and its count."
        )
        hist_row.addWidget(self._histogram_csv_cb)
        hist_row.addStretch()
        mod_vbox.addLayout(hist_row)

        def _histogram_toggled(_state=None):
            enabled = self._histogram_cb.isChecked()
            self._histogram_format.setEnabled(enabled)
            self._histogram_csv_cb.setEnabled(enabled)
        self._histogram_cb.stateChanged.connect(_histogram_toggled)
        _histogram_toggled()

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
                   self._labels_cb, self._histogram_cb, self._report_cb,
                   self._histogram_csv_cb):
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
        # A histogram export writes a counts file and an image per layer,
        # plus a CSV when asked for
        if self._histogram_cb.isChecked():
            n_mods += 3 if self._histogram_csv_cb.isChecked() else 2
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
        self.histogram_format = self._histogram_format.currentData()
        self.histogram_csv = self._histogram_csv_cb.isChecked()
        self.export_report = self._report_cb.isChecked()
        self.accept()


class FigureExportDialog(FigureSaveDialog):
    """The format pop-up, plus the options of the histogram + slice figure.

    After exec_() returns Accepted, read ``fmt``, ``dpi``, ``save_csv``,
    ``show_legend``, ``outline_highlights`` and ``include_active_roi``.
    """

    def __init__(self, has_active_roi=False, parent=None):
        self._legend_cb = QCheckBox("Show legends")
        self._legend_cb.setChecked(True)

        self._outline_cb = QCheckBox("Outline the slice highlights")
        self._outline_cb.setChecked(True)
        self._outline_cb.setToolTip(
            "Draw a thin contour around each highlighted label, so it stays\n"
            "readable in print and in greyscale."
        )

        self._active_cb = QCheckBox("Include ROIs drawn but not saved")
        self._active_cb.setChecked(has_active_roi)
        self._active_cb.setEnabled(has_active_roi)

        super().__init__(
            "Export Histogram + Slice Figure",
            csv_label="Also save the label counts as CSV",
            note=("Left: the local histogram of this timepoint with every "
                  "label's selection on top. Right: the slice on screen with "
                  "the same labels highlighted."),
            parent=parent,
            extra_widgets=(QLabel("<b>Figure options</b>"), self._legend_cb,
                           self._outline_cb, self._active_cb),
        )

    @property
    def show_legend(self):
        return self._legend_cb.isChecked()

    @property
    def outline_highlights(self):
        return self._outline_cb.isChecked()

    @property
    def include_active_roi(self):
        return self._active_cb.isEnabled() and self._active_cb.isChecked()
