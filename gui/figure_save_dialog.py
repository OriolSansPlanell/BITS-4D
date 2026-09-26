"""
figure_save_dialog.py - The small pop-up every analysis figure goes through

Asks the figure format (SVG by default), the resolution, and whether to also
save the plotted numbers as CSV, then asks where. The last answers are
remembered for the session, so exporting several figures in a row asks the
same question with the same defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QLabel, QRadioButton, QSpinBox, QVBoxLayout,
)

from utils.figure_io import DEFAULT_FORMAT, FIGURE_FORMATS, figure_path


@dataclass
class FigureOutput:
    """Where and how to save one figure (and, optionally, its data)."""

    base: Path                   # path without extension
    fmt: str                     # "svg", "pdf", "png" or "tif"
    dpi: int
    save_csv: bool

    @property
    def figure_path(self) -> Path:
        return figure_path(self.base, self.fmt)

    def data_path(self, suffix: str = "") -> Path:
        """CSV path next to the figure; *suffix* tells several CSVs apart."""
        return self.base.with_name(self.base.name + suffix + ".csv")


class FigureSaveDialog(QDialog):
    """Format, resolution and CSV choice for one figure."""

    #: Remembered between exports (for this session)
    last_format = DEFAULT_FORMAT
    last_dpi = 300
    last_save_csv = True

    def __init__(self, title: str, csv_label: Optional[str] = None,
                 note: str = "", parent=None, extra_widgets=()):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        if note:
            info = QLabel(note)
            info.setWordWrap(True)
            info.setStyleSheet("color: #555;")
            # A wrapped label in a dialog is otherwise sized for the wrong
            # width and its first lines are cut off
            info.setMinimumHeight(info.heightForWidth(400))
            layout.addWidget(info)

        layout.addWidget(QLabel("<b>Figure format</b>"))
        self._format_group = QButtonGroup(self)
        self._format_buttons = {}
        for ext, label in FIGURE_FORMATS:
            button = QRadioButton(label)
            self._format_group.addButton(button)
            self._format_buttons[ext] = button
            layout.addWidget(button)
        chosen = type(self).last_format
        if chosen not in self._format_buttons:
            chosen = DEFAULT_FORMAT
        self._format_buttons[chosen].setChecked(True)

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("Resolution (DPI):"))
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(72, 1200)
        self._dpi_spin.setSingleStep(50)
        self._dpi_spin.setValue(type(self).last_dpi)
        self._dpi_spin.setToolTip(
            "For PNG and TIFF, the image resolution. In SVG and PDF, the\n"
            "resolution of any image inside the figure (a histogram);\n"
            "lines and text stay vector graphics."
        )
        dpi_row.addWidget(self._dpi_spin)
        dpi_row.addStretch()
        layout.addLayout(dpi_row)

        self._csv_cb = QCheckBox(csv_label or "Also save the data as CSV")
        self._csv_cb.setChecked(type(self).last_save_csv)
        self._csv_cb.setToolTip(
            "Write the numbers the figure is drawn from, next to the figure,\n"
            "so it can be redrawn or checked in another program."
        )
        self._csv_cb.setVisible(csv_label is not None)
        layout.addWidget(self._csv_cb)

        # Options particular to one figure go between the common ones and
        # the buttons
        for widget in extra_widgets:
            layout.addWidget(widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def fmt(self) -> str:
        for ext, button in self._format_buttons.items():
            if button.isChecked():
                return ext
        return DEFAULT_FORMAT

    @property
    def dpi(self) -> int:
        return self._dpi_spin.value()

    @property
    def save_csv(self) -> bool:
        return not self._csv_cb.isHidden() and self._csv_cb.isChecked()

    def remember(self):
        cls = type(self)
        cls.last_format = self.fmt
        cls.last_dpi = self.dpi
        if not self._csv_cb.isHidden():
            cls.last_save_csv = self._csv_cb.isChecked()


def ask_figure_output(parent, title: str, default_name: str,
                      csv_label: Optional[str] = "Also save the data as CSV",
                      note: str = "",
                      dialog: Optional[FigureSaveDialog] = None,
                      ) -> Optional[FigureOutput]:
    """Pop-up for the format, then the save dialog. None if cancelled.

    Pass ``csv_label=None`` for a figure that has no data to save, or a
    ready-made *dialog* (a subclass with extra options) to use instead.
    """
    if dialog is None:
        dialog = FigureSaveDialog(title, csv_label=csv_label, note=note,
                                  parent=parent)
    if dialog.exec_() != QDialog.Accepted:
        return None
    dialog.remember()
    fmt = dialog.fmt
    label = dict(FIGURE_FORMATS)[fmt].split(" — ")[0]
    suggested = str(figure_path(default_name, fmt))
    path, _filter = QFileDialog.getSaveFileName(
        parent, title, suggested,
        f"{label} (*.{fmt});;All files (*)",
    )
    if not path:
        return None
    base = figure_path(path, fmt).with_suffix("")
    return FigureOutput(base=base, fmt=fmt, dpi=dialog.dpi,
                        save_csv=dialog.save_csv)
