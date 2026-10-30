"""The axial profile: both modalities averaged slice by slice along the
view axis, inside the virtual crop.

A layered sample — electrode | separator | foil | plunger — is one-
dimensional along its stack axis, and the mean of every slice through the
pellet traces the layers far more legibly than a histogram cloud does: a
material boundary is a step in one or both curves. The panel sits under
the slice viewer, follows its view axis and its crop, and gives two things
back:

* **a range** dragged on the profile becomes the crop's extent along the
  stack axis, so the histogram shows exactly the slices between two steps;
* **a click** jumps the slice viewer to that slice.

The computation is :func:`utils.crop_box.axial_profile`; this module only
draws it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

AXIS_TITLE = {'z': 'Z (slice index)', 'y': 'Y (row index)', 'x': 'X (column index)'}
NEUTRON_COLOUR = '#1f5fbf'
XRAY_COLOUR = '#c43c2a'


class AxialProfileWidget(QWidget):
    """Mean neutron and X-ray value per slice along the view axis."""

    # The user dragged a range on the profile: (axis, lo, hi) — half-open,
    # full-resolution indices along that axis
    range_selected = pyqtSignal(str, int, int)
    # The user asked for the range to be released (full extent again)
    range_cleared = pyqtSignal(str)
    # The user clicked a slice: (axis, index)
    index_clicked = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile: Optional[dict] = None
        self.axis = 'z'
        self.current_index: Optional[int] = None
        self.selected_range: Optional[Tuple[int, int]] = None
        self._span = None
        self._press = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        head = QHBoxLayout()
        self.title_label = QLabel("<b>Axial profile</b> — mean per slice along the view axis")
        head.addWidget(self.title_label, stretch=1)
        self.std_cb = QCheckBox("± σ band")
        self.std_cb.setToolTip("Shade one standard deviation of the voxels in each slice")
        self.std_cb.setChecked(False)
        self.std_cb.toggled.connect(lambda _: self.redraw())
        head.addWidget(self.std_cb)
        self.clear_range_btn = QPushButton("Full extent")
        self.clear_range_btn.setToolTip(
            "Release the range selected on the profile: the crop spans the\n"
            "whole stack axis again"
        )
        self.clear_range_btn.setEnabled(False)
        self.clear_range_btn.clicked.connect(self._on_clear_range)
        head.addWidget(self.clear_range_btn)
        layout.addLayout(head)

        self.fig = Figure(figsize=(6, 1.9))
        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(120)
        self.canvas.setMaximumHeight(230)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        self.info_label = QLabel("Load a dataset to see the profile")
        self.info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        layout.addWidget(self.info_label)
        self._setup_span()
        self.redraw()

    # ── data in ───────────────────────────────────────────────────────────
    def set_profile(self, profile: dict, current_index: Optional[int] = None,
                    selected_range: Optional[Tuple[int, int]] = None,
                    cropped: bool = False):
        """Show a profile from :func:`utils.crop_box.axial_profile`."""
        self.profile = profile
        self.axis = str(profile.get('axis', self.axis))
        self.current_index = current_index
        self.selected_range = selected_range
        self.clear_range_btn.setEnabled(selected_range is not None)
        where = "inside the crop" if cropped else "whole cross-section"
        self.title_label.setText(
            f"<b>Axial profile along {self.axis.upper()}</b> — mean per slice, {where}"
        )
        self.redraw()

    def set_current_index(self, index: Optional[int]):
        self.current_index = index
        self.redraw()

    # ── drawing ───────────────────────────────────────────────────────────
    def _setup_span(self):
        from matplotlib.widgets import SpanSelector
        try:
            self._span = SpanSelector(
                self.ax, self._on_span, 'horizontal', useblit=False,
                props=dict(alpha=0.18, facecolor='#f2b134'),
                interactive=True, drag_from_anywhere=True, minspan=0.5,
                button=[1],
            )
        except TypeError:                     # older matplotlib signatures
            self._span = SpanSelector(self.ax, self._on_span, 'horizontal',
                                      useblit=False)

    def redraw(self):
        ax, ax2 = self.ax, self.ax2
        ax.clear()
        ax2.clear()
        ax.set_xlabel(AXIS_TITLE.get(self.axis, self.axis))
        ax.set_ylabel('neutron', color=NEUTRON_COLOUR, fontsize=9)
        ax2.set_ylabel('X-ray', color=XRAY_COLOUR, fontsize=9)
        ax.tick_params(axis='y', colors=NEUTRON_COLOUR, labelsize=8)
        ax2.tick_params(axis='y', colors=XRAY_COLOUR, labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, alpha=0.3)

        prof = self.profile
        if prof is None or prof['index'].size == 0:
            ax.text(0.5, 0.5, 'no profile', transform=ax.transAxes,
                    ha='center', va='center', color='#888')
            self.canvas.draw_idle()
            return

        idx = prof['index']
        n, x = prof['neutron'], prof['xray']
        ax.plot(idx, n, '-', color=NEUTRON_COLOUR, lw=1.3)
        ax2.plot(idx, x, '-', color=XRAY_COLOUR, lw=1.3)
        if self.std_cb.isChecked():
            ax.fill_between(idx, n - prof['neutron_std'], n + prof['neutron_std'],
                            color=NEUTRON_COLOUR, alpha=0.12, lw=0)
            ax2.fill_between(idx, x - prof['xray_std'], x + prof['xray_std'],
                             color=XRAY_COLOUR, alpha=0.12, lw=0)
        # keep the x range tight to the slices that exist
        ax.set_xlim(idx[0] - 0.5, idx[-1] + 0.5)

        if self.selected_range is not None:
            lo, hi = self.selected_range
            ax.axvspan(lo - 0.5, hi - 0.5, color='#f2b134', alpha=0.18, lw=0)
            ax.axvline(lo - 0.5, color='#f2b134', lw=1)
            ax.axvline(hi - 0.5, color='#f2b134', lw=1)
        if self.current_index is not None:
            ax.axvline(self.current_index, color='k', lw=0.9, ls='--')

        finite = np.isfinite(n)
        if finite.any():
            info = (f"{idx[0]}–{idx[-1]}: neutron {np.nanmin(n):.4g}…{np.nanmax(n):.4g}, "
                    f"X-ray {np.nanmin(x):.4g}…{np.nanmax(x):.4g}")
            if self.selected_range is not None:
                lo, hi = self.selected_range
                info += f" | selected {lo}–{hi - 1} ({hi - lo} slices)"
            info += " | drag to select a range, click to go to a slice"
        else:
            info = "no finite voxel in the profile"
        self.info_label.setText(info)
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    # ── interaction ───────────────────────────────────────────────────────
    def _on_span(self, xmin, xmax):
        if self.profile is None:
            return
        lo = int(np.floor(min(xmin, xmax) + 0.5))
        hi = int(np.floor(max(xmin, xmax) + 0.5)) + 1
        idx = self.profile['index']
        lo = max(int(idx[0]), lo)
        hi = min(int(idx[-1]) + 1, hi)
        if hi <= lo:
            return
        self.selected_range = (lo, hi)
        self.clear_range_btn.setEnabled(True)
        self.range_selected.emit(self.axis, lo, hi)

    def _on_clear_range(self):
        self.selected_range = None
        self.clear_range_btn.setEnabled(False)
        if self._span is not None:
            try:
                self._span.clear()
            except Exception:
                pass
        self.range_cleared.emit(self.axis)
        self.redraw()

    def _on_press(self, event):
        if event.inaxes not in (self.ax, self.ax2) or event.button != 1:
            self._press = None
            return
        self._press = (event.x, event.y, event.xdata)

    def _on_release(self, event):
        """A press-and-release without movement is a click: go to that slice."""
        if self._press is None or event.button != 1:
            return
        x0, y0, xdata = self._press
        self._press = None
        if xdata is None or self.profile is None:
            return
        moved = abs(event.x - x0) > 3 or abs(event.y - y0) > 3
        if moved:
            return
        idx = self.profile['index']
        i = int(np.clip(round(xdata), idx[0], idx[-1]))
        self.index_clicked.emit(self.axis, i)
