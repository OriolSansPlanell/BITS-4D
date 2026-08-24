"""The materials panel: what was defined, what may change, and the run.

Everything the time-series segmentation needs lives here rather than behind a
modal dialog, because the decisions are about the *materials* and are easier
to make while looking at them. Materials arrive from two places and are
treated identically once here:

* regions drawn on the histogram, and
* clusters found by K-means and copied across.

The one judgement the software cannot make is which materials are allowed to
change. A casing, a support, a structural metal cannot; the phases under
study can and should. Marking the first kind gives every other result an
independent check — if something that cannot change appears to change, the
segmentation is wrong rather than the sample. Nothing else in the panel is
load-bearing in the same way, which is why it is the first column the eye
lands on.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

#: What the behaviour column offers, and what each choice means.
CHANGES = "Changes"
UNCHANGED = "Stays unchanged"
BEHAVIOURS = (CHANGES, UNCHANGED)

#: Smoothing presets. The user picks a word; the number stays internal.
SMOOTHING_CHOICES = ("Auto", "Off", "Low", "Medium", "High")
SMOOTHING_VALUES = {"Off": 0.0, "Low": 0.5, "Medium": 1.0, "High": 3.0}

_STATUS_STYLE = {
    "pass": "color: #1a7f37;",
    "warn": "color: #9a6700;",
    "fail": "color: #b42318;",
    "idle": "color: gray; font-style: italic;",
}


def describe_strength(value) -> str:
    """A smoothing number, said in words.

    The number is a property of the method, not of the sample, and showing it
    invites tuning by eye — which is how a minority material gets smoothed
    away without anyone noticing.
    """
    if value is None or value <= 0:
        return "Off"
    if value < 0.75:
        return "Low"
    if value < 2.0:
        return "Medium"
    return "High"


class MaterialPanel(QWidget):
    """Define materials, say which may change, and run the series.

    Emits
    -----
    refresh_requested
        Re-read the materials from the current segmentation.
    copy_clusters_requested
        Copy the latest K-means clusters in as materials.
    preview_requested
        Run the current timepoint only.
    run_requested
        Run every timepoint.
    """

    refresh_requested = pyqtSignal()
    copy_clusters_requested = pyqtSignal()
    preview_requested = pyqtSignal()
    run_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[dict] = []
        self._build()

    # ── construction ─────────────────────────────────────────────────────
    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        intro = QLabel(
            "<b>Materials</b><br>"
            "Each material you define is measured at every timepoint. The "
            "definitions stay fixed and voxels move between them, so a change "
            "in a volume is a change in the sample."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555; font-size: 9pt;")
        layout.addWidget(intro)

        # ── the materials themselves ─────────────────────────────────────
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Material", "From", "Voxels", "During the experiment"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table)

        self.control_hint = QLabel(
            "Set anything that cannot change during the experiment — a "
            "casing, a support, a structural metal — to <i>Stays "
            "unchanged</i>. We use those as a control: if one of them moves, "
            "something is wrong with the segmentation rather than with the "
            "sample.<br>"
            "<b>Do not</b> mark a material that reacts. Its real change would "
            "be read as an instrument effect and taken off every other "
            "material."
        )
        self.control_hint.setWordWrap(True)
        self.control_hint.setStyleSheet("color: #555; font-size: 9pt;")
        layout.addWidget(self.control_hint)

        source_row = QHBoxLayout()
        self.refresh_btn = QPushButton("↻ Refresh from segmentation")
        self.refresh_btn.setToolTip(
            "Re-read the materials from the regions currently segmented."
        )
        self.refresh_btn.clicked.connect(self.refresh_requested.emit)
        source_row.addWidget(self.refresh_btn)

        self.copy_clusters_btn = QPushButton("⇱ Copy K-means clusters")
        self.copy_clusters_btn.setEnabled(False)
        self.copy_clusters_btn.setToolTip(
            "Bring the clusters found by 3-D K-means in as materials.\n"
            "They behave exactly like drawn regions afterwards, including\n"
            "the control setting, so a cluster you recognise as the casing\n"
            "can be marked as unchanging like any other."
        )
        self.copy_clusters_btn.clicked.connect(self.copy_clusters_requested.emit)
        source_row.addWidget(self.copy_clusters_btn)
        layout.addLayout(source_row)

        # ── how it is run ────────────────────────────────────────────────
        options = QGroupBox("Settings")
        options_layout = QVBoxLayout(options)

        smoothing_row = QHBoxLayout()
        smoothing_row.addWidget(QLabel("Spatial smoothing:"))
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(SMOOTHING_CHOICES)
        self.smoothing_combo.setToolTip(
            "Uses neighbouring voxels to clean up noisy assignments.\n\n"
            "Auto is recommended: it tries a range and keeps the strongest\n"
            "setting that costs no material any of its volume. Too much\n"
            "smoothing erases a small material entirely, and nothing later\n"
            "would show that it had."
        )
        smoothing_row.addWidget(self.smoothing_combo, 1)
        options_layout.addLayout(smoothing_row)

        self.smoothing_result = QLabel("")
        self.smoothing_result.setStyleSheet("color: #555; font-size: 9pt;")
        self.smoothing_result.setWordWrap(True)
        options_layout.addWidget(self.smoothing_result)

        from PyQt5.QtWidgets import QCheckBox

        self.mixed_check = QCheckBox("Look for mixed boundaries")
        self.mixed_check.setChecked(True)
        self.mixed_check.setToolTip(
            "Where two materials touch, some voxels contain a bit of both.\n"
            "This flags any material that behaves like such a boundary\n"
            "rather than like a phase, so you can decide what to do."
        )
        options_layout.addWidget(self.mixed_check)

        self.lock_check = QCheckBox("Lock material definitions")
        self.lock_check.setChecked(True)
        self.lock_check.setToolTip(
            "Material properties do not change during an experiment, so the\n"
            "definitions are kept fixed and voxels move between them.\n"
            "Recommended.\n\n"
            "Unticking this lets the definitions themselves follow the data,\n"
            "which is only appropriate when the instrument is known to move —\n"
            "and it can absorb a real change in the sample."
        )
        self.lock_check.toggled.connect(self._on_lock_toggled)
        options_layout.addWidget(self.lock_check)

        self.lock_warning = QLabel("")
        self.lock_warning.setWordWrap(True)
        self.lock_warning.setStyleSheet("color: #9a6700; font-size: 9pt;")
        options_layout.addWidget(self.lock_warning)
        layout.addWidget(options)

        # ── run ──────────────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.preview_btn = QPushButton("▶ Preview this timepoint")
        self.preview_btn.setEnabled(False)
        self.preview_btn.setToolTip(
            "Segment the current timepoint only, so you can look at the "
            "result before committing to the whole series."
        )
        self.preview_btn.clicked.connect(self.preview_requested.emit)
        run_row.addWidget(self.preview_btn)

        self.run_btn = QPushButton("▶▶ Run all timepoints")
        self.run_btn.setEnabled(False)
        self.run_btn.setToolTip(
            "Measure every timepoint against these materials."
        )
        self.run_btn.clicked.connect(self.run_requested.emit)
        run_row.addWidget(self.run_btn)
        layout.addLayout(run_row)

        # ── what happened ────────────────────────────────────────────────
        results = QGroupBox("Result")
        results_layout = QVBoxLayout(results)
        self.status_label = QLabel("Nothing run yet.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(_STATUS_STYLE["idle"])
        results_layout.addWidget(self.status_label)

        self.findings_label = QLabel("")
        self.findings_label.setWordWrap(True)
        self.findings_label.setTextFormat(Qt.RichText)
        self.findings_label.setStyleSheet("font-size: 9pt;")
        results_layout.addWidget(self.findings_label)
        layout.addWidget(results)

        layout.addStretch()

    def _on_lock_toggled(self, checked):
        self.lock_warning.setText(
            "" if checked else
            "Definitions will be allowed to move. Check the control materials "
            "carefully in the result — this setting can absorb a real change "
            "in the sample and report it as no change at all."
        )

    # ── materials ────────────────────────────────────────────────────────
    def set_materials(self, materials: Sequence[dict]) -> None:
        """Populate the table.

        *materials* is a sequence of ``{'name', 'source', 'voxels'}``. A
        behaviour already chosen for a name is preserved across a refresh, so
        re-reading the segmentation never silently discards the one decision
        the user had to make themselves.
        """
        previous = {row["name"]: row["behaviour"] for row in self._rows}
        self._rows = []
        self.table.setRowCount(0)

        for material in materials:
            name = str(material["name"])
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(
                row, 1, QTableWidgetItem(str(material.get("source", "drawn")))
            )
            voxels = material.get("voxels")
            self.table.setItem(
                row, 2,
                QTableWidgetItem("" if voxels is None else f"{int(voxels):,}"),
            )
            combo = QComboBox()
            combo.addItems(BEHAVIOURS)
            combo.setToolTip(
                "'Stays unchanged' marks this as a control: we check it "
                "afterwards and warn if it moved."
            )
            combo.setCurrentText(previous.get(name, CHANGES))
            self.table.setCellWidget(row, 3, combo)
            self._rows.append({
                "name": name,
                "source": material.get("source", "drawn"),
                "voxels": voxels,
                "combo": combo,
                "behaviour": combo.currentText(),
            })

        has_materials = bool(self._rows)
        self.preview_btn.setEnabled(has_materials)
        self.run_btn.setEnabled(has_materials)
        if not has_materials:
            self.set_status(
                "No materials yet — draw a region on the histogram and "
                "segment it, or copy the K-means clusters.", "idle",
            )

    def material_names(self) -> List[str]:
        return [row["name"] for row in self._rows]

    def control_materials(self) -> List[str]:
        """Materials the user says cannot change."""
        return [
            row["name"] for row in self._rows
            if row["combo"].currentText() == UNCHANGED
        ]

    def set_behaviour(self, name: str, behaviour: str) -> None:
        if behaviour not in BEHAVIOURS:
            raise ValueError(f"behaviour must be one of {BEHAVIOURS}")
        for row in self._rows:
            if row["name"] == name:
                row["combo"].setCurrentText(behaviour)
                row["behaviour"] = behaviour
                return
        raise KeyError(f"No material named {name!r}")

    def set_clusters_available(self, available: bool) -> None:
        self.copy_clusters_btn.setEnabled(bool(available))

    # ── settings ─────────────────────────────────────────────────────────
    @property
    def smoothing_mode(self) -> str:
        return "auto" if self.smoothing_combo.currentText() == "Auto" else "manual"

    @property
    def smoothing_strength(self) -> Optional[float]:
        """None when Auto — the value is chosen by measurement, not here."""
        choice = self.smoothing_combo.currentText()
        return None if choice == "Auto" else SMOOTHING_VALUES[choice]

    @property
    def find_mixed_boundaries(self) -> bool:
        return self.mixed_check.isChecked()

    @property
    def lock_definitions(self) -> bool:
        return self.lock_check.isChecked()

    def settings(self) -> dict:
        return {
            "control_materials": self.control_materials(),
            "smoothing_mode": self.smoothing_mode,
            "smoothing_strength": self.smoothing_strength,
            "find_mixed_boundaries": self.find_mixed_boundaries,
            "lock_definitions": self.lock_definitions,
        }

    # ── results ──────────────────────────────────────────────────────────
    def set_status(self, message: str, level: str = "idle") -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            _STATUS_STYLE.get(level, _STATUS_STYLE["idle"])
        )

    def set_smoothing_result(self, value, chosen_automatically: bool = True) -> None:
        word = describe_strength(value)
        self.smoothing_result.setText(
            f"Smoothing used: <b>{word}</b>"
            + (" (chosen automatically)" if chosen_automatically else "")
        )

    def set_findings(self, findings: Sequence[str]) -> None:
        if not findings:
            self.findings_label.setText("")
            return
        items = "".join(f"<li>{text}</li>" for text in findings)
        self.findings_label.setText(f"<ul>{items}</ul>")

    def show_health_report(self, report) -> None:
        """Render a health report from :mod:`model.health_check`."""
        level = getattr(report.status, "value", "idle")
        self.set_status(report.headline(), level)
        self.set_findings([
            finding.message + (f" {finding.detail}" if finding.detail else "")
            for finding in report.findings
            if getattr(finding.status, "value", "pass") != "pass"
        ])

    def clear_result(self) -> None:
        self.set_status("Nothing run yet.", "idle")
        self.set_findings([])
        self.smoothing_result.setText("")
