"""Dialog: place materials from their attenuation coefficients.

Two materials already drawn in the scan (with known coefficients) calibrate
each modality's grey scale; every other material listed is then placed on
the histogram from its coefficients alone — a phase with no region to draw
(not yet formed, too thin, hidden) can still be tracked.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

Pair = Tuple[float, float]


def _number(item) -> Optional[float]:
    if item is None:
        return None
    text = item.text().strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


class PhysicsMaterialsDialog(QDialog):
    """Collect reference and target coefficients.

    *materials* are the names drawn in the scan (candidate references);
    *previous* restores an earlier entry ``{"references": {...},
    "targets": {...}}``.
    """

    def __init__(self, materials: Sequence[str], previous: Optional[dict] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Materials from Attenuation Coefficients")
        previous = previous or {}
        references = previous.get("references", {})
        targets = previous.get("targets", {})

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Each modality's grey value is taken as a linear function of the "
            "attenuation coefficient. Two (or more) materials you have drawn, "
            "whose coefficients you know, fix that line; the materials below "
            "are then placed from their coefficients alone.\n\n"
            "Use one unit for every entry (e.g. cm⁻¹). For X-rays, enter the "
            "effective coefficient at your scan's spectrum — the prediction is "
            "only as good as that number, and the health check will tell you "
            "if a predicted material is never found."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ── references ───────────────────────────────────────────────────
        ref_box = QGroupBox("Reference materials (drawn in the scan)")
        ref_layout = QVBoxLayout(ref_box)
        self.reference_table = QTableWidget(len(materials), 4)
        self.reference_table.setHorizontalHeaderLabels(
            ["Use", "Material", "Neutron μ", "X-ray μ"])
        for row, name in enumerate(materials):
            check = QCheckBox()
            check.setChecked(name in references)
            self.reference_table.setCellWidget(row, 0, check)
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.reference_table.setItem(row, 1, item)
            values = references.get(name, ("", ""))
            for column, value in enumerate(values, start=2):
                self.reference_table.setItem(row, column,
                                             QTableWidgetItem(str(value)))
        self.reference_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        ref_layout.addWidget(self.reference_table)
        layout.addWidget(ref_box)

        # ── targets ──────────────────────────────────────────────────────
        target_box = QGroupBox("Materials to place")
        target_layout = QVBoxLayout(target_box)
        self.target_table = QTableWidget(0, 3)
        self.target_table.setHorizontalHeaderLabels(
            ["Material", "Neutron μ", "X-ray μ"])
        self.target_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        for name, values in targets.items():
            self.add_target(name, *values)
        target_layout.addWidget(self.target_table)
        buttons = QHBoxLayout()
        add = QPushButton("Add")
        add.clicked.connect(lambda: self.add_target())
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_target)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        target_layout.addLayout(buttons)
        layout.addWidget(target_box)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self.resize(560, 520)

    def add_target(self, name: str = "", neutron="", xray="") -> None:
        row = self.target_table.rowCount()
        self.target_table.insertRow(row)
        for column, value in enumerate((name, neutron, xray)):
            self.target_table.setItem(row, column, QTableWidgetItem(str(value)))

    def set_reference(self, name: str, neutron: float, xray: float) -> None:
        for row in range(self.reference_table.rowCount()):
            if self.reference_table.item(row, 1).text() == name:
                self.reference_table.cellWidget(row, 0).setChecked(True)
                self.reference_table.setItem(row, 2, QTableWidgetItem(str(neutron)))
                self.reference_table.setItem(row, 3, QTableWidgetItem(str(xray)))
                return
        raise KeyError(name)

    def _remove_target(self) -> None:
        row = self.target_table.currentRow()
        if row >= 0:
            self.target_table.removeRow(row)

    def values(self) -> Tuple[Dict[str, Pair], Dict[str, Pair], list]:
        """``(references, targets, problems)``."""
        problems = []
        references: Dict[str, Pair] = {}
        for row in range(self.reference_table.rowCount()):
            if not self.reference_table.cellWidget(row, 0).isChecked():
                continue
            name = self.reference_table.item(row, 1).text()
            n = _number(self.reference_table.item(row, 2))
            x = _number(self.reference_table.item(row, 3))
            if n is None or x is None:
                problems.append(f"Reference '{name}' needs both coefficients.")
            else:
                references[name] = (n, x)
        targets: Dict[str, Pair] = {}
        for row in range(self.target_table.rowCount()):
            name_item = self.target_table.item(row, 0)
            name = name_item.text().strip() if name_item else ""
            n = _number(self.target_table.item(row, 1))
            x = _number(self.target_table.item(row, 2))
            if not name and n is None and x is None:
                continue
            if not name or n is None or x is None:
                problems.append(f"Row {row + 1} of the materials to place "
                                "needs a name and both coefficients.")
            elif name in references:
                problems.append(f"'{name}' is already drawn; it cannot also "
                                "be placed from coefficients.")
            else:
                targets[name] = (n, x)
        if len(references) < 2:
            problems.append("Choose at least two reference materials.")
        if not targets:
            problems.append("Add at least one material to place.")
        return references, targets, problems

    def _accept(self) -> None:
        _references, _targets, problems = self.values()
        if problems:
            QMessageBox.warning(self, "Materials from Coefficients",
                                "\n".join(problems))
            return
        self.accept()
