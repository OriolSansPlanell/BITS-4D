"""The manual window: a contents tree, a search box and a reading pane.

Non-modal on purpose. A manual you have to close before you can try the thing
it describes is a manual nobody reads twice, so this stays open beside the
application and remembers where you were.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import manual_content


class ManualWindow(QDialog):
    """Browsable, searchable documentation for the application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BiTS 4D — Manual")
        self.resize(940, 700)
        # Non-modal, and independent of the main window's lifetime
        self.setWindowFlags(
            self.windowFlags() | Qt.Window | Qt.WindowMinMaxButtonsHint
        )
        self._build()
        self.show_section(manual_content.SECTIONS[0]["id"])

    # ── construction ─────────────────────────────────────────────────────
    def _build(self):
        layout = QVBoxLayout(self)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText(
            "e.g. control materials, smoothing, drift, Mahalanobis"
        )
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._on_search)
        search_row.addWidget(self.search_box, 1)

        self.save_btn = QPushButton("Save as text…")
        self.save_btn.setToolTip("Write the whole manual to a text file.")
        self.save_btn.clicked.connect(self._on_save)
        search_row.addWidget(self.save_btn)
        layout.addLayout(search_row)

        splitter = QSplitter(Qt.Horizontal)

        self.contents = QTreeWidget()
        self.contents.setHeaderLabel("Contents")
        self.contents.setMinimumWidth(230)
        self.contents.setMaximumWidth(320)
        self.contents.itemSelectionChanged.connect(self._on_selection)
        splitter.addWidget(self.contents)

        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(True)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("color: gray; font-size: 9pt;")
        footer.addWidget(self.result_label, 1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)
        layout.addLayout(footer)

        self._populate(manual_content.SECTIONS)

    def _populate(self, sections):
        """Rebuild the contents tree from a list of sections."""
        self.contents.blockSignals(True)
        self.contents.clear()
        groups = {}
        for section in sections:
            group = section.get("group", "Manual")
            if group not in groups:
                node = QTreeWidgetItem([group])
                node.setFlags(node.flags() & ~Qt.ItemIsSelectable)
                self.contents.addTopLevelItem(node)
                groups[group] = node
            leaf = QTreeWidgetItem([section["title"]])
            leaf.setData(0, Qt.UserRole, section["id"])
            groups[group].addChild(leaf)
        self.contents.expandAll()
        self.contents.blockSignals(False)

    # ── behaviour ────────────────────────────────────────────────────────
    def show_section(self, section_id: str) -> None:
        """Display one section and select it in the contents."""
        section = manual_content.get_section(section_id)
        self.viewer.setHtml(manual_content.render(section))
        self.viewer.verticalScrollBar().setValue(0)

        self.contents.blockSignals(True)
        for index in range(self.contents.topLevelItemCount()):
            group = self.contents.topLevelItem(index)
            for child_index in range(group.childCount()):
                child = group.child(child_index)
                if child.data(0, Qt.UserRole) == section_id:
                    self.contents.setCurrentItem(child)
        self.contents.blockSignals(False)

    def current_section_id(self) -> Optional[str]:
        item = self.contents.currentItem()
        return None if item is None else item.data(0, Qt.UserRole)

    def _on_selection(self):
        section_id = self.current_section_id()
        if section_id:
            section = manual_content.get_section(section_id)
            self.viewer.setHtml(manual_content.render(section))
            self.viewer.verticalScrollBar().setValue(0)

    def _on_search(self, text):
        matches = manual_content.search(text)
        self._populate(matches)
        if not text.strip():
            self.result_label.setText("")
            return
        if matches:
            self.result_label.setText(
                f"{len(matches)} section(s) mention every word."
            )
            self.show_section(matches[0]["id"])
        else:
            self.result_label.setText(
                "Nothing matches every word — try fewer of them."
            )

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Manual", "BiTS4D_manual.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(manual_content.as_plain_text())
        except OSError as error:
            QMessageBox.warning(self, "Could Not Save", str(error))
            return
        self.result_label.setText(f"Saved to {path}")
