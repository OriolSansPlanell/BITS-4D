"""A layout that wraps its widgets onto the next line when it runs out of room.

Qt's box layouts do not wrap: a row of fifteen controls demands the sum of
their widths as a hard minimum, and the whole window inherits it. That is how
this application came to need a 3800-pixel-wide screen — not because anything
was large, but because several toolbars were long and could only ever be one
line tall.

A flow layout reports the width of its *widest single item* as its minimum
and spends extra height instead, so the same controls fit a laptop screen in
three rows and a wide monitor in one.

Standard Qt flow-layout construction, with the height-for-width contract
implemented so a containing splitter or scroll area can size it correctly.
"""

from __future__ import annotations

from PyQt5.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PyQt5.QtWidgets import QLayout, QSizePolicy, QWidgetItem


class FlowLayout(QLayout):
    """Left-to-right layout that wraps onto new rows as needed.

    Parameters
    ----------
    margin
        Outer margin in pixels.
    h_spacing, v_spacing
        Gaps between items. ``-1`` takes the style's default.
    """

    def __init__(self, parent=None, margin=0, h_spacing=6, v_spacing=4):
        super().__init__(parent)
        self._items = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(QMargins(margin, margin, margin, margin))

    # ── QLayout plumbing ─────────────────────────────────────────────────
    def addItem(self, item):
        self._items.append(item)

    def addStretch(self, _stretch=0):
        """Accepted and ignored.

        A flow layout wraps rather than distributing slack along one line, so
        there is nothing for a stretch to do. Accepting it means existing
        box-layout code can be switched over without editing every call.
        """
        return None

    def addSpacing(self, _size=0):
        return None

    def addLayout(self, layout):
        """Nest a layout, so a label and its spin box can stay together."""
        self.addChildLayout(layout)
        self.addItem(layout)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        # Never ask for more width: wrapping is the whole point.
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        """The widest single item — not the sum of all of them."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    # ── the wrapping itself ──────────────────────────────────────────────
    def _spacing(self, horizontal: bool) -> int:
        value = self._h_spacing if horizontal else self._v_spacing
        if value >= 0:
            return value
        parent = self.parent()
        if parent is None:
            return 6
        if parent.isWidgetType():
            return parent.style().pixelMetric(
                QSizePolicy.PushButton
                if horizontal else QSizePolicy.PushButton,
                None, parent,
            )
        return self.spacing()

    def _layout(self, rect, apply: bool) -> int:
        """Place the items; return the total height needed."""
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = effective.x()
        y = effective.y()
        row_height = 0

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing(True)
            if row_height and next_x - self._spacing(True) > effective.right():
                # Does not fit on this row: start a new one.
                x = effective.x()
                y = y + row_height + self._spacing(False)
                next_x = x + hint.width() + self._spacing(True)
                row_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            row_height = max(row_height, hint.height())

        return y + row_height - rect.y() + margins.bottom()


def wrap_in_flow(widgets, margin=0, h_spacing=6, v_spacing=4) -> FlowLayout:
    """Convenience: a :class:`FlowLayout` already holding *widgets*."""
    layout = FlowLayout(margin=margin, h_spacing=h_spacing, v_spacing=v_spacing)
    for widget in widgets:
        if isinstance(widget, QLayout):
            layout.addLayout(widget)
        elif widget is not None:
            layout.addWidget(widget)
    return layout
