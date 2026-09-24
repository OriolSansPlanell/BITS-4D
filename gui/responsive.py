"""
responsive.py - Layout helpers that let the window fit laptop screens

The window used to be laid out for a very wide monitor: long single rows of
buttons and side-by-side plots gave it a minimum width of several thousand
pixels. These helpers remove the fixed minimums instead of shrinking fonts:

* :class:`FlowLayout` wraps a row of controls onto as many lines as the
  available width needs, like words in a paragraph.
* :func:`flow_row` puts widgets into a FlowLayout on their own container, the
  form that plays well inside ordinary box layouts.
* :class:`PlotPanes` shows plots side by side when there is room and as
  tabs, one at a time, when there is not.
* :func:`scrollable` wraps a panel so that, when the window really is too
  small, the panel scrolls instead of forcing the window to grow.
* :func:`fit_to_screen` sizes a top-level window to the screen it opens on.
"""

from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtWidgets import (
    QApplication, QFrame, QLayout, QScrollArea, QSizePolicy, QSplitter,
    QWidget,
)


class FlowLayout(QLayout):
    """Left-to-right layout that wraps onto new lines when out of width."""

    def __init__(self, parent=None, margin=0, h_spacing=6, v_spacing=4):
        super().__init__(parent)
        self._items = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        while self.count():
            self.takeAt(0)

    # QLayout interface -----------------------------------------------------
    def addItem(self, item):
        self._items.append(item)

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
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        # Preferred: everything on one line
        width = 0
        height = 0
        for item in self._items:
            hint = item.sizeHint()
            width += hint.width() + self._h_spacing
            height = max(height, hint.height())
        margins = self.contentsMargins()
        return QSize(
            max(0, width - self._h_spacing) + margins.left() + margins.right(),
            height + margins.top() + margins.bottom(),
        )

    def minimumSize(self):
        # Minimum: the widest single item, since everything else can wrap
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _do_layout(self, rect, test_only):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(),
                             -margins.right(), -margins.bottom())
        x, y = area.x(), area.y()
        line_height = 0
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if next_x - self._h_spacing > area.right() + 1 and line_height > 0:
                x = area.x()
                y += line_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def flow_row(*widgets, h_spacing=6, v_spacing=4):
    """A container widget holding *widgets* in a wrapping row."""
    container = _FlowContainer()
    layout = FlowLayout(container, margin=0,
                        h_spacing=h_spacing, v_spacing=v_spacing)
    for widget in widgets:
        layout.addWidget(widget)
    policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    policy.setHeightForWidth(True)
    container.setSizePolicy(policy)
    return container


def group(*widgets):
    """Keep several widgets together on one line of a flow row.

    A label and the control it names should wrap as a unit.
    """
    from PyQt5.QtWidgets import QHBoxLayout
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    for widget in widgets:
        layout.addWidget(widget)
    return container


class _FlowContainer(QWidget):
    """Holder of a FlowLayout that asks for the height its wrapping needs.

    Splitters and scroll areas ignore height-for-width, so without this a
    wrapped row is given the height of a single line and its buttons
    overlap whatever comes next.
    """

    def resizeEvent(self, event):
        super().resizeEvent(event)
        layout = self.layout()
        if layout is not None:
            needed = layout.heightForWidth(event.size().width())
            if needed > 0 and needed != self.minimumHeight():
                self.setMinimumHeight(needed)


class PlotPanes(QWidget):
    """Plots side by side when there is room, otherwise one at a time in tabs.

    Stacking two plots in a short column squeezes both flat, so on a narrow
    or short panel (a laptop screen) the panes become tabs instead. Nothing
    is re-parented: in tab mode the panes that are not current are hidden.
    """

    def __init__(self, parent=None, side_ratio=1.7, tab_ratio=1.45):
        super().__init__(parent)
        from PyQt5.QtWidgets import QTabBar, QVBoxLayout
        self._side_ratio = side_ratio
        self._tab_ratio = tab_ratio
        self._headings = []
        self.tab_bar = QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.currentChanged.connect(self._show_current)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.tab_bar)
        layout.addWidget(self.splitter, stretch=1)
        self.tabbed = False
        self.tab_bar.hide()

    def add_pane(self, widget, title, heading=None):
        """Add *widget*; *heading* (if any) is hidden while shown as tabs."""
        self.splitter.addWidget(widget)
        self.tab_bar.addTab(title)
        self._headings.append(heading)

    def _panes(self):
        return [self.splitter.widget(i) for i in range(self.splitter.count())]

    def _side_by_side_width(self):
        return (sum(p.minimumSizeHint().width() for p in self._panes())
                + self.splitter.handleWidth() * max(0, self.splitter.count() - 1))

    def set_tabbed(self, tabbed):
        tabbed = bool(tabbed)
        if tabbed == self.tabbed:
            return
        self.tabbed = tabbed
        self.tab_bar.setVisible(tabbed)
        for heading in self._headings:
            if heading is not None:
                heading.setVisible(not tabbed)
        self._show_current()

    def _show_current(self, *_args):
        current = self.tab_bar.currentIndex()
        for index, pane in enumerate(self._panes()):
            pane.setVisible(not self.tabbed or index == current)

    def wants_tabs(self, width, height):
        """Tab mode for the given size, with hysteresis around the switch."""
        if width < self._side_by_side_width():
            return True
        if height <= 0:
            return self.tabbed
        ratio = width / float(height)
        if self.tabbed:
            return ratio < self._side_ratio
        return ratio < self._tab_ratio

    def minimumSizeHint(self):
        # Never demand the side-by-side width: with less, show tabs instead
        hint = super().minimumSizeHint()
        panes = self._panes()
        if panes:
            hint.setWidth(max(p.minimumSizeHint().width() for p in panes))
        return hint

    def resizeEvent(self, event):
        size = event.size()
        self.set_tabbed(self.wants_tabs(size.width(), size.height()))
        super().resizeEvent(event)


def scrollable(widget, horizontal=True):
    """Wrap *widget* in a frameless scroll area that resizes it."""
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarAsNeeded if horizontal else Qt.ScrollBarAlwaysOff
    )
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    return area


def fit_to_screen(window, preferred=QSize(1800, 960), margin=0.94):
    """Size *window* to its screen: the preferred size, or what fits.

    Returns the size chosen. On a laptop screen smaller than the preferred
    size the window opens maximised, so nothing is off-screen.
    """
    screen = window.screen() if hasattr(window, "screen") else None
    if screen is None:
        screen = QApplication.primaryScreen()
    if screen is None:
        window.resize(preferred)
        return preferred
    available = screen.availableGeometry()
    width = min(preferred.width(), int(available.width() * margin))
    height = min(preferred.height(), int(available.height() * margin))
    window.resize(width, height)
    window.move(
        available.x() + (available.width() - width) // 2,
        available.y() + (available.height() - height) // 2,
    )
    small = (available.width() < preferred.width()
             or available.height() < preferred.height())
    window._bits_small_screen = small
    return QSize(width, height)
