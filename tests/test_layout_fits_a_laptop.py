"""The window has to fit the screen people actually have.

This started as a real problem: the application demanded a minimum width of
nearly 3900 pixels, so on any normal monitor part of it was simply
unreachable. Nothing in it was large — several control rows were long, and a
Qt box layout cannot wrap, so each row's total width became a hard minimum
that the whole window inherited.

These tests pin the sizes down, because a single new non-wrapping row of
buttons is enough to bring the problem back and nothing else would notice.
"""

import os

import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QRect, QSize, Qt  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QPushButton,
    QWidget,
)

from gui.flow_layout import FlowLayout, wrap_in_flow  # noqa: E402

#: The screen a 14-inch laptop usually has, minus a little for a task bar.
LAPTOP = (1366, 740)
#: The smallest screen worth supporting at all.
SMALL_LAPTOP = (1280, 700)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow

    widget = BiTS4DMainWindow()
    widget.show()
    qapp.processEvents()
    return widget


# ── the flow layout itself ───────────────────────────────────────────────────

def test_a_flow_layout_asks_for_its_widest_item_not_their_sum(qapp):
    host = QWidget()
    buttons = [QPushButton("A rather long button label") for _ in range(10)]
    layout = wrap_in_flow(buttons)
    host.setLayout(layout)

    widest = max(button.sizeHint().width() for button in buttons)
    total = sum(button.sizeHint().width() for button in buttons)

    assert layout.minimumSize().width() <= widest + 8
    assert layout.minimumSize().width() < total / 4


def test_it_spends_height_instead_of_width(qapp):
    host = QWidget()
    layout = wrap_in_flow([QPushButton("Button") for _ in range(12)])
    host.setLayout(layout)

    wide = layout.heightForWidth(1200)
    narrow = layout.heightForWidth(300)
    assert narrow > wide, "a narrower layout must get taller, not clip"
    assert layout.hasHeightForWidth()


def test_every_item_is_placed_inside_the_area_it_was_given(qapp):
    host = QWidget()
    buttons = [QPushButton(f"Item {index}") for index in range(9)]
    layout = wrap_in_flow(buttons)
    host.setLayout(layout)
    layout.setGeometry(QRect(0, 0, 320, 400))

    for button in buttons:
        assert button.geometry().left() >= 0
        assert button.geometry().right() <= 320 + 2, button.text()


def test_stretch_calls_are_accepted_so_existing_rows_can_be_converted(qapp):
    layout = FlowLayout()
    layout.addWidget(QLabel("x"))
    layout.addStretch()          # meaningless here, but must not raise
    layout.addSpacing(20)
    assert layout.count() == 1


def test_items_can_be_taken_back_out(qapp):
    layout = FlowLayout()
    first = QLabel("a")
    layout.addWidget(first)
    layout.addWidget(QLabel("b"))
    assert layout.count() == 2
    layout.takeAt(0)
    assert layout.count() == 1
    assert layout.itemAt(5) is None
    assert layout.takeAt(5) is None


# ── the window ───────────────────────────────────────────────────────────────

def test_the_window_fits_a_fourteen_inch_screen(window):
    minimum = window.minimumSizeHint()
    assert minimum.width() <= LAPTOP[0], (
        f"needs {minimum.width()}px of width; a 14-inch laptop has "
        f"{LAPTOP[0]}"
    )
    assert minimum.height() <= LAPTOP[1], (
        f"needs {minimum.height()}px of height; a 14-inch laptop has "
        f"about {LAPTOP[1]} once the task bar is out"
    )


@pytest.mark.parametrize("size", [LAPTOP, SMALL_LAPTOP, (1600, 900)])
def test_the_window_actually_resizes_to_that_size(window, qapp, size):
    window.resize(*size)
    qapp.processEvents()
    assert window.width() <= size[0]
    assert window.height() <= size[1] + 4      # a couple of pixels of chrome


def test_no_single_panel_demands_more_than_its_share(window):
    """One panel that will not shrink is enough to break the whole window."""
    for index in range(window.main_splitter.count()):
        panel = window.main_splitter.widget(index)
        assert panel.minimumSizeHint().width() <= 620, (
            f"panel {index} ({type(panel).__name__}) will not go below "
            f"{panel.minimumSizeHint().width()}px"
        )


def test_the_histogram_and_slice_canvases_can_shrink(qapp):
    from gui.dual_histogram_widget import DualHistogramWidget
    from gui.main_window import SliceViewerWidget

    histograms = DualHistogramWidget()
    viewer = SliceViewerWidget()
    assert histograms.minimumSizeHint().width() <= 560
    assert viewer.minimumSizeHint().width() <= 420


def test_the_time_strip_is_one_row_not_a_stack(qapp):
    """It spans the window now, so four stacked rows waste 150px of height."""
    from gui.time_navigation_widget import TimeNavigationWidget

    widget = TimeNavigationWidget(num_timepoints=26)
    assert widget.minimumSizeHint().height() <= 80


# ── the two arrangements ─────────────────────────────────────────────────────

def test_compact_folds_the_second_histogram_away(window, qapp):
    window.set_compact_layout(True)
    qapp.processEvents()
    assert not window.dual_histogram.local_is_visible()

    window.set_compact_layout(False)
    qapp.processEvents()
    assert window.dual_histogram.local_is_visible()


def test_folding_is_reversible_from_the_splitter_alone(window, qapp):
    """Folded, not removed: the handle has to bring it back."""
    window.show_local_histogram(False)
    qapp.processEvents()
    assert not window.dual_histogram.local_is_visible()

    splitter = window.dual_histogram.hist_splitter
    assert splitter.count() == 2, "the panel must still be in the splitter"
    window.dual_histogram.set_local_visible(True)
    assert window.dual_histogram.local_is_visible()


def test_the_arrangement_is_chosen_from_the_screen(window, monkeypatch):
    """A wide monitor should not open in the laptop arrangement."""
    monkeypatch.setattr(
        type(window), "_screen_size", lambda self: (1366, 768)
    )
    window._apply_layout_for_screen()
    assert window._compact_layout

    monkeypatch.setattr(
        type(window), "_screen_size", lambda self: (2560, 1440)
    )
    window._apply_layout_for_screen()
    assert not window._compact_layout


def test_the_window_never_opens_larger_than_the_screen(window, monkeypatch):
    monkeypatch.setattr(
        type(window), "_screen_size", lambda self: (1366, 768)
    )
    window._apply_layout_for_screen()
    assert window.width() <= 1366
    assert window.height() <= 768


def test_panels_can_be_hidden_and_brought_back(window, qapp):
    window.show_tool_panel(False)
    qapp.processEvents()
    assert not window.right_tabs.isVisible()
    window.show_tool_panel(True)
    qapp.processEvents()
    assert window.right_tabs.isVisible()

    window.show_time_strip(False)
    qapp.processEvents()
    assert not window.time_group.isVisible()
    window.show_time_strip(True)
    qapp.processEvents()
    assert window.time_group.isVisible()


def test_the_view_menu_offers_the_layout_controls(window):
    labels = []
    for action in window.menuBar().actions():
        if action.text() == "View":
            labels = [entry.text() for entry in action.menu().actions()]
    assert labels, "there is no View menu"
    assert any("Compact" in label for label in labels)
    assert any("histogram" in label for label in labels)
    assert any("Tool panel" in label for label in labels)
    assert any("Fit window" in label for label in labels)


def test_the_menu_and_the_layout_stay_in_step(window, qapp):
    window.set_compact_layout(True)
    assert window.compact_action.isChecked()
    assert not window.local_hist_action.isChecked()

    window.set_compact_layout(False)
    assert not window.compact_action.isChecked()
    assert window.local_hist_action.isChecked()


def test_the_time_strip_spans_the_window(window):
    """Not stacked inside one column, where it would eat that column's height."""
    assert window.time_group.parentWidget() is window.centralWidget()
    assert window.time_group.width() >= window.main_splitter.width() - 20
