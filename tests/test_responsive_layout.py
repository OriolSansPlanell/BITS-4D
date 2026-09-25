"""The window must fit a laptop screen.

Regression: long single rows of controls and side-by-side plots gave the
main window a minimum width of 3839 px, so on a 14" or 16" laptop part of it
was always off-screen.
"""

import os

import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QPushButton  # noqa: E402

#: A 14" laptop at 150 % scaling (1920x1080 physical) offers about this much
LAPTOP = (1280, 720)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    from gui import BiTS4DMainWindow
    return BiTS4DMainWindow()


def test_window_minimum_fits_a_laptop_screen(window):
    hint = window.minimumSizeHint()
    assert hint.width() <= LAPTOP[0], hint
    assert hint.height() <= LAPTOP[1], hint


def test_window_opens_within_the_screen(window, qapp):
    available = qapp.primaryScreen().availableGeometry()
    assert window.width() <= available.width()
    assert window.height() <= available.height()


def test_each_panel_fits_its_share_of_a_laptop_screen(window):
    width = LAPTOP[0]
    assert window.dual_histogram.minimumSizeHint().width() <= 0.34 * width
    assert window.slice_viewer.minimumSizeHint().width() <= 0.44 * width


def test_plots_become_tabs_when_narrow(qapp):
    from PyQt5.QtWidgets import QLabel
    from gui.responsive import PlotPanes

    panes = PlotPanes()
    global_pane, local_pane = QLabel("global"), QLabel("local")
    for pane in (global_pane, local_pane):
        pane.setMinimumSize(200, 150)
    panes.add_pane(global_pane, "Global")
    panes.add_pane(local_pane, "Local")
    panes.resize(1000, 400)
    panes.show()
    qapp.processEvents()
    assert not panes.tabbed
    assert global_pane.isVisible() and local_pane.isVisible()

    panes.resize(380, 400)
    qapp.processEvents()
    assert panes.tabbed
    # One plot at a time; the tab bar switches between them
    assert global_pane.isVisible() and not local_pane.isVisible()
    panes.tab_bar.setCurrentIndex(1)
    assert local_pane.isVisible() and not global_pane.isVisible()

    panes.resize(1000, 400)
    qapp.processEvents()
    assert not panes.tabbed
    assert global_pane.isVisible() and local_pane.isVisible()
    panes.close()


def test_histogram_panel_uses_plot_panes(window):
    from gui.responsive import PlotPanes
    panes = window.dual_histogram.plot_panes
    assert isinstance(panes, PlotPanes)
    assert panes.splitter.count() == 2


def test_wrapped_rows_get_the_height_they_need(qapp):
    from gui.responsive import flow_row

    buttons = [QPushButton(f"Button number {i}") for i in range(8)]
    row = flow_row(*buttons)
    row.resize(200, 10)
    row.show()
    qapp.processEvents()
    bottom = max(b.geometry().bottom() for b in buttons)
    assert row.minimumHeight() > bottom, "wrapped lines would overlap"
    row.hide()


def test_spatial_tools_can_be_folded_away(window):
    viewer = window.slice_viewer
    viewer.set_spatial_tools_visible(False)
    assert viewer.spatial_tools_row.isHidden()
    assert not viewer.spatial_tools_btn.isChecked()
    viewer.spatial_tools_btn.setChecked(True)
    assert not viewer.spatial_tools_row.isHidden()
