"""Shared test set-up.

The figure-format pop-up is modal: a test that reaches it without stubbing
it would wait forever. Make that fail at once instead, and reset the
choices the pop-up remembers between exports so tests stay independent.
Tests that go through the pop-up stub ``FigureSaveDialog.exec_`` themselves.
"""

import pytest


@pytest.fixture(autouse=True)
def _figure_dialog_never_blocks(monkeypatch):
    try:
        from gui.figure_save_dialog import FigureSaveDialog
    except ImportError:          # no PyQt5: nothing modal can open
        yield
        return

    def _unexpected(self):
        raise AssertionError(
            "The figure-format pop-up opened in a test that did not stub it"
        )

    monkeypatch.setattr(FigureSaveDialog, "exec_", _unexpected)
    monkeypatch.setattr(FigureSaveDialog, "last_format", "svg")
    monkeypatch.setattr(FigureSaveDialog, "last_dpi", 300)
    monkeypatch.setattr(FigureSaveDialog, "last_save_csv", True)
    yield
