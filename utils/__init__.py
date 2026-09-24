"""
utils package - Utility modules for BiTS 4D

The progress-dialog helpers need PyQt5; they are imported on first use so
the numerical utilities (and the headless test suite) work without Qt.
"""

from .config import *
from .roi_manager import ROIManager

_PROGRESS_DIALOG_NAMES = (
    'ProgressDialog',
    'WorkerThread',
    'run_with_progress',
    'show_loading_message',
    'run_batch_operation',
    'ProgressCallback',
)


def __getattr__(name):
    if name in _PROGRESS_DIALOG_NAMES:
        from . import progress_dialog
        return getattr(progress_dialog, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['ROIManager', *_PROGRESS_DIALOG_NAMES]
