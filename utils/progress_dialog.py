"""Cancelable progress dialogs and worker-thread orchestration."""

from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable, Optional

from PyQt5.QtCore import QEventLoop, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

from .cancellation import CancellationToken, OperationCancelled, OperationFailed


class ProgressDialog(QProgressDialog):
    """Progress dialog that remains visible while cancellation is acknowledged."""

    def __init__(
        self,
        title: str,
        message: str,
        maximum: int = 100,
        parent=None,
    ) -> None:
        super().__init__(message, "Cancel", 0, maximum, parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumDuration(250)
        self.setAutoClose(False)
        self.setAutoReset(False)

    def update_progress(self, value: int, message: Optional[str] = None) -> None:
        self.setValue(max(self.minimum(), min(int(value), self.maximum())))
        if message:
            self.setLabelText(message)


def _accepts_keyword(operation: Callable, keyword: str) -> bool:
    """Return whether a callable accepts a keyword or arbitrary kwargs."""
    try:
        signature = inspect.signature(operation)
    except (TypeError, ValueError):
        return False
    if keyword in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


class WorkerThread(QThread):
    """Execute one operation and report progress without blocking the GUI."""

    progress_updated = pyqtSignal(int, str)
    operation_completed = pyqtSignal(object)
    operation_failed = pyqtSignal(str)
    operation_cancelled = pyqtSignal()

    def __init__(self, operation: Callable, *args, **kwargs) -> None:
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = dict(kwargs)
        self.result: Any = None
        self.error: Optional[str] = None
        self.traceback_text: Optional[str] = None
        self.token = CancellationToken()

    def request_cancel(self) -> None:
        self.token.cancel()
        self.requestInterruption()

    def run(self) -> None:
        try:
            kwargs = dict(self.kwargs)
            if (
                "progress_callback" not in kwargs
                and _accepts_keyword(self.operation, "progress_callback")
            ):
                kwargs["progress_callback"] = self.report_progress
            if (
                "cancel_check" not in kwargs
                and _accepts_keyword(self.operation, "cancel_check")
            ):
                kwargs["cancel_check"] = self.token.raise_if_cancelled

            self.token.raise_if_cancelled()
            self.result = self.operation(*self.args, **kwargs)
            self.token.raise_if_cancelled()
            self.operation_completed.emit(self.result)
        except OperationCancelled:
            self.operation_cancelled.emit()
        except BaseException as exc:  # worker boundary: preserve full traceback
            self.error = str(exc)
            self.traceback_text = traceback.format_exc()
            self.operation_failed.emit(self.error)

    def report_progress(self, value: int, message: str = "") -> None:
        if self.isInterruptionRequested():
            self.token.cancel()
        self.token.raise_if_cancelled()
        self.progress_updated.emit(int(value), str(message))


def run_with_progress(
    parent,
    title: str,
    message: str,
    operation: Callable,
    *args,
    **kwargs,
) -> Any:
    """Run an operation and raise explicit cancellation/failure exceptions.

    Cancellation is cooperative: every progress callback and optional
    ``cancel_check`` call is a safe cancellation checkpoint.  The dialog no
    longer disappears while the worker continues in the background, and the
    caller can distinguish cancellation from failure instead of retrying the
    operation synchronously.
    """
    progress = ProgressDialog(title, message, 100, parent)
    worker = WorkerThread(operation, *args, **kwargs)
    loop = QEventLoop()
    state = {"result": None, "status": "running", "error": None}

    def finish(status: str, result: Any = None, error: Optional[str] = None) -> None:
        state.update(result=result, status=status, error=error)
        if status == "completed":
            progress.setValue(progress.maximum())
        progress.close()
        if loop.isRunning():
            loop.quit()

    worker.progress_updated.connect(progress.update_progress)
    worker.operation_completed.connect(
        lambda result: finish("completed", result=result)
    )
    worker.operation_cancelled.connect(lambda: finish("cancelled"))
    worker.operation_failed.connect(
        lambda error: finish("failed", error=error)
    )

    def request_cancel() -> None:
        if state["status"] != "running":
            return
        progress.setLabelText("Cancelling at the next safe checkpoint...")
        progress.setCancelButton(None)
        worker.request_cancel()

    progress.canceled.connect(request_cancel)
    worker.start()
    progress.show()
    loop.exec_()
    worker.wait()

    if state["status"] == "cancelled":
        raise OperationCancelled("Operation cancelled by user")
    if state["status"] == "failed":
        details = worker.traceback_text or state["error"] or "Unknown worker error"
        QMessageBox.critical(
            parent,
            "Operation failed",
            f"{state['error'] or 'Unknown error'}\n\nSee the console/log for details.",
        )
        raise OperationFailed(details)
    return state["result"]


class ProgressCallback:
    """Map completed steps to percentage updates."""

    def __init__(self, total_steps: int) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.total_steps = int(total_steps)
        self.current_step = 0
        self.callback: Optional[Callable[[int, str], None]] = None

    def set_callback(self, callback: Callable[[int, str], None]) -> None:
        self.callback = callback

    def _emit(self, message: str) -> None:
        if self.callback is not None:
            percentage = int(
                100 * min(max(self.current_step, 0), self.total_steps)
                / self.total_steps
            )
            self.callback(percentage, message)

    def update(self, steps: int = 1, message: str = "") -> None:
        self.current_step += int(steps)
        self._emit(message)

    def set_step(self, step: int, message: str = "") -> None:
        self.current_step = int(step)
        self._emit(message)

    def reset(self) -> None:
        self.current_step = 0


def show_loading_message(parent, title: str, message: str):
    progress = QProgressDialog(message, None, 0, 0, parent)
    progress.setWindowTitle(title)
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setCancelButton(None)
    progress.show()
    QApplication.processEvents()
    return progress


def run_batch_operation(
    parent,
    title: str,
    items: list,
    operation: Callable,
    item_name: str = "item",
) -> list:
    results = []
    total = len(items)
    progress = ProgressDialog(
        title, f"Processing {item_name} 0/{total}", max(total, 1), parent
    )
    progress.show()
    try:
        for index, item in enumerate(items):
            if progress.wasCanceled():
                raise OperationCancelled("Batch operation cancelled by user")
            results.append(operation(item))
            progress.update_progress(
                index + 1, f"Processing {item_name} {index + 1}/{total}"
            )
            QApplication.processEvents()
    finally:
        progress.close()
    return results
