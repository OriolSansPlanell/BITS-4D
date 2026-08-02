"""Cooperative cancellation primitives shared by GUI workers and services."""

from __future__ import annotations

from threading import Event


class OperationCancelled(RuntimeError):
    """Raised when a user-requested cancellation reaches a safe checkpoint."""


class OperationFailed(RuntimeError):
    """Raised after a worker operation reports a failure to the GUI."""


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise OperationCancelled("Operation cancelled by user")
