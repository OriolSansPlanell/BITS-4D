"""
progress_dialog.py - Progress dialog utilities for long-running operations

Provides progress feedback for:
- Dataset loading
- Histogram computation
- Segmentation operations
"""

from PyQt5.QtWidgets import QProgressDialog, QApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from typing import Callable, Any, Optional
import time


class ProgressDialog(QProgressDialog):
    """
    Enhanced progress dialog with better defaults
    """
    
    def __init__(self, title: str, message: str, 
                 maximum: int = 100, parent=None):
        super().__init__(message, "Cancel", 0, maximum, parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumDuration(500)  # Show after 500ms
        self.setAutoClose(True)
        self.setAutoReset(True)
        
    def update_progress(self, value: int, message: Optional[str] = None):
        """Update progress value and optionally message"""
        self.setValue(value)
        if message:
            self.setLabelText(message)
        QApplication.processEvents()


class WorkerThread(QThread):
    """
    Worker thread for running long operations without blocking UI
    """
    
    # Signals
    progress_updated = pyqtSignal(int, str)
    operation_completed = pyqtSignal(object)
    operation_failed = pyqtSignal(str)
    
    def __init__(self, operation: Callable, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
        
    def run(self):
        """Execute the operation"""
        try:
            # Add progress callback to kwargs if operation supports it
            if 'progress_callback' not in self.kwargs:
                self.kwargs['progress_callback'] = self.report_progress
            
            self.result = self.operation(*self.args, **self.kwargs)
            self.operation_completed.emit(self.result)
            
        except Exception as e:
            self.error = str(e)
            self.operation_failed.emit(self.error)
    
    def report_progress(self, value: int, message: str = ""):
        """Report progress back to main thread"""
        self.progress_updated.emit(value, message)


def run_with_progress(parent, title: str, message: str,
                     operation: Callable, *args, **kwargs) -> Any:
    """
    Run an operation with a progress dialog
    
    Args:
        parent: Parent widget
        title: Progress dialog title
        message: Progress dialog message
        operation: Function to execute
        *args, **kwargs: Arguments for operation
        
    Returns:
        Result from operation or None if cancelled/failed
    """
    # Create progress dialog
    progress = ProgressDialog(title, message, 100, parent)
    
    # Create worker thread
    worker = WorkerThread(operation, *args, **kwargs)
    
    # Connect signals
    result_container = {'result': None, 'success': False}
    
    def on_progress(value, msg):
        progress.update_progress(value, msg)
    
    def on_completed(result):
        result_container['result'] = result
        result_container['success'] = True
        progress.setValue(100)
    
    def on_failed(error):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(parent, "Error", f"Operation failed: {error}")
        progress.cancel()
    
    worker.progress_updated.connect(on_progress)
    worker.operation_completed.connect(on_completed)
    worker.operation_failed.connect(on_failed)
    
    # Start worker
    worker.start()
    
    # Show progress dialog (blocks until done or cancelled)
    progress.exec_()
    
    # Wait for worker to finish
    worker.wait()
    
    if result_container['success']:
        return result_container['result']
    else:
        return None


class ProgressCallback:
    """
    Helper class for reporting progress from operations
    """
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.callback = None
        
    def set_callback(self, callback: Callable[[int, str], None]):
        """Set the callback function"""
        self.callback = callback
    
    def update(self, steps: int = 1, message: str = ""):
        """Increment progress"""
        self.current_step += steps
        if self.callback:
            percentage = int((self.current_step / self.total_steps) * 100)
            self.callback(percentage, message)
    
    def set_step(self, step: int, message: str = ""):
        """Set absolute step value"""
        self.current_step = step
        if self.callback:
            percentage = int((self.current_step / self.total_steps) * 100)
            self.callback(percentage, message)
    
    def reset(self):
        """Reset progress to 0"""
        self.current_step = 0


# Convenience functions for common operations

def show_loading_message(parent, title: str, message: str):
    """
    Show an indeterminate progress dialog for quick operations
    
    Returns the dialog object - caller should call dialog.close() when done
    """
    progress = QProgressDialog(message, None, 0, 0, parent)
    progress.setWindowTitle(title)
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setCancelButton(None)
    progress.show()
    QApplication.processEvents()
    return progress


def run_batch_operation(parent, title: str, items: list,
                       operation: Callable, item_name: str = "item") -> list:
    """
    Run an operation on a list of items with progress feedback
    
    Args:
        parent: Parent widget
        title: Progress dialog title
        items: List of items to process
        operation: Function that takes an item and returns result
        item_name: Name for items in progress message
        
    Returns:
        List of results
    """
    results = []
    total = len(items)
    
    progress = ProgressDialog(title, f"Processing {item_name} 0/{total}", total, parent)
    
    for i, item in enumerate(items):
        if progress.wasCanceled():
            break
        
        try:
            result = operation(item)
            results.append(result)
        except Exception as e:
            print(f"Error processing {item_name} {i}: {e}")
            results.append(None)
        
        progress.update_progress(i + 1, f"Processing {item_name} {i+1}/{total}")
    
    return results
