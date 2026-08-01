"""GUI components for BiTS 4D."""

from .main_window import BiTS4DMainWindow, SliceViewerWidget
from .runtime_fixes import apply_runtime_fixes
from .time_navigation_widget import TimeNavigationWidget
from .dual_histogram_widget import DualHistogramWidget

apply_runtime_fixes(BiTS4DMainWindow, SliceViewerWidget)

__all__ = ["BiTS4DMainWindow", "TimeNavigationWidget", "DualHistogramWidget"]
