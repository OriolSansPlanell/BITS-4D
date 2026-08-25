"""ROI-based segmentation for 4D datasets.

The classifier that used to live here has moved to
:mod:`segmentation.legacy`; see that package for why. Time-series
segmentation is in :mod:`model.locked`.
"""

from .segmentation_engine_4d import SegmentationEngine4D

__all__ = [
    'SegmentationEngine4D',
]
