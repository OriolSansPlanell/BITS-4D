"""Superseded methods, kept runnable so earlier work can be reproduced.

Nothing here is on the application's segmentation path any more.

**Why the classifier was retired.** Its training labels came from
point-in-polygon tests on the (neutron, X-ray) histogram, so every label was
already a closed-form function of the voxel's two intensities. Training a
classifier on that target means fitting a function you have exactly:
formally, no additional feature can carry information about a label that the
intensities already determine. That is not a tuning problem, and it is why
the classifier and the regions it was trained from agreed almost everywhere,
differing only in the shape of the boundary between them.

It is also unnecessary on physical grounds. Neutron and X-ray attenuation
coefficients are material constants, so where a material sits in the plane is
fixed by physics rather than learned. What the histogram genuinely cannot
supply is *spatial* information — building it discards every spatial
relationship in the volume — and that is what the spatial smoothing in
:mod:`model.spatial_prior` restores.

Use :mod:`model.locked` instead. This package exists for reproducing figures
and for method comparisons, not for new work.
"""

from segmentation.legacy.random_forest_4d import (
    RandomForestSegmentation4D,
    labels_from_kmeans,
    labels_from_manual,
    labels_from_otsu,
)

__all__ = [
    "RandomForestSegmentation4D",
    "labels_from_manual",
    "labels_from_kmeans",
    "labels_from_otsu",
]
