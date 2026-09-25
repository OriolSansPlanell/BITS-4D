"""A synthetic 4-D battery-like cell with exact ground truth.

Geometry (per slice, identical along z): air around a cylindrical aluminium
casing (inert), filled with electrolyte, holding a lithium electrode slab.
From the first timepoint on, a **product layer** grows at the
lithium/electrolyte interface, consuming lithium: absent at T0, then thicker
at every step. That is the case the reviewers ask about — a phase that
appears during the experiment and is thinner than any ROI early on.

The measured images are made the way a scanner makes them: the ideal
(piecewise-constant) attenuation map is blurred (partial volume at every
interface) and noise is added. Optional artefacts:

* ``gain_drift`` — the X-ray channel's gain changes by this fraction per
  timepoint (instrument drift);
* ``cupping`` — beam-hardening-like radial darkening of the X-ray channel,
  as a fraction of the signal at the rim (non-Gaussian, position-dependent
  class clouds);
* ``misregistration`` — the X-ray volume is shifted by this many voxels
  (z, y, x) relative to the neutron volume;
* ``closed_ends`` — structure along z as well (see :func:`_geometry`);
* ``two_state_lithium`` — the lower half of the lithium electrode sits at a
  second (neutron, X-ray) position, so the class's cloud has two lobes: a
  non-Gaussian class, as a material in two states or a streak artefact
  produces.

Labels are exact integers per voxel and per timepoint (the ideal map before
blurring), so every method is scored against the same truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

#: Class order: label k+1 is CLASSES[k]; 0 is never used by the truth.
CLASSES = ("Air", "Aluminium", "Electrolyte", "Lithium", "Product")

#: (neutron, X-ray) attenuation-like signal of each class
SIGNAL = {
    "Air": (400.0, 300.0),
    "Aluminium": (700.0, 1500.0),
    "Electrolyte": (1100.0, 700.0),
    "Lithium": (1700.0, 450.0),
    "Product": (1350.0, 1000.0),
}

#: Per-channel noise standard deviation
NOISE = (60.0, 60.0)


@dataclass
class Phantom:
    """A generated series: volumes ``(T, Z, Y, X)`` and exact labels."""

    neutron: np.ndarray
    xray: np.ndarray
    labels: np.ndarray                      # (T, Z, Y, X), 1..len(CLASSES)
    classes: Tuple[str, ...] = CLASSES
    params: Dict[str, object] = field(default_factory=dict)

    @property
    def num_timepoints(self) -> int:
        return int(self.labels.shape[0])

    def mask(self, name: str, timepoint: int) -> np.ndarray:
        return self.labels[timepoint] == (self.classes.index(name) + 1)

    def volume_fractions(self) -> Dict[str, List[float]]:
        return {
            name: [float(np.mean(self.mask(name, t)))
                   for t in range(self.num_timepoints)]
            for name in self.classes
        }

    def roi_masks(self, timepoint: int = 0, erode: int = 1) -> Dict[str, np.ndarray]:
        """The masks a careful user would draw: each class's truth, eroded
        so the region stays clear of the partial-volume boundary.

        A class thinner than ``2 * erode + 1`` voxels at *timepoint* has an
        empty ROI — exactly what happens to a phase that has only just
        appeared.
        """
        structure = ndimage.generate_binary_structure(3, 1)
        masks = {}
        for name in self.classes:
            mask = self.mask(name, timepoint)
            if erode:
                mask = ndimage.binary_erosion(mask, structure, iterations=erode)
            masks[name] = mask
        return masks


class PhantomDataset:
    """Minimal dataset interface (what the segmenters read)."""

    def __init__(self, phantom: Phantom):
        self.neutron_data = phantom.neutron
        self.xray_data = phantom.xray
        self.num_timepoints = phantom.num_timepoints
        self.shape = phantom.neutron.shape

    def get_volume_at_time(self, timepoint):
        return self.neutron_data[timepoint], self.xray_data[timepoint]


def _geometry(shape, timepoint, product_growth, closed_ends=False):
    """Ideal label volume for one timepoint.

    By default the cell is extruded: every slice is the same. With
    *closed_ends* its contents stop short of the ends (air above and below)
    and the electrode leans by one voxel over the depth, so the structure
    varies along z too — which measuring an offset along z needs. Air, not
    aluminium, closes the ends: an aluminium/lithium partial-volume voxel
    would sit right on the product's signature.
    """
    depth, height, width = shape
    volume = np.empty(shape, dtype=np.int8)
    for z in range(depth):
        lean = int(round(z / max(depth - 1, 1))) if closed_ends else 0
        volume[z] = _slice_geometry((height, width), timepoint,
                                    product_growth, lean)
    if not closed_ends:
        return volume
    caps = max(2, depth // 6)
    yy, xx = np.mgrid[:height, :width]
    inside = np.hypot(yy - (height - 1) / 2.0,
                      xx - (width - 1) / 2.0) < 0.46 * min(height, width)
    for z in list(range(caps)) + list(range(depth - caps, depth)):
        volume[z][inside] = CLASSES.index("Air") + 1
    return volume


def _slice_geometry(plane, timepoint, product_growth, lean=0):
    """One slice of the cell; *lean* shifts the electrode in x."""
    height, width = plane
    yy, xx = np.mgrid[:height, :width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = np.hypot(yy - cy, xx - cx)
    outer = 0.46 * min(height, width)
    wall = max(2.0, 0.06 * min(height, width))

    labels = np.full((height, width), CLASSES.index("Air") + 1, dtype=np.int8)
    inside = radius < outer
    labels[inside] = CLASSES.index("Aluminium") + 1
    labels[radius < outer - wall] = CLASSES.index("Electrolyte") + 1

    # Lithium electrode: a vertical slab left of centre
    slab_left = int(cx - 0.30 * outer) + lean
    slab_right = int(cx - 0.05 * outer) + lean
    cavity = radius < outer - wall
    lithium = cavity & (xx >= slab_left) & (xx < slab_right)
    labels[lithium] = CLASSES.index("Lithium") + 1

    # Product grows on the electrolyte-facing side of the lithium,
    # consuming it: thickness 0 at T0, then product_growth voxels per step
    thickness = int(round(product_growth * timepoint))
    if thickness > 0:
        product = cavity & (xx >= slab_right - thickness) & (xx < slab_right)
        labels[product] = CLASSES.index("Product") + 1
    return labels


def make_phantom(
    shape: Sequence[int] = (12, 64, 64),
    timepoints: int = 6,
    product_growth: float = 1.0,
    blur: float = 0.8,
    noise: Tuple[float, float] = NOISE,
    gain_drift: float = 0.0,
    cupping: float = 0.0,
    misregistration: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    two_state_lithium: Tuple[float, float] = None,
    closed_ends: bool = False,
    seed: int = 0,
) -> Phantom:
    """Generate a series. Every argument is recorded in ``params``."""
    shape = tuple(int(v) for v in shape)
    rng = np.random.default_rng(seed)
    neutron = np.empty((timepoints,) + shape, dtype=np.float32)
    xray = np.empty((timepoints,) + shape, dtype=np.float32)
    labels = np.empty((timepoints,) + shape, dtype=np.int8)

    table_n = np.zeros(len(CLASSES) + 1)
    table_x = np.zeros(len(CLASSES) + 1)
    for index, name in enumerate(CLASSES, start=1):
        table_n[index], table_x[index] = SIGNAL[name]

    _, height, width = shape
    yy, xx = np.mgrid[:height, :width]
    radius = np.hypot(yy - (height - 1) / 2.0, xx - (width - 1) / 2.0)
    radial = (radius / radius.max()) ** 2          # 0 centre .. 1 corner

    for t in range(timepoints):
        truth = _geometry(shape, t, product_growth, closed_ends)
        labels[t] = truth
        ideal_n = table_n[truth]
        ideal_x = table_x[truth]
        if two_state_lithium is not None:
            lower = (truth == CLASSES.index("Lithium") + 1) & (
                np.arange(height)[None, :, None] >= height // 2)
            ideal_n = np.where(lower, two_state_lithium[0], ideal_n)
            ideal_x = np.where(lower, two_state_lithium[1], ideal_x)
        if blur > 0:
            ideal_n = ndimage.gaussian_filter(ideal_n, blur)
            ideal_x = ndimage.gaussian_filter(ideal_x, blur)
        if cupping:
            # Centre darker than the rim, proportionally to the signal
            ideal_x = ideal_x * (1.0 - cupping * (1.0 - radial))[None]
        ideal_x = ideal_x * (1.0 + gain_drift * t)
        if any(misregistration):
            ideal_x = ndimage.shift(ideal_x, misregistration, order=1,
                                    mode="nearest")
        neutron[t] = ideal_n + rng.normal(0.0, noise[0], shape)
        xray[t] = ideal_x + rng.normal(0.0, noise[1], shape)

    return Phantom(
        neutron=neutron, xray=xray, labels=labels,
        params=dict(shape=shape, timepoints=timepoints,
                    product_growth=product_growth, blur=blur,
                    noise=tuple(noise), gain_drift=gain_drift,
                    cupping=cupping, two_state_lithium=two_state_lithium,
                    closed_ends=closed_ends,
                    misregistration=tuple(misregistration), seed=seed),
    )
