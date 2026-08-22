"""Composable feature sets for the Random Forest.

The old feature ladder nests: ``advanced`` adds normalised ``(Z, Y, X)``
coordinates, and ``expert`` adds texture *on top of* ``advanced``. Texture is
therefore unreachable without the coordinates — so any experiment that wanted
to ask "does texture help?" was forced to also take a frozen T0 anchor, and
came back unable to separate the two effects.

Those coordinates are normalised once and are **identical at every
timepoint**. A model that leans on them has memorised where materials were at
T0; it will look excellent on T0 and decay quietly thereafter, and the decay
will not show up in any histogram-space metric.

:class:`FeatureSpec` makes each group independent. The legacy names still
work and produce **exactly** the same columns in the same order, so existing
models, saved payloads and notebooks reproduce bit-for-bit.

Geometry has three settings rather than a flag:

``'none'``
    No spatial coordinates. Time-invariant; the honest default for anything
    that will be applied to a timepoint other than the one it trained on.
``'absolute'``
    Coordinates normalised by the volume extent — the legacy behaviour, and a
    T0 anchor. Warned about when used across timepoints.
``'relative'``
    Distance and direction from the *sample's own* centre of mass at that
    timepoint. Still spatial, but it moves with the sample instead of pinning
    it, so it survives a shift in the field of view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import laplace, sobel, uniform_filter

# Feature groups that do not change between timepoints, so a model leaning on
# them is remembering T0 rather than measuring.
ANCHORED_FEATURE_PREFIXES = ("coord_",)


@dataclass
class FeatureSpec:
    """Which feature groups to extract.

    Attributes
    ----------
    intensity
        The two raw values, ``n`` and ``x``.
    projections
        ``n/x``, ``n+x``, ``n−x``, ``hypot(n, x)`` — the linear combinations
        the histogram itself is built on.
    texture_scales
        Radii for local mean and standard deviation, e.g. ``(1, 2, 4)``. The
        old code offered a single 3-voxel window; compact, mossy and
        dendritic morphologies differ at different scales, and one window
        cannot see that.
    laplacian
        Second derivative of the neutron volume (legacy ``expert`` feature).
    gradient
        ``|∇n|``, ``|∇x|`` and their **coherence** — the cosine between the
        two gradient directions. A real material interface produces aligned
        edges in both modalities; a ring artifact or a beam-hardening streak
        usually produces an edge in only one. That distinction is not
        expressible in any other feature here.
    structure
        Structure-tensor shape descriptors (anisotropy and planarity) at each
        texture scale — local *shape*, which is what separates a compact
        deposit from a dendritic one.
    geometry
        ``'none'``, ``'absolute'`` or ``'relative'`` (see the module note).
    """

    intensity: bool = True
    projections: bool = True
    texture_scales: Tuple[int, ...] = ()
    laplacian: bool = False
    gradient: bool = False
    structure: bool = False
    geometry: str = "none"

    def __post_init__(self):
        if self.geometry not in {"none", "absolute", "relative"}:
            raise ValueError(
                f"geometry must be 'none', 'absolute' or 'relative', "
                f"not {self.geometry!r}"
            )
        self.texture_scales = tuple(int(radius) for radius in self.texture_scales)
        if any(radius < 1 for radius in self.texture_scales):
            raise ValueError("texture radii must be at least 1")

    # ── description ──────────────────────────────────────────────────────
    def feature_names(self) -> List[str]:
        names: List[str] = []
        if self.intensity:
            names += ["neutron", "xray"]
        if self.projections:
            names += ["ratio", "sum", "difference", "magnitude"]
        if self.geometry == "absolute":
            names += ["coord_z", "coord_y", "coord_x"]
        elif self.geometry == "relative":
            names += ["radial_distance", "offset_z", "offset_y", "offset_x"]
        for radius in self.texture_scales:
            names += [
                f"neutron_mean_r{radius}", f"neutron_std_r{radius}",
                f"xray_mean_r{radius}", f"xray_std_r{radius}",
            ]
        if self.laplacian:
            names += ["neutron_laplacian"]
        if self.gradient:
            names += ["neutron_gradient", "xray_gradient", "gradient_coherence"]
        if self.structure:
            for radius in self.texture_scales or (1,):
                names += [
                    f"anisotropy_r{radius}", f"planarity_r{radius}",
                ]
        return names

    @property
    def n_features(self) -> int:
        return len(self.feature_names())

    @property
    def is_time_invariant(self) -> bool:
        """True when nothing here pins the model to one timepoint's geometry."""
        return self.geometry != "absolute"

    def anchored_features(self) -> List[str]:
        return [
            name for name in self.feature_names()
            if name.startswith(ANCHORED_FEATURE_PREFIXES)
        ]

    def describe(self) -> str:
        parts = []
        if self.intensity:
            parts.append("intensity")
        if self.projections:
            parts.append("projections")
        if self.texture_scales:
            parts.append(
                "texture r=" + ",".join(str(r) for r in self.texture_scales)
            )
        if self.laplacian:
            parts.append("laplacian")
        if self.gradient:
            parts.append("gradient+coherence")
        if self.structure:
            parts.append("structure tensor")
        parts.append(f"geometry={self.geometry}")
        return ", ".join(parts)


# Legacy levels, reproducing the old column order exactly.
LEGACY_SPECS: Dict[str, FeatureSpec] = {
    "basic": FeatureSpec(),
    "advanced": FeatureSpec(geometry="absolute"),
    "expert": FeatureSpec(
        geometry="absolute", texture_scales=(1,), laplacian=True
    ),
}

# Recommended combinations.
PRESETS: Dict[str, FeatureSpec] = {
    "histogram": FeatureSpec(),
    "texture": FeatureSpec(texture_scales=(1, 2, 4), gradient=True, structure=True),
    "anchored": LEGACY_SPECS["expert"],
    "relative": FeatureSpec(
        texture_scales=(1, 2), gradient=True, geometry="relative"
    ),
}


def resolve_spec(spec) -> FeatureSpec:
    """Accept a :class:`FeatureSpec`, a legacy level name or a preset name."""
    if isinstance(spec, FeatureSpec):
        return spec
    if isinstance(spec, str):
        if spec in LEGACY_SPECS:
            return LEGACY_SPECS[spec]
        if spec in PRESETS:
            return PRESETS[spec]
        raise ValueError(
            f"Unknown feature spec {spec!r}; choose one of "
            f"{sorted(set(LEGACY_SPECS) | set(PRESETS))} or pass a FeatureSpec"
        )
    raise TypeError("spec must be a FeatureSpec or a name")


# ── derived volumes ──────────────────────────────────────────────────────────

def local_moments(volume, radius: int):
    """Local mean and standard deviation, in two separable filter passes.

    Exactly the quantities a per-voxel ``std`` callback would produce, without
    running Python once per voxel.
    """
    array = np.asarray(volume, dtype=np.float32)
    size = 2 * int(radius) + 1
    mean = uniform_filter(array, size=size, mode="nearest")
    mean_square = uniform_filter(array * array, size=size, mode="nearest")
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return mean, np.sqrt(variance, dtype=np.float32)


def gradient_magnitude(volume):
    array = np.asarray(volume, dtype=np.float32)
    components = [sobel(array, axis=axis, mode="nearest") for axis in range(3)]
    return np.sqrt(sum(component * component for component in components))


def gradient_coherence(neutron, xray, epsilon: float = 1e-6):
    """|cos θ| between the two modalities' gradient directions.

    Near 1 where both see the same edge — a real interface. Near 0 where only
    one does, which is the signature of a ring artifact or a beam-hardening
    streak rather than a material boundary.
    """
    first = np.asarray(neutron, dtype=np.float32)
    second = np.asarray(xray, dtype=np.float32)
    a = [sobel(first, axis=axis, mode="nearest") for axis in range(3)]
    b = [sobel(second, axis=axis, mode="nearest") for axis in range(3)]
    dot = sum(u * v for u, v in zip(a, b))
    norm_a = np.sqrt(sum(u * u for u in a))
    norm_b = np.sqrt(sum(v * v for v in b))
    return np.abs(dot) / (norm_a * norm_b + np.float32(epsilon))


def structure_descriptors(volume, radius: int):
    """Anisotropy and planarity of the local structure tensor.

    The tensor is the outer product of the gradient, smoothed over the
    neighbourhood; its eigenvalue ratios describe local *shape* — whether a
    neighbourhood is blob-like, sheet-like or filament-like.
    """
    array = np.asarray(volume, dtype=np.float32)
    gradients = [sobel(array, axis=axis, mode="nearest") for axis in range(3)]
    size = 2 * int(radius) + 1

    tensor = np.empty(array.shape + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(i, 3):
            smoothed = uniform_filter(
                gradients[i] * gradients[j], size=size, mode="nearest"
            )
            tensor[..., i, j] = smoothed
            tensor[..., j, i] = smoothed

    eigenvalues = np.linalg.eigvalsh(tensor)          # ascending
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = eigenvalues[..., 2]
    middle = eigenvalues[..., 1]
    smallest = eigenvalues[..., 0]
    total = largest + middle + smallest + np.float32(1e-12)
    anisotropy = (largest - smallest) / total          # Westin
    planarity = 2.0 * (middle - smallest) / total
    return anisotropy.astype(np.float32), planarity.astype(np.float32)


# ── extraction ───────────────────────────────────────────────────────────────

def extract_features_at_indices(
    neutron_volume,
    xray_volume,
    indices,
    spec,
    sample_mask=None,
) -> np.ndarray:
    """Feature matrix for the voxels at *indices*.

    *sample_mask* defines the sample for ``geometry='relative'``; without it
    the centre of mass is taken over the whole volume.
    """
    spec = resolve_spec(spec)
    neutron = np.asarray(neutron_volume)
    xray = np.asarray(xray_volume)
    if neutron.shape != xray.shape:
        raise ValueError(
            f"Shape mismatch: neutron {neutron.shape} vs X-ray {xray.shape}"
        )
    if neutron.ndim != 3:
        raise ValueError("Feature extraction requires paired 3-D volumes")

    indices = np.asarray(indices, dtype=np.int64)
    neutron_flat = neutron.reshape(-1).astype(np.float32, copy=False)
    xray_flat = xray.reshape(-1).astype(np.float32, copy=False)
    n = neutron_flat[indices]
    x = xray_flat[indices]

    columns: List[np.ndarray] = []
    if spec.intensity:
        columns += [n, x]
    if spec.projections:
        columns += [
            n / (x + np.float32(1e-6)),
            n + x,
            n - x,
            np.hypot(n, x),
        ]

    if spec.geometry == "absolute":
        z, y, x_coord = np.unravel_index(indices, neutron.shape)
        denominators = [max(size - 1, 1) for size in neutron.shape]
        columns += [
            z.astype(np.float32) / denominators[0],
            y.astype(np.float32) / denominators[1],
            x_coord.astype(np.float32) / denominators[2],
        ]
    elif spec.geometry == "relative":
        z, y, x_coord = np.unravel_index(indices, neutron.shape)
        if sample_mask is None:
            centre = np.array(
                [(size - 1) / 2.0 for size in neutron.shape], dtype=np.float32
            )
        else:
            coordinates = np.nonzero(np.asarray(sample_mask, dtype=bool))
            centre = np.array(
                [float(axis.mean()) if axis.size else 0.0 for axis in coordinates],
                dtype=np.float32,
            )
        scale = np.float32(max(max(neutron.shape) - 1, 1))
        offset_z = (z.astype(np.float32) - centre[0]) / scale
        offset_y = (y.astype(np.float32) - centre[1]) / scale
        offset_x = (x_coord.astype(np.float32) - centre[2]) / scale
        columns += [
            np.sqrt(offset_z ** 2 + offset_y ** 2 + offset_x ** 2),
            offset_z, offset_y, offset_x,
        ]

    for radius in spec.texture_scales:
        for volume in (neutron, xray):
            mean, deviation = local_moments(volume, radius)
            columns.append(mean.reshape(-1)[indices])
            columns.append(deviation.reshape(-1)[indices])

    if spec.laplacian:
        columns.append(
            laplace(neutron.astype(np.float32, copy=False), mode="nearest")
            .reshape(-1)[indices]
        )

    if spec.gradient:
        columns.append(gradient_magnitude(neutron).reshape(-1)[indices])
        columns.append(gradient_magnitude(xray).reshape(-1)[indices])
        columns.append(
            gradient_coherence(neutron, xray).reshape(-1)[indices]
        )

    if spec.structure:
        for radius in spec.texture_scales or (1,):
            anisotropy, planarity = structure_descriptors(neutron, radius)
            columns.append(anisotropy.reshape(-1)[indices])
            columns.append(planarity.reshape(-1)[indices])

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.all(np.isfinite(matrix)):
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix
