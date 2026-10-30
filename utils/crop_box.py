"""Virtual crop: an axis-aligned box of voxels that everything else can follow.

A crop is drawn as a rectangle on the slice shown in the viewer and
propagated along the view axis — the axis perpendicular to that slice,
"the stack axis" of whatever plane is displayed. It is virtual: no voxel is
copied or deleted; the box is a set of index ranges that the histogram
engine, the segmentation engine, the axial profile and the viewer's
overlays all read.

Conventions (see docs/architecture.md): volumes are ``(Z, Y, X)``; the slice
viewer shows, for view axis

* ``'z'`` — the ``(Y, X)`` plane: image rows are Y, image columns are X;
* ``'y'`` — the ``(Z, X)`` plane: rows are Z, columns are X;
* ``'x'`` — the ``(Z, Y)`` plane: rows are Z, columns are Y.

The viewer may display median-binned copies of the data (``bin_factor``
> 1); a rectangle drawn there is converted to full-resolution indices here,
so a ``CropBox`` is always in full-resolution voxel coordinates. Ranges are
half-open ``[lo, hi)`` like NumPy slices.

Nothing here imports Qt, so the whole crop logic is testable headless.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

AXIS_INDEX: Dict[str, int] = {'z': 0, 'y': 1, 'x': 2}

# For each view axis, the volume axes shown as (image rows, image columns)
PLANE_AXES: Dict[str, Tuple[str, str]] = {
    'z': ('y', 'x'),
    'y': ('z', 'x'),
    'x': ('z', 'y'),
}

Range = Tuple[int, int]


def _check_axis(axis: str) -> str:
    if axis not in AXIS_INDEX:
        raise ValueError(f"axis must be 'z', 'y' or 'x', got {axis!r}")
    return axis


def _clip_range(lo: int, hi: int, size: int) -> Range:
    """Clip a half-open range to ``[0, size)`` keeping at least one voxel."""
    lo = int(max(0, min(lo, size - 1)))
    hi = int(max(lo + 1, min(hi, size)))
    return lo, hi


@dataclass(frozen=True)
class CropBox:
    """An axis-aligned box of full-resolution voxels, ``[lo, hi)`` per axis."""

    shape: Tuple[int, int, int]
    z: Range
    y: Range
    x: Range

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def full(cls, shape: Sequence[int]) -> "CropBox":
        """The box covering the whole volume."""
        shape = cls._shape3(shape)
        return cls(shape, (0, shape[0]), (0, shape[1]), (0, shape[2]))

    @classmethod
    def from_plane_rectangle(
        cls,
        axis: str,
        rect: Sequence[float],
        shape: Sequence[int],
        bin_factor: int = 1,
        along: Optional[Range] = None,
    ) -> "CropBox":
        """Box from a rectangle drawn on the displayed plane.

        Parameters
        ----------
        axis
            The view axis of the slice the rectangle was drawn on.
        rect
            ``(x1, y1, x2, y2)`` in *display* pixel coordinates of that
            plane (``x`` = image column, ``y`` = image row; any corner
            order). Pixel ``p`` covers ``[p, p + 1)``, so a rectangle from
            2.3 to 5.6 takes pixels 2..5 inclusive.
        shape
            Full-resolution volume shape ``(Z, Y, X)``.
        bin_factor
            Display binning of the plane the rectangle was drawn on; display
            pixel ``p`` corresponds to full-resolution indices
            ``[p·f, (p + 1)·f)``.
        along
            Optional ``(lo, hi)`` full-resolution range along the view axis;
            the full extent when omitted — the rectangle is propagated
            through the whole stack.
        """
        axis = _check_axis(axis)
        shape = cls._shape3(shape)
        f = int(bin_factor)
        if f < 1:
            raise ValueError("bin_factor must be a positive integer")
        x1, y1, x2, y2 = (float(v) for v in rect)
        col_lo, col_hi = math.floor(min(x1, x2)), math.ceil(max(x1, x2))
        row_lo, row_hi = math.floor(min(y1, y2)), math.ceil(max(y1, y2))
        if col_hi <= col_lo:
            col_hi = col_lo + 1
        if row_hi <= row_lo:
            row_hi = row_lo + 1

        row_axis, col_axis = PLANE_AXES[axis]
        ranges = {
            row_axis: (row_lo * f, row_hi * f),
            col_axis: (col_lo * f, col_hi * f),
            axis: (0, shape[AXIS_INDEX[axis]]) if along is None
            else (int(along[0]), int(along[1])),
        }
        return cls(
            shape,
            _clip_range(*ranges['z'], shape[0]),
            _clip_range(*ranges['y'], shape[1]),
            _clip_range(*ranges['x'], shape[2]),
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "CropBox":
        return cls(
            tuple(int(v) for v in data['shape']),
            tuple(int(v) for v in data['z']),
            tuple(int(v) for v in data['y']),
            tuple(int(v) for v in data['x']),
        )

    @staticmethod
    def _shape3(shape: Sequence[int]) -> Tuple[int, int, int]:
        shape = tuple(int(v) for v in shape)
        if len(shape) == 4:            # (T, Z, Y, X) given: use the volume part
            shape = shape[1:]
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError(f"shape must be (Z, Y, X) with positive sizes, got {shape}")
        return shape  # type: ignore[return-value]

    def __post_init__(self):
        shape = self._shape3(self.shape)
        object.__setattr__(self, 'shape', shape)
        for name, size in zip(('z', 'y', 'x'), shape):
            lo, hi = getattr(self, name)
            lo, hi = int(lo), int(hi)
            if not (0 <= lo < hi <= size):
                raise ValueError(
                    f"range {name}=({lo}, {hi}) is not inside [0, {size})"
                )
            object.__setattr__(self, name, (lo, hi))

    # ── derived forms ─────────────────────────────────────────────────────
    def bounds(self, axis: str) -> Range:
        return getattr(self, _check_axis(axis))

    def with_range(self, axis: str, lo: int, hi: int) -> "CropBox":
        """The same box with a new ``[lo, hi)`` range along *axis*."""
        axis = _check_axis(axis)
        size = self.shape[AXIS_INDEX[axis]]
        if hi < lo:
            lo, hi = hi, lo
        new = {name: getattr(self, name) for name in ('z', 'y', 'x')}
        new[axis] = _clip_range(int(math.floor(lo)), int(math.ceil(hi)), size)
        return CropBox(self.shape, new['z'], new['y'], new['x'])

    def with_full_range(self, axis: str) -> "CropBox":
        axis = _check_axis(axis)
        return self.with_range(axis, 0, self.shape[AXIS_INDEX[axis]])

    def slices(self) -> Tuple[slice, slice, slice]:
        return (slice(*self.z), slice(*self.y), slice(*self.x))

    def crop(self, volume: np.ndarray) -> np.ndarray:
        """A view of the voxels inside the box (3-D, or 4-D with time first)."""
        volume = np.asarray(volume) if not hasattr(volume, 'shape') else volume
        if volume.ndim == 4:
            return volume[(slice(None),) + self.slices()]
        if volume.ndim != 3:
            raise ValueError(f"expected a 3-D or 4-D volume, got {volume.ndim}-D")
        return volume[self.slices()]

    def mask(self) -> np.ndarray:
        """Boolean ``(Z, Y, X)`` mask of the box."""
        m = np.zeros(self.shape, dtype=bool)
        m[self.slices()] = True
        return m

    def embed(self, cropped: np.ndarray, fill=0) -> np.ndarray:
        """Place an array shaped like the box back into a full-size volume."""
        cropped = np.asarray(cropped)
        if cropped.shape != self.size_zyx:
            raise ValueError(
                f"array shape {cropped.shape} is not the box shape {self.size_zyx}"
            )
        out = np.full(self.shape, fill, dtype=cropped.dtype)
        out[self.slices()] = cropped
        return out

    @property
    def size_zyx(self) -> Tuple[int, int, int]:
        return (self.z[1] - self.z[0], self.y[1] - self.y[0], self.x[1] - self.x[0])

    @property
    def num_voxels(self) -> int:
        return int(np.prod(self.size_zyx))

    @property
    def fraction(self) -> float:
        return self.num_voxels / float(np.prod(self.shape))

    def is_full(self) -> bool:
        return self.size_zyx == self.shape

    def rectangle_on_plane(self, axis: str, bin_factor: int = 1
                           ) -> Tuple[float, float, float, float]:
        """The box's footprint on the plane viewed along *axis*.

        Returns ``(x1, y1, x2, y2)`` in display pixel coordinates (columns,
        rows) of that plane, so the viewer can draw it on any of the three
        planes: on the plane the crop was drawn on it is the rectangle, on
        the other two it is the band the propagated box cuts through them.
        """
        axis = _check_axis(axis)
        f = float(bin_factor)
        row_axis, col_axis = PLANE_AXES[axis]
        r_lo, r_hi = self.bounds(row_axis)
        c_lo, c_hi = self.bounds(col_axis)
        return (c_lo / f, r_lo / f, c_hi / f, r_hi / f)

    def to_dict(self) -> Dict:
        return {'shape': list(self.shape), 'z': list(self.z),
                'y': list(self.y), 'x': list(self.x)}

    def describe(self) -> str:
        parts = []
        for name in ('z', 'y', 'x'):
            lo, hi = getattr(self, name)
            size = self.shape[AXIS_INDEX[name]]
            parts.append(f"{name} {lo}–{hi - 1}" if (lo, hi) != (0, size)
                         else f"{name} all")
        return f"{', '.join(parts)} ({100 * self.fraction:.1f}% of the volume)"


def axial_profile(
    neutron: np.ndarray,
    xray: np.ndarray,
    axis: str,
    box: Optional[CropBox] = None,
    statistic: str = 'mean',
) -> Dict[str, np.ndarray]:
    """Per-slice statistic of both modalities along *axis*, inside *box*.

    This is the one-dimensional view of a layered sample: for a planar
    stack (electrodes, separators, foils) the mean of each slice through
    the pellet traces the layers directly, and a change of material shows
    as a step in one or both curves. Slices are read one at a time, so a
    memory-mapped volume is never loaded whole.

    Returns a dict with ``index`` (full-resolution slice indices along the
    axis), ``neutron`` and ``xray`` (the statistic per slice, NaN where the
    slice has no finite voxel), ``neutron_std`` / ``xray_std`` (standard
    deviation per slice) and ``count`` (finite voxels per slice).
    """
    axis = _check_axis(axis)
    if neutron.shape != xray.shape or neutron.ndim != 3:
        raise ValueError("neutron and xray must be 3-D volumes of the same shape")
    if box is None:
        box = CropBox.full(neutron.shape)
    elif tuple(box.shape) != tuple(neutron.shape):
        raise ValueError(f"box shape {box.shape} does not match volume {neutron.shape}")
    if statistic not in ('mean', 'median'):
        raise ValueError("statistic must be 'mean' or 'median'")

    ax = AXIS_INDEX[axis]
    lo, hi = box.bounds(axis)
    index = np.arange(lo, hi)
    out = {k: np.full(index.size, np.nan) for k in
           ('neutron', 'xray', 'neutron_std', 'xray_std')}
    count = np.zeros(index.size, dtype=np.int64)
    sl = list(box.slices())
    reducer = np.nanmean if statistic == 'mean' else np.nanmedian
    with np.errstate(all='ignore'):
        for k, i in enumerate(index):
            sl[ax] = i
            n = np.asarray(neutron[tuple(sl)], dtype=np.float64)
            x = np.asarray(xray[tuple(sl)], dtype=np.float64)
            finite = np.isfinite(n) & np.isfinite(x)
            c = int(np.count_nonzero(finite))
            count[k] = c
            if c:
                n, x = n[finite], x[finite]
                out['neutron'][k] = reducer(n)
                out['xray'][k] = reducer(x)
                out['neutron_std'][k] = n.std()
                out['xray_std'][k] = x.std()
    return {'axis': axis, 'index': index, 'count': count, **out}
