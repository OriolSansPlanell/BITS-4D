"""
roi_manager.py - Region of Interest management

Handles polygon and rectangular ROIs for histogram-based segmentation.
Supports both a single "active" ROI (for classic segmentation) and a list
of named class ROIs (for multi-class Random Forest training).
"""

import numpy as np
from matplotlib.path import Path
from typing import Optional, Tuple, List
import json


# Tab-10 colours for the first 10 classes, then cycle
_CLASS_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
]

def _class_color(class_id: int) -> str:
    return _CLASS_COLORS[class_id % len(_CLASS_COLORS)]


class ROIManager:
    """
    Manages Region of Interest (ROI) for segmentation.

    Two layers:
      • Single active ROI  – the ROI currently drawn on the histogram,
        used by the classic segmentation engine (``is_inside_roi``).
      • Named ROI list     – a list of {name, class_id, roi_type,
        points/rectangle, color} dicts used for multi-class RF training.

    ``is_inside_roi()`` returns the union of **every** defined ROI —
    all named class ROIs plus the active one.  This guarantees that the
    region highlighted on the histogram is exactly the region that
    segmentation selects (a previous version silently ignored the active
    ROI once named ROIs existed, so the displayed selection and the
    segmented voxels disagreed).

    All containment tests are evaluated in histogram data coordinates:
    x = neutron intensity, y = X-ray intensity — the same convention the
    histogram canvases use for display and mouse input.
    """

    def __init__(self):
        # --- single active ROI (backward compat) ---
        self.polygon_points: Optional[np.ndarray] = None
        self.rectangle: Optional[Tuple[float, float, float, float]] = None
        self.roi_type: Optional[str] = None  # 'polygon' | 'rectangle'

        # --- named multi-class ROI list ---
        self.named_rois: List[dict] = []
        self._next_class_id: int = 1

    # ─────────────────────────── single active ROI ───────────────────────────

    def set_polygon_roi(self, points: np.ndarray) -> None:
        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 points")
        # Always store an independent float copy so the caller's array cannot
        # be mutated through this ROI (or vice versa).
        self.polygon_points = np.array(points, dtype=float)
        self.roi_type = 'polygon'

    def get_active_vertices(self) -> Optional[np.ndarray]:
        """Snapshot of the active ROI's outline, or None if there is none.

        Returns a fresh array: callers keep these as a record of what was
        segmented, so later edits to the ROI must not alter them.
        """
        if self.roi_type == 'polygon':
            return np.array(self.polygon_points, dtype=float)
        if self.roi_type == 'rectangle':
            x_min, y_min, x_max, y_max = self.rectangle
            return np.array(
                [[x_min, y_min], [x_max, y_min],
                 [x_max, y_max], [x_min, y_max]], dtype=float
            )
        return None

    def set_rectangle_roi(self, x_min: float, y_min: float,
                          x_max: float, y_max: float) -> None:
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid rectangle coordinates")
        self.rectangle = (x_min, y_min, x_max, y_max)
        self.roi_type = 'rectangle'

    def has_roi(self) -> bool:
        """True if any ROI is currently in play (active or a visible class).

        Hidden classes do not count: they are neither drawn nor segmented,
        so with everything hidden and no active ROI there is nothing to do.
        """
        return self.roi_type is not None or len(self.get_visible_named_rois()) > 0

    def is_inside_roi(self, neutron_values: np.ndarray,
                      xray_values: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask for points inside *any* active ROI.

        The mask is the union of every *visible* named class ROI and the
        active ROI (when one is drawn), so it always matches what is shown
        on the histogram canvases.
        """
        if not self.has_roi():
            raise ValueError("No ROI defined")

        result = np.zeros(neutron_values.shape, dtype=bool)
        for roi in self.get_visible_named_rois():
            result |= self._mask_for_named_roi(roi, neutron_values, xray_values)
        if self.roi_type == 'polygon':
            result |= self._polygon_mask(
                self.polygon_points, neutron_values, xray_values
            )
        elif self.roi_type == 'rectangle':
            result |= self._rectangle_mask(
                self.rectangle, neutron_values, xray_values
            )
        return result

    @staticmethod
    def _polygon_mask(points: np.ndarray, x: np.ndarray,
                      y: np.ndarray) -> np.ndarray:
        """Vectorized point-in-polygon test with a bounding-box prefilter.

        ``Path.contains_points`` is O(N·V) in the number of tested points
        and polygon vertices; restricting it to points inside the
        polygon's bounding box makes segmentation of full volumes fast
        when the ROI covers a small part of the intensity range.
        """
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        candidates = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        result = np.zeros(x.shape, dtype=bool)
        if not np.any(candidates):
            return result
        flat_candidates = candidates.ravel()
        test_points = np.column_stack(
            [x.ravel()[flat_candidates], y.ravel()[flat_candidates]]
        )
        inside = Path(points).contains_points(test_points)
        flat_result = result.ravel()
        flat_result[flat_candidates] = inside
        return flat_result.reshape(x.shape)

    @staticmethod
    def _rectangle_mask(rectangle: Tuple[float, float, float, float],
                        x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_min, y_min, x_max, y_max = rectangle
        return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    def get_roi_bounds(self) -> Tuple[float, float, float, float]:
        """Bounding box (x_min, y_min, x_max, y_max) of the active ROI."""
        if self.roi_type is None:
            raise ValueError("No active ROI defined")
        if self.roi_type == 'polygon':
            x_min = self.polygon_points[:, 0].min()
            x_max = self.polygon_points[:, 0].max()
            y_min = self.polygon_points[:, 1].min()
            y_max = self.polygon_points[:, 1].max()
            return (x_min, y_min, x_max, y_max)
        return self.rectangle

    def get_roi_area(self) -> float:
        if not self.has_roi():
            return 0.0
        if self.roi_type == 'rectangle':
            x_min, y_min, x_max, y_max = self.rectangle
            return (x_max - x_min) * (y_max - y_min)
        elif self.roi_type == 'polygon':
            x = self.polygon_points[:, 0]
            y = self.polygon_points[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return 0.0

    def clear_roi(self) -> None:
        """Clear the single active ROI (named ROIs are unaffected)."""
        self.polygon_points = None
        self.rectangle = None
        self.roi_type = None

    # ──────────────────────────── named ROI list ─────────────────────────────

    def add_named_roi(self, name: str, class_id: Optional[int] = None,
                      color: Optional[str] = None) -> int:
        """
        Save the current active ROI as a named class ROI.

        Parameters
        ----------
        name     : human-readable label (e.g. "Lithium", "Electrolyte")
        class_id : integer class label for RF; auto-assigned if None
        color    : hex colour string; auto-assigned if None

        Returns the assigned class_id.
        """
        if self.roi_type is None:
            raise ValueError("No active ROI to save as named class.")

        if class_id is None:
            class_id = self._next_class_id
        self._next_class_id = max(self._next_class_id, class_id) + 1

        if color is None:
            color = _class_color(class_id - 1)

        entry: dict = {
            'name': name,
            'class_id': class_id,
            'roi_type': self.roi_type,
            'color': color,
            'visible': True,
        }
        if self.roi_type == 'polygon':
            entry['points'] = self.polygon_points.copy()
        else:
            entry['rectangle'] = self.rectangle

        self.named_rois.append(entry)
        return class_id

    def remove_named_roi(self, index: int) -> None:
        """Remove named ROI at position *index* in the list."""
        if 0 <= index < len(self.named_rois):
            del self.named_rois[index]

    def clear_named_rois(self) -> None:
        """Remove all named ROIs and reset the class counter."""
        self.named_rois.clear()
        self._next_class_id = 1

    def has_named_rois(self) -> bool:
        return len(self.named_rois) > 0

    def get_named_rois(self) -> List[dict]:
        """Return a copy of the named ROI list (including hidden ones)."""
        return list(self.named_rois)

    def get_visible_named_rois(self) -> List[dict]:
        """Named ROIs that are currently shown and segmented."""
        return [roi for roi in self.named_rois if roi.get('visible', True)]

    def set_named_roi_visible(self, index: int, visible: bool) -> None:
        """Show/hide one class.

        Hidden classes are not drawn on the histogram and not segmented, so
        the display and the segmentation result stay in agreement.
        """
        if 0 <= index < len(self.named_rois):
            self.named_rois[index]['visible'] = bool(visible)

    def take_named_roi(self, index: int) -> dict:
        """Move a stored class back into the active ROI slot for editing.

        The entry is removed from the class list and becomes the active ROI,
        so it is never counted twice. The returned dict still holds its
        ``name``/``class_id``/``color``, which lets the caller restore them
        when saving it back as a class.
        """
        if not 0 <= index < len(self.named_rois):
            raise IndexError(f"No named ROI at index {index}")

        entry = self.named_rois.pop(index)
        if entry['roi_type'] == 'polygon':
            self.set_polygon_roi(np.asarray(entry['points']))
        else:
            self.set_rectangle_roi(*entry['rectangle'])
        return entry

    def _mask_for_named_roi(self, roi: dict,
                             x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask for one entry in named_rois."""
        if roi['roi_type'] == 'polygon':
            return self._polygon_mask(np.asarray(roi['points']), x, y)
        return self._rectangle_mask(roi['rectangle'], x, y)

    def _mask_for_active_roi(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask for the single active ROI (raises if none)."""
        if self.roi_type == 'polygon':
            return self._polygon_mask(self.polygon_points, x, y)
        if self.roi_type == 'rectangle':
            return self._rectangle_mask(self.rectangle, x, y)
        raise ValueError("No active ROI defined")

    def get_multi_class_labels(self,
                                neutron_vol: np.ndarray,
                                xray_vol: np.ndarray) -> np.ndarray:
        """
        Build an integer label array from all defined ROIs.

        Returns an int32 array shaped like the input volumes where:
          0  = background (not covered by any ROI)
          N  = class_id of the named ROI that covers this voxel.

        If two named ROIs overlap, the last one in the list wins.
        An active (unsaved) ROI is included as the next free class id,
        so the labels always cover everything shown on the histogram.
        With no named ROIs, the active ROI alone is class 1.
        """
        labels = np.zeros(neutron_vol.shape, dtype=np.int32)

        visible = self.get_visible_named_rois()
        for roi in visible:
            mask = self._mask_for_named_roi(roi, neutron_vol, xray_vol)
            labels[mask] = roi['class_id']
        if self.roi_type is not None:
            active_class = (
                max(r['class_id'] for r in visible) + 1 if visible else 1
            )
            labels[self._mask_for_active_roi(neutron_vol, xray_vol)] = active_class

        return labels

    def get_named_roi_overlays(self):
        """
        Return overlays list in the format expected by HistogramCanvas:
        [(name, vertices_Nx2, color), ...]

        Only visible classes are returned, matching what is segmented.
        Rectangles are converted to 4-vertex polygons for uniform drawing.
        """
        overlays = []
        for roi in self.get_visible_named_rois():
            color = roi.get('color', '#ff0000')
            label = f"Class {roi['class_id']}: {roi['name']}"
            if roi['roi_type'] == 'polygon':
                vertices = roi['points']
            else:
                x_min, y_min, x_max, y_max = roi['rectangle']
                vertices = np.array([
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max],
                ])
            overlays.append((label, vertices, color))
        return overlays

    # ──────────────────────────── persistence ────────────────────────────────

    def save_to_dict(self) -> dict:
        data: dict = {}
        # Active single ROI
        if self.roi_type == 'polygon':
            data['active'] = {'type': 'polygon',
                              'points': self.polygon_points.tolist()}
        elif self.roi_type == 'rectangle':
            data['active'] = {'type': 'rectangle',
                              'bounds': list(self.rectangle)}
        else:
            data['active'] = {'type': None}

        # Named ROIs
        named = []
        for roi in self.named_rois:
            entry = {k: v for k, v in roi.items()
                     if k not in ('points', 'rectangle')}
            if roi['roi_type'] == 'polygon':
                entry['points'] = roi['points'].tolist()
            else:
                entry['rectangle'] = list(roi['rectangle'])
            named.append(entry)
        data['named_rois'] = named
        return data

    def load_from_dict(self, data: dict) -> None:
        # Support old format (just a 'type' key at root)
        if 'type' in data:
            roi_type = data.get('type')
            if roi_type is None:
                self.clear_roi()
            elif roi_type == 'polygon':
                self.set_polygon_roi(np.array(data['points']))
            elif roi_type == 'rectangle':
                self.set_rectangle_roi(*data['bounds'])
            return

        # New format
        active = data.get('active', {'type': None})
        roi_type = active.get('type')
        if roi_type == 'polygon':
            self.set_polygon_roi(np.array(active['points']))
        elif roi_type == 'rectangle':
            self.set_rectangle_roi(*active['bounds'])
        else:
            self.clear_roi()

        self.named_rois.clear()
        for entry in data.get('named_rois', []):
            roi = dict(entry)
            if roi['roi_type'] == 'polygon':
                roi['points'] = np.array(roi['points'])
            else:
                roi['rectangle'] = tuple(roi['rectangle'])
            self.named_rois.append(roi)
        if self.named_rois:
            self._next_class_id = max(r['class_id'] for r in self.named_rois) + 1

    def save_to_file(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.save_to_dict(), f, indent=2)

    def load_from_file(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.load_from_dict(data)

    def __repr__(self) -> str:
        parts = []
        if self.roi_type:
            parts.append(f"active={self.roi_type}")
        if self.named_rois:
            parts.append(f"named={len(self.named_rois)}")
        return f"ROIManager({', '.join(parts) or 'empty'})"

