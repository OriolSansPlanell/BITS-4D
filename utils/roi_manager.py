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
      • Single active ROI  – backward-compatible, used by the classic
        segmentation engine (``is_inside_roi``).
      • Named ROI list     – a list of {name, class_id, roi_type,
        points/rectangle, color} dicts used for multi-class RF training.

    When named ROIs are present ``has_roi()`` returns True and
    ``is_inside_roi()`` returns the *union* of every named ROI, so the
    classic segmentation pipeline continues to work unchanged.
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
        self.polygon_points = np.array(points)
        self.roi_type = 'polygon'

    def set_rectangle_roi(self, x_min: float, y_min: float,
                          x_max: float, y_max: float) -> None:
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid rectangle coordinates")
        self.rectangle = (x_min, y_min, x_max, y_max)
        self.roi_type = 'rectangle'

    def has_roi(self) -> bool:
        """True if any ROI is defined (single or named)."""
        return self.roi_type is not None or len(self.named_rois) > 0

    def is_inside_roi(self, neutron_values: np.ndarray,
                      xray_values: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask for points inside *any* defined ROI.

        If named ROIs are present, returns the union of all of them.
        Otherwise falls back to the single active ROI.
        """
        if not self.has_roi():
            raise ValueError("No ROI defined")

        if self.named_rois:
            result = np.zeros(neutron_values.shape, dtype=bool)
            for roi in self.named_rois:
                result |= self._mask_for_named_roi(roi, neutron_values, xray_values)
            return result

        # Single-ROI fallback
        if self.roi_type == 'polygon':
            return self._is_inside_polygon(neutron_values, xray_values)
        return self._is_inside_rectangle(neutron_values, xray_values)

    def _is_inside_polygon(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        points = np.column_stack([x.ravel(), y.ravel()])
        path = Path(self.polygon_points)
        return path.contains_points(points).reshape(original_shape)

    def _is_inside_rectangle(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_min, y_min, x_max, y_max = self.rectangle
        return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    def get_roi_bounds(self) -> Tuple[float, float, float, float]:
        if not self.has_roi():
            raise ValueError("No ROI defined")
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
        """Return a copy of the named ROI list."""
        return list(self.named_rois)

    def _mask_for_named_roi(self, roi: dict,
                             x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask for one entry in named_rois."""
        if roi['roi_type'] == 'polygon':
            pts = roi['points']
            original_shape = x.shape
            flat = np.column_stack([x.ravel(), y.ravel()])
            return Path(pts).contains_points(flat).reshape(original_shape)
        else:
            x_min, y_min, x_max, y_max = roi['rectangle']
            return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    def get_multi_class_labels(self,
                                neutron_vol: np.ndarray,
                                xray_vol: np.ndarray) -> np.ndarray:
        """
        Build an integer label array from all named ROIs.

        Returns a (Z, Y, X) int32 array where:
          0  = background (not covered by any named ROI)
          N  = class_id of the named ROI that covers this voxel.

        If two named ROIs overlap, the last one in the list wins.
        If no named ROIs are defined but a single active ROI exists,
        it is treated as class 1 with everything else as background.
        """
        labels = np.zeros(neutron_vol.shape, dtype=np.int32)

        if self.named_rois:
            for roi in self.named_rois:
                mask = self._mask_for_named_roi(roi, neutron_vol, xray_vol)
                labels[mask] = roi['class_id']
        elif self.roi_type is not None:
            # Single active ROI → class 1
            mask = self.is_inside_roi(neutron_vol, xray_vol)
            labels[mask] = 1

        return labels

    def get_named_roi_overlays(self):
        """
        Return overlays list in the format expected by HistogramCanvas:
        [(name, vertices_Nx2, color), ...]

        Rectangles are converted to 4-vertex polygons for uniform drawing.
        """
        overlays = []
        for roi in self.named_rois:
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

