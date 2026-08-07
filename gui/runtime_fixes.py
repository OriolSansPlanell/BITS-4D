"""Targeted runtime fixes kept separate from the legacy main-window module.

The GUI is currently monolithic.  These overrides isolate correctness fixes
without copying the entire window implementation and can be removed once the
window is split into controllers and services.
"""

from __future__ import annotations

from typing import Type

import numpy as np

_APPLIED = False


def _apply_correctness_fixes(main_window_cls: Type, slice_viewer_cls: Type) -> None:
    """Install behavioural corrections on the legacy window classes."""
    # Backward-compatible alias for a typo in the 3-D univariate region-growing
    # path.  The canonical attribute remains ``view_mode``.
    if not hasattr(slice_viewer_cls, "current_view_mode"):
        slice_viewer_cls.current_view_mode = property(
            lambda self: self.view_mode,
            lambda self, value: setattr(self, "view_mode", value),
        )

    def _on_roi_updated(self):
        roi_manager = self.dual_histogram.get_roi_manager()
        has_roi = roi_manager.has_roi()

        # Clearing the active editor ROI must not delete computed segmentation
        # layers.  It only removes the transient slice highlight when no saved
        # ROI remains.
        if not has_roi and self.dataset is not None:
            self.slice_viewer._clear_highlight()

        self.segment_current_btn.setEnabled(has_roi)
        self.segment_all_btn.setEnabled(has_roi and self.mode == "4D")
        self.selection_manager.enable_save_button(has_roi)
        if has_roi:
            self.status_bar.showMessage("ROI defined - ready to segment")
        elif not any(self.segmentation_masks.values()):
            self.export_current_btn.setEnabled(False)
            self.export_all_btn.setEnabled(False)

    main_window_cls._on_roi_updated = _on_roi_updated

    def _segment_all_volumes(self):
        from PyQt5.QtWidgets import QMessageBox
        from utils.progress_dialog import run_with_progress
        from utils.roi_manager import ROIManager

        if not self.dataset:
            return
        roi_manager = self.dual_histogram.get_roi_manager()
        if not roi_manager.has_roi():
            QMessageBox.warning(self, "No ROI", "Please define an ROI first")
            return
        if (
            QMessageBox.question(
                self,
                "Confirm Batch Segmentation",
                f"Segment all {self.dataset.num_timepoints} timepoints?",
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        # Segment every ROI shown on the histogram: all named class ROIs plus
        # the active (unsaved) one, so batch results match the display.
        if roi_manager.has_named_rois():
            roi_specs = self._enumerate_roi_specs(roi_manager)
        else:
            roi_specs = None

        def operation(progress_callback):
            results: dict[int, list[tuple[np.ndarray, object, str]]] = {}
            timepoints = self.dataset.num_timepoints
            class_count = len(roi_specs) if roi_specs else 1
            total_steps = timepoints * class_count
            completed = 0

            for timepoint in range(timepoints):
                neutron, xray = self.dataset.get_volume_at_time(timepoint)
                layers = []
                if roi_specs:
                    for roi in roi_specs:
                        manager = ROIManager()
                        if roi["roi_type"] == "polygon":
                            manager.set_polygon_roi(roi["points"])
                        else:
                            manager.set_rectangle_roi(*roi["rectangle"])
                        mask = self.segmentation_engine.segment_volume(
                            neutron, xray, manager
                        )
                        layers.append((mask, roi["color"], roi["name"]))
                        completed += 1
                        progress_callback(
                            int(100 * completed / total_steps),
                            f"T={timepoint}: {roi['name']}",
                        )
                else:
                    mask = self.segmentation_engine.segment_volume(
                        neutron, xray, roi_manager
                    )
                    color = self._next_roi_color(roi_manager, 0)
                    name = self._current_roi_name(roi_manager)
                    layers.append((mask, color, name))
                    completed += 1
                    progress_callback(
                        int(100 * completed / total_steps),
                        f"Segmented timepoint {timepoint + 1}/{timepoints}",
                    )
                results[timepoint] = layers
            return results

        segmented = run_with_progress(
            self,
            "Batch Segmentation",
            f"Segmenting {self.dataset.num_timepoints} timepoints...",
            operation,
        )
        if segmented is None:
            return

        import matplotlib.colors as mcolors

        active_vertices = self._active_roi_vertices(roi_manager)
        for timepoint, layers in segmented.items():
            existing = self.segmentation_masks.get(timepoint, [])
            new_names = {name for _mask, _color, name in layers}
            # Replace earlier layers with the same name so repeated batch runs
            # stay idempotent instead of stacking duplicate overlays.
            destination = [
                layer for layer in existing if layer[2] not in new_names
            ]
            for mask, color, name in layers:
                try:
                    red, green, blue, _ = mcolors.to_rgba(color)
                    color = (red, green, blue, 0.5)
                except Exception:
                    color = self._OVERLAY_COLORS[len(destination) % len(self._OVERLAY_COLORS)]
                destination.append((mask, color, name))
            self.segmentation_masks[timepoint] = destination

            # Record the exact histogram outline of each ROI-derived layer so
            # the histogram overlay shows the drawn shape, not a derived hull.
            if roi_specs:
                for roi in roi_specs:
                    self._record_layer_shape(
                        timepoint, roi["name"], self._roi_spec_vertices(roi)
                    )
            elif active_vertices is not None:
                for _mask, _color, name in layers:
                    self._record_layer_shape(timepoint, name, active_vertices)

        self.export_current_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)
        self._update_current_timepoint(self.dataset.current_timepoint)
        QMessageBox.information(
            self,
            "Segmentation Complete",
            f"Segmented {len(segmented)} timepoint(s) with "
            f"{len(roi_specs) if roi_specs else 1} class layer(s).",
        )

    main_window_cls._segment_all_volumes = _segment_all_volumes

    def _refresh_histograms(self):
        """Recompute histograms only when no valid global histogram exists.

        CPU and GPU accumulation produce identical counts, so a plain
        backend switch keeps every cached histogram valid.
        """
        if self.dataset is None or self.histogram_engine is None:
            return
        if self.global_histogram is not None:
            return
        self._compute_global_histogram()
        if self.global_histogram is not None:
            self._update_current_timepoint(self.dataset.current_timepoint)

    # GPU controls differ between legacy BiTS versions. Install compatible
    # handlers even when the original main-window class does not define them.
    original_cpu_toggle = getattr(
        main_window_cls, "_on_cpu_gpu_toggled", None
    )

    def _on_cpu_gpu_toggled(self, force_cpu=False):
        force_cpu = bool(force_cpu)

        if original_cpu_toggle is not None:
            original_cpu_toggle(self, force_cpu)
        else:
            self.force_cpu = force_cpu
            engine = getattr(self, "histogram_engine", None)

            if engine is not None:
                try:
                    import torch
                    engine.use_gpu = bool(
                        not force_cpu and torch.cuda.is_available()
                    )
                except (ImportError, RuntimeError):
                    engine.use_gpu = False

            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                mode = "CPU" if force_cpu else "GPU if available"
                status_bar.showMessage(f"Processing mode: {mode}")

        _refresh_histograms(self)

    main_window_cls._on_cpu_gpu_toggled = _on_cpu_gpu_toggled

    original_gpu_change = getattr(
        main_window_cls, "_on_gpu_device_changed", None
    )

    def _on_gpu_device_changed(self, gpu_id=0):
        gpu_id = int(gpu_id)

        if original_gpu_change is not None:
            original_gpu_change(self, gpu_id)
        else:
            self.gpu_device = gpu_id

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_id)
            except (ImportError, RuntimeError, ValueError):
                pass

            engine = getattr(self, "histogram_engine", None)
            if engine is not None and not getattr(
                self, "force_cpu", False
            ):
                try:
                    import torch
                    engine.use_gpu = bool(torch.cuda.is_available())
                except (ImportError, RuntimeError):
                    engine.use_gpu = False

            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(
                    f"Selected GPU device {gpu_id}"
                )

        _refresh_histograms(self)

    main_window_cls._on_gpu_device_changed = _on_gpu_device_changed

def apply_runtime_fixes(main_window_cls: Type, slice_viewer_cls: Type) -> None:
    """Apply all runtime overrides exactly once (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    _apply_correctness_fixes(main_window_cls, slice_viewer_cls)

    from utils.cancellation import OperationCancelled, OperationFailed
    from utils.progress_dialog import run_with_progress
    
    def _remove_qt_checked_argument(args, kwargs):
        """
        Remove the optional boolean emitted by QAction.triggered and
        QPushButton.clicked when wrapping zero-argument Qt slots.
        """
        if not kwargs and len(args) == 1 and isinstance(args[0], bool):
            return (), {}

        return args, kwargs

    def _handle_operation_exception(self, exc: BaseException) -> None:
        if isinstance(exc, OperationCancelled):
            message = "Operation cancelled"
        else:
            message = "Operation failed; see the error dialog/log"
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(message)

    def _guard(method):
        def guarded(self, *args, **kwargs):
            args, kwargs = _remove_qt_checked_argument(args, kwargs)
    
            try:
                return method(self, *args, **kwargs)
            except (OperationCancelled, OperationFailed) as exc:
                _handle_operation_exception(self, exc)
                return None
    
        guarded.__name__ = getattr(method, "__name__", "guarded_operation")
        guarded.__doc__ = getattr(method, "__doc__")
        return guarded

    # Replace global histogram computation so worker cancellation/failure never
    # triggers the legacy synchronous retry path.
    def _compute_global_histogram(self):
        if self.dataset is None or self.histogram_engine is None:
            return None

        def operation(progress_callback):
            return self.histogram_engine.compute_global_histogram(
                self.dataset.neutron_data,
                self.dataset.xray_data,
                progress_callback=progress_callback,
            )

        try:
            histogram = run_with_progress(
                self,
                "Computing Global Histogram",
                "Analyzing the dataset...",
                operation,
            )
        except (OperationCancelled, OperationFailed) as exc:
            self._bits_histogram_abort = exc
            self.global_histogram = None
            return None

        self._bits_histogram_abort = None
        self.global_histogram = histogram
        self.dual_histogram.set_global_histogram(histogram)
        self.status_bar.showMessage("Global histogram computed")
        return histogram

    main_window_cls._compute_global_histogram = _compute_global_histogram

    original_load_dataset = main_window_cls.load_dataset

    def load_dataset(self, *args, **kwargs):
        self._bits_histogram_abort = None
        args, kwargs = _remove_qt_checked_argument(args, kwargs)
        try:
            result = original_load_dataset(self, *args, **kwargs)
        except (OperationCancelled, OperationFailed) as exc:
            _handle_operation_exception(self, exc)
            return None

        # The legacy loader catches histogram exceptions internally.  Use the
        # explicit abort flag set by the replacement method to roll back the
        # partially initialized dataset rather than continuing with stale state.
        if self._bits_histogram_abort is not None:
            _handle_operation_exception(self, self._bits_histogram_abort)
            self.dataset = None
            self.histogram_engine = None
            self.global_histogram = None
            self.segmentation_masks.clear()
            self._clear_layer_shapes()
            self.time_navigation.setEnabled(False)
            self.segment_current_btn.setEnabled(False)
            self.segment_all_btn.setEnabled(False)
            return None
        return result

    main_window_cls.load_dataset = load_dataset

    # Every remaining slot that invokes run_with_progress now handles explicit
    # cancellation/failure uniformly instead of leaking a traceback into Qt.
    for name in (
        "_segment_current_volume",
        "_segment_all_volumes",
        "_run_otsu_segment",
        "_rf_train",
        "_rf_predict_current",
        "_rf_predict_all",
    ):
        method = getattr(main_window_cls, name, None)
        if method is not None:
            setattr(main_window_cls, name, _guard(method))
