# Bug-fix and cleanup notes

## K-means dtype crash on float32 volumes (latest)

**Symptom:** 3-D Auto-Detect crashed in `KMeans3D.cluster_volume` with
"Buffer dtype mismatch, expected 'const float' but got 'double'".

**Root cause:** the K-means model is fitted on float64 feature samples, but
prediction chunks kept the volume's dtype. Integer volumes were silently
upcast to float64 by StandardScaler; float32 volumes — which the binned
display copies introduced — pass through as float32, and scikit-learn's
compiled kernel rejects float32 data against float64 cluster centers.

**Fix:** prediction chunks are cast to float64 to match the fitted model.
Regression-tested with float32 volumes (the test reproduces the crash
without the fix).

## Big-endian TIFF crash

**Symptom:** loading certain TIFF datasets crashed histogram computation on
the GPU with "given numpy array has byte order different from the native
byte order".

**Root cause:** big-endian TIFFs (common from some instruments and older
ImageJ exports) yield NumPy arrays in the file's byte order; PyTorch cannot
build tensors from non-native arrays.

**Fix:** the loader normalizes in-RAM data to native byte order at load
time via an in-place byteswap (no extra memory), and the GPU histogram path
converts each chunk to native float64 before tensor conversion — covering
read-only big-endian memory maps that cannot be swapped in place. Values
are bit-identical after conversion (regression-tested against a real
big-endian TIFF round trip).

## Big-dataset mode

- **Histograms computed once, served from memory.** After loading, every
  timepoint's local histogram is precomputed into a cache sized to hold the
  whole series; time navigation and the evolution export read from memory
  instead of re-scanning volumes.
- **Median-binned display volumes.** When one volume exceeds 1 GiB
  (`config.DISPLAY_MAX_VOLUME_BYTES`), all timepoints are binned by block
  median with the smallest factor that fits the budget. The slice viewer,
  spatial selections, statistics, and time-series tracking work on the
  binned copies; segmentation, RF, and exports stay at full resolution.
  Histogram-space ROIs bridge the two, so selections made on binned display
  data segment the original voxels exactly.
- **Masks scaled to the display grid.** Full-resolution segmentation layers
  are block-any binned (cached) before overlay so they align with the
  binned slices; display-space cluster masks are upscaled (block
  replication + edge padding) before RF training.
- **Histogram evolution export.** *Analytics → Histogram Evolution vs First
  Timepoint* saves an image of log10(h_t+1) − log10(h_0+1) per timepoint on
  a shared diverging scale (red = bins gaining voxels, blue = losing),
  plus the T0 reference panel.
- **Responsiveness.** Slice-slider redraws are debounced (30 ms) so
  dragging tracks smoothly; a duplicated base-image render per timepoint
  switch was removed; timepoint changes now cost one cached-histogram fetch
  plus one binned-slice render.


This pass audited the whole codebase for correctness, performance, and
maintainability issues. Summary of what changed and why.

## Bug fixes

### Histogram selection vs. segmentation mismatch (the reported bug)

**Symptom:** the region selected on the histogram and the region actually
segmented could differ.

**Root cause:** once at least one *named class ROI* existed,
`ROIManager.is_inside_roi()` returned the union of the named ROIs only — a
freshly drawn (active) ROI was displayed on the histogram but silently
ignored by segmentation. The GUI's multi-class segment actions had the same
blind spot: they iterated only over named ROIs.

**Fix:**

- `ROIManager.is_inside_roi()` now returns the union of **all** ROIs —
  named classes plus the active one — so segmentation always matches the
  displayed selection.
- `BiTS4DMainWindow._enumerate_roi_specs()` (used by *Segment Current* and
  *Segment All*) enumerates named ROIs **and** the active ROI, which becomes
  its own "Active ROI" layer.
- `ROIManager.get_multi_class_labels()` includes the active ROI as the next
  free class id.
- Regression tests: `tests/test_roi_manager.py`,
  `tests/test_selection_segmentation_consistency.py` (the latter also locks
  in the display ↔ data coordinate convention end to end: histogram peak
  position on screen → rectangle drawn there → exact voxel set segmented).

### Histogram overlay showed a smaller, rounder region than the drawn ROI

**Symptom:** after segmenting, the dashed overlay drawn back on the histogram
was a rounder, smaller shape than the rectangle/polygon the user drew,
suggesting the wrong voxels had been selected.

**Root cause:** the voxel selection itself was exact, but
`_update_rf_histogram_overlays` did not draw the ROI — it re-derived a
**convex hull of the segmented voxels' intensity pairs**, trimmed to the
2nd–98th percentile per axis and subsampled to 5 000 points
(`create_convex_hull_roi_3d(percentile=98)`). The hull of the *data inside*
the ROI is necessarily smaller than the ROI (sparse corners drop out, tails
are trimmed), so the display contradicted the selection.

**Fix:** every layer created from a histogram ROI now records its exact
outline (`segmentation_layer_shapes`), and the histogram overlay draws that
recorded shape. The hull remains only as a fallback for layers that have no
drawn shape (RF predictions, Otsu, K-means classes), where it is an honest
approximation of where the class lives in histogram space.

### Old ROIs reappeared and stacked when re-segmenting

**Symptom:** draw ROI → Segment → Clear Active ROI → draw a new ROI →
Segment: the old ROI's segmentation reappeared alongside the new one; a
third repetition showed three regions.

**Root cause:** segmentation layers are intentionally kept when the active
ROI is cleared, but the single-ROI segment path silently *appended* a new
layer on every press, so every past ROI accumulated with no way to discard
them.

**Fix:** when previous layers exist for the timepoint, *Segment Current* now
asks whether to **keep** them (the new ROI is added as an extra, uniquely
named layer) or **replace** them (previous layers and their recorded
outlines are dropped), with Cancel leaving everything untouched. Batch
segmentation and Otsu replace same-name layers, staying idempotent.

### Startup crash without a CUDA GPU

`_create_menu_bar` imported `QActionGroup` inside an `if self.available_gpus:`
branch, making the name an unbound local when no GPU was present, and the
GPU status check called `force_cpu_action.setChecked(True)` before the status
bar existed, firing a toggle handler that dereferenced it. On a machine
without CuPy/CUDA the window crashed during construction. The import was
moved to function scope and the initial check no longer fires the signal.

### CPU/GPU toggle destroyed histogram state

Switching *Force CPU Processing* or the GPU device re-instantiated
`HistogramEngine4D`, losing the global data range and all cached histograms;
subsequent local-histogram updates failed with "Must compute global histogram
first". The handlers now flip `engine.use_gpu` in place — CPU and GPU produce
identical counts, so caches stay valid and nothing is recomputed.

### Saved selections lost their histogram ROI

`_save_current_selection` read `roi_manager.polygon_roi` /
`rectangle_roi` / `get_rectangle_coords()` — attributes that do not exist on
`ROIManager` — so the histogram ROI was silently dropped from every saved
selection. It now reads the real attributes (`polygon_points`, `rectangle`)
and stores rectangles as 4-vertex polygons.

### Duplicated 4D-mode menu action

A stray duplicated block re-registered `mode_4d_action` and connected its
`triggered` signal a second time, so switching to 4D mode ran the handler
twice (including its dataset-clearing confirmation logic). Removed.

### Cancelled operations re-ran synchronously

`load_dataset` and the legacy `_compute_global_histogram` retried the whole
operation synchronously on the GUI thread whenever the progress-dialog worker
returned `None` — including when the user had *cancelled*. The fallbacks were
removed; cancellation now simply cancels.

### Clearing the active ROI deleted segmentation layers

The legacy `_on_roi_updated` popped all stored segmentation layers for the
current timepoint whenever the active ROI was cleared. It now only removes
the transient slice highlight (matching the runtime-fix override, which is
kept as the canonical behaviour).

### Repeated segmentation stacked duplicate layers

Pressing *Segment Current* / *Segment All* repeatedly appended a new copy of
every class layer each time. Layers are now replaced by name, making both
actions idempotent.

## Performance

- **Polygon containment:** `ROIManager` now prefilters candidate voxels with
  the polygon's bounding box before running
  `matplotlib.path.Path.contains_points` (~19× faster on a 3.3M-voxel volume
  with a typical small ROI; exact results, verified by test).
- **Histogram canvas redraws:** `HistogramCanvas.update_plot` performed four
  full draw cycles per update (plus `QTimer` re-draws, `processEvents`,
  duplicated ROI drawing, and per-update debug printing). It now draws once
  via `draw_idle()`.
- **ROI finalisation** no longer triggers a second redraw of the same canvas;
  the `roi_updated` signal chain performs the single refresh.
- **Auto Range** updates both spinboxes with signals blocked, applying the
  new range once instead of three times.
- **Backend switching** no longer recomputes histograms (see above).
- Hot-path debug printing (slice display, histogram updates, statistics
  refresh, timepoint navigation) was removed; error reporting is kept.

## Cleanup / maintainability

- Removed committed editor backups (`main_window.py.kmeans-rf-backup`,
  `*.before-*`) and `__pycache__` directories; added a `.gitignore`.
- `gui/runtime_fixes.py`: collapsed the confusing double definition of
  `apply_runtime_fixes` into one idempotent entry point, and shared the ROI
  enumeration helper with the main window.
- Removed the redundant *Sync to Global* button (its only effect was a
  redraw that already happens on every ROI update).
- Fixed misleading tuple unpacking in `_auto_detect_3d`
  (`cluster_volume` returns stats, not the model).
- `requirements.txt` now reflects real usage (scikit-image is required for
  Otsu; torch is the actual GPU dependency).
- `HistogramEngine4D.precompute_all_local_histograms` forwards its
  `cancel_check` into each per-timepoint computation.

## Test suite

`python -m pytest` — 20 tests covering: chunked histogram equality with the
NumPy reference, ROI containment and union semantics, bounding-box prefilter
equivalence, selection↔segmentation parity end to end, batch-vs-single
segmentation equality, ROI persistence round-trips, K-means and RF pipeline
behaviour, cancellation, and memmap loading.
