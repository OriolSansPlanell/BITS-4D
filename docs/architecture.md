# BiTS 4D architecture

This document describes how the application is organised, the data and
coordinate conventions every module must respect, and how to extend it.

## Module map

```
main.py
└── gui/                          PyQt5 presentation layer
    ├── main_window.py            BiTS4DMainWindow + SliceViewerWidget
    ├── runtime_fixes.py          Behavioural overrides applied at import time
    ├── dual_histogram_widget.py  Global/local histogram canvases + ROI tools
    ├── selection_manager.py      Saved selections (mask + histogram ROI)
    ├── statistics_panel.py       Live per-selection statistics
    └── time_navigation_widget.py Timepoint slider / playback

Computation layer (GUI-independent, scriptable):
    data/data_loader_4d.py        Memory-aware TIFF loading (memmap for big data)
    data/dataset_4d.py            Dataset4D container (T, Z, Y, X)
    histograms/histogram_engine_4d.py
                                  Chunked CPU/GPU 2-D histogram accumulation
    segmentation/segmentation_engine_4d.py
                                  ROI → voxel mask application + statistics
    segmentation/random_forest_4d.py
                                  Memory-bounded RF training / prediction
    segmentation/kmeans_class_conversion.py
                                  K-means clusters → RF training layers
    utils/roi_manager.py          Histogram-space ROI state + containment tests
    utils/clustering_3d.py        Scale-aware K-means on paired volumes
    utils/region_growing.py       2-D flood fill (uni/bivariate)
    utils/region_growing_3d.py    3-D connected region growing
    utils/value_extractor.py      Spatial rectangle → intensity values
    utils/progress_dialog.py      Worker thread + cancellable progress dialogs
    utils/cancellation.py         CancellationToken, OperationCancelled/Failed
    utils/selection_library.py    Selection persistence + CSV/Excel export
    utils/config.py               Application-wide defaults
```

The GUI never re-implements numerics: every widget delegates to the
computation layer, which is why the whole pipeline can also be driven from a
script or a notebook (see the README's "Library usage" section).

## Data conventions

- Volumes are NumPy arrays shaped `(T, Z, Y, X)`; 3-D mode stores a singleton
  time dimension (`T == 1`) so every code path handles one layout.
- Neutron and X-ray arrays must always have identical shapes; the loaders and
  engines validate this.
- Non-finite voxels (NaN/Inf) are excluded from histograms and reported via
  `HistogramData.ignored_voxels`.
- Large datasets are memory-mapped automatically above
  `config.MEMMAP_THRESHOLD_GB`; engines iterate in chunks
  (`config.HISTOGRAM_CHUNK_VOXELS`) and never materialise a full flattened
  copy of the data.

## Histogram coordinate conventions (important!)

Everything in the selection pipeline meets in one coordinate system, and any
new code must preserve it:

- **Data coordinates:** a point is `(x, y) = (neutron intensity, X-ray
  intensity)`.
- **Storage:** `HistogramEngine4D` computes `np.histogram2d(neutron, xray)`
  and stores the **transpose**, so `HistogramData.histogram[row, col]` counts
  voxels with X-ray in bin `row` and neutron in bin `col`.
  `x_edges`/`x_centers` are neutron bin edges/centres, `y_edges`/`y_centers`
  are X-ray bin edges/centres.
- **Display:** `HistogramCanvas` renders with
  `imshow(hist, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0],
  y_edges[-1]])`, which puts neutron on the x-axis and X-ray on the y-axis.
  Mouse events therefore deliver ROI vertices directly in data coordinates.
- **Containment:** `ROIManager.is_inside_roi(neutron_values, xray_values)`
  tests `(neutron, xray)` pairs against ROIs stored in those same
  coordinates. `SegmentationEngine4D` simply forwards volumes to it.

**Draw ROIs filled, not as bare outlines.** Containment is decided by the
winding rule (`Path.contains_points`), and matplotlib fills by the same
rule, so a translucent fill shows exactly which region will be segmented.
An outline alone does not: in a self-crossing polygon an area can be ringed
by edges yet have winding number 0, looking enclosed while being excluded.
Drawing code that adds ROI patches should keep the fill for that reason, and
`polygon_self_intersects()` flags the crossing case when a polygon is
finished. Interactive drawing must also ignore non-left clicks and any click
made while the navigation toolbar's pan/zoom tool is active, or those drags
inject stray vertices into the polygon.

Because display, ROI storage, and containment all share this convention, a
region drawn on the histogram selects exactly the voxels whose intensity
pairs fall inside it. `tests/test_selection_segmentation_consistency.py`
locks this property in; keep it green when touching any of these modules.

### ROI semantics

`ROIManager` holds two layers of state:

- an **active ROI** (`roi_type`, `polygon_points` / `rectangle`) — the shape
  currently drawn/edited on the canvases, and
- a list of **named class ROIs** (`named_rois`) — saved classes for
  multi-material workflows, each with a `visible` flag.

**Visibility is not just a display flag**: a hidden class is excluded from
the overlays *and* from segmentation (`has_roi`, `is_inside_roi`,
`get_multi_class_labels`, `get_named_roi_overlays`,
`_enumerate_roi_specs`). Keeping those in step is what preserves the
invariant that the selection shown on the histogram is the selection that
gets segmented — any new consumer of `named_rois` should filter through
`get_visible_named_rois()`.

The same applies to the **segmentation layers already computed** from a
class. A layer carries the name of the ROI that produced it, which is how
`BiTS4DMainWindow._visible_layers(timepoint)` matches it back to that class's
tick. Read layers through that helper — not `segmentation_masks` directly —
wherever they are displayed or trained on, or unticking a class will leave
its old segmentation on screen. The masks are kept rather than deleted, so
re-ticking restores the layer instantly.

Removing a class is the destructive case, and the panel does not own the
layers, so the two sides are wired explicitly:
`DualHistogramWidget.layer_count_provider` (set by the window) reports how
many layers a class produced, and `class_removed(name, discard)` carries the
user's answer back. The panel asks only when layers exist, and never decides
by itself whether to throw them away.

`take_named_roi(index)` moves a class back into the active slot for
reshaping and removes it from the list, so a class being edited is never
counted twice; the returned entry carries the name/class_id/colour needed
to restore its identity when it is saved again.

**Snapshot ROI geometry whenever it is stored or handed off.** Segmentation
layers, saved classes and worker-thread specs all keep a record of the ROI
that produced them, and an ROI can be edited afterwards. Use
`get_active_vertices()` (or an explicit `np.array(..., dtype=float)`) rather
than `np.asarray`, which returns the *same object* for a float64 array and
silently aliases the live ROI — that aliasing made an edit rewrite the
record of an already-segmented layer, so the histogram and the mask
disagreed. For the same reason `EditableROIHandler` replaces the point array
on each drag instead of mutating it in place.

`is_inside_roi()` returns the union of the visible classes and the active
ROI, and the GUI's segment actions enumerate visible classes **plus** the
active one
(`BiTS4DMainWindow._enumerate_roi_specs`), so the segmentation result always
matches the selection shown on screen. Polygon containment uses a
bounding-box prefilter before `matplotlib.path.Path.contains_points`, which
is an order of magnitude faster on full volumes when the ROI is small.

## Big-dataset display pipeline

Loading a dataset performs three preparation passes (each cancellable, each
optional — cancelling degrades speed, never correctness):

1. **Global histogram** over all timepoints (chunked, CPU or GPU).
2. **Local-histogram cache**: every timepoint's histogram is computed once
   (`HistogramEngine4D.precompute_all_local_histograms`) into an LRU cache
   sized to hold all timepoints, so time navigation never recomputes them.
3. **Display pyramid** (`utils/display_downsampler.py`): if one volume
   exceeds `config.DISPLAY_MAX_VOLUME_BYTES` (1 GiB), every timepoint is
   median-binned by the smallest integer factor that fits the budget. The
   binned float32 copies are what the slice viewer shows.

Two coordinate spaces follow from this, and the split is strict:

- **Display space** (binned): slice viewer, spatial rectangle/region-grow
  selections, saved selection masks, per-slice statistics, time-series
  tracking, and 3-D auto-detect clustering. `SliceViewerWidget.
  display_bin_factor` and `BiTS4DMainWindow._current_display_volumes()`
  give access; `DisplayDownsampler.bin_mask` (block-any) scales
  full-resolution layer masks onto the display grid for overlays, and
  `upscale_mask` maps display-space cluster masks back to full resolution
  when they must feed the RF trainer.
- **Full resolution**: histogram computation, ROI segmentation, Otsu, RF
  training/prediction, and all mask/volume exports. Histogram-space ROIs are
  the bridge — they are resolution-independent, so a selection made while
  looking at binned data segments the original voxels exactly.

`HistogramEngine4D.compute_masked_histogram()` builds the histogram of one
segmented class on the **global** histogram's bin grid, which is what lets
exported per-class histograms be compared bin-for-bin across classes and
timepoints; `utils/histogram_export.py` writes them (`.npy` counts + `.png`)
together with the shared edges. Anything producing a histogram meant for
comparison should go through that method rather than binning independently.

`utils/histogram_evolution.py` renders the temporal analyses from the cached
local histograms — cumulative panels (vs T0), incremental panels (vs the
previous timepoint) and marginal kymographs per modality. The GUI exposes
all three under *Analytics → Histogram Time Analysis*, driven by the shared
`_run_histogram_time_analysis()` helper, which gathers the histograms and
delegates rendering. `notebooks/joint_hist_4d-5.ipynb` on `main` is the
reference implementation for these figures.

## Slice-viewer overlays

Segmentation layers are **3-D masks**, and the viewer keeps them that way.
`SliceViewerWidget.mask_overlays` holds `(name, mask, color)` entries whose
mask is either a 3-D volume (re-sliced by `_slice_mask_for_display` on every
redraw, so the highlight follows the slice index and the viewing plane) or a
2-D single-slice mask (region growing, saved selections — shown only while
the displayed slice matches). Never store a pre-sliced 2-D view of a 3-D
layer: that was the cause of highlights going stale on scroll and vanishing
on plane changes.

Overlay entries may carry an optional fourth element, `(axis, slice_index)`,
pinning a 2-D mask to the plane it was created on. Always set it for
single-slice masks: a shape check alone is not enough, because on an
isotropic volume every plane produces the same slice shape and a stale mask
would be drawn over the wrong plane. Selections that come from a 3-D
operation (3-D k-means, 3-D region growing) should instead store
`Selection.spatial_mask_3d`, which the viewer re-slices like any other 3-D
layer.

Two independent sources feed the overlay list — segmentation layers and
visible saved selections — so they are merged by
`BiTS4DMainWindow._compose_slice_overlays()`. Anything updating one source
must go through that composer (`_refresh_slice_overlays()` for an
overlay-only update, `_apply_segmentation_overlays()` when the base image
changes too) rather than calling `set_mask_overlays()` with its own list,
which would erase the other source.

## Execution model

Long operations (loading, histogram accumulation, segmentation, RF
training/prediction) run in a `WorkerThread` behind
`utils.progress_dialog.run_with_progress`. Cancellation is cooperative:
engines accept `progress_callback(value, message)` and `cancel_check()`
parameters and call them at chunk boundaries; `cancel_check` raises
`OperationCancelled` at the next safe checkpoint. New engine code should
follow the same pattern:

```python
def my_operation(data, progress_callback=None, cancel_check=None):
    for i, chunk in enumerate(chunks):
        if cancel_check:
            cancel_check()
        ...
        if progress_callback:
            progress_callback(int(100 * i / len(chunks)), "working...")
```

`gui/runtime_fixes.py` applies behavioural overrides to the legacy
main-window class at import time (see `gui/__init__.py`). New corrections
belong either directly in the widget modules or, when the monolithic
`main_window.py` makes that risky, in `runtime_fixes.py` with a comment
explaining what they replace.

## GPU usage

`HistogramEngine4D` accumulates on CUDA (via PyTorch) when available; the
binning formula is identical to the CPU path so both backends produce the
same counts, and CUDA out-of-memory errors trigger a transparent CPU retry.
Toggling *Force CPU Processing* or switching GPUs mutates `engine.use_gpu`
in place — cached histograms remain valid, so no recomputation occurs.

## Extending BiTS 4D

- **New segmentation method:** produce `(mask_3d, color, name)` layers and
  store them in `BiTS4DMainWindow.segmentation_masks[timepoint]`; the slice
  viewer, histogram overlays, RF training, and exporters all consume that
  format (see `_run_otsu_segment` for a template).
- **New ROI shape:** add storage + a `_<shape>_mask()` static method to
  `ROIManager`, extend `is_inside_roi`, `get_named_roi_overlays`, and the
  (de)serialisation methods, and add drawing support to `HistogramCanvas`.
- **New export format:** follow `utils/selection_library.py` — pure functions
  taking selections + arrays, no GUI imports.
- **Testing:** engines are GUI-free, so test them directly with pytest. GUI
  logic can be exercised offscreen (`QT_QPA_PLATFORM=offscreen`).
