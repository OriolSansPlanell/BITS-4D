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

Because display, ROI storage, and containment all share this convention, a
region drawn on the histogram selects exactly the voxels whose intensity
pairs fall inside it. `tests/test_selection_segmentation_consistency.py`
locks this property in; keep it green when touching any of these modules.

### ROI semantics

`ROIManager` holds two layers of state:

- an **active ROI** (`roi_type`, `polygon_points` / `rectangle`) — the shape
  currently drawn/edited on the canvases, and
- a list of **named class ROIs** (`named_rois`) — saved classes for
  multi-material workflows.

`is_inside_roi()` returns the union of *all* of them, and the GUI's segment
actions enumerate named ROIs **plus** the active one
(`BiTS4DMainWindow._enumerate_roi_specs`), so the segmentation result always
matches the selection shown on screen. Polygon containment uses a
bounding-box prefilter before `matplotlib.path.Path.contains_points`, which
is an order of magnitude faster on full volumes when the ROI is small.

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
