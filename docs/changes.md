# Bug-fix and cleanup notes

## Removing a class asks about its segmentation (latest)

Unticking a class hides the segmentation computed from it, but *removing*
the class left that layer behind with nothing in the panel controlling it.
Rather than pick an outcome silently, **Remove** (and **Clear All**) now ask
when layers exist:

* **Yes** — remove the class and discard its segmentation,
* **No** — remove the class but keep its segmentation as an ordinary layer,
* **Cancel** — keep everything.

A class with no segmentation still gets the plain confirmation, so the extra
question only appears when there is something to lose.

The panel owns the ROIs and the main window owns the layers, so they are
wired together explicitly: `DualHistogramWidget.layer_count_provider` reports
how many layers a class produced (the panel offers nothing to discard
without it), and `class_removed(name, discard)` tells the window what the
user chose. Discarding also clears that layer's recorded outline and its
binned-mask and derived-hull cache entries.

## Unticking a class left its segmentation on screen (latest)

**Symptom:** segment an ROI, untick it in the selection panel, segment a
second ROI — the first region was still highlighted.

**Root cause:** the tick governed the *ROI* (whether it is drawn and whether
future segmentation covers it) but not the *layer* already computed from it.
Segmentation layers live in `segmentation_masks[timepoint]` and were all
displayed unconditionally, so the first ROI's mask stayed visible. Segmenting
again kept it too, since the layer-replacement step only drops layers whose
name is being re-created.

**Fix:** layers are matched to the class that produced them (they carry its
name) and filtered by its tick — `_visible_layers()` is now the single way
layers are read for display and training:

- the slice-viewer highlights,
- the histogram layer outlines,
- and Random Forest training, which no longer trains on a class the user
  cannot see. The completion dialog reports how many hidden classes were
  excluded, and the "no segmentation" warning says when layers exist but are
  all unticked.

Masks are **kept**, not deleted, so ticking the class back on restores its
layer immediately. Toggling a tick now refreshes the viewer and histogram at
once rather than waiting for the next timepoint change.

**Also fixed:** the histogram canvases had the same "two writers, one list"
problem the slice viewer had — `_refresh_named_roi_overlays` (class ROIs) and
`_update_rf_histogram_overlays` (layer hulls) overwrote each other. The
latter now composes both, and skips a hull for any layer that repeats a class
ROI's shape so a segmented class is not outlined twice.

## 3-D region grow built its ROI from a single slice (latest)

**Symptom:** growing a region through the volume produced a histogram ROI
much narrower than the region selected, so segmenting with it recovered only
part of it. Visible in the log as a 3-D grow finding 4,659 voxels while the
ROI was derived from the 173 pixels on the displayed slice.

**Root cause:** `SliceViewerWidget` computes and stores
`region_grow_mask_3d`, but "Create Histogram ROI from Selection" emitted
only `region_grow_mask` — the 2-D slice currently on screen. The convex hull
therefore described 3.7% of the selected voxels, covering only the intensity
spread that happens to appear on that one slice.

**Fix:** the 3-D mask is emitted when the region was grown through the
volume, and the main window extracts values from the whole volume when the
mask is 3-D (2-D masks keep the slice path). Because a 3-D mask can hold
millions of voxels, `create_convex_hull_roi` now deduplicates intensity
pairs above 100k points before hulling — exact, since the hull of a set
equals the hull of its distinct points.

**Also fixed, found while testing:** a region of *uniform* intensity (a
saturated or single-valued phase) has zero spread, so the percentage margin
padded it by zero and the fallback bounding box had no area — the ROI
selected none of its own voxels. `_bounding_box_roi()` now falls back to an
absolute pad, guaranteeing a usable region.

## A drawn polygon selected less than its outline enclosed (latest)

**Symptom:** an area clearly inside a polygon ROI's outline was not
segmented.

**Root cause — two faults compounding:**

1. **Stray vertices.** `HistogramCanvas.on_mouse_press` checked neither the
   mouse button nor the navigation toolbar. Every pan or zoom-rectangle drag
   *also* dropped a polygon vertex where the drag started — so zooming in to
   place a vertex precisely, the natural thing to do on a dense histogram,
   silently corrupted the polygon into a self-crossing shape. Right- and
   middle-clicks added vertices too.
2. **Winding-rule mismatch.** ROIs were drawn as outlines only, but
   containment is decided by the winding rule
   (`Path.contains_points`). In a self-crossing polygon a region can be
   ringed by edges yet have winding number 0, so it looks enclosed while
   being excluded from the selection. The drawing and the segmentation
   genuinely disagreed, and nothing on screen showed it.

**Fix:**

- Only a plain left-click places a vertex, and clicks are ignored while the
  toolbar's pan/zoom tool is active (or the canvas widget lock is held), so
  the polygon stays what the user actually drew.
- ROIs and class overlays are now drawn **filled** as well as outlined.
  Matplotlib fills with the same winding rule that decides containment
  (verified by rasterizing and comparing against `contains_points`), so the
  shaded area *is* the region that will be segmented — any excluded pocket
  is now plainly visible.
- Finishing a polygon whose edges cross warns explicitly that part of the
  enclosed area may be left out, instead of letting the segmentation come
  out quietly wrong. `utils.roi_manager.polygon_self_intersects()` does the
  detection.

The bounding-box prefilter added earlier was checked against
`Path.contains_points` over 200 random concave polygons and matched exactly,
ruling it out as a cause.

## Edited ROIs rewrote what had already been segmented (latest)

**Symptom:** after editing a histogram region, the segmented volume no
longer matched the selection, and training the Random Forest showed the
region as "modified" with parts of the ROI apparently unselected.

**Root cause:** the outline recorded for a segmented layer was the *same
NumPy array object* as the live ROI (`np.asarray` returns the input
unchanged for a float64 array, and `EditableROIHandler` mutated
`polygon_points` in place while dragging). Editing a vertex therefore
rewrote the record of an already-segmented layer retroactively: the
histogram drew the *edited* shape while the mask — and the RF labels built
from it — came from the *original* shape. The two genuinely disagreed.

**Fix:** ROI geometry is now snapshotted at every boundary where it is
stored or handed off.

- `EditableROIHandler` replaces the point array instead of mutating it, so
  snapshots taken earlier stay valid.
- `ROIManager.set_polygon_roi` stores an independent float copy, and the new
  `get_active_vertices()` returns a fresh outline.
- `_record_layer_shape`, `_roi_spec_vertices` and `_enumerate_roi_specs`
  copy, so a worker thread segmenting in the background cannot see the ROI
  change under it.

A layer's overlay therefore keeps showing the ROI that produced its mask.
Editing the active ROI after segmenting no longer alters the stored layer —
re-run "Segment Current" to apply the new shape.

## Unsaved selections now appear in the selection panel

Only saved classes were listed, so a region being drawn was invisible in
the panel. The active ROI is now shown as an italic "✎ (unsaved selection)"
row after the saved classes. It has no class id or visibility checkbox
until it is saved; *Remove* discards it (leaving classes untouched), *Only
This* hides every saved class so it is segmented alone, and double-clicking
it saves it as a class.


## Selection panel and previous-timepoint marginals (latest)

**Selection panel.** The named-class list under the histograms became a
proper management panel: each row has a visibility checkbox, and the
buttons are *Edit*, *Remove*, *Clear All*, *Show All*, *Hide All* and
*Only This*.

- **Visibility governs both display and segmentation.** A hidden class is
  neither drawn on the histogram nor segmented, so what is shown always
  equals what is segmented — the invariant the whole selection pipeline
  rests on. `has_roi()`, `is_inside_roi()`, `get_multi_class_labels()`,
  `get_named_roi_overlays()` and `_enumerate_roi_specs()` all filter on it.
  *Only This* therefore isolates one class to work with in one click.
- **Edit** moves a class back into the active ROI slot
  (`ROIManager.take_named_roi`) so it can be reshaped, and removes it from
  the list so it is never counted twice. Saving it again restores its
  original name, class id and colour.
- Visibility is persisted with the ROIs; files written before this change
  have no `visible` key and default to visible.

A crash was found and fixed while testing the panel: toggling a checkbox
rebuilt the list from inside that item's own `itemChanged` handler, which
deletes the item currently emitting the signal — a use-after-free that
aborted the process. The row is now restyled in place. (A missing
`_refresh_named_roi_overlays` after a refactor also surfaced this way:
an `AttributeError` raised inside a Qt slot aborts rather than propagating,
so both bugs presented as a hard crash.)

**Marginal change vs previous timepoint.** *Analytics → Histogram Time
Analysis* gains "Marginal Change vs Previous Timepoint", alongside the
existing vs-T0 version. `compute_marginal_changes()` takes the same
`reference` argument as the joint-histogram figures; in previous mode T0 is
blank (no predecessor) and each later column shows only that step's change,
so a band that moves once stands out instead of persisting in every later
column. Bins empty in the denominator stay blank rather than saturating the
colour scale.

## K-means highlight showed the previous plane (latest)

**Symptom:** after K-means segmentation, changing the visualization plane
left the highlight showing the previous plane's voxels.

**Root cause:** cluster selections stored only the **2-D mask slice**
extracted when clustering ran. The viewer's guard was a shape check, and on
an isotropic volume every plane yields the same slice shape — so the stale
2-D mask passed the check and was drawn over the new plane. (The same class
of bug as the segmentation-layer one, in the selection path.)

**Fix:** `Selection` now carries `spatial_mask_3d` plus the
`source_axis`/`source_slice_index` the 2-D mask belongs to. 3-D clusters
hand the viewer their volume mask, which is re-sliced on every redraw so the
highlight follows the plane; genuinely 2-D selections are pinned to their
own plane and slice and are never drawn elsewhere. Overlay entries may now
carry that plane as an optional fourth element. Region-grow selections and
the K-means→RF conversion record the same provenance.

## New temporal histogram analyses

*Analytics → Histogram Time Analysis* now holds three functions:

- **Evolution vs First Timepoint** — cumulative drift, `log10(h_t+1) −
  log10(h_0+1)` (the existing figure, moved into the submenu).
- **Change vs Previous Timepoint (incremental)** — `log10(h_t+1) −
  log10(h_{t-1}+1)`, so the steps where change actually happens stand out
  instead of being buried in cumulative drift.
- **Marginal Evolution (Neutron / X-ray)** — kymographs of each modality's
  1-D histogram against time, coloured by `log2(m_t / m_0)`, following the
  reference notebook (`notebooks/joint_hist_4d-5.ipynb`). Each timepoint is
  count-normalized first so differing finite-voxel counts stay comparable,
  and the colour scale uses the 99th percentile of |change| so a few extreme
  bins do not flatten the rest. This separates a shift in neutron from a
  shift in X-ray, which the joint histogram can hide.

Note that `HistogramData.histogram` is stored `[xray_bin, neutron_bin]`,
transposed relative to the notebook's `[neutron, xray]`; the neutron
marginal therefore sums axis 0 and the X-ray marginal axis 1. A test with a
single-modality change pins this orientation down.


## Slice highlight did not follow slices, planes or timepoints (latest)

**Symptom:** after segmenting, the highlight in the slice viewer was wrong
or absent when scrolling through slices, switching viewing plane, or after
"Segment All Timepoints".

**Root cause:** `SliceViewerWidget` stored the *2-D slices* of the masks
that happened to be current when segmentation ran. Every later redraw
re-used those stale 2-D arrays: scrolling within a plane painted a previous
slice's voxels (the shapes still matched, so nothing complained), and
switching plane made `mask.shape != current_slice.shape`, silently skipping
the overlay entirely.

**Fix:** the viewer now holds the **3-D masks** and re-slices them on every
redraw (`_slice_mask_for_display`), so one segmentation covers all slices
and all planes, for the current timepoint and for every timepoint after a
batch run. 2-D single-slice masks (region growing, saved selections) are
still supported and shown only on a matching slice. The info label now
reports the voxels actually highlighted on the displayed slice, replacing a
counter that read a legacy attribute which was always `None`.

## Selection changes erased the segmentation highlight

`_update_histogram_overlays` (selection manager) replaced the viewer's whole
overlay list, so toggling "Show All on Histogram" or editing selections
wiped the segmentation layers until the next timepoint change. Overlays from
the two sources are now **composed** (`_compose_slice_overlays`) instead of
overwriting one another.

## Dead duplicate batch-segmentation implementation removed

`main_window._segment_all_volumes` (73 lines) was unreachable —
`gui/runtime_fixes.py` replaces it on the class at import time. Two
divergent implementations of the same action is a standing bug source, so
the dead one was removed and replaced with a pointer to the live one.

## Performance

- Convex hulls derived for layers without a drawn ROI (RF, Otsu, K-means)
  are cached per layer; they previously re-gathered every segmented voxel's
  intensities from the full-resolution volume on each timepoint switch.
- The binned-display-mask and derived-hull caches are now cleared alongside
  the layers they describe, so they cannot grow without bound.
- A timepoint switch paints once instead of rendering overlays twice.


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
