# Bug-fix and cleanup notes

## The classifier removed, a materials panel, and a built-in manual (latest)

**The classifier is gone from the application.** Its tab, menu entries,
handlers and state are removed; `segmentation.legacy` keeps the module
importable so earlier figures and method comparisons still run, but nothing
in the application reaches it. The reason is recorded in
`segmentation/legacy/__init__.py` and in the manual: its training labels came
from point-in-polygon tests on the histogram, so every label was already an
exact function of the two intensities, and no other feature can carry
information about a target the intensities already determine. Attenuation
coefficients are material constants, so the boundary did not need learning
either. What the histogram genuinely cannot supply is spatial information —
building it discards spatial arrangement by construction — and that is what
the smoothing term provides.

**The materials panel** replaces it (`gui/material_panel.py`). Every
material, whether drawn on the histogram or copied from a K-means clustering,
is listed with its source, its size and a toggle: *Changes* or *Stays
unchanged*. Marking the second kind is the one judgement the software cannot
make for the user, and it is what gives every other result an independent
check. Smoothing, preview, run and the health-check readout are on the same
panel, which replaces the modal dialog the settings used to live behind.

The panel emits signals and knows nothing about datasets, so it is testable
without one. `set_materials()` carries an existing behaviour choice across a
refresh — re-reading the segmentation must not silently discard the control
setting.

**A built-in manual** (*Help → Manual*, F1). Twenty sections in two groups:
*How to*, which assumes you know segmentation and nothing else, and
*Mathematics*, which states what is actually computed — the histogram and its
sufficient statistics, the winding rule, Mahalanobis distance and the match
score, the smoothing cost and where its matrix comes from, how the smoothing
strength is chosen, the validity rule, mean-shift drift estimation, the
partial-volume model, every metric, and why there is no classifier. Non-modal,
searchable, and exportable as plain text. `tests/test_ui_language.py` checks
the complement of the interface rule: the manual is required to *contain*
the vocabulary the interface hides.

**A regression caught while removing the classifier.**
`_update_rf_histogram_overlays` was named for the classifier but drew the
class outlines for *every* segmentation layer. Deleting it with the rest of
the classifier removed the outlines from the histogram entirely. It is
restored as `_update_class_histogram_overlays`, with the classifier-specific
half dropped, and called wherever layers change.

## Locked material definitions, guards and a health check

Implements the revised v17 specification, which withdraws the assumption that
the class parameters should be fitted. The user-facing result is
[workflow.md](workflow.md); the method note is
[model_segmentation.md](model_segmentation.md).

**The design change.** Neutron and X-ray attenuation are material constants,
so where a material sits on the histogram is fixed by physics rather than
estimated from data. Letting each class mean float is therefore wrong: a
class centroid that moves is a class absorbing material that should have left
it. Locked mode is now the default — class positions come from the drawn
regions and stay put, and only the assignment of voxels changes between
timepoints. Three failure modes become structurally impossible rather than
guarded against: classes cannot merge, identities cannot permute (classes
*are* the regions, in region order), and timepoints are independent so there
is no coupling to oscillate. The fitted path remains available under
Advanced, off by default and with a warning.

**A real bug in the previous release.** The validity mask rejected a voxel
only when *both* channels held the sentinel value, so a region the neutron
instrument measured and the X-ray instrument did not passed as valid. Where
the two fields of view differ, that region is large, static and pinned to
zero in one axis, and whichever material is nearest absorbs it. A voxel now
needs data in **both** channels; `channel_coverage()` reports the overlap and
the Check Data step surfaces it on load.

**Automatic smoothing strength.** Spatial smoothing was a parameter with a
default, which is the one setting that can destroy a result invisibly — too
strong and a minority material is simply erased with everything downstream
still looking healthy. `auto_smoothing()` now measures instead of guessing:
it sweeps a grid and keeps the strongest setting at which no material loses
volume and the control materials stay put, and it retains the whole sweep as
the evidence for that choice.

**Guards and a health check.** Over-smoothing, too many unmatched voxels, a
voxel budget that does not close, and a spatial pass that cycles instead of
settling now stop the run rather than producing numbers. A health check runs
before results are shown, with control materials as the null control, and
every finding names the material involved and suggests an action.

**Interface language.** No term from the derivation appears in any
user-facing string, and `tests/test_ui_language.py` parses every string
literal in the GUI to keep it that way.

**Corrections to the specification as written.**

* The retention guard is specified as "a class loses > 50 % of its
  unsmoothed volume". Implemented literally against the *first timepoint* it
  would abort on genuine physics — a material that really shrinks is the
  measurement, not a fault. Retention is therefore compared against the
  unsmoothed result **at the same timepoint**, which isolates what smoothing
  did from what the sample did. There is a test for each direction.
* "Unclassified" conflated two different things. Voxels that were measured
  and matched nothing mean a material may be missing; voxels that were never
  measured mean the instruments disagree about coverage. They are counted
  separately, and the advice differs: unmatched voxels that appear gradually
  are drift, not a missing material, and the message says so and points at
  Check Instrument Stability.
* The learned boundary cost `V(k,l) = -log(n_kl / Σ_m n_km)` carries a
  per-row offset, so a class whose own voxels are less reliably adjacent — a
  thin or scattered one — pays a standing penalty for existing on top of the
  penalty for bordering anything. That is a bias against exactly the classes
  most at risk of being smoothed away. The diagonal is removed by default;
  the raw form is available and there is a test showing the difference.
* Monotone cost decrease cannot be *asserted* for a synchronous update, which
  can settle into a two-cycle rather than converging — the oscillation the
  specification reports from an earlier run. The update is damped, the cost
  is traced every sweep, and non-monotone behaviour is reported and can be
  made fatal, rather than being claimed as guaranteed.
* Unclassified needed a boundary cost of its own. Free would let smoothing
  flood unmatched voxels across the volume; prohibitive would push them into
  a real material, which is the outcome Unclassified exists to prevent. It is
  priced at the typical cost of any other boundary.
* The docstring fix in §12 does not apply here: this codebase has no such
  docstring, and its feature counts are 6 / 9 / 14, not 13.

**Worth recording as a finding.** The learned boundary costs turn out to
*protect* a minority material that genuinely borders its neighbour — its
boundary is cheap, so smoothing has no incentive to remove it even though it
is small. A uniform cost cannot express that. What over-smoothing actually
destroys is a **finely dispersed** phase, whose voxels have no neighbours to
lean on. Both cases are tested.

## Model-based time-series segmentation

Implements the BiTS v17 proposal. The plan's assessment — including which of
its claims held against this codebase, which were addressed to a different
one, and the seven places it was corrected — is in
[v17_plan_evaluation.md](v17_plan_evaluation.md); the method and its controls
are in [model_segmentation.md](model_segmentation.md).

**The change.** A histogram ROI drawn at T0 is a frozen partition of the
(neutron, X-ray) plane and cannot follow the drift every long series carries;
it keeps segmenting, reports no error, and progressively segments the wrong
thing. The new `model/` package keeps the manual selection as a *prior*
instead: a Normal-Inverse-Wishart prior on each mixture component, centred on
the moments of the voxels that ROI selects, whose strength interpolates
between "frozen at T0" and an unconstrained mixture. Around it sit a validity
mask, drift tracking on inert anchor classes, an MRF whose boundary costs are
counted from the T0 labels, a temporal transition, and partial-volume
components for classes that are really mixing lines.

**Corrections to the plan as specified.** The hardcoded `neutron_floor=3000`
became an opt-in policy (a floor tuned on one dataset deletes a real phase in
the next); the histogram cache stores per-bin *moments* rather than counts,
because fitting on bin centres inflates every covariance by the Sheppard
bias; drift is applied to the model rather than by rewriting every volume,
which keeps exports in native units; κ₀ is exposed as a dimensionless
strength in [0, 1] scaled by class size; the anchoring index is documented as
an upper bound and paired with a permutation version; and the mutually
contradictory acceptance criteria were resolved by versioning the feature
spec and the RF capacity limits.

**Bugs found while building it.**

* *M-step double-counting.* The first mixture implementation weighted the
  per-bin moment sums by the bin count a second time, inflating every
  covariance by roughly the mean bin occupancy until the uniform outlier
  component won every voxel. `cache.sums` and `cache.scatter` already carry
  the count; they take the bare responsibility.
* *Drift tracking gave up where it was needed most.* The plausibility guard
  compared each anchor against its T0 position, so on a series whose
  cumulative drift exceeds a few σ every anchor was rejected exactly when the
  drift was largest. The search is now cumulative — it starts from the
  previous estimate and checks the step since then.
* *Two anchors on one mode.* Neither moves implausibly far on its own, so the
  distance guard sees nothing wrong, but the drift is then estimated from one
  mode counted twice. All anchors involved in a collision are rejected;
  picking one would be a guess dressed as a measurement.
* *A uniform class reported elongation 0.* Already fixed in the previous
  round for `E_k`; the same degenerate ratio appears in the mixel machinery
  and is guarded there too.
* *`interface_area` counted a class's own interior faces* when compared with
  itself. Only voxels in exactly one of the two masks count now.
* *`f_rind` and `n_interior_components` measured different erosion depths*,
  so a small displaced object scored as a boundary rind with zero surviving
  components — both signals failing at once. They now share a depth, and the
  scale-dependence of `f_rind` is documented.
* *Tightened RF capacity broke small training sets.* `min_samples_leaf=50`
  with 80 training samples stops the forest splitting at all. The limit is
  now a ceiling scaled to the training set, so the intent survives at every
  dataset size.

**Also new.** `utils/metrics_spatial.py` adds the metric family the software
had none of — every previous quantity was computed in the histogram plane, so
a speckled or displaced class was structurally invisible.
`utils/validation.py` replaces in-sample accuracy with block
cross-validation and adds bootstrap bands, the anchoring index and the
temporal generalisation matrix. `segmentation/features.py` decouples texture
features from frozen T0 coordinates — in the old ladder texture was
unreachable without them, which made any experiment about either one
uninterpretable — while reproducing the legacy feature columns bit-for-bit.

**Not implemented, deliberately:** component birth-by-novelty (cannot be
validated without data containing a genuinely new phase) and the fuel-cell
path (where a dry reference exists, Beer–Lambert gives water thickness
directly and segmenting first throws that away — which is the source plan's
own advice).

## Quality metrics and the segmentation report

**Metrics.** *Analytics → Histogram Time Analysis → Histogram &
Segmentation Metrics…* computes the ground-truth-free subset of the
reference metric tables and writes a CSV plus a multi-panel evolution plot.
Two scopes are covered: the global histogram, and every timepoint using its
own cached local histogram and its own class masks. Hidden classes are
excluded, matching what segmentation does. Full metric list and CSV layout:
[docs/metrics.md](metrics.md).

`eps_k`, `CE` and `O_ab` are **not** computed. All three measure distance to
a known material position or label, which measured data does not have —
reporting them would mean inventing a reference. `CD` (mean class-centroid
drift against T0) is provided instead as the honest time-series analogue of
`CE`, and `DB`, the per-class spreads and `E_k` are computed as ordinary
internal cluster indices over the classes the user actually created.

*Edge case found while testing:* a perfectly uniform class has zero spread
on both axes, and the elongation ratio returned 0 — an isotropic class
reported as maximally flat. `E_k` is now 1.0 when both spreads vanish.

**Segmentation report.** The export dialogs gain a "Text report" option
(on by default) that writes `segmentation_report.txt` next to the exported
volumes: the class legend (integer value in the label volumes, class name,
total voxels, mean volume fraction), voxel counts and volume fractions per
class and timepoint, the histogram selections the classes came from, and the
settings involved — bin count, neutron and X-ray ranges, Random Forest
training accuracy and class names, and whether display binning was active
(with a note that segmentation and export are always full resolution).

*Latent bug fixed alongside it:* the batch export numbered label values
per timepoint, so if a class was absent at one timepoint every later class
shifted down by one and a label value meant different classes in different
files. The export now fixes a single `class_order` up front and uses it for
every timepoint, which is also what the report documents.

## Per-class histogram export and class names in file names

**Bimodal histogram per class.** The export dialogs gain a "Bimodal
histogram of the class" option. For every selected class and every exported
timepoint it computes the 2-D neutron/X-ray histogram of just that class's
segmented voxels and writes `<timepoint>_<class>_hist.npy` (counts) plus a
`.png` for quick viewing.

`HistogramEngine4D.compute_masked_histogram()` does the work on the
full-resolution volumes, reusing the **bin count and data range of the
global histogram**, so every exported histogram shares its edges exactly and
the files can be stacked or compared bin-for-bin across classes and time. It
is chunked like the other accumulators, so a large class costs no more peak
memory. The shared edges are written once per export as
`histogram_edges_neutron.npy` / `histogram_edges_xray.npy`, with a short
README recording that counts are laid out `[X-ray bin, neutron bin]`.

**Class names in file names.** Exports already used a layer's name, so a
class called "Lithium" produced `timepoint_000_Lithium_*`. Random Forest
predictions were the exception — they were named `RF class 1` regardless of
what the training class was called. `_rf_train` now records which training
layer became each class id, and `_rf_class_label()` turns that into
`RF: Lithium`, which flows through the prediction layers, the slice-viewer
legend, the histogram outlines and the exported file names. Names are
sanitized for the filesystem (`utils.histogram_export.sanitize_name`), so
"Solid Electrolyte" becomes `Solid_Electrolyte` and "Li/graphite" becomes
`Li_graphite` rather than creating a stray directory.

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
