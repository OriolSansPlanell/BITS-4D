# Model-based time-series segmentation — how it works

For how to *use* it, read [workflow.md](workflow.md) first. This page is the
method behind it.

*Analytics → Time Series Segmentation → Track Materials Across Time…*

## Two modes

**Locked (default).** Class positions are fixed, taken from the regions drawn
at the reference timepoint. Per timepoint the software scores every voxel
against every fixed material and resolves the assignment spatially. Nothing
is estimated, so timepoints are independent and three failure modes are
structurally impossible rather than guarded against: classes cannot merge,
identities cannot permute, and there is no coupling between timepoints to
oscillate. This is the right default because attenuation coefficients are
material constants — a class that moves is a class absorbing material that
should have left it.

**Adaptive (advanced, off by default).** The definitions themselves are
allowed to move, anchored to where they were drawn. Appropriate only when the
instrument genuinely drifts. The rest of this page describes the machinery
that mode uses.

## The problem this solves

A histogram ROI drawn once at T0 is a **frozen partition** of the
(neutron, X-ray) plane. Over a long series the whole cloud migrates — beam
current, detector gain, scatter build-up, a reconstruction change — and the
frozen boundary cannot follow it. It keeps segmenting, reports no error, and
is progressively segmenting the wrong thing. A classifier trained at T0 has
exactly the same problem for exactly the same reason, plus one more: if its
features include normalised coordinates, part of its decision function is
memory of *where things were* rather than evidence about what they are.

The fix is not a better boundary. It is to stop treating the manual selection
as a constraint and start treating it as a **prior**:

```
p(labels, fractions | data) ∝ mixture likelihood × spatial prior × temporal transition
                              └─ anchored on your ROIs
                                                 └─ learned from your T0 labels
                                                                  └─ how fast a class may move
```

Your work enters twice — once as the moments each component is anchored on,
once as the cost of each class boundary — and the model is free to follow the
data from there, as far as you allow and no further.

## The one control that matters

**Anchor strength**, a number from 0 to 1.

| Strength | Behaviour |
| --- | --- |
| `1.0` | Components frozen at their T0 position. This *is* the fixed-ROI method. |
| `0.9` | Held firmly; moves only where the data insists. |
| `0.5` | The T0 selection counts for exactly as much as the data (default). |
| `0.0` | Unconstrained mixture: follows the drift, and also the noise. May swap two classes and lose their identity. |

The fixed ROI and a free mixture are not rival methods — they are the two
ends of this dial. The strength is scaled by each class's own T0 size, so the
same setting means the same thing for a 1 % phase and a 40 % one, and on
every dataset.

Released from its anchor a component follows whatever mass is nearest, which
on a drifted histogram may be a neighbour's. Movement is what the strength
controls; *being right* is what the anchor buys.

## Anchor classes and drift

Pick the classes that **cannot change chemically** during the experiment — an
inert container, a support, a structural metal. Any movement of such a class
is instrumental by definition, and that movement corrects every other class.

Choose them on physics *and* evidence: a good anchor has a flat segmented
volume across the series and no role in the reaction.

> **Never anchor on a reacting phase.** Its real change would be subtracted
> from every other class as if it were drift — the physics would be fitted
> away. The software cannot tell the difference, which is why the choice is
> yours and there is no default.

Locating an anchor at a later timepoint needs no labels: it is a dense
isolated mode, so a mean-shift started from where it was last seen finds it.
Two things can go wrong, and both are reported rather than hidden:

- an anchor that moves implausibly far has latched onto something else and is
  dropped;
- **two anchors that converge on the same mode** are both dropped. Neither
  moved suspiciously far on its own, but the drift would be estimated from
  one mode counted twice, and nothing local says which is the impostor.

Both are far less likely when the series is stepped through timepoint by
timepoint, which is what the segmenter does — a small step is never
ambiguous. Estimating in one jump across a large drift is what breaks.

The drift is applied to the **model**, not to your data. No volume is
rewritten, so every histogram, statistic and export stays in native intensity
units and remains comparable with everything produced before.

*Analytics → Model-Based Segmentation → Estimate Instrumental Drift…* runs
the tracker on its own and writes a CSV of the per-timepoint shift, so you
can look at the drift before deciding what to do about it.

## Spatial coherence

A mixture classifies each voxel from its two intensities and nothing else, so
its raw output is speckled. The MRF fixes that, and its costs are **learned
from your T0 labels**: boundaries that occur constantly in your own
segmentation stay cheap, ones that never occur are expensive. A generic
smoothing model charges the same for both.

`beta` sets the strength. `0` turns it off and shows the raw mixture labels —
useful once, to see what the spatial term is doing.

Two solvers, chosen automatically from a memory budget:

| Solver | Memory | Quality |
| --- | --- | --- |
| mean-field | `K × 4` bytes/voxel (~1.4 GB at 38 M voxels, 9 classes) | better |
| ICM | ~9 bytes/voxel, independent of `K` | greedier |

## Between timepoints

**Memory** is how much of each timepoint's prior comes from the previous
timepoint rather than from T0. `0` re-anchors to T0 every time; `1` is a pure
random walk that stops looking back at your selection; `0.5` blends them.
Both terms are Gaussian, so the blend is closed-form.

Two safeguards:

- a component that moves further in one step than the instrumental noise
  floor allows has its movement **clipped**, and the event is reported.
  Reactive classes are permitted to move faster by an explicit factor rather
  than by accident;
- a component whose weight collapses goes **dormant**, not deleted. If the
  phase returns it is the *same* component, so its time series is continuous
  across the gap.

## Partial-volume classes

Some "classes" are not phases at all: they are mixing lines running between
two pure phases, made of voxels containing part of each. The giveaway is
elongation — an anisotropy around 2.4 where real phases sit at 1.1–1.2.

A hard label for such a voxel is ill-posed. The answer for a voxel that is
40 % lithium is not "lithium" or "indium", it is 0.4. Ticking *Look for
mixing lines* flags any class that is both elongated **and** aligned with the
line joining two others, and reports `E[α]` per voxel as a fractional map.
Requiring both conditions is what separates a genuine partial-volume ridge
from a phase that merely happens to be anisotropic.

This also explains, rather than merely tolerating, the usual discrepancy
between a fixed polygon and a trained classifier: they disagree about a
boundary shell whose true membership is fractional, so neither was right and
the disagreement was never evidence about the interior.

## Abstention

Two mechanisms let the model decline:

- the **outlier component**, a uniform density that absorbs padding,
  artifacts and materials you did not model, instead of forcing them into
  whichever real class is nearest;
- **low-confidence rejection**, which leaves a voxel unassigned rather than
  giving it the best of a bad set of options.

Unassigned voxels are labelled `-1` and counted in the summary. For a method
whose purpose is tracking change, "I don't recognise this" is information.

## Which voxels count

Padding, NaN and saturated voxels are not measurements, and nothing used to
exclude them. A padding region large enough to rival a real phase gets
absorbed into whichever class is nearest, inflating its spread and dragging
its centroid.

The default policy rejects only what is provably not a measurement:
non-finite values and the exact sentinel the padding was written with
(usually 0). A hard intensity floor is available but **off by default** — one
tuned on another dataset will silently delete a genuinely low-attenuation
phase.

The rejected fraction is recorded per timepoint. A step in it is an
acquisition change — a shifted field of view, a different reconstruction, a
detector fault — and absolute volume comparisons across such a step are not
valid.

## Scripting it

The engine has no GUI dependency:

```python
import numpy as np
from data import Dataset4D
from histograms import HistogramEngine4D
from model import (
    DriftTracker, ROIAnchoredMixture, ROIDerivedMRF, SequentialSegmenter,
)
from model.temporal import DriftTransition

dataset = Dataset4D(neutron_4d, xray_4d)
histogram = HistogramEngine4D(bins=256).compute_global_histogram(
    neutron_4d, xray_4d
)

segmenter = SequentialSegmenter(
    mixture=ROIAnchoredMixture(outlier_component=True, reject_margin=0.5),
    mrf=ROIDerivedMRF(beta=1.0, n_sweeps=5),
    temporal=DriftTransition(memory=0.5),
    drift_tracker=DriftTracker(anchor_classes=["Aluminium", "Steel"]),
)
segmenter.prepare(
    neutron_4d[0], xray_4d[0], class_masks_at_t0,
    histogram.x_edges, histogram.y_edges,
    anchor_strength=0.5,
)
result = segmenter.run(dataset)

for entry in result:
    print(entry.timepoint, entry.voxel_counts, entry.drift.describe())

# The component trajectories *are* the evolution analysis
trajectories = result.parameter_trajectories()
```

Every timepoint emits a full parameter set, so there is no separate pass
needed to find out how the histogram changed.

## Checking the result

- **Spatial metrics** (*Analytics → Histogram Time Analysis → Spatial
  Metrics…*) — a speckled or displaced class is invisible to any
  histogram-plane metric and obvious here: connected components,
  surface-to-volume, centre of mass and its drift.
- **`disagreement_topology`** — comparing two segmentations, erode what they
  disagree about. A shell that vanishes in two voxels means they agree on
  morphology and differ only on fractional boundary voxels. Compact clumps
  that survive are a real disagreement. Read `f_rind` and
  `n_interior_components` together: a small displaced object erodes away like
  a shell, so the component count is the robust signal.
- **Block cross-validation** (*Analytics → Model-Based Segmentation →
  Block Cross-Validation…*) — honest classifier accuracy. Voxels are
  spatially autocorrelated, so an in-sample, random-fold or out-of-bag score
  asks the model to recognise data it has effectively already seen. Expect a
  substantially lower number, and quote that one.
- **Anchoring index** — the share of a classifier's decision function carried
  by features identical at every timepoint. Above 0.20, a fifth of the
  segmentation is T0 geometric memory.

## What is not here

**Component birth.** A new phase appearing mid-series is not detected
automatically. Birth-by-novelty cannot be validated without data containing a
genuinely new phase, and a birth rule that fires on noise is worse than none.
Dormancy and resurrection *are* implemented, which covers a phase that
disappears and returns.

**The fuel-cell path.** Where a dry reference scan exists, do not classify
first: water thickness follows directly from Beer–Lambert,
`t_w = −ln(I_wet/I_dry)/Σ_w` — quantitative, calibratable, no training, and
it handles sub-resolution saturation natively. In a GDL with 10–50 µm pores
most voxels are a carbon/water/air mixture where a hard label is physically
meaningless, so segmenting first throws away the quantity you actually want.
Threshold the thickness map afterwards if discrete regions are needed.
