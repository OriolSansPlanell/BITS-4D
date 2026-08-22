# Quality metrics

Two families, sharing one CSV schema so their outputs concatenate:

- **histogram metrics** — *Analytics → Histogram Time Analysis → Histogram &
  Segmentation Metrics…* (this page);
- **spatial metrics** — *Analytics → Histogram Time Analysis → Spatial
  Metrics…* ([below](#spatial-metrics)).

## Histogram metrics

*Analytics → Histogram Time Analysis → Histogram & Segmentation Metrics…*

Pick a CSV filename; BiTS 4D writes that file plus `<name>_evolution.png`
alongside it. Two scopes are analysed:

- **global** — the whole-dataset bimodal histogram, with the classes of the
  first segmented timepoint;
- **timepoint** — one row per timepoint, using that timepoint's own local
  histogram and its own class masks.

Hidden classes (unticked in the selection panel) are excluded, exactly as
they are from segmentation.

## What is measured, and what is not

Every metric here is **ground-truth-free**: it can be computed from the data
and your own classes, with no phantom or reference labelling.

The reference tables (`metrics_table.py`, `metrics_table_morphology.py`) also
define `eps_k`, `CE` and `O_ab`. Those measure the distance between a
recovered class and a *known* material position or label, which does not
exist for measured data — computing them would mean inventing a reference,
so they are **not** reported. In their place, `CD` reports how far the class
centroids have drifted from the first timepoint: the same "how far off is
it" question, anchored on something the data actually provides.

## Metrics

### Histogram shape (from the histogram alone)

| Metric | Unit | Meaning | Better |
| --- | --- | --- | --- |
| `S_h` | dimensionless | Spread along neutron at fixed X-ray. High values mean a horizontal streak — typically misalignment between the two modalities. | lower |
| `S_v` | dimensionless | Spread along X-ray at fixed neutron. High values mean a vertical streak — typically ring artifacts. | lower |
| `S_d` | Pearson ρ, [-1, 1] | Correlation between the modalities. A strong diagonal smear points at beam hardening or scatter. | closer to 0 |
| `A_x` | (mean − median)/σ | Skew of the X-ray marginal; flags cupping. | closer to 0 |
| `Delta_n` | intensity | Shift of the mean neutron intensity since T0; flags scatter build-up or a changing sample. Blank for the global row and zero at T0. | closer to 0 |

### Segmentation classes (internal cluster indices)

| Metric | Unit | Meaning | Better |
| --- | --- | --- | --- |
| `DB` | dimensionless | Davies–Bouldin index over the classes in the histogram plane: how separable they are. Blank with fewer than two classes. | lower |
| `CD` | intensity | Mean class-centroid drift against T0. | lower |
| `n_classes` | count | Classes contributing to the row. | — |

### Per class

| Metric | Unit | Meaning | Better |
| --- | --- | --- | --- |
| `voxels_k` | voxels | Segmented voxels in the class (always the true count, even when the statistics were subsampled). | — |
| `centroid_n_k`, `centroid_x_k` | intensity | Mean neutron / X-ray intensity of the class. | — |
| `sigma_n_k`, `sigma_x_k` | intensity | Class spread along each axis. | lower |
| `E_k` | dimensionless, ≥ 1 | Elongation — the larger spread over the smaller. 1 is isotropic. | lower |
| `drift_k` | intensity | Distance this class has moved in the histogram since T0. Blank at T0 itself. | lower |

## The CSV

Long format, one value per line, so it loads straight into pandas or a
spreadsheet without reshaping:

```
scope,timepoint,metric,class,value,label,unit,meaning,better_when
global,,S_h,,3.41,Horizontal streak score,dimensionless,...,lower
timepoint,0,DB,,0.42,Davies-Bouldin index ...,dimensionless,...,lower
timepoint,0,voxels_k,Lithium,144.0,Class voxel count,voxels,...,n/a
```

`class` is empty for scalar metrics and names the class for per-class ones;
`value` is empty where a metric could not be computed (no reference, fewer
than two classes, empty histogram). The four trailing columns describe the
metric so the file stands on its own.

```python
import pandas as pd
table = pd.read_csv("metrics.csv")
db = table[(table.scope == "timepoint") & (table.metric == "DB")]
db.plot(x="timepoint", y="value")
```

## The evolution plot

One panel per metric against timepoint. Scalar metrics are drawn as a single
series with the global value as a dashed grey reference; per-class metrics
draw one line per class. Metrics that are blank everywhere are skipped, and
the plot is only written when there are at least **two** timepoints — a
single timepoint has nothing to evolve.

## Scale

Class statistics on classes larger than 2 million voxels are computed on a
random subsample of that size. Mean, spread and elongation are unchanged
within sampling error, and `voxels_k` still reports the full count.

---

# Spatial metrics

*Analytics → Histogram Time Analysis → Spatial Metrics…*

Everything above lives in the (neutron, X-ray) plane. That is a real gap,
because a segmentation failure is usually a **spatial** one: a class that is
right in aggregate but scattered into a thousand speckles, a deposit whose
centre of mass has walked across the sample, a rim that belongs to a boundary
rather than to a phase. None of those move the histogram much, so a
histogram-only metric set cannot see them.

| Metric | Unit | Meaning | Better |
| --- | --- | --- | --- |
| `com_z_k`, `com_y_k`, `com_x_k` | voxels | Centre of mass of the class. | — |
| `com_drift_k` | voxels | How far it has physically moved since the first timepoint it appears in. | — |
| `rg_k` | voxels | Radius of gyration — grows when a compact deposit becomes diffuse. | — |
| `n_components_k` | count | Connected components. A coherent phase is a few; a noisy segmentation is thousands. | lower |
| `largest_frac_k` | fraction | Share of the class in its single biggest piece. | higher |
| `sa_vol_k` | faces/voxel | Boundary roughness. Speckle drives it up. | lower |
| `interface_kl` | faces | Shared surface between a class **pair** (the `class` column holds `A\|B`). The quantity that governs reaction kinetics at a boundary. | — |

## Comparing two segmentations: rind or blob?

`utils.metrics_spatial.disagreement_topology` answers the question two
methods always raise. Erode what they disagree about and watch what survives:

| Metric | Meaning |
| --- | --- |
| `disagreement_voxels` | Voxels labelled differently. |
| `f_rind` | Share gone within two erosions. Near 1 = a boundary shell. |
| `n_interior_components` | Compact clumps still standing at that depth. |

A shell one or two voxels thick vanishes immediately: the methods agree about
where the material is and differ only on partial-volume voxels, whose
membership is fractional anyway — so the disagreement was never evidence
about the interior. Clumps that survive are a real disagreement, and worth
investigating.

**Read both.** `f_rind` is scale-dependent: a genuinely displaced *small*
object erodes away in two voxels just as a shell does. `n_interior_components`
does not have that problem, so a high `f_rind` with a non-zero component count
is still a real disagreement — about something thin.

## Honest accuracy and error bars

`utils/validation.py` covers the two quantities that do not mean what they
appear to:

- **`block_cross_validation`** — holds out contiguous 3-D blocks instead of
  random voxels. Voxels are spatially autocorrelated, so an in-sample,
  random-fold or out-of-bag score asks a model to recognise data it has
  effectively already seen, which is why such numbers sit above 95 %
  regardless of quality. Reports per-class IoU and Cohen's κ.
- **`bootstrap_bands`** — resamples the training subsample and the random
  seed and reports a band, which is what turns "these two methods differ by
  6 %" into a claim that can be true or false rather than merely observed.
  `difference_within_band` settles it.
- **`anchoring_index`** — the share of a classifier's decision function
  carried by features identical at every timepoint. Above 0.20, a fifth of
  the segmentation is T0 geometric memory rather than measurement. The
  impurity-based figure is an *upper bound* (impurity importance over-weights
  continuous high-cardinality features, which normalised coordinates are);
  `permutation_anchoring_index` computes it on held-out blocks instead.
- **`temporal_generalisation_matrix`** — train at several timepoints, predict
  all of them. A flat row generalises; a row decaying with `|t_pred − t_train|`
  is extrapolating, and `staleness_half_life` turns that slope into a
  re-anchoring interval.
