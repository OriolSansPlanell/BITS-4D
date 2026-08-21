# Quality metrics

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
