# Evaluation of the BiTS v17 implementation plan

**Verdict: the method is sound and worth building; the plan's code-level
claims are addressed to a different codebase and roughly a third of its
"fixes" were already in place here.** The method (§4–§8) is implemented, with
seven corrections listed at the end. Two parts (§7.2, §7.3) are deliberately
not built, for reasons the plan itself gives.

---

## 1. What was checked

Every code-level claim in the plan was checked against this repository
(`BITS-4D` at the commit this document was written on), not taken on trust.

### Claims that hold

| Plan | Claim | Verified against |
| --- | --- | --- |
| §1 | T0 labels are a pure function of `(n, x)`, so texture is conditionally uninformative | `ROIManager.is_inside_roi(neutron_values, xray_values)` is point-in-polygon in the (n, x) plane. Structural, not incidental. |
| §1, §9.1 | Texture is unreachable without frozen T0 geometry | `_extract_features_at_indices`: `advanced`/`expert` add normalised `(z, y, x)`; `expert` adds texture *on top*. Confirmed. |
| §1 | RF `expert` carries geometry identical at every timepoint | Coordinates normalised by volume extent, recomputed identically per timepoint. Confirmed. |
| §3.1 | Nothing excludes non-measurement voxels | No validity mask existed anywhere. Padding, NaN and saturation all entered histograms, fits and exports. Confirmed. |
| §3.3 | No drift compensation of any kind | Confirmed — there was none. |
| §9.2 | No gradient, cross-modal coherence or structure-tensor features | Feature list was `n, x, n/x, n+x, n−x, hypot, [z,y,x], [mean, std ×2], laplace(n)`. Confirmed. |
| §9.3 | `min_samples_leaf=5`, no depth cap | Confirmed. |
| §10 | **v16 has no spatial metric of any kind** | `utils/histogram_metrics.py` is entirely histogram-plane. Confirmed, and it is the sharpest observation in the document. |
| §11 | Reported accuracy is in-sample | `training_accuracy` is `mean(predict(x_train) == y_train)`. Confirmed. |

### Claims that do not hold here

| Plan | Claim | Reality in this repo |
| --- | --- | --- |
| §2 | Module map (`bits/core/universal_dataset.py`, `ml/universal_rf.py`, `improved_rf.py`, `create_labels_from_rois()`, `GUI_USER_GUIDE.md`) | **None of these files or functions exist.** The layout is `data/`, `histograms/`, `segmentation/`, `utils/`, `gui/`. |
| §3.4 | `generic_filter(vol, np.std, size=3)` is a Python callback over 38 M voxels | **Already fixed.** The code already used `uniform_filter` mean/mean-square. |
| §9.3 | Set `class_weight='balanced_subsample'` | **Already set.** |
| §9.2 | "`sobel` is imported but used only in `_extract_features_single`" | No `sobel` import existed at all. |
| §9.4 | Expert docstring says `+8`, implements `+4`; expert is 13 features | No such docstring. Feature counts here are 6 / 9 / **14**. |
| §11 | Only in-sample accuracy is available | An OOB score was also computed. It leaks too — the bootstrap draws from the same autocorrelated voxels — so the critique survives, but the plan overstates the starting point. |

The evidence-base CSV schema quoted in §10 (`scope, timepoint, metric, class,
value, label, unit, meaning, better_when`) *is* this repository's — it is the
output format of `utils/histogram_metrics.write_metrics_csv`. So the plan's
**measurements** come from here while its **code trace** comes from a sibling
"universal" variant. Every module path had to be re-derived; the diagnoses
themselves survived that translation.

## 2. Is the method right?

Yes, and one idea in it is genuinely good.

**The strongest claim is correct.** A manually drawn ROI and a free Gaussian
mixture are the two limits of a single model: put a Normal-Inverse-Wishart
prior on each component centred on the ROI's moments, and the prior strength
κ₀ interpolates between "frozen at T0" (κ₀ → ∞) and "unconstrained" (κ₀ → 0).
This reframes a methodological argument as a dial, and the dial is
measurable. The implementation reproduces both limits exactly — verified by
test, including that the movement is monotone in between.

**Fitting on the histogram rather than the voxels is correct.** A mixture
likelihood depends on the data only through per-component sums of `1`, `v`
and `v vᵀ`, which are additive and can be accumulated per bin once.

**The MRF is necessary, not optional.** A per-voxel mixture has no spatial
term at all; its raw labels are speckled and would be worse than the
classifier being replaced. This is stated plainly in the plan and it is right.

**Drift tracking on inert anchors is the core of the answer** to improving
time-series segmentation, and it works: on a synthetic series where the
histogram drifts by a full class spacing, a frozen boundary loses a class
entirely by the fourth timepoint (Air → 0 voxels, absorbed into Aluminium),
while the anchored model tracks all three classes to within a handful of
voxels — including a genuine shrinkage that is *not* drift.

## 3. Corrections made

Seven changes to what the plan specifies. Each is a defect in the plan, not a
matter of taste.

**1. `neutron_floor=3000` is a magic number from one dataset.**
A hard intensity floor tuned on one experiment silently deletes a genuinely
low-attenuation phase in the next one, and the stated derivation ("Air peak
12 836 − 3σ") only yields 3000 for that σ. `ValidityPolicy` defaults to
rejecting what is *provably* not a measurement — non-finite values and the
exact sentinel the padding was written with — which is what actually removes
the 97 k-voxel artifact. Floors exist but are opt-in, and `estimate_floor()`
derives a suggestion from the data's own lower tail.

**2. "262 144 bins… with no loss" is false.**
Fitting on bin *centres* inflates every covariance by the bin variance
(h²/12 per axis — the Sheppard bias). `HistogramCache` therefore stores per
bin the count **and the first and second moments of the voxels in it**, so
the M-step is algebraically the voxel-level one. The only approximation left
is that voxels sharing a bin share a responsibility. Verified by test against
the voxels directly, to 1e-10.

**3. `DriftTracker.normalise(vol1, vol2, t)` rewrites the data.**
That costs a full pass over every voxel at every timepoint and — worse —
silently changes the units of every downstream histogram, statistic and
export, so numbers stop being comparable with anything produced before. The
drift is instead applied to the **model**: a component anchored at
`(μ₀, Σ₀)` is anchored at `(s⊙μ₀ + d, S Σ₀ Sᵀ)`. Equivalent for the fit,
touches no voxel. Value-space normalisation remains available and explicit.

**4. The plan does not say how an anchor is located at time *t*.**
`estimate(dataset, t)` appears to need the segmentation it is meant to
enable. It does not: an anchor is a dense isolated mode, so a mean-shift
started from its last known position finds it with no labels. Two guards were
needed and are not in the plan:

- the search must be **cumulative** (start from the previous estimate, check
  the step since then). Checking against T0 instead means every anchor is
  rejected as implausible exactly when cumulative drift is largest — the
  estimator gives up where it is most needed;
- **anchors that converge on the same mode** must all be rejected. Neither
  moved implausibly far on its own, so the distance guard sees nothing wrong,
  yet the drift is being estimated from one mode counted twice. Which one is
  the impostor cannot be told from the anchors alone, so picking one would be
  a guess dressed as a measurement.

**5. κ₀ is unusable as a GUI control.**
As a raw pseudo-count, `κ₀ = 1000` means something completely different for a
4 000-voxel class than for a 4 000 000-voxel one — and the plan proposes
exposing it as a slider. `anchor_strength_to_kappa` maps a dimensionless
`strength ∈ [0, 1]` onto `κ₀ = n·s/(1−s)`, scaled by each class's own size,
so the same slider position means the same thing at every class size and on
every dataset.

**6. `A > 0.20` rests on a biased estimator.**
`feature_importances_` is impurity-based and over-weights continuous
high-cardinality features — which normalised coordinates are precisely. The
anchoring index computed that way is an **upper bound**;
`permutation_anchoring_index` computes it on held-out blocks instead, and
both are reported.

**7. The acceptance criteria contradict each other.**
§13 requires v16 notebooks to "reproduce the current CSVs bit-for-bit" while
§9.3 changes `min_samples_leaf`, `max_depth` and the sampling scheme, and
§3.1 changes which voxels enter every fit. Those cannot both hold. Resolved
by versioning: `LEGACY_SPECS` reproduces the old feature columns exactly
(verified bit-for-bit by test), capacity limits are constructor arguments
with the old values available, and the validity mask is documented as an
intentional break rather than a silent one.

Two further additions the plan does not call for:

- **The model can abstain.** The plan has the outlier component but no
  per-voxel rejection at the output. For a method whose whole purpose is
  tracking drift, the honest failure mode is saying "I don't recognise this",
  so labels carry `-1` for voxels no class claims.
- **A memory-scalable MRF.** The plan says "chunked over z", but mean-field
  needs a `[Z, Y, X, K]` responsibility array — 1.4 GB for 38 M voxels and 9
  classes, before temporaries. An ICM solver is provided that keeps hard
  labels instead and costs ~9 bytes per voxel *regardless of K*; `refine`
  picks between them from a memory budget.

Two smaller corrections found while testing:

- **BIC/ICL must use the voxel count**, not the number of occupied bins.
  Otherwise the criterion depends on the histogram resolution rather than the
  data. Storing the within-bin scatter makes the log-likelihood a genuine
  per-voxel quantity, so this is well defined.
- **`f_rind` is scale-dependent.** A genuinely displaced *small* object
  erodes away in two voxels just as a boundary shell does. The component
  count is the robust signal, and both are now measured at the same erosion
  depth so they describe the same thing.

## 4. Deliberately not implemented

**§7.2 `BirthDeathTransition` (birth by novelty).** Component *birth* is the
one part of the design that cannot be validated without data containing a
genuinely new phase, and an untested birth rule that fires on noise is worse
than no birth rule. What *is* implemented is **dormancy**: a component whose
weight collapses is frozen rather than deleted, so a phase that disappears
and returns keeps its identity and its time series stays continuous. That is
most of the practical value and it is testable.

**§7.3 fuel-cell path (`reference_subtraction.py`, robust PCA).** The plan's
own advice here is the right one and does not need this machinery: where a
dry reference scan exists, water thickness follows directly from
Beer–Lambert, `t_w = −ln(I_wet/I_dry)/Σ_w` — quantitative, calibratable, no
training, and it handles sub-resolution saturation natively. Segmenting first
would throw that away. There is also no fuel-cell data here to validate
against.

**§11 manual ground truth.** Labelling 3–5 ROIs at T0/T12/T25 blind is a
human task. The machinery that consumes it — block cross-validation, the
temporal generalisation matrix, per-class IoU and κ — is built and tested.

**§12 task 1, §13 acceptance runs.** Those are measurements on the Li-In
dataset, which is not in this repository. The tools to make them are here;
the numbers have to come from the data.

## 5. §14 — the open question, restated

The plan asks whether the T9–T11 changes are the sample or the instrument. It
is now answerable from inside the software rather than only from the
acquisition log: `validity_report()` records the rejected fraction per
timepoint and `find_acquisition_steps()` flags a jump in it. A step in *how
much of the volume is not a measurement* is an acquisition change, and
absolute volume comparisons across it are not valid. The 97 k zero-valued
voxels appearing at T10 would show up there directly. Check that before
reading T0→T25 as a continuous trend.

## 6. Where to read more

- [`docs/model_segmentation.md`](model_segmentation.md) — how to use the
  model, what each control does, and the limits of each part.
- [`docs/metrics.md`](metrics.md) — the metric registries, including the new
  spatial family.
- [`docs/architecture.md`](architecture.md) — module map.
