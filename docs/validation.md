# Validation

Every number on this page is produced by

```bash
python -m validation.run          # ~4 min on 4 cores; tables → validation/results/
python -m validation.scaling      # time and memory of the spatial pass
```

with the default arguments (seed 0). `validation/results/` holds the CSVs of
the run quoted here. Nothing on this page comes from data that is not in the
repository. Claims made elsewhere about the battery series BiTS was developed
on (26 timepoints, not distributed) are labelled as such where they appear,
and are not relied on here.

## The phantom

`validation/phantom.py` builds a 4-D cell with **exact labels**: air around
a cylindrical aluminium casing (wall 3–4 voxels), filled with electrolyte,
holding a lithium electrode slab. A **reaction product** grows on the
electrode's electrolyte-facing side, consuming lithium: **absent at T0**, one
voxel thick at T1, one more voxel each step. The default is 6 timepoints of
12 × 64 × 64 voxels.

| class | neutron | X-ray |
|---|---|---|
| Air | 400 | 300 |
| Aluminium | 700 | 1500 |
| Electrolyte | 1100 | 700 |
| Lithium | 1700 | 450 |
| Product | 1350 | 1000 |

The measured volumes are made as a scanner makes them: the ideal map is
blurred (Gaussian σ = 0.8 voxel — partial volume at every interface) and
Gaussian noise is added (σ = 60 per channel by default, one sixth of the
closest class spacing). Scenarios add X-ray gain drift, beam-hardening-like
cupping, an X-ray/neutron offset, structure along z, or a second state of
lithium.

**Regions.** Every method that needs user input gets the same thing a careful
user would draw: each class's true mask, eroded by one voxel so the region
stays clear of the partial-volume rim. A class thinner than three voxels
therefore has an empty region — the product at T0 and T1.

**Scoring.** Dice per class per timepoint, averaged over the series. A class
correctly absent (the product at T0) scores 1; predicting any voxel of it
there scores 0, so the product's mean over 6 timepoints is at most 0.83 for a
method that puts even a few voxels of it at T0 — every method here does.

## 1. Against other methods

| method | Air | Aluminium | Electrolyte | Lithium | Product | mean |
|---|---|---|---|---|---|---|
| **BiTS (locked, auto smoothing)** | 0.975 | 0.941 | 0.994 | 0.983 | **0.788** | **0.936** |
| Pixel classifier (random forest, ilastik features) | 0.985 | 0.932 | 0.969 | 0.915 | 0.726 | 0.905 |
| Random walker (seeded with the regions) | **0.999** | **0.994** | 0.995 | 0.866 | 0.506 | 0.872 |
| K-means per timepoint\* | 0.966 | 0.958 | 0.977 | 0.975 | 0.315 | 0.838 |
| Multi-Otsu (neutron) | 0.978 | 0.919 | 0.775 | 0.922 | 0.254 | 0.769 |
| GMM per timepoint\* | 0.937 | 0.855 | 0.975 | 0.587 | 0.234 | 0.718 |
| Multi-Otsu (X-ray) | 0.898 | 0.829 | 0.769 | 0.355 | 0.259 | 0.622 |
| Otsu (neutron, 2 classes) | 0.825 | 0.000 | 0.876 | 0.000 | 0.167 | 0.373 |

\* Unsupervised: clusters are named by Hungarian matching against the truth —
the most favourable naming possible, so these rows are **upper bounds**. In
use the naming is manual and can be worse.

Reading it honestly:

- BiTS has the best mean, but **not the best on every class**. The random
  walker is better on the thick, well-separated phases (air, casing), where
  its edge-following excels; BiTS loses a few percent there to its
  Unclassified label (3.8% of voxels — partial-volume voxels that match no
  class well) and to the casing's partial-volume rim.
- **The advantage is on the thin, growing phase** (product 0.79 vs 0.73 for
  the random forest, 0.51 for the random walker) and on lithium, whose
  signature it shares partly with the product's. Both are where the
  bivariate signature matters and spatial context does not help.
- The random forest uses the same regions with spatial features; it is the
  strongest baseline and within 0.03 of BiTS.
- Over five noise realisations (`seeds` scenario) the mean Dice is
  0.932 ± 0.007 for BiTS (range 0.919–0.937), 0.906 ± 0.002 for the random
  forest and 0.872 ± 0.000 for the random walker; on the product
  0.775 ± 0.017, 0.735 ± 0.012 and 0.505 ± 0.001. The ranking holds at every
  seed; BiTS varies most, through the automatic smoothing choice.
- **Not included: a U-Net** or other deep network. Training one needs
  labelled volumes, which is exactly what this setting does not have; one
  trained on the regions alone would be a pixel classifier with more
  parameters. A comparison against a network trained on fully annotated
  volumes would answer a different question.

## 2. Noise

| noise σ | BiTS | random forest | random walker | BiTS unclassified |
|---|---|---|---|---|
| 40 | 0.886 | **0.919** | 0.874 | 10.8% |
| 60 | **0.936** | 0.905 | 0.872 | 3.8% |
| 90 | **0.915** | 0.903 | 0.865 | 0.9% |
| 120 | 0.868 | **0.892** | 0.856 | 0.1% |

**BiTS is not best at every noise level, and not monotone in noise.** At low
noise the class clouds are narrow and the partial-volume voxels between them
(blur 0.8 voxel) match no class: 10.8% of voxels end Unclassified and count
as errors. That is by design — a voxel that is 40% lithium has no correct
single label — but it costs Dice, and a method that must assign every voxel
scores higher. The *mixed boundaries* option reports those voxels as
fractions instead; it is not scored here. At high noise the random forest's
smoothed features win on the product.

## 3. Smoothing strength

Mean Dice against a fixed strength β, and what *Auto* chose:

| β | 0 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 | auto |
|---|---|---|---|---|---|---|---|---|---|
| σ = 60 | 0.922 | 0.929 | 0.932 | 0.934 | 0.936 | 0.936 | 0.937 | 0.937 | 0.936 (β = 8) |
| σ = 90 | 0.909 | 0.919 | 0.924 | 0.928 | 0.929 | 0.930 | 0.931 | 0.932 | 0.915 (β = 0.25) |

- Accuracy is **flat from β ≈ 4 upwards**, so the choice is not fragile.
  At σ = 60 Auto stops at β = 8 because the next value changes only 0.15% of
  labels ("labels settled"); it no longer runs to the top of the grid.
- At σ = 90 the **thin-sheet guard** stops it at β = 0.25: stronger smoothing
  erodes the one-voxel product layer at T1 (product Dice at T1 0.50 at
  β = 0.25, 0.42 at β = 32) while improving the thicker layer later. The
  guard is deliberately conservative — it protects a phase that is just
  forming at a cost of 0.017 mean Dice here. Set the strength by hand if the
  thin stage is not of interest.

## 4. A phase absent at T0

| product defined | product | mean |
|---|---|---|
| at T0 only (region empty → no class) | 0.167 | 0.803 |
| **where it exists** (first timepoint its eroded region has voxels, T3) | **0.788** | **0.936** |

Previously a class with an empty region was dropped without a word; it is now
defined at its own timepoint, or reported as a health-check failure if it
cannot be defined at all (`tests/test_smoothing_and_class_birth.py`).

## 5. Instrument drift

X-ray gain drifting by a fixed fraction per timepoint (8 timepoints):

| gain drift per step | locked | adaptive, frozen | adaptive, anchored |
|---|---|---|---|
| 0 | **0.939** | 0.714 | 0.733 |
| +3% | **0.894** | 0.720 | 0.742 |
| +6% | **0.792** | 0.715 | 0.739 |

- **Locked mode is not drift-proof**: it loses 0.15 mean Dice at 6% per step,
  as designed — it assumes fixed class positions, and its health check
  reports the resulting unmatched voxels as drift rather than hiding them.
- The adaptive modes score lower overall on this phantom: they cannot define
  a class after T0 (product ≈ 0.125) and lose lithium to the product.
  Anchored drift tracking beats frozen boundaries by 0.02–0.03 at every
  drift level.
- The collapse described in `docs/v17_plan_evaluation.md` — a frozen
  boundary losing air entirely from the third timepoint — is reproduced on
  its own synthetic series (a shift of a full class spacing,
  `tests/test_model_segmentation.py::test_frozen_boundary_loses_air_as_documented`),
  **not** on this phantom, whose gain drift is gentler for air.

## 6. X-ray/neutron misalignment

Scored on the voxels both volumes cover after correction:

| X-ray offset (z, y, x) | uncorrected | after *Check Alignment* | offset measured |
|---|---|---|---|
| none | 0.936 | 0.936 | (0.00, +0.01, 0.00) → none applied |
| (0, 0.5, 0.5) | 0.911 | 0.911 | (0.00, −0.44, −0.60) → none applied |
| (0, 1, 1) | 0.854 | **0.934** | (0.00, −0.97, −1.01) → (0, −1, −1) |
| (0, 2, 2) | 0.885 | **0.936** | (0.00, −1.98, −2.01) → (0, −2, −2) |
| (2, 0, 0), structure along z | 0.814 | **0.858** | (−2.05, +0.01, −0.01) → (−2, 0, 0) |

- **A one-voxel offset costs 0.08 mean Dice and 0.24 on the thin product**;
  correction restores the aligned result. (A two-voxel offset scores higher
  than a one-voxel one uncorrected: the regions are drawn on the misaligned
  data, and at two voxels they fit the smeared clouds better — an offset
  does not degrade the result smoothly.)
- The correction applies **whole voxels only**. Interpolating a half-voxel
  shift away averaged out X-ray noise, narrowed the clouds and scored 0.861
  — worse than leaving it (0.911; a separate comparison, same scoring
  region). The sub-voxel remainder is reported.
- On the default phantom (identical along z) the z component is reported as
  unmeasurable, not guessed.
- In the z-offset case the correction leaves two slices unmeasured at each
  end; that, not the estimate, is why it does not return to 0.936.

## 7. Class shapes

| scenario | one Gaussian | mixture (BIC, ≤ 3) | components chosen |
|---|---|---|---|
| lithium in two states | 0.904 (product 0.695) | **0.914 (product 0.729)** | lithium 3, others 1 |
| cupping 0.3 (beam hardening) | 0.935 | 0.935 | all 1 |

The mixture model helps where a class's cloud straddles another's (the
single ellipse of two-state lithium covers the product). It changes nothing
when every class is one blob — cupping of 0.3 widens clouds without making
them multimodal, and BIC keeps one component. Artefacts that make clouds
overlap outright (strong streaks) are not corrected by any option here; the
streak and smear metrics (`docs/metrics.md`) flag them.

## 8. Materials from attenuation coefficients

The product given no region, placed instead from its coefficients (calibrated
on air and aluminium):

| product defined by | product | mean |
|---|---|---|
| its region (where it exists) | 0.788 | 0.936 |
| nothing | 0.167 | 0.803 |
| coefficients, exact | 0.752 | 0.923 |
| coefficients, +5% | 0.793 | 0.937 |
| coefficients, +15% | 0.647 | 0.899 |

Exact coefficients come within 0.04 of a drawn region. A 5% error happens to
score slightly better here (the reference spread is wider than the product's
own); 15% costs 0.1. In practice the X-ray coefficient at the effective
energy of a polychromatic spectrum is the uncertain one.

## 9. Scaling of the spatial pass

`python -m validation.scaling`: 5 classes, 5 sweeps, one core-bound Python
process on a 4-core cloud container. Peak memory is what the pass allocates
(`tracemalloc`), excluding the input volumes.

| volume | solver | time | peak memory | bytes / voxel |
|---|---|---|---|---|
| 64³ (0.26 M) | mean-field | 0.6 s | 47 MiB | 188 |
| | mean-field, slabs of 32 | 0.5 s | 30 MiB | 121 |
| | ICM | 0.4 s | 19 MiB | 75 |
| 128³ (2.1 M) | mean-field | 5.7 s | 376 MiB | 188 |
| | mean-field, slabs of 32 | 4.1 s | 144 MiB | 72 |
| | ICM | 3.3 s | 150 MiB | 75 |
| 192³ (7.1 M) | mean-field | 21.6 s | 1.24 GiB | 188 |
| | mean-field, slabs of 32 | 14.6 s | 332 MiB | 49 |
| | ICM | 14.4 s | 505 MiB | 75 |
| 256³ (16.8 M) | mean-field | 53 s | 2.94 GiB | 188 |
| | mean-field, slabs of 32 | 44 s | 607 MiB | 38 |
| | ICM | 50 s | 1.17 GiB | 75 |

- Time is linear in the number of voxels (≈ 0.4–0.65 µs per voxel and
  sweep); slabs are not slower — they are faster, the working set being
  smaller.
- Whole-volume mean-field needs ≈ `32·K + 30` bytes per voxel (126 at K = 3,
  188 at K = 5, 316 at K = 9). With slabs, the volume-wide cost is the
  4-byte label output and the rest is bounded by the slab: a 38 M-voxel,
  9-class volume that needs ≈ 11 GiB in one pass fits a 2 GiB budget in
  slabs. Slab results are identical to the whole-volume pass
  (`tests/test_chunked_mrf.py`).
- This is the spatial pass alone, per timepoint and per smoothing value
  tried. Timepoints are independent in locked mode but are processed one
  after another.

## What this does not show

- **Real data.** The phantom has piecewise-constant materials, Gaussian noise
  and Gaussian blur. Real reconstructions add ring and streak artefacts,
  non-Gaussian noise, texture within a material and interfaces that are not
  planar. The benchmark shows the method behaves as intended where its
  assumptions hold, and how it degrades when three of them do not (drift,
  offset, non-Gaussian classes); it is not evidence of accuracy on a
  particular instrument.
- **The adaptive (mixture) mode** is only scored in the drift scenario.
- **Deep networks** are not compared (above).
- One seed per scenario except `seeds` (five, for the three strongest
  methods on the default phantom); differences smaller than about 0.01 in
  the other tables are within the seed-to-seed variation.
