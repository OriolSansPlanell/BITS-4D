# Segmenting 4D multimodal data with BiTS — the workflow

Nine steps. Steps 1 and 3 are the ones you already know; the rest are quick,
and each exists because skipping it is how a run goes wrong quietly.

```
 1. LOAD              neutron + X-ray volumes, all timepoints
 2. CHECK DATA        automatic — what is actually measurable
 3. DEFINE MATERIALS  draw regions on the histogram at T0
 4. MARK CONTROLS     tick what should not change
 5. LINK BOUNDARIES   which pairs mix  (optional)
 6. PREVIEW           one timepoint, smoothing chosen for you
 7. RUN THE SERIES    all timepoints
 8. HEALTH CHECK      automatic, before you see any numbers
 9. RESULTS           volumes, maps, export
```

---

## 1. Load

*File → Load 4D Dataset.* Neutron and X-ray stacks with identical shapes:
`(T, Z, Y, X)` for 4-D, `(Z, Y, X)` for 3-D (*Settings → Data Mode*). The
histogram is computed on load.

## 2. Check data

*Analytics → Time Series Segmentation → Check Data…* — and automatically on
load.

It answers three questions:

- how much of the array holds real measurements;
- **whether both instruments cover the same region.** If the neutron and
  X-ray fields of view differ, the non-overlapping part has a value in one
  channel and nothing in the other. Treated as data it forms a large, static
  blob pinned to zero in one axis that some material will absorb, inflating
  that material's spread and dragging its position. Those voxels are
  excluded — a material can only be identified where both measurements
  exist;
- whether the amount of usable data **changes part-way through**. A step
  there is an acquisition change: a shifted field of view, a different
  reconstruction, a detector fault. Volume comparisons across that point are
  not meaningful, and the series should be analysed as two segments.

Read this before anything else. It is cheap and it decides whether the rest
of the numbers mean anything.

### Check alignment

*Analytics → Time Series Segmentation → Check Alignment…* The bivariate
histogram pairs each neutron voxel with the X-ray voxel at the same index,
so it assumes the two volumes are co-registered. If they are offset by even
one voxel, every interface pairs two different materials: the clouds smear
towards each other and a thin phase can vanish. On the synthetic validation
a one-voxel offset costs 0.08 mean Dice and almost a third of the thin
phase's accuracy ([validation.md](validation.md)).

The check measures the offset of the X-ray volume at the first, middle and
last timepoints by maximising the mutual information of the two volumes
(the modalities have different contrast, so their dependence is compared,
not their values). An axis along which the sample has no structure is
reported as unmeasurable rather than guessed. If a whole-voxel offset is
found, it offers to correct it — once for the series if the offset is
constant (a mounting offset), per timepoint if it changes (something
moved). The correction moves the X-ray volumes by whole voxels, in memory
only; the vacated edge is treated as unmeasured. A sub-voxel remainder is
reported but not interpolated: resampling averages away X-ray noise, which
on the validation phantom cost more accuracy than the half-voxel offset
itself.

Only a rigid translation is measured. Rotation, scaling or deformation
between the instruments need a registration tool (elastix, ANTs) before
loading; resampling a volume scanned at a different resolution onto the
other's grid is `model.registration.resample_to_shape`.

## 3. Define materials

Draw a rectangle or polygon on the global histogram — neutron on the x-axis,
X-ray on the y-axis — and save it with a name (*Lithium*, *Separator*,
*Aluminium*). Draw one per material. Use *✂ Segment Current* to see them in
the slice viewer and adjust until they look right at the first timepoint.

You can draw several regions before saving: each stays on the histogram in
its own colour, and *Save as Class* asks for a name for each and makes one
class per region. To adjust a region, **click it** on either histogram and
move it with the **arrow keys** (one bin per press, **Shift** for ten);
**Backspace** or **Delete** removes it — for a saved class, after asking.

Two things worth knowing:

- **The names you give here are the names in every output**, in this order,
  and the integer values in the exported label volumes follow the same order.
  Nothing renames or reorders them.
- **These regions are definitions, not boundaries.** What the software takes
  from them is where each material sits on the histogram and how much it
  spreads. It does not re-use the polygon edges as a fixed partition, which
  is why a voxel near a boundary can be assigned sensibly instead of by which
  side of a hand-drawn line it fell on.

Anything you draw can also come from the slice viewer: draw a box or grow a
region there and convert it into a histogram region.

Or let K-means propose the regions (*Auto Seg* tab, or *🔍 Auto-Detect*
under the slice), at the scale the question needs:

| Scope | Clusters | Use it when |
| --- | --- | --- |
| **Slice** | the slice on screen | a quick look at which phases a slice holds |
| **Volume** | every voxel of this timepoint | defining materials at the reference timepoint; *Segment All* extends them to every timepoint |
| **Time series** | every timepoint, one shared clustering | following phases through the experiment, including phases that exist only in some timepoints |

Every scope puts its clusters straight into the selection panel, one class
per cluster, so they can be ticked off, renamed and edited like drawn regions
without drawing anything. A slice or volume cluster's region is its exact
K-means region, so segmenting with it reproduces the cluster at any
timepoint.

The time-series scope writes a layer per cluster at every timepoint, keeps
each cluster's colour throughout, and flags the phases present only in some
timepoints (a reaction product, a deposit). *Export Cluster Timeline* writes
each cluster's share of the sample over time. A large instrument drift can
look like a new phase, so run *Check Instrument Stability* when one appears
unexpectedly.

### Phases that appear during the experiment

A material does not have to exist at the first timepoint. If its region
selects nothing at the reference timepoint — a reaction product that has not
formed yet, a layer still thinner than the region — it is defined at the
first timepoint where it has voxels, and the panel shows that timepoint next
to its source ("drawn (T3)"). The health check then expects it to be absent
before, and says so. A material that cannot be defined anywhere is **not
dropped silently**: the run goes ahead without it and the health check
fails, naming it and why.

### Materials from attenuation coefficients

For a phase you cannot draw at all, *Analytics → Time Series Segmentation →
Add Materials from Attenuation Coefficients…* places it from its neutron and
X-ray attenuation coefficients. Tick two (or more) materials you have drawn
whose coefficients you know — air and aluminium are ideal — and enter theirs;
they fix a straight line from coefficient to grey value for each instrument.
Then list the materials to place. Their spread is taken from the reference
materials (it is a property of the instrument, not the material). The
calibration and the predicted positions are shown before anything runs.

Enter X-ray coefficients at the effective energy of your spectrum — the
prediction is only as good as that number. On the validation phantom a
predicted phase scored within 0.04 Dice of a drawn one with exact
coefficients, and lost a further 0.1 with coefficients 15% off
([validation.md](validation.md)). A predicted material that is never found
is a warning in the health check.

### Irregular material shapes

By default each material is described by a single elliptical cloud. Tick
*Allow irregular material shapes* on the Materials tab when a material
occupies more than one place on the histogram — two states of the same
phase, or a cloud bent by an artefact — and its single ellipse would cover a
neighbour. Each material then gets as many sub-clouds (up to three) as its
own voxels justify (by the Bayesian information criterion); a material that
is one cloud stays one cloud, so ticking it costs nothing when it is not
needed.

## 4. Mark control materials

On the **🧱 Materials** tab every material is listed with where it came from,
how many voxels it holds, and a setting in the last column. Set the materials
that **cannot change during the experiment** — a casing, a support, a
structural metal — to *Stays unchanged*.

Materials copied from a K-means clustering appear here alongside the drawn
ones and behave identically, so a cluster you recognise as the casing can be
marked as a control like any other.

One click each, and it buys an independent check on everything else: if a
control material's volume moves, the segmentation is wrong, not the sample.
Without it you have no way to tell a real change from a broken one.

> Do not tick something that reacts. Its real change would be treated as an
> instrument effect and subtracted from every other material. The software
> cannot tell the difference, which is why there is no default.

## 5. Link boundaries *(optional)*

Where two materials touch, some voxels contain a bit of both. Those voxels
have no correct label — the answer for a voxel that is 40 % lithium is not
"lithium", it is 0.4 — so instead of forcing one, the software can report how
much of each is present.

Leave *Look for boundaries that behave like a mix of two materials* ticked and
it will tell you which of your materials look like a boundary rather than a
phase: elongated, and pointing along the line joining two others. You decide
what to do about it. Skipping this costs nothing else.

## 6. Preview

**▶ Preview this timepoint** on the Materials tab runs the current timepoint
only. Look at the result beside the raw slice before committing to the whole
series.

Leave **Smoothing strength** on **Auto**. Smoothing uses neighbouring voxels
to clean up noisy assignments, and it is the one setting that can destroy a
result invisibly: too strong and a small material is simply erased, with
everything downstream still looking healthy. Auto does not guess — it tries a
range and applies three rules:

- **no material may lose volume** to smoothing, and **no thin sheet may be
  eroded** (a structure one or two voxels thick keeps at least 80% of
  itself);
- these are checked at the reference timepoint **and** where phases that
  appear later are still thin, and in the middle and at the end of the
  series;
- among the settings that pass, it stops at the first one after which **more
  smoothing changes almost nothing** (under 0.2% of voxels). If the labels
  are still changing at the top of the range, the range is extended; if they
  never settle, the health check warns that the chosen value sits at the
  ceiling.

The chosen value, the reason, and how much the labels would change at the
next setting are reported with the results. On the validation phantom the
accuracy is flat over a wide range of settings, so the choice is not
fragile; at high noise the thin-sheet rule holds smoothing back to protect a
one-voxel layer, at a small cost to the thicker parts
([validation.md](validation.md)).

## 7. Run the series

**▶▶ Run all timepoints.** Every timepoint is measured against the same fixed material definitions.
Voxels move between materials; the definitions do not move.

That is the design decision that makes the results mean something:
attenuation coefficients are material constants, so a material that appears
to move is a material absorbing something that should have left it. Because
the definitions are fixed, **timepoints are completely independent** — the
result is identical run forwards, backwards, or one timepoint at a time.

## 8. Health check

Runs automatically before you see any numbers, and reports in plain
sentences. Each failure names the material involved and what to do:

| Check | What a failure means |
| --- | --- |
| materials defined | A material could not be defined from its region anywhere, and is missing from the results. |
| every material present at every timepoint | Something vanished. The report says whether smoothing did it. A material defined at a later timepoint, or placed from coefficients, is expected to be absent before it forms. |
| smoothing strength | How the value was chosen; a warning if the labels were still changing at the top of the tested range. |
| control materials stable | A material you said would not change, changed. Something is wrong with the segmentation. |
| unmatched voxels | Voxels that matched nothing. Either a material is missing, or the measurement has drifted. |
| voxel budget | Every voxel counted exactly once. A failure here is a bug — please report it. |
| usable data stable | The acquisition changed part-way through. |
| linked boundaries | A material you linked does not actually sit between its two neighbours. |

Some conditions stop the run outright rather than warning: smoothing removing
a material, too many unmatched voxels, a voxel budget that does not close, or
a spatial cleanup that cycles instead of settling. In those cases nothing is
applied and you are told why — a results screen full of numbers describing
the wrong voxels is worse than no results.

**If it reports drift**, the message will say so and point at *Check
Instrument Stability*. The distinction it draws is between voxels that never
matched (a missing material — visible from the first timepoint) and voxels
that stopped matching as the series went on (the measurement moved). They
need opposite responses.

## 9. Results

Segmented layers appear in the slice viewer for every timepoint, and everything
downstream works on them:

- **Export** (*File → Export*) — masked volumes, binary masks, label maps,
  per-material histograms, and `segmentation_report.txt` recording each
  material's name, its value in the label volumes, and its voxel count at
  every timepoint.
- **Histogram + slice figure** (*File → Export Histogram + Slice Figure*,
  Ctrl+Shift+F) — one SVG with the local histogram and every material's
  selection on the left, and the slice on screen with the same materials
  highlighted on the right. It shows exactly what you are looking at: pick the
  timepoint, plane and slice first, and untick any material you want left
  out.
- **Quality metrics** — *Analytics → Histogram Time Analysis*: histogram
  metrics, spatial metrics (position, spread, how many pieces a material is
  in, contact area between materials), and the histogram evolution images.
  See [metrics.md](metrics.md).

---

## When the drift is real

Locked definitions are right when the instrument is stable, which is the
normal case and the default. If *Check Instrument Stability* shows the
histogram genuinely moving — beam current, detector gain, scatter build-up —
you have three options, in order of preference:

1. **Re-draw the materials part-way through** and analyse the series in
   segments. Simplest, and it keeps every result anchored to something you
   looked at.
2. **Correct for the movement.** *Check Instrument Stability* measures it
   from the control materials and writes it out; the correction is applied to
   the material definitions, never to your data, so exports stay in their
   original intensity units.
3. **Untick "Lock material definitions"** (Advanced). The definitions are
   then allowed to follow the data, anchored to where you drew them. Use this
   last: a definition free to move can absorb a real change in the sample and
   report it as no change at all. Check the control materials carefully
   afterwards.

## Quick reference

| I want to… | Go to |
| --- | --- |
| read about any of this from inside the app | Help → Manual (F1) |
| see what is actually measurable | Analytics → Time Series Segmentation → Check Data |
| segment the whole series | 🧱 Materials tab → Run all timepoints |
| find out whether the instrument moved | Analytics → Time Series Segmentation → Check Instrument Stability |
| measure shape, position, contact area | Analytics → Histogram Time Analysis → Spatial Metrics |
| export volumes and a written report | File → Export |
| a figure of the histogram next to the slice | File → Export Histogram + Slice Figure |
| move or delete a region | click it on the histogram, then arrow keys / Backspace |

## The one-line summary

Draw your materials once, say which ones cannot change, let the software pick
the smoothing, and read the health check before you read the numbers.
