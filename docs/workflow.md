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

## 3. Define materials

Draw a rectangle or polygon on the global histogram — neutron on the x-axis,
X-ray on the y-axis — and save it with a name (*Lithium*, *Separator*,
*Aluminium*). Draw one per material. Use *✂ Segment Current* to see them in
the slice viewer and adjust until they look right at the first timepoint.

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
range and keeps the strongest setting that costs no material any of its
volume, and shows you the sweep it used.

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
| every material present at every timepoint | Something vanished. The report says whether smoothing did it. |
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

## The one-line summary

Draw your materials once, say which ones cannot change, let the software pick
the smoothing, and read the health check before you read the numbers.
