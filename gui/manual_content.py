"""The text of the in-application manual.

Kept apart from the window that shows it so the content can be checked,
searched and exported without a display. Every section is a dict with a
``title``, a ``body`` in HTML, and optional ``keywords`` that widen the
search beyond the words actually printed.

Two audiences share this file, and the split is deliberate: the *How to*
sections assume you know segmentation and nothing else, while the
*Mathematics* sections state exactly what is computed, for anyone who has to
defend a number in a paper. Neither is a summary of the other — the second
gives the formula the first describes in words.
"""

from __future__ import annotations

from typing import Dict, List

_CSS = """
<style>
  body { font-family: sans-serif; font-size: 10pt; line-height: 1.5; }
  h1 { font-size: 15pt; margin-bottom: 2px; }
  h2 { font-size: 12pt; margin-top: 16px; color: #1a4f7a; }
  h3 { font-size: 10.5pt; margin-top: 12px; color: #333; }
  code, .m { font-family: monospace; background: #f4f4f4; padding: 1px 3px; }
  .m { display: block; margin: 8px 0 8px 16px; padding: 8px;
       border-left: 3px solid #c8d8e8; background: #f7fafc; }
  .note { background: #fff8e5; border-left: 3px solid #e8c88a;
          padding: 6px 10px; margin: 10px 0; }
  .warn { background: #fdeeee; border-left: 3px solid #d88;
          padding: 6px 10px; margin: 10px 0; }
  table { border-collapse: collapse; margin: 8px 0; }
  th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left;
           font-size: 9.5pt; }
  th { background: #f0f4f8; }
  dt { font-weight: bold; margin-top: 6px; }
</style>
"""


# ── how to ───────────────────────────────────────────────────────────────────

_GETTING_STARTED = """
<h1>Getting started</h1>
<p>BiTS segments <b>paired</b> neutron and X-ray tomography. Neither modality
alone separates every material: neutrons see hydrogen and lithium strongly and
metals weakly, X-rays the other way round. Plotting each voxel as a point in
the <i>(neutron, X-ray)</i> plane separates materials that overlap completely
in either channel on its own.</p>

<h2>The whole workflow in nine steps</h2>
<ol>
<li><b>Load</b> the two stacks — <i>File → Load 4D Dataset</i>.</li>
<li><b>Check the data</b> — automatic on load. Says how much is measurable
    and whether both instruments cover the same region.</li>
<li><b>Define materials</b> — draw a region per material on the histogram and
    press <i>✂ Segment Current</i>.</li>
<li><b>Mark controls</b> — on the <i>Materials</i> tab, set anything that
    cannot change to <i>Stays unchanged</i>.</li>
<li><b>Link boundaries</b> (optional) — leave <i>Look for mixed
    boundaries</i> ticked.</li>
<li><b>Preview</b> one timepoint and look at it.</li>
<li><b>Run all timepoints.</b></li>
<li><b>Read the health check</b> — it appears before the numbers do.</li>
<li><b>Export</b> volumes, label maps and a written report.</li>
</ol>

<h2>On a laptop screen</h2>
<p>The window opens at a size that fits your screen and rearranges itself as
it gets smaller, so it works on a 14" or 16" laptop:</p>
<ul>
<li>When the histogram column is narrow, the global and local histograms
    become two tabs above one plot instead of sitting side by side.</li>
<li>Rows of buttons wrap onto extra lines instead of pushing the window
    wider, and a column that still does not fit scrolls.</li>
<li>The histogram's display settings (log scale, ROI visibility, colour
    range) are in the <b>Display ▾</b> drop-down.</li>
<li>The spatial selection tools under the slice can be folded away with
    <b>🛠 Spatial tools</b>; on a small screen they start folded.</li>
<li>Time navigation is one bar along the bottom of the window.</li>
</ul>
<p>Every divider between panels can be dragged.</p>

<h2>What the two axes are</h2>
<p>The horizontal axis is <b>neutron</b> intensity, the vertical is
<b>X-ray</b>. A region you draw is a set of intensity pairs, so it selects
every voxel whose pair falls inside it, wherever that voxel is in the volume.
That is why a selection made on one timepoint can be applied to all of
them.</p>

<div class="note">The region is drawn <i>filled</i>, not as an outline,
because containment is decided by the winding rule. In a polygon that crosses
itself an area can be ringed by edges and still be outside. What is shaded is
what gets segmented.</div>
"""

_DEFINING = """
<h1>Defining materials</h1>

<h2>Drawing a region</h2>
<ol>
<li>Choose <b>Rectangle</b> or <b>Polygon</b> on the <i>Manual ROI</i>
    tab.</li>
<li>Drag (rectangle) or click each vertex and close the shape (polygon) on
    the global histogram.</li>
<li>Press <b>Save as class</b> and name it — <i>Lithium</i>,
    <i>Separator</i>, <i>Aluminium</i>.</li>
<li>Repeat for every material. You can also draw several regions first and
    save them together: each region you draw stays on the histogram in its
    own colour, and <b>Save as class</b> then asks for one name per region
    and makes one class each.</li>
<li>Press <b>✂ Segment Current</b> to see them in the slice viewer.</li>
</ol>

<div class="note">The names you give here are the names in every output, in
this order, and they set the integer values in the exported label volumes.
Nothing renames or reorders them later.</div>

<h2>Moving and removing a region with the keyboard</h2>
<p>Click inside any region on either histogram — saved or not — to select it;
it gets a black-and-white dashed outline, and its row is highlighted in the
selection panel. Then:</p>
<table>
<tr><th>Key</th><th>Effect</th></tr>
<tr><td>← → ↑ ↓</td><td>Move the region by one histogram bin.</td></tr>
<tr><td>Shift + arrow</td><td>Move it by ten bins.</td></tr>
<tr><td>Backspace or Delete</td><td>Remove it. Removing a saved class asks
    first, and asks what to do with any segmentation made from it.</td></tr>
</table>
<p>Click an empty part of the histogram to deselect. Moving a saved class
changes its definition; segmentation already computed from it keeps the
outline it was made with until you segment again.</p>

<h2>Editing one afterwards</h2>
<p>In the selection panel, select a class and press <b>Edit</b>. It returns to
the canvas with draggable vertices (anything you were drawing stays, as its
own unsaved region). Save it again when you are done. Untick a
class to hide it — a hidden class is excluded from segmentation too, so what
you see on the histogram is always exactly what gets segmented.</p>

<h2>Starting again from a clean view</h2>
<p><b>🧹 Clear Highlight</b> (under the slice, with the spatial tools) hides
every coloured overlay — segmentation layers, saved selections, the grown
region — and unticks every class, so the next region you draw is segmented,
and shown, on its own. Nothing is deleted: tick a class to bring its layers
back. A layer with no class behind it (from Otsu, or an unsaved region)
comes back when you segment it again.</p>

<h2>From the slice viewer instead</h2>
<p>If a material is easier to point at in space than in the histogram: draw a
box on a slice, or use <b>Region Grow</b> (2-D or full 3-D), then convert the
selection into a histogram region. The software takes the intensity pairs of
the voxels you picked and builds the region around them.</p>

<h2>From K-means</h2>
<p>K-means groups voxels by their (neutron, X-ray) values without you drawing
anything. On the <i>Auto Seg</i> tab (or with <b>🔍 Auto-Detect</b> under the
slice) choose a <b>scope</b> and the number of clusters, then press
<b>Run K-means</b>:</p>
<table>
<tr><th>Scope</th><th>What it clusters</th><th>What you get</th></tr>
<tr><td><b>Slice</b></td><td>The pixels of the slice on screen.
    Seconds.</td><td>A layer per cluster on that slice. <b>Segment
    Current</b> extends the clusters to the whole volume.</td></tr>
<tr><td><b>Volume</b></td><td>Every voxel of this timepoint.</td><td>A layer
    per cluster at this timepoint. <b>Segment All</b> extends them to every
    timepoint.</td></tr>
<tr><td><b>Time series</b></td><td>Every timepoint at once, with one shared
    set of clusters.</td><td>A segmentation layer per cluster at every
    timepoint, in the same colour throughout, plus a separate cluster for
    each phase that is present only in some timepoints.</td></tr>
</table>
<p>Every scope puts its clusters straight into the <b>selection panel</b>
under the histogram, one class per cluster, so there is nothing to draw or
save by hand. From then on a cluster is a class like any drawn one: tick it
off to hide it (and leave it out of segmentation), rename it with a
double-click, edit or remove it, and on the Materials tab mark it <i>Stays
unchanged</i> if it is the casing. Rename them to something meaningful before
you run the series. Running the same scope again replaces its earlier
clusters.</p>
<p>A slice or volume cluster's region on the histogram is its exact K-means
region, so segmenting with it gives back exactly the voxels K-means put in
the cluster — at any timepoint. Time-series clusters are marked
<i>[layers]</i>: a phase found only in some timepoints cuts a hole in its
neighbour's region, which no single outline can describe, so their stored
layers are the segmentation and <i>Segment Current</i> leaves them as they
are.</p>

<h3>Phases present only in some timepoints</h3>
<p>With <i>Find phases present only in some timepoints</i> ticked, the
time-series scope also looks for regions of the histogram that fill at some
timepoints and are empty at others — a reaction product, a deposit, a phase
that forms and dissolves. Each becomes a cluster named <i>Transient phase
1, 2, …</i> however small it is; plain K-means would miss it, because it
spends its clusters on the bulk of the voxels. The summary says where each
one is present (for example <i>T2–T5</i>), and <b>Export Cluster
Timeline</b> writes every cluster's share of the sample at every timepoint
(CSV) with a plot (SVG).</p>
<div class="note">Instrument drift moves intensities too, and a large drift
can look like a new phase. Run <i>Check Instrument Stability</i> first when
a transient phase appears where you did not expect one.</div>

<h2>What a definition actually is</h2>
<p>Not the polygon. What the software keeps is where the selected voxels sit
in the plane and how much they spread — a centre and a shape. That is why a
voxel near a boundary can be assigned sensibly instead of by which side of a
hand-drawn line it happened to fall on.</p>
"""

_CONTROLS = """
<h1>Control materials</h1>

<p>On the <i>Materials</i> tab, every material has a setting in the last
column:</p>
<table>
<tr><th>Setting</th><th>Meaning</th></tr>
<tr><td><b>Changes</b></td><td>This material may grow, shrink or move. The
    default.</td></tr>
<tr><td><b>Stays unchanged</b></td><td>This material cannot change during the
    experiment.</td></tr>
</table>

<h2>Why it matters</h2>
<p>A segmentation has no way to check itself. Marking one material as
unchanging supplies that check for free: if its volume moves anyway, the
segmentation is wrong rather than the sample. The health check does this
automatically and tells you which material and by how much.</p>

<p>Good controls are inert and structural — a casing, a support beam, a
steel pin, the surrounding air. Look for a flat volume curve across the
series and no role in the reaction.</p>

<div class="warn"><b>Do not mark a material that reacts.</b> Its real change
would be read as an instrument effect and taken off every other material —
the physics you are trying to measure would be fitted away. The software
cannot tell the difference, which is why there is no default and the choice
is yours.</div>

<p>Marking nothing is allowed, and the health check will say so: you simply
have no independent check on the result.</p>
"""

_RUNNING = """
<h1>Running the series</h1>

<h2>Preview first</h2>
<p><b>▶ Preview this timepoint</b> runs the current timepoint only. Look at
it beside the raw slice before committing. Some checks — the control
materials, and whether the usable data changes — need the whole series to
mean anything, and the preview says so.</p>

<h2>Spatial smoothing</h2>
<p>Leave it on <b>Auto</b>. Smoothing uses neighbouring voxels to clean up
noisy assignments, and it is the one setting that can destroy a result
invisibly: too strong and a small material is simply erased with everything
downstream still looking healthy.</p>
<p>Auto does not guess. It segments the reference timepoint at a range of
strengths and keeps the strongest one at which no material loses volume and
the control materials stay put. The strength it chose is reported in
words.</p>

<h2>What happens per timepoint</h2>
<ol>
<li>Work out which voxels were measured by both instruments.</li>
<li>Build this timepoint's histogram on the shared grid.</li>
<li>Score every voxel against every material definition.</li>
<li>Resolve the assignment using the neighbours.</li>
</ol>
<p>Nothing is estimated, so timepoints are completely independent: the result
is identical run forwards, backwards or one at a time.</p>

<h2>Lock material definitions</h2>
<p>On by default and should stay on. Attenuation coefficients are material
constants, so where a material sits in the plane is fixed. A definition that
moves is a definition absorbing something that should have left it.</p>
<p>Unticking it lets the definitions follow the data. Only appropriate when
the instrument is known to move — and it can absorb a real change in the
sample and report it as no change at all.</p>
"""

_HEALTH = """
<h1>The health check</h1>

<p>Runs automatically before any numbers are shown. Each finding names the
material involved and says what to do.</p>

<table>
<tr><th>Check</th><th>Failing means</th></tr>
<tr><td>All materials present</td><td>Something disappeared. The report says
    whether smoothing did it — if it is still there without smoothing, reduce
    the smoothing.</td></tr>
<tr><td>Control materials stable</td><td>Something you said would not change,
    changed. Do not trust the other volumes until you know why.</td></tr>
<tr><td>Unmatched voxels</td><td>Voxels matched no material. Either one is
    missing, or the measurement drifted.</td></tr>
<tr><td>Voxel budget</td><td>Every voxel counted exactly once. A failure here
    is a bug — please report it.</td></tr>
<tr><td>Usable data stable</td><td>The acquisition changed part-way through;
    comparing volumes across that point is not meaningful.</td></tr>
<tr><td>Mixed boundaries</td><td>A material does not sit between the two you
    linked it to. It may be mislabelled.</td></tr>
</table>

<h2>Some problems stop the run</h2>
<p>Smoothing removing a material, too many unmatched voxels, a voxel budget
that does not close, or a spatial cleanup that cycles instead of settling:
nothing is applied and you are told why. A results screen full of numbers
describing the wrong voxels is worse than no results.</p>

<h2>Missing material, or drift?</h2>
<p>Both look like unmatched voxels, and they need opposite responses:</p>
<ul>
<li><b>Unmatched from the very first timepoint</b> — a material is missing.
    Look at where those voxels fall on the histogram and draw a region for
    them.</li>
<li><b>Unmatched voxels accumulating over the series</b> — the measurement
    moved. Run <i>Check Instrument Stability</i>.</li>
</ul>
<p>The software distinguishes them and says which.</p>
"""

_DRIFT = """
<h1>When the instrument moves</h1>

<p><i>Analytics → Time Series Segmentation → Check Instrument Stability…</i>
measures how far the histogram moved at each timepoint, using the materials
you marked as unchanging. It writes a CSV of the shift.</p>

<p>Locating a control material at a later timepoint needs no labels: it is a
dense isolated peak, so a search started from where it was last seen finds
it. Two things are reported rather than hidden — a control that moved
implausibly far has latched onto something else and is dropped, and two
controls that end up in the same place are both dropped, because the drift
would otherwise be measured from one peak counted twice.</p>

<h2>What to do about it</h2>
<ol>
<li><b>Re-draw the materials part-way through</b> and analyse the series in
    segments. Simplest, and every result stays anchored to something you
    looked at.</li>
<li><b>Correct for it.</b> The correction is applied to the material
    definitions, never to your data, so exports stay in their original
    intensity units.</li>
<li><b>Untick "Lock material definitions".</b> Last resort — see above.</li>
</ol>

<div class="note">A step in the <i>amount of usable data</i> is different
again: that is an acquisition change, and absolute volume comparisons across
it are not valid regardless of which method you use. Check the acquisition
log.</div>
"""

_MIXED = """
<h1>Mixed boundaries</h1>

<p>Where two materials touch, a voxel spanning the interface contains part of
each. Its intensity pair sits on the line between the two pure materials, and
there is no correct hard label for it — the answer for a voxel that is 40 %
lithium is not "lithium", it is 0.4.</p>

<p>With <b>Look for mixed boundaries</b> ticked, the software flags any
material that behaves like such a boundary: elongated, <i>and</i> pointing
along the line joining two others. Requiring both is what separates a genuine
boundary from a material that merely happens to be elongated.</p>

<p>You are told which and between what. It is a prompt to look, not an
automatic change — a flagged material may be mislabelled, or it may be a
phase in its own right that happens to lie between two others.</p>

<h2>Why this explains an old puzzle</h2>
<p>Two segmentation methods applied to the same data typically disagree about
a shell one or two voxels thick around each material. That is not evidence
that either is wrong about where the material is: those voxels are genuine
mixtures, and the two methods simply put a hard threshold in slightly
different places. <i>Spatial Metrics</i> can confirm it — see
<b>Rind or blob?</b>.</p>
"""

_EXPORT = """
<h1>Exporting and measuring</h1>

<h2>Export</h2>
<p><i>File → Export</i>, or the <i>Export</i> tab. You choose which materials
and which outputs:</p>
<ul>
<li><b>Binary masks</b> — one 0/255 volume per material.</li>
<li><b>Masked volumes</b> — the neutron or X-ray data with everything else
    zeroed.</li>
<li><b>Label maps</b> — one integer volume; each material's value is fixed
    across the whole series, so a value means the same thing in every
    file.</li>
<li><b>Per-material histograms</b> — on the main histogram's grid, so they
    stack and compare bin for bin.</li>
<li><b>segmentation_report.txt</b> — each material's name, its label value,
    its voxel count at every timepoint, and the settings used.</li>
</ul>
<p>Files are named after your materials — <code>Lithium</code>, not
<code>Class 1</code>.</p>

<h2>Histogram + slice figure</h2>
<p><i>File → Export Histogram + Slice Figure</i> (Ctrl+Shift+F) saves one
image of what is on screen:</p>
<ul>
<li><b>Left</b> — the histogram of the current timepoint, with the same
    scale and range as on screen, and every material's selection drawn on
    top of it.</li>
<li><b>Right</b> — the slice currently displayed, with the same materials
    highlighted in the same colours. The legend gives each material's share
    of that slice.</li>
</ul>
<p>Choose the timepoint, plane and slice before exporting, and untick any
material you want left out — hidden materials are left out of both panels.
You can also include the regions you have drawn but not saved, set the
resolution, and turn the legends and the outlines around the highlights on or
off.</p>
<p>Like every figure (below), it is saved as <b>SVG</b> by default, and the
label counts can be saved alongside as CSV.</p>

<h2>Saving any figure: format and CSV</h2>
<p>Every analysis figure — the histogram time analyses, both metric plots,
the histogram + slice figure, the K-means cluster timeline and the time-series
plot — first opens a small window asking:</p>
<ul>
<li><b>Figure format</b> — <b>SVG</b> (recommended: outlines, text and
    legends stay vector graphics and the text stays editable in Inkscape or
    Illustrator), PDF, PNG or TIFF.</li>
<li><b>Resolution</b> — for PNG and TIFF, the image resolution; in SVG and
    PDF, the resolution of any image inside the figure (a histogram), while
    lines and text stay vector.</li>
<li><b>Also save the data as CSV</b> — the numbers the figure is drawn from,
    written next to it with the same name, so the plot can be redrawn or
    checked in another program.</li>
</ul>
<p>The window remembers your last answers. What the CSV holds:</p>
<table>
<tr><th>Figure</th><th>CSV</th></tr>
<tr><td>Histogram evolution / change vs previous</td><td><code>name.csv</code>:
    per timepoint, the share of voxels that changed bin against the
    reference; <code>name_bins.csv</code>: every non-empty bin with both
    counts and the plotted log-difference.</td></tr>
<tr><td>Marginal evolution / change</td><td>Per timepoint, modality and
    intensity bin: its share of the voxels and the plotted log2
    change.</td></tr>
<tr><td>Histogram &amp; segmentation metrics, spatial metrics</td><td>Every
    metric value (the long-format metrics table).</td></tr>
<tr><td>Histogram + slice figure</td><td>Per label: voxels inside its
    histogram region, and pixels highlighted on the slice.</td></tr>
<tr><td>K-means cluster timeline</td><td>Per timepoint and cluster: voxels,
    share, and whether it is present.</td></tr>
<tr><td>Time-series plot</td><td>The plotted values per selection and
    timepoint.</td></tr>
</table>
<p>In <i>File → Export</i>, the per-class histograms have their own format
choice and a <i>counts as CSV</i> option, since one export writes many of
them.</p>

<h2>Quality metrics</h2>
<p><i>Analytics → Histogram Time Analysis</i>:</p>
<ul>
<li><b>Histogram &amp; Segmentation Metrics</b> — streak and smear scores,
    marginal asymmetry, class separability, spread, elongation, drift.</li>
<li><b>Spatial Metrics</b> — centre of mass and its drift, radius of
    gyration, connected components, surface-to-volume, and the contact area
    between each pair of materials. A speckled or displaced material is
    invisible to any histogram metric and obvious here.</li>
<li><b>Histogram evolution / marginal evolution</b> — images of how the
    histogram changed, against the first timepoint or against the previous
    one.</li>
</ul>
<p>Both metric families share one CSV layout, so their outputs
concatenate.</p>
"""

_TROUBLE = """
<h1>If something looks wrong</h1>

<dl>
<dt>A material vanished part-way through the series</dt>
<dd>Read the health check: it says whether smoothing did it. If yes, reduce
the smoothing strength. If the material is absent with or without smoothing,
either it really is gone or its region no longer covers it — check
instrument stability.</dd>

<dt>A control material's volume changed</dt>
<dd>Something is wrong. The most common causes are: it is not actually inert;
its region overlaps a neighbour; or the instrument drifted. Check instrument
stability first.</dd>

<dt>Lots of unmatched voxels</dt>
<dd>Unmatched from the start means a missing material — look at where they
fall on the histogram. Unmatched voxels that build up over the series mean
drift.</dd>

<dt>The result is speckled</dt>
<dd>Smoothing is off or very low. Set it to Auto.</dd>

<dt>The result looks over-smoothed</dt>
<dd>Set smoothing to Low or Off and compare. Auto will not pick a setting
that costs a material volume, but it cannot know which detail you care
about.</dd>

<dt>A large static region got absorbed into a material</dt>
<dd>Run <i>Check Data</i>. If the two instruments have different fields of
view, the non-overlapping part is excluded — but only if it is written as
zero or NaN. A reconstruction that pads with something else needs that value
setting explicitly.</dd>

<dt>The volumes jump at one timepoint</dt>
<dd><i>Check Data</i> reports a step in the amount of usable data. That is an
acquisition change; analyse the series as two segments.</dd>

<dt>Everything is slow</dt>
<dd>Volumes over 1 GiB are automatically binned for <i>display</i> only —
segmentation and export always run at full resolution.</dd>
</dl>
"""

# ── mathematics ──────────────────────────────────────────────────────────────

_MATH_HISTOGRAM = """
<h1>The bivariate histogram</h1>

<p>Each voxel <i>i</i> contributes the pair
<span class="m">x<sub>i</sub> = (n<sub>i</sub>, x<sub>i</sub>) ∈ ℝ²</span>
of neutron and X-ray intensity. The histogram counts how many voxels fall in
each cell of a regular <i>B×B</i> grid over the data range:</p>

<span class="m">H[b<sub>x</sub>, b<sub>n</sub>] = #{ i : n<sub>i</sub> ∈ N<sub>b<sub>n</sub></sub> and x<sub>i</sub> ∈ X<sub>b<sub>x</sub></sub> }</span>

<p>It is stored as <code>[X-ray bin, neutron bin]</code> and drawn with
neutron horizontal, so a point on screen is directly a pair of intensities.
The global histogram uses every timepoint; the local one uses the current
timepoint. Both share one bin grid, which is what lets counts be compared
across time.</p>

<h2>Why a histogram is not enough on its own</h2>
<p>The histogram is a <b>marginal</b> distribution: constructing it discards
every spatial relationship in the volume. A clean sample and pure noise can
have identical histograms. Anything that reads only the histogram —
including a classifier trained on histogram-derived labels — inherits that
limit exactly. Recovering what was discarded requires a spatial term, which
is what the smoothing below supplies.</p>

<h2>Per-bin sufficient statistics</h2>
<p>Scoring a voxel needs only per-bin sums, so for each occupied bin
<i>m</i> the software stores</p>
<span class="m">c<sub>m</sub> = Σ<sub>i∈m</sub> 1,&nbsp;&nbsp;
s<sub>m</sub> = Σ<sub>i∈m</sub> v<sub>i</sub>,&nbsp;&nbsp;
S<sub>m</sub> = Σ<sub>i∈m</sub> v<sub>i</sub> v<sub>i</sub><sup>T</sup></span>
<p>Keeping the first and second moments — rather than just the count —
means every quantity derived from a bin is the one you would get from the
voxels themselves. Using bin <i>centres</i> instead would inflate every
covariance by the bin variance <i>h²/12</i> per axis (the Sheppard bias).</p>
"""

_MATH_CONTAINMENT = """
<h1>Region containment</h1>

<p>A drawn region <i>R</i> is a polygon in the (n, x) plane. A voxel belongs
to it when</p>
<span class="m">(n<sub>i</sub>, x<sub>i</sub>) ∈ R</span>
<p>decided by the <b>non-zero winding rule</b>: cast a ray from the point and
sum the signed crossings of the polygon's edges; the point is inside when the
sum is not zero.</p>

<p>This is the same rule the drawing code fills with, which is why the shaded
area is exactly the segmented set. It also has a consequence worth knowing:
in a polygon that crosses itself, a region can be enclosed by edges and still
have winding number zero — visually surrounded, mathematically outside. The
software warns when a polygon self-intersects for that reason.</p>

<p>A rectangle is the axis-aligned special case
<span class="m">n<sub>min</sub> ≤ n<sub>i</sub> ≤ n<sub>max</sub> and x<sub>min</sub> ≤ x<sub>i</sub> ≤ x<sub>max</sub></span></p>

<p>Several visible regions form a union: a voxel is selected when it is
inside any of them.</p>
"""

_MATH_MATERIALS = """
<h1>Material definitions and the match score</h1>

<h2>The definition</h2>
<p>For each material <i>k</i>, the voxels its region selects at the reference
timepoint give</p>
<span class="m">μ<sub>k</sub> = (1/|R<sub>k</sub>|) Σ<sub>i∈R<sub>k</sub></sub> v<sub>i</sub>
&nbsp;&nbsp;&nbsp;
Σ<sub>k</sub> = (1/(|R<sub>k</sub>|−1)) Σ<sub>i∈R<sub>k</sub></sub> (v<sub>i</sub>−μ<sub>k</sub>)(v<sub>i</sub>−μ<sub>k</sub>)<sup>T</sup></span>

<p>μ<sub>k</sub> is where the material sits; Σ<sub>k</sub> is how far it
spreads and in what direction. These are held <b>fixed</b> for the whole
series.</p>

<h3>Why fixed</h3>
<p>Neutron and X-ray attenuation coefficients are material constants. The map
from a material to a position in the plane is set by physics, not estimated
from data. So a material centroid that moves between timepoints is a material
<i>absorbing something that should have left it</i>: voxels must migrate
between fixed materials, not materials chase voxels.</p>

<h2>The match score</h2>
<p>A voxel's agreement with material <i>k</i> is the log of a Gaussian
density at its intensity pair:</p>
<span class="m">U<sub>i</sub>(k) = −½ [ (v<sub>i</sub>−μ<sub>k</sub>)<sup>T</sup> Σ<sub>k</sub><sup>−1</sup> (v<sub>i</sub>−μ<sub>k</sub>) + log|Σ<sub>k</sub>| + 2 log 2π ] + log π<sub>k</sub></span>

<p>The quadratic form is the squared <b>Mahalanobis distance</b> — distance
measured in units of the material's own spread, so a material that is broad
in neutron and narrow in X-ray is judged accordingly rather than by plain
Euclidean distance. π<sub>k</sub> is the material's share of the reference
volume and only breaks ties.</p>

<p>Because the score depends on the voxel only through its intensity pair, it
is evaluated once per occupied <b>bin</b> — a few hundred thousand times
rather than tens of millions — and scoring a voxel is then a table lookup.
That is what makes the method fast.</p>

<h2>Unclassified</h2>
<p>A uniform density over the histogram's support, with a small weight
<i>π<sub>0</sub></i>, competes with the materials:</p>
<span class="m">U<sub>i</sub>(0) = −log|𝒳| + log π<sub>0</sub></span>
<p>A voxel whose best material match is worse than that baseline is genuinely
unexplained, and is labelled <b>Unclassified</b> (value 0) rather than being
given to whichever material happens to be nearest. Without it, a large region
of padding or an unmodelled material is silently absorbed into a real
one.</p>
"""

_MATH_SMOOTHING = """
<h1>Spatial smoothing</h1>

<p>The match score treats every voxel independently, so its raw output is
speckled. Smoothing adds a term that prefers a voxel to agree with its
neighbours. The total cost of a labelling <b>L</b> is</p>

<span class="m">E(L) = Σ<sub>i</sub> −U<sub>i</sub>(L<sub>i</sub>)
 + β Σ<sub>i</sub> Σ<sub>j∈N(i)</sub> w<sub>ij</sub> V(L<sub>i</sub>, L<sub>j</sub>)</span>

<p>with <i>N(i)</i> the six face-neighbours. The first term is the evidence,
the second the price of disagreement. β is the smoothing strength.</p>

<h2>The boundary cost V, learned from your own labels</h2>
<p>A uniform penalty charges the same for every boundary. Instead, the
face-adjacencies in the reference labelling are counted,
<i>n<sub>kl</sub></i>, and</p>
<span class="m">V(k,l) = −log[ (n<sub>kl</sub>+ε) / Σ<sub>m</sub>(n<sub>km</sub>+ε) ],&nbsp;&nbsp;
V ← ½(V + V<sup>T</sup>)</span>

<p>Boundaries that occur constantly in your own segmentation become cheap;
ones that never occur become expensive. The per-row offset is removed so that
<i>V(k,k) = 0</i> — without that, a material whose own voxels are less
reliably adjacent (a thin or scattered one) would pay a standing penalty just
for existing, biasing against exactly the materials most at risk of being
smoothed away.</p>

<p>This has a useful side effect: a small material that genuinely borders its
neighbour has a cheap boundary, so smoothing has no incentive to remove it.
What over-smoothing destroys is a <i>finely dispersed</i> phase, whose voxels
have no neighbours to lean on.</p>

<h2>Edge weights</h2>
<span class="m">w<sub>ij</sub> = exp( −‖v<sub>i</sub> − v<sub>j</sub>‖² / 2σ<sub>g</sub>² )</span>
<p>so a real interface — where the intensities change sharply — is cheap to
cross, and noise is not.</p>

<h2>Solving it</h2>
<p>Minimising <i>E</i> exactly is intractable, so one of two approximations
is used, chosen from a memory budget:</p>
<ul>
<li><b>Mean-field.</b> Keep a soft assignment <i>r<sub>ik</sub></i> and
iterate
<span class="m">r<sub>ik</sub> ∝ exp( U<sub>i</sub>(k) − β Σ<sub>j∈N(i)</sub> w<sub>ij</sub> Σ<sub>l</sub> r<sub>jl</sub> V(k,l) )</span>
Holds several arrays of <i>K</i> numbers per voxel (measured: about
32·<i>K</i> + 30 bytes per voxel). The update is <b>damped</b>, mixing each
sweep with the previous one — an undamped synchronous sweep can settle into a
two-cycle that flips a whole region back and forth forever and looks, from
outside, like an unstable segmentation.</li>
<li><b>ICM.</b> Keep hard labels and give each voxel its best label given its
current neighbours. About 75 bytes per voxel whatever <i>K</i> is; greedier.</li>
</ul>
<p>A volume that does not fit is processed in slabs along z, each padded
with one more slice than there are sweeps. Every voxel is updated from its
neighbours' previous state, so after <i>n</i> sweeps it depends only on voxels
within <i>n</i> steps: the padded slabs give exactly the whole-volume result,
and mean-field is used at any size.</p>
<p>The total cost is recorded every sweep and should fall. If it rises, the
refinement is cycling rather than settling, and that is reported.</p>
"""

_MATH_AUTOSMOOTH = """
<h1>Choosing the smoothing strength</h1>

<p>β is the most destructive parameter in the method: raise it far enough and
a small material disappears, with every downstream number still looking
healthy. It is therefore chosen by measurement, with a limit on each side.</p>

<p>For each candidate β on a grid, each checked timepoint is segmented and
compared against the same timepoint segmented with β = 0:</p>

<span class="m">retention<sub>k</sub>(β) = |{ i : L<sub>i</sub><sup>β</sup> = k }| / |{ i : L<sub>i</sub><sup>0</sup> = k }|</span>

<h2>Lower guards: what smoothing may not do</h2>
<ul>
<li>every material keeps at least a set share of its unsmoothed volume
    (default 80 %);</li>
<li><b>thin sheets survive</b>: the voxels of each material that belong to
    a structure one or two voxels thick (those a 3-D opening would remove,
    in components of at least 30 voxels) keep at least 80 % of
    themselves — a volume guard alone lets a one-voxel layer be eaten
    long before the material's total volume notices;</li>
<li>every control material's volume changes by no more than a small
    tolerance, and the unmatched fraction stays below its limit.</li>
</ul>
<p>These are checked at the reference timepoint, at every timepoint where a
material that appears later was defined (where it is thinnest), and in the
middle and at the end of the series.</p>

<h2>Upper rule: enough, not more</h2>
<p>The label change from one grid value to the next,</p>
<span class="m">Δ(β<sub>j</sub>) = fraction of voxels whose label differs between β<sub>j−1</sub> and β<sub>j</sub></span>
<p>falls as smoothing takes hold. The chosen β is the first acceptable value
with Δ below 0.2 %: beyond it more smoothing changes almost nothing, so there
is no reason to pay for it. If the top of the grid is reached with the labels
still changing, the grid is doubled (twice at most); if they never settle,
the health check says the choice sits at the ceiling. A failed guard after
an acceptable value stops the search there.</p>

<div class="note"><b>Note what the comparison is against.</b> Retention is
measured at the <i>same timepoint</i> with and without smoothing — never
against the first timepoint. Compared against the first, a material that
genuinely shrinks would look as though smoothing had destroyed it, and the
software would refuse to report exactly the change it exists to
measure.</div>

<p>The whole sweep is retained, with the reason for the choice and the change
the next value would have made (its sensitivity). A chosen number is only
trustworthy if you can see the curve it came from.</p>
"""

_ALIGNMENT = """
<h1>Alignment and materials you cannot draw</h1>

<h2>Check the alignment first</h2>
<p><i>Analytics → Time Series Segmentation → Check Alignment…</i> The
histogram pairs each neutron voxel with the X-ray voxel at the same position,
so the two volumes must sit on the same grid. An offset of one voxel pairs
two different materials at every interface: the clouds smear together and a
thin phase can vanish.</p>
<p>The check measures the offset at the first, middle and last timepoints and
offers to correct it — once, if it is constant (a mounting offset), or per
timepoint if it changes. Correction moves the X-ray volumes by whole voxels
in memory; files on disk are not changed. Offsets under a voxel are reported
but left alone: interpolating them away does more harm than good. Only
shifts are measured — a rotation or a scale difference must be removed with
a registration tool before loading.</p>

<h2>A material that appears later</h2>
<p>Draw it where it exists. A material whose region selects nothing at the
first timepoint is defined at the first timepoint where it does, and the
panel shows that timepoint next to it. The health check then expects it to
be absent before. A material that cannot be defined at all is reported as a
failure, never left out silently.</p>

<h2>A material you cannot draw at all</h2>
<p><i>Add Materials from Attenuation Coefficients…</i> places it from its
neutron and X-ray attenuation coefficients. Pick two drawn materials whose
coefficients you know (air and aluminium are ideal) to calibrate each
instrument's grey scale, then list the materials to place. Use the X-ray
coefficient at the effective energy of your spectrum.</p>

<h2>A material in two places on the histogram</h2>
<p>Tick <i>Allow irregular material shapes</i> on the Materials tab. Each
material is then described by as many clouds as its own voxels support (up
to three), instead of one ellipse that may cover a neighbour.</p>
"""

_MATH_ALIGNMENT = """
<h1>Alignment, calibration and material shapes</h1>

<h2>Measuring the offset</h2>
<p>The two modalities have different contrast, so their values cannot be
compared directly. What they share is dependence: when aligned, the X-ray
value is best predicted by the neutron value. The mutual information of the
joint histogram</p>
<span class="m">I(N; X) = Σ p(n, x) · log[ p(n, x) / (p(n) p(x)) ]</span>
<p>is largest at the correct alignment. Both volumes are smoothed lightly
(σ = 1 voxel) so noise does not dominate, then I is maximised over
whole-voxel shifts, one axis at a time, on a fixed sample of voxels (whole
voxels need no interpolation, which would itself raise I). A parabola
through I at the best shift and its two neighbours gives the sub-voxel
part. An axis counts as measured only if I two voxels either side of the
peak is at least 1 % lower; along an axis without structure the component
is reported as zero, not guessed.</p>

<h2>Calibrating grey values to coefficients</h2>
<p>For each modality, grey = g · μ + o. With reference materials r of known
μ<sub>r</sub> and measured mean grey value m<sub>r</sub>, (g, o) is the
least-squares fit; with more than two references the root-mean-square misfit
is reported. A predicted material's centre is g · μ + o per modality, and
its spread is the root-mean-square spread of the reference regions — noise
and partial volume, properties of the instrument.</p>

<h2>Materials as mixtures of clouds</h2>
<p>With irregular shapes allowed, a material's voxels (standardised per
channel) are fitted with Gaussian mixtures of 1, 2 and 3 components, and
the one with the lowest Bayesian information criterion is kept:</p>
<span class="m">BIC = −2 log L + p log n</span>
<p>The match score of a bin is then the log of the weighted sum of the
component densities, so a material occupies only where its voxels are,
instead of the ellipse that covers all of them.</p>
"""

_MATH_VALIDATION = """
<h1>How the method was validated</h1>

<p>Every quantitative claim can be reproduced from the repository:
<code>python -m validation.run</code> builds a synthetic 4-D cell with exact
labels — air, an aluminium casing, electrolyte, a lithium electrode and a
reaction product that is absent at the first timepoint and grows one voxel
per step — blurs and adds noise to it as a scanner would, and scores BiTS
and seven other methods (Otsu, multi-level Otsu, K-means, Gaussian mixture,
random walker, a random-forest pixel classifier) by Dice overlap per
material and timepoint,</p>
<span class="m">Dice(A, B) = 2 |A ∩ B| / (|A| + |B|)</span>
<p>(1 when a material is correctly absent). It also varies the noise, the smoothing strength,
the alignment, instrument drift, and artefacts, and reports each.</p>

<p>Results and their limits are in <code>docs/validation.md</code>. The
phantom is simpler than a real cell; unsupervised baselines are named with
the truth (the most favourable naming possible), and no deep network is
included, since training one needs labelled volumes this setting does not
have.</p>
"""

_MATH_VALIDITY = """
<h1>Which voxels count</h1>

<p>A reconstructed volume contains values that are not measurements: zero
padding outside the reconstruction circle, NaN from a failed slice, detector
saturation. In a <i>paired</i> dataset there is a fourth: a region one
instrument covers and the other does not.</p>

<p>A voxel is usable only where both instruments measured:</p>
<span class="m">valid<sub>i</sub> = finite(n<sub>i</sub>) ∧ finite(x<sub>i</sub>) ∧ (n<sub>i</sub> ≠ sentinel) ∧ (x<sub>i</sub> ≠ sentinel)</span>

<p>Rejecting a voxel only when <i>both</i> channels are empty misses the
paired case entirely — a region with real neutron data and no X-ray data
would pass, forming a large static blob pinned to zero in one axis that some
material then absorbs, inflating its spread and dragging its position.</p>

<h2>Intensity floors</h2>
<p>A hard floor is available but <b>off by default</b>, because one tuned on
one dataset will silently delete a genuinely low-attenuation phase in the
next. When you want one, it is derived from the data's own lower tail:</p>
<span class="m">floor = q<sub>0.001</sub> − 3 · 1.4826 · MAD</span>
<p>using the median absolute deviation, which padding cannot skew as badly as
the plain standard deviation.</p>

<h2>The acquisition-change alarm</h2>
<p>The rejected fraction is recorded per timepoint. A step in it means the
acquisition changed — a shifted field of view, a different reconstruction, a
detector fault — and absolute volume comparisons across that point are not
valid whatever method produced them.</p>
"""

_MATH_DRIFT = """
<h1>Instrument drift</h1>

<p>Over a long series the whole cloud can migrate: beam current, detector
gain, scatter build-up. A material that cannot change chemically cannot
really move, so any movement of its peak is instrumental by definition.</p>

<h2>Finding a control material without labels</h2>
<p>A control is a dense isolated peak, so its position at time <i>t</i> is
found by <b>mean shift</b> from where it was last seen: iterate</p>
<span class="m">c ← Σ<sub>m</sub> c<sub>m</sub> K(v̄<sub>m</sub> − c) v̄<sub>m</sub> / Σ<sub>m</sub> c<sub>m</sub> K(v̄<sub>m</sub> − c)</span>
<p>with a Gaussian kernel <i>K</i> whose width is a multiple of the
material's own σ, over the bin means v̄<sub>m</sub> weighted by their counts
c<sub>m</sub>. No segmentation is needed.</p>

<p>The search is <b>cumulative</b> — each timepoint starts from the previous
estimate. Checking against the first timepoint instead would reject every
control as implausible exactly when the accumulated drift is largest, which
is when it matters most.</p>

<h2>Combining and guarding</h2>
<p>The drift is a weighted mean of the per-control displacements. With two or
more controls at different positions a per-axis gain can also be fitted by
weighted least squares; with one it is not identifiable and is left at 1.</p>
<p>Two guards. A control that moves further than a set number of its own σ in
one step has latched onto something else and is dropped. <b>Two controls that
converge on the same peak are both dropped</b> — neither moved suspiciously
far on its own, but the drift would then be estimated from one peak counted
twice, and nothing local says which is the impostor.</p>

<h2>Applied to the model, not to your data</h2>
<p>A material anchored at (μ, Σ) is anchored at
<span class="m">(s ⊙ μ + d, S Σ S<sup>T</sup>),&nbsp;&nbsp;S = diag(s)</span>
at time <i>t</i>. Equivalent to correcting the data, but no voxel is touched,
so every histogram, statistic and export stays in its native intensity
units.</p>
"""

_MATH_PARTIAL = """
<h1>Mixed boundaries — the mathematics</h1>

<p>A voxel that is fraction α of material <i>a</i> and (1−α) of <i>b</i> has
mean and covariance</p>
<span class="m">m(α) = α μ<sub>a</sub> + (1−α) μ<sub>b</sub><br>
C(α) = α² Σ<sub>a</sub> + (1−α)² Σ<sub>b</sub> + σ<sub>n</sub>² I</span>

<p>As α runs from 0 to 1, m(α) traces the straight segment between the two
pure materials. That is why such a class looks <b>elongated</b> in the
histogram: elongation is a prediction of the model, not an anomaly.</p>

<p>Discretising α on <i>J</i> ≈ 10 steps keeps it a finite mixture:</p>
<span class="m">p<sub>ab</sub>(v) ≈ (1/J) Σ<sub>j</sub> 𝒩( v ; m(α<sub>j</sub>), C(α<sub>j</sub>) )</span>

<p>and the reported fraction per voxel is the posterior mean</p>
<span class="m">α̂<sub>i</sub> = Σ<sub>j</sub> r<sub>ij</sub> α<sub>j</sub> / Σ<sub>j</sub> r<sub>ij</sub></span>

<p>a continuous, physically meaningful quantity — local composition —
replacing a categorical label that was never well defined at a boundary.</p>

<h2>The check</h2>
<p>A declared pair is verified by requiring the elongated component's
principal axis to align with the line joining its parents to within 15°:</p>
<span class="m">θ = arccos | e<sub>1</sub> · (μ<sub>a</sub> − μ<sub>b</sub>)/‖μ<sub>a</sub> − μ<sub>b</sub>‖ |</span>
<p>with e<sub>1</sub> the leading eigenvector of the component's covariance.
Elongation alone is not enough: a material can be anisotropic without being a
mixture.</p>

<h2>Elongation</h2>
<span class="m">E<sub>k</sub> = √(λ<sub>1</sub>/λ<sub>2</sub>)</span>
<p>the square root of the eigenvalue ratio of Σ<sub>k</sub>. 1 is circular.
Genuine compact phases sit around 1.1–1.2; a value above about 1.5 with the
right alignment is the signature of a mixing line.</p>
"""

_MATH_METRICS = """
<h1>The metrics</h1>

<p>Every metric here is <b>ground-truth-free</b>: computable from the data
and your own materials, with no phantom or reference labelling.</p>

<h2>Histogram shape</h2>
<table>
<tr><th>Metric</th><th>Definition</th><th>Flags</th></tr>
<tr><td>S<sub>h</sub></td><td>max row variance / mean column variance of the
    normalised histogram</td><td>a horizontal streak — misalignment between
    the two modalities</td></tr>
<tr><td>S<sub>v</sub></td><td>the transpose of the above</td><td>a vertical
    streak — ring artifacts</td></tr>
<tr><td>S<sub>d</sub></td><td>Pearson ρ between the two channels</td>
    <td>a diagonal smear — beam hardening or scatter</td></tr>
<tr><td>A<sub>x</sub></td><td>(mean − median)/σ of the X-ray marginal</td>
    <td>skew — cupping</td></tr>
<tr><td>Δ<sub>n</sub></td><td>shift of the mean neutron intensity since the
    first timepoint</td><td>scatter build-up or a changing sample</td></tr>
</table>

<h2>Separability</h2>
<p>The <b>Davies–Bouldin index</b> over the materials:</p>
<span class="m">DB = (1/K) Σ<sub>k</sub> max<sub>l≠k</sub> [ (σ<sub>k</sub>+σ<sub>l</sub>) / ‖μ<sub>k</sub>−μ<sub>l</sub>‖ ]</span>
<p>Lower is better separated. Over materials you already defined this is an
ordinary internal index and needs no ground truth.</p>

<h2>Spatial metrics</h2>
<p>Computed in the volume, not the plane — a speckled or displaced material
barely moves the histogram, so a histogram-only metric set cannot see it:
centre of mass and its drift, radius of gyration
<span class="m">R<sub>g</sub> = √( (1/N) Σ<sub>i</sub> ‖r<sub>i</sub> − r̄‖² )</span>
connected components, largest-component fraction, surface-to-volume ratio,
and the interface area between each pair of materials — the quantity that
governs reaction kinetics at a boundary.</p>

<h2>Rind or blob?</h2>
<p>When two segmentations disagree, erode the disagreement:</p>
<span class="m">f<sub>rind</sub> = 1 − |erode²(A △ B)| / |A △ B|</span>
<p>A shell one or two voxels thick vanishes immediately — the two methods
agree about where the material is and differ only on genuinely fractional
boundary voxels. Compact clumps that survive are a real disagreement about
the interior.</p>
<div class="note">Read f<sub>rind</sub> together with the surviving component
count. f<sub>rind</sub> is scale-dependent: a genuinely displaced <i>small</i>
object erodes away just like a shell does, so the component count is the
robust signal.</div>

<h2>Honest validation</h2>
<p>Voxels are spatially autocorrelated — a voxel's neighbour is very nearly a
copy of it. Any score measured on the training voxels, on a random k-fold of
them, or out-of-bag from them, asks a model to recognise data it has
effectively already seen, which is why such numbers sit above 95 % regardless
of quality. Holding out contiguous 3-D <b>blocks</b> removes the near-copies
and gives a substantially lower, honest figure. Agreement is reported as
Cohen's kappa (κ),</p>
<span class="m">κ = (p<sub>o</sub> − p<sub>e</sub>) / (1 − p<sub>e</sub>)</span>
<p>which corrects for the agreement chance alone would produce.</p>
"""

_MATH_KMEANS = """
<h1>K-means at three scales</h1>

<h2>Slice and volume</h2>
<p>Each voxel is the point <i>v = (n, x)</i>. Both channels are standardised
first, <i>z = (v − μ) / σ</i> per channel, because neutron and X-ray values
can differ by orders of magnitude and unscaled K-means would then separate
along the larger channel only. K-means minimises</p>
<span class="m">Σ<sub>i</sub> min<sub>k</sub> ‖z<sub>i</sub> −
c<sub>k</sub>‖²</span>
<p>(k-means++ start, ten restarts, the best kept). The volume scope fits on
up to 250 000 sampled voxels and assigns every voxel to its nearest centre.
Non-finite values are left unassigned.</p>

<h2>Time series: persistent phases</h2>
<p>The series is clustered on the histogram plane. With
<i>h<sub>t</sub>(b)</i> the count in bin <i>b</i> at timepoint <i>t</i> and
<i>N<sub>t</sub></i> that timepoint's total, each bin is weighted by</p>
<span class="m">w(b) = (1/T) Σ<sub>t</sub> h<sub>t</sub>(b) /
N<sub>t</sub></span>
<p>— the pooled histogram with every timepoint counting equally, which is
what K-means on an equal number of voxels from each timepoint would see.
Weighted K-means on the bin centres (standardised with the same weights)
gives the persistent clusters. With <i>Give small phases more weight</i> the
weight is <i>log(1 + w(b)·N̄)</i> instead: a small phase then competes on the
area it covers rather than its voxel count, at the price of splitting a broad
phase more readily. Every bin of the grid is assigned to its nearest centre,
so labelling a voxel is a bin lookup — identical to assigning it to the
nearest centre directly, up to the bin width.</p>

<h2>Time series: transient phases</h2>
<p>K-means cannot find a phase holding a small fraction of one timepoint: the
objective is dominated by the bulk of the voxels, so no centre moves to it.
Such a phase is identified by time instead. Each <i>h<sub>t</sub></i> is
smoothed with a Gaussian (σ = 1 bin); a bin is <i>occupied</i> at
<i>t</i> when the smoothed count is ≥ 0.5. Bins occupied at some but fewer
than half of the timepoints are grown by ⌈3σ⌉+1 bins into bins that are not
persistently occupied (so the phase's tails stay with it) and grouped into
connected islands. With <i>C<sub>t</sub></i> an island's total count at
<i>t</i>, the island is a transient phase when</p>
<ul>
<li><i>max<sub>t</sub> C<sub>t</sub> ≥ 10</i> voxels;</li>
<li><i>max<sub>t</sub> C<sub>t</sub> ≥ m + 5√(m + 1)</i>, with <i>m</i> the
    median of <i>C<sub>t</sub></i> — five times the counting noise above what
    the island usually holds;</li>
<li><i>min<sub>t</sub> C<sub>t</sub> ≤ 0.2 max<sub>t</sub> C<sub>t</sub></i> —
    it (nearly) vanishes at some timepoint.</li>
</ul>
<p>The last two conditions reject the flickering rim of a broad persistent
phase: its bins come and go individually, but its total barely changes. Each
island's bins are assigned to a new cluster, numbered after the persistent
ones; its centre is the peak-count-weighted mean of its bins.</p>

<h2>Presence</h2>
<p>Cluster <i>k</i> is <i>present</i> at <i>t</i> when it holds at least
0.1 % of that timepoint's voxels. The per-timepoint counts come straight from
the histograms (<i>Σ<sub>b ∈ k</sub> h<sub>t</sub>(b)</i>), so they match the
label volumes exactly.</p>
"""

_MATH_WHY = """
<h1>Why there is no classifier</h1>

<p>Earlier versions trained a Random Forest on the segmentation and used it
to label the remaining timepoints. That path has been removed, for two
reasons that are worth stating precisely.</p>

<h2>1. The target contained nothing to learn</h2>
<p>Training labels came from point-in-polygon tests on the histogram, so
every label was already an exact function of the voxel's two intensities:</p>
<span class="m">L<sub>i</sub> = f(n<sub>i</sub>, x<sub>i</sub>)</span>
<p>Conditioned on those two numbers, no other feature can carry information
about the label:</p>
<span class="m">I( L ; any other feature | n, x ) = 0</span>
<p>Training a classifier on such a target means fitting a function you
already have in closed form. Texture and gradient features have nothing to
fit; position features are worse than useless, because they are identical at
every timepoint and so encode a fixed memory of where things were at the
first one. The measurable effect was a boundary halo — the classifier drew a
smooth partition where the polygons had corners and gaps — and nothing
else.</p>

<h2>2. It was unnecessary on physical grounds</h2>
<p>Attenuation coefficients are material constants. The map from material to
histogram position is fixed by physics, not learned from data. Fitting a
decision boundary to it adds variance and buys nothing.</p>

<h2>What was genuinely missing — and what replaced it</h2>
<p>What the histogram cannot supply is spatial information, because building
it discards spatial arrangement by construction. That gap is real, and it is
what the smoothing term fills: it draws on <i>p(L<sub>i</sub> |
L<sub>N(i)</sub>)</i>, which is not derivable from the histogram at any
resolution.</p>

<p>The size of the gap is worth knowing. Fitting Gaussians to two materials
separated by about 2σ gives a Bayes error near 6 % — meaning roughly a sixth
of the smaller material is <i>irreducibly</i> ambiguous on intensity alone.
Moving the decision boundary anywhere conserves that error; a fixed
partition, a learned partition and a provably optimal partition all suffer it
equally. The only escape is that an ambiguous voxel's neighbours are usually
not ambiguous.</p>

<p>The classifier remains importable from <code>segmentation.legacy</code>
for reproducing earlier figures and for method comparisons.</p>
"""

_REFERENCES = """
<h1>References</h1>

<p>The method combines standard pieces; these are the primary sources.</p>

<dl>
<dt>Zhang, Brady &amp; Smith (2001)</dt>
<dd><i>Segmentation of brain MR images through a hidden Markov random field
model and the expectation-maximization algorithm.</i> IEEE TMI 20(1). — The
combination of a per-voxel intensity model with a spatial smoothing term, and
the mean-field solution.</dd>

<dt>Van Leemput, Maes, Vandermeulen &amp; Suetens (2003)</dt>
<dd><i>A unifying framework for partial volume segmentation of brain MR
images.</i> IEEE TMI 22(1). — The mixed-boundary model.</dd>

<dt>Santago &amp; Gage (1993)</dt>
<dd><i>Quantification of MR brain images by mixture density and partial
volume modeling.</i> IEEE TMI 12(3). — The original partial-volume
density.</dd>

<dt>Besag (1986)</dt>
<dd><i>On the statistical analysis of dirty pictures.</i> JRSS B 48(3). —
ICM.</dd>

<dt>Boykov, Veksler &amp; Zabih (2001)</dt>
<dd><i>Fast approximate energy minimization via graph cuts.</i> IEEE TPAMI
23(11). — The graph-cut alternative to mean-field.</dd>

<dt>Davies &amp; Bouldin (1979)</dt>
<dd><i>A cluster separation measure.</i> IEEE TPAMI 1(2).</dd>

<dt>Roberts et al. (2017)</dt>
<dd><i>Cross-validation strategies for data with temporal, spatial,
hierarchical, or phylogenetic structure.</i> Ecography 40(8). — Why blocked
cross-validation is required for autocorrelated data.</dd>
</dl>

<h2>Further reading in this repository</h2>
<ul>
<li><code>docs/workflow.md</code> — the workflow, in more detail.</li>
<li><code>docs/model_segmentation.md</code> — the method notes.</li>
<li><code>docs/metrics.md</code> — every metric and the CSV layout.</li>
<li><code>docs/architecture.md</code> — module map and conventions.</li>
</ul>
"""


SECTIONS: List[Dict[str, str]] = [
    {"id": "start", "group": "How to", "title": "Getting started",
     "body": _GETTING_STARTED,
     "keywords": "begin first steps overview workflow axes neutron xray"},
    {"id": "define", "group": "How to", "title": "Defining materials",
     "body": _DEFINING,
     "keywords": "roi region polygon rectangle draw class kmeans k-means "
                 "clear highlight hide "
                 "cluster edit slice volume time series transient phase "
                 "timeline "
                 "select move arrow keys keyboard backspace delete several"},
    {"id": "controls", "group": "How to", "title": "Control materials",
     "body": _CONTROLS,
     "keywords": "inert unchanged null check casing support steel"},
    {"id": "run", "group": "How to", "title": "Running the series",
     "body": _RUNNING,
     "keywords": "preview smoothing auto lock timepoints run"},
    {"id": "health", "group": "How to", "title": "The health check",
     "body": _HEALTH,
     "keywords": "check warning failure refuse unmatched budget"},
    {"id": "align", "group": "How to",
     "title": "Alignment and materials you cannot draw", "body": _ALIGNMENT,
     "keywords": "alignment offset registration shift coefficient attenuation "
                 "calibration appear later product irregular shape"},
    {"id": "drift", "group": "How to", "title": "When the instrument moves",
     "body": _DRIFT,
     "keywords": "drift stability shift gain beam detector"},
    {"id": "mixed", "group": "How to", "title": "Mixed boundaries",
     "body": _MIXED,
     "keywords": "partial volume fraction interface boundary alloy"},
    {"id": "export", "group": "How to", "title": "Exporting and measuring",
     "body": _EXPORT,
     "keywords": "save tiff csv label map report metrics figure image png pdf "
                 "svg vector histogram slice side by side"},
    {"id": "trouble", "group": "How to", "title": "If something looks wrong",
     "body": _TROUBLE,
     "keywords": "problem troubleshoot speckle vanished slow jump"},

    {"id": "m_hist", "group": "Mathematics", "title": "The bivariate histogram",
     "body": _MATH_HISTOGRAM,
     "keywords": "bins marginal sufficient statistics moments sheppard"},
    {"id": "m_contain", "group": "Mathematics", "title": "Region containment",
     "body": _MATH_CONTAINMENT,
     "keywords": "winding rule point in polygon self intersect"},
    {"id": "m_material", "group": "Mathematics",
     "title": "Material definitions and the match score",
     "body": _MATH_MATERIALS,
     "keywords": "mahalanobis covariance mean unclassified uniform"},
    {"id": "m_smooth", "group": "Mathematics", "title": "Spatial smoothing",
     "body": _MATH_SMOOTHING,
     "keywords": "neighbours cost matrix adjacency damping icm energy"},
    {"id": "m_auto", "group": "Mathematics",
     "title": "Choosing the smoothing strength", "body": _MATH_AUTOSMOOTH,
     "keywords": "retention sweep grid automatic beta thin sheet converged "
                 "ceiling sensitivity"},
    {"id": "m_align", "group": "Mathematics",
     "title": "Alignment, calibration and material shapes",
     "body": _MATH_ALIGNMENT,
     "keywords": "mutual information registration calibration least squares "
                 "bic mixture components"},
    {"id": "m_validation", "group": "Mathematics",
     "title": "How the method was validated", "body": _MATH_VALIDATION,
     "keywords": "validation benchmark phantom dice ground truth baseline"},
    {"id": "m_valid", "group": "Mathematics", "title": "Which voxels count",
     "body": _MATH_VALIDITY,
     "keywords": "mask padding nan saturation field of view floor mad"},
    {"id": "m_drift", "group": "Mathematics", "title": "Instrument drift",
     "body": _MATH_DRIFT,
     "keywords": "mean shift kernel anchor cumulative collision gain"},
    {"id": "m_partial", "group": "Mathematics",
     "title": "Mixed boundaries — the mathematics", "body": _MATH_PARTIAL,
     "keywords": "alpha fraction elongation eigenvector alignment"},
    {"id": "m_metrics", "group": "Mathematics", "title": "The metrics",
     "body": _MATH_METRICS,
     "keywords": "davies bouldin kappa cohen gyration rind iou validation agreement"},
    {"id": "m_kmeans", "group": "Mathematics",
     "title": "K-means at three scales", "body": _MATH_KMEANS,
     "keywords": "kmeans k-means cluster standardise weighted histogram "
                 "transient island poisson presence timeline"},
    {"id": "m_why", "group": "Mathematics", "title": "Why there is no classifier",
     "body": _MATH_WHY,
     "keywords": "random forest classifier mutual information bayes error legacy"},
    {"id": "refs", "group": "Mathematics", "title": "References",
     "body": _REFERENCES,
     "keywords": "papers citation bibliography sources"},
]


def section_ids() -> List[str]:
    return [section["id"] for section in SECTIONS]


def get_section(section_id: str) -> Dict[str, str]:
    for section in SECTIONS:
        if section["id"] == section_id:
            return section
    raise KeyError(f"No manual section named {section_id!r}")


def render(section: Dict[str, str]) -> str:
    """One section as a standalone HTML document."""
    return _CSS + section["body"]


def search(query: str) -> List[Dict[str, str]]:
    """Sections whose title, keywords or text contain every word of *query*.

    Requiring *every* word rather than any keeps a two-word query from
    returning most of the manual.
    """
    words = [word for word in query.lower().split() if word]
    if not words:
        return list(SECTIONS)
    matches = []
    for section in SECTIONS:
        haystack = " ".join([
            section["title"], section.get("keywords", ""), section["body"]
        ]).lower()
        if all(word in haystack for word in words):
            matches.append(section)
    return matches


def as_plain_text() -> str:
    """The whole manual with the markup stripped, for saving or grepping."""
    import re

    parts = []
    for section in SECTIONS:
        text = re.sub(r"<br\s*/?>", "\n", section["body"])
        text = re.sub(r"</(p|li|h1|h2|h3|dd|dt|tr)>", "\n", text)
        text = re.sub(r"<li>", "  - ", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        parts.append(
            f"{'=' * 70}\n{section['group']} — {section['title']}\n{'=' * 70}\n"
            + text.strip()
        )
    return "\n\n".join(parts) + "\n"
