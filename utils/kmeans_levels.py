"""
kmeans_levels.py - K-means at three scales: slice, volume, time series

The three levels trade speed for reach:

* **slice** — the pixels of one displayed slice. Seconds; good enough to see
  which phases a slice holds, or to seed histogram regions.
* **volume** — every voxel of one timepoint (``KMeans3D.cluster_volume``).
* **time series** — one clustering shared by *every* timepoint, so a cluster
  keeps its identity (and colour) through time, including phases that exist
  only in some frames.

The time-series level works on the histogram plane rather than raw voxels,
in two stages:

1. **Persistent phases** — K-means on the pooled histogram, every timepoint
   weighted equally (each frame's histogram is normalised by its own size).
2. **Transient phases** — islands of the histogram occupied in only some
   timepoints, whose total voxel count collapses at some timepoint. Each
   becomes a cluster of its own however small it is. K-means cannot do this
   by itself: it spends its centres on the bulk of the voxels, so a phase
   holding a fraction of a percent of one frame never wins one.

Every voxel is then labelled by looking up its bin — what ``KMeans.predict``
returns for that bin's centre — at one pass over the data per timepoint.

Nothing here imports Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SCOPES = ("slice", "volume", "series")

#: A cluster counts as present at a timepoint once it holds this fraction of
#: that timepoint's voxels.
DEFAULT_PRESENCE_FRACTION = 0.001


def _paired_features(neutron, xray):
    neutron = np.asarray(neutron, dtype=np.float64).reshape(-1)
    xray = np.asarray(xray, dtype=np.float64).reshape(-1)
    if neutron.shape != xray.shape:
        raise ValueError("Neutron and X-ray arrays must have the same shape")
    return np.column_stack((neutron, xray))


# ── slice ────────────────────────────────────────────────────────────────────

@dataclass
class SliceClustering:
    labels: np.ndarray            # 2-D, -1 where a pixel was not finite
    centers: np.ndarray           # (k, 2) in (neutron, X-ray) units
    pixel_counts: np.ndarray      # (k,)
    scale: np.ndarray = None      # (2,) channel standardisation used


def kmeans_cell_polygon(centers, scale, index, bounds) -> Optional[np.ndarray]:
    """The region of the (neutron, X-ray) plane that K-means gives *index*.

    K-means on standardised channels assigns a point to the nearest centre
    in the metric ``sum_d ((v_d - c_d) / scale_d)**2``. The set of points
    nearer to centre *i* than to every other centre is an intersection of
    half-planes — a convex polygon. It is clipped to *bounds*
    ``(x_min, y_min, x_max, y_max)`` (the histogram range).

    Testing a voxel against this polygon gives exactly the cluster K-means
    assigns it, which is what lets a cluster become an ordinary class.
    Returns None if the cell does not reach inside *bounds*.
    """
    centers = np.asarray(centers, dtype=float)
    weights = 1.0 / np.square(np.asarray(scale, dtype=float))
    x_min, y_min, x_max, y_max = bounds
    polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    own = centers[index]
    for other_index, other in enumerate(centers):
        if other_index == index:
            continue
        # |v-own|^2_w <= |v-other|^2_w  <=>  a . v <= b
        a = 2.0 * weights * (other - own)
        b = float(np.sum(weights * (other ** 2 - own ** 2)))
        polygon = _clip_half_plane(polygon, a, b)
        if len(polygon) < 3:
            return None
    return np.array(polygon, dtype=float)


def _clip_half_plane(polygon, a, b):
    """Sutherland–Hodgman: keep the part of *polygon* where a . v <= b."""
    result = []
    count = len(polygon)
    for position in range(count):
        current = np.asarray(polygon[position])
        following = np.asarray(polygon[(position + 1) % count])
        current_in = a @ current <= b
        following_in = a @ following <= b
        if current_in:
            result.append(tuple(current))
        if current_in != following_in:
            denominator = a @ (following - current)
            if denominator != 0:
                t = (b - a @ current) / denominator
                result.append(tuple(current + t * (following - current)))
    return result


def cluster_slice(neutron_slice, xray_slice, n_clusters: int,
                  random_state: int = 42,
                  max_fit_samples: int = 200_000) -> SliceClustering:
    """K-means on one slice's (neutron, X-ray) pixel pairs.

    Both channels are standardised first: raw neutron and X-ray values can
    differ by orders of magnitude, and unscaled K-means would then split
    along the larger channel only.
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    shape = np.asarray(neutron_slice).shape
    features = _paired_features(neutron_slice, xray_slice)
    finite = np.all(np.isfinite(features), axis=1)
    usable = features[finite]
    if len(np.unique(usable, axis=0)) < n_clusters:
        raise ValueError(
            f"The slice has fewer than {n_clusters} distinct intensity pairs"
        )
    rng = np.random.default_rng(random_state)
    fit = usable
    if len(fit) > max_fit_samples:
        fit = fit[rng.choice(len(fit), max_fit_samples, replace=False)]
    scaler = StandardScaler().fit(fit)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(scaler.transform(fit))

    labels = np.full(features.shape[0], -1, dtype=np.int32)
    labels[finite] = model.predict(scaler.transform(usable))
    labels = labels.reshape(shape)
    counts = np.array([np.count_nonzero(labels == k) for k in range(n_clusters)])
    centers = scaler.inverse_transform(model.cluster_centers_)
    return SliceClustering(labels=labels, centers=centers, pixel_counts=counts,
                           scale=np.asarray(scaler.scale_, dtype=float))


# ── time series ──────────────────────────────────────────────────────────────

@dataclass
class SeriesClustering:
    """One clustering of the histogram plane, shared by every timepoint."""

    x_edges: np.ndarray                 # neutron bin edges
    y_edges: np.ndarray                 # X-ray bin edges
    bin_labels: np.ndarray              # (n_xray_bins, n_neutron_bins)
    centers: np.ndarray                 # (k, 2) in (neutron, X-ray) units
    counts: np.ndarray                  # (T, k) voxels per cluster
    totals: np.ndarray                  # (T,) voxels histogrammed per timepoint
    occupied: np.ndarray                # (n_xray_bins, n_neutron_bins) bool
    presence_fraction: float = DEFAULT_PRESENCE_FRACTION
    emphasise_small: bool = False
    #: Clusters 0..n_persistent-1 come from K-means; the rest are transient
    #: phases found as islands of the histogram
    n_persistent: int = 0
    timepoints: List[int] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return int(self.centers.shape[0])

    @property
    def fractions(self) -> np.ndarray:
        """(T, k) share of each timepoint's voxels in each cluster."""
        totals = np.maximum(self.totals, 1)[:, None]
        return self.counts / totals

    @property
    def presence(self) -> np.ndarray:
        """(T, k) True where a cluster is present at a timepoint."""
        return self.fractions >= self.presence_fraction

    def is_transient_phase(self, cluster: int) -> bool:
        """True for a cluster found as a transient island (not K-means)."""
        return cluster >= self.n_persistent

    def cluster_name(self, cluster: int) -> str:
        if self.is_transient_phase(cluster):
            return f"Transient phase {cluster - self.n_persistent + 1}"
        return f"Series cluster {cluster}"

    def present_at(self, cluster: int) -> List[int]:
        """Timepoints at which *cluster* is present."""
        return [self.timepoints[i]
                for i in np.flatnonzero(self.presence[:, cluster])]

    def transient_clusters(self) -> List[int]:
        """Clusters present at some timepoints but not at all of them."""
        presence = self.presence
        return [k for k in range(self.n_clusters)
                if presence[:, k].any() and not presence[:, k].all()]

    def label_volume(self, neutron, xray) -> np.ndarray:
        """Cluster of every voxel, by bin lookup; -1 where not finite.

        Values outside the histogram range go to the nearest edge bin, the
        same bin they would have been counted in.
        """
        neutron = np.asarray(neutron)
        xray = np.asarray(xray)
        if neutron.shape != xray.shape:
            raise ValueError("Neutron and X-ray arrays must have the same shape")
        n_bins_x = len(self.x_edges) - 1
        n_bins_y = len(self.y_edges) - 1
        finite = np.isfinite(neutron) & np.isfinite(xray)
        ix = np.searchsorted(self.x_edges, neutron, side='right') - 1
        iy = np.searchsorted(self.y_edges, xray, side='right') - 1
        np.clip(ix, 0, n_bins_x - 1, out=ix)
        np.clip(iy, 0, n_bins_y - 1, out=iy)
        labels = self.bin_labels[iy, ix].astype(np.int8 if self.n_clusters < 127
                                                 else np.int16)
        labels[~finite] = -1
        return labels

    def outline(self, cluster: int) -> Optional[np.ndarray]:
        """Histogram outline (Nx2) of the occupied bins of *cluster*.

        K-means cells are convex, so the convex hull of the cluster's
        occupied bins is its region on the histogram.
        """
        from scipy.spatial import ConvexHull, QhullError

        iy, ix = np.nonzero((self.bin_labels == cluster) & self.occupied)
        if len(ix) == 0:
            return None
        x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        half_x = 0.5 * float(self.x_edges[1] - self.x_edges[0])
        half_y = 0.5 * float(self.y_edges[1] - self.y_edges[0])
        # Bin corners, so a one-bin cluster still has an area
        points = np.concatenate([
            np.column_stack((x_centers[ix] + dx, y_centers[iy] + dy))
            for dx in (-half_x, half_x) for dy in (-half_y, half_y)
        ])
        try:
            hull = ConvexHull(points)
            return points[hull.vertices]
        except QhullError:
            return None

    def timeline_rows(self) -> List[dict]:
        """One row per (timepoint, cluster): voxels, fraction, present."""
        rows = []
        fractions = self.fractions
        presence = self.presence
        for t_index, timepoint in enumerate(self.timepoints):
            for k in range(self.n_clusters):
                rows.append({
                    "timepoint": int(timepoint),
                    "cluster": int(k),
                    "name": self.cluster_name(k),
                    "transient_phase": self.is_transient_phase(k),
                    "neutron_center": float(self.centers[k, 0]),
                    "xray_center": float(self.centers[k, 1]),
                    "voxels": int(self.counts[t_index, k]),
                    "fraction": float(fractions[t_index, k]),
                    "present": bool(presence[t_index, k]),
                })
        return rows

    def write_timeline_csv(self, path) -> None:
        """Per-timepoint voxel count and fraction of every cluster."""
        import csv
        rows = self.timeline_rows()
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def describe_presence(present: Sequence[int], all_timepoints: Sequence[int]) -> str:
    """"all timepoints", "T3", "T2–T5" or "T1, T4–T6"."""
    present = sorted(int(t) for t in present)
    if not present:
        return "no timepoint"
    if len(present) == len(all_timepoints):
        return "all timepoints"
    runs, start, previous = [], present[0], present[0]
    for value in present[1:] + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        runs.append(f"T{start}" if start == previous else f"T{start}–T{previous}")
        if value is not None:
            start = previous = value
    return ", ".join(runs)


def plot_timeline(result: "SeriesClustering", path, colors=None,
                  names=None, dpi: Optional[int] = None) -> None:
    """Share of the sample in every cluster against time, saved to *path*.

    One line per cluster, in the cluster's overlay colour; transient phases
    are drawn dashed, and a cluster is not drawn at timepoints where it is
    absent. The share axis is logarithmic so small phases stay readable. The
    format follows the extension (SVG recommended).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(7.5, 4.2))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    # Log scale: a phase holding 0.1 % must be as readable as one holding
    # 60 %. Timepoints where a cluster is absent are left as gaps.
    fractions = 100.0 * result.fractions
    shown = np.where(result.presence, fractions, np.nan)
    for k in range(result.n_clusters):
        color = colors[k] if colors is not None else None
        if color is not None and len(color) > 3:
            color = tuple(color[:3])
        axes.plot(
            result.timepoints, shown[:, k],
            color=color, linewidth=2,
            linestyle='--' if result.is_transient_phase(k) else '-',
            marker='o', markersize=4,
            label=names[k] if names else result.cluster_name(k),
        )
    axes.set_yscale('log')
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel("Timepoint")
    axes.set_ylabel("Share of voxels (%, log scale)")
    axes.set_title("K-means clusters over the series")
    axes.grid(True, which='both', alpha=0.25, linewidth=0.5)
    axes.legend(fontsize=8, loc="best")
    figure.tight_layout()
    from utils.figure_io import save_figure
    save_figure(figure, path, dpi)


def series_bin_weights(histograms: Sequence[np.ndarray],
                       emphasise_small: bool = False) -> np.ndarray:
    """Weight of every histogram bin for the persistent-phase clustering.

    Each timepoint's histogram is normalised by its own voxel count and the
    results are averaged, so every timepoint counts equally whatever its
    size — the histogram equivalent of pooling the same number of voxels
    from each frame. With *emphasise_small* the weight is log-compressed,
    which lets a small but persistent phase compete with a large one (at the
    price of splitting a broad phase more readily).
    """
    stack = np.stack([np.asarray(h, dtype=np.float64) for h in histograms])
    totals = np.maximum(stack.reshape(len(stack), -1).sum(axis=1), 1.0)
    mean_share = (stack / totals[:, None, None]).mean(axis=0)
    if emphasise_small:
        # Scale to "voxels of an average timepoint" before compressing
        return np.log1p(mean_share * totals.mean())
    return mean_share


def find_transient_islands(histograms: Sequence[np.ndarray],
                           max_share: float = 0.5,
                           smoothing: float = 1.0,
                           occupied_level: float = 0.5,
                           min_voxels: int = 10,
                           vanish_ratio: float = 0.2) -> List[np.ndarray]:
    """Regions of the histogram occupied only at some timepoints.

    A bin is *transiently occupied* when, after light smoothing, it holds
    data at fewer than ``max_share`` of the timepoints. Connected groups of
    such bins are candidate islands. An island is kept as a transient phase
    when, summed over its bins,

    * it holds at least ``min_voxels`` voxels at its peak timepoint, and
    * at some timepoint it drops below ``vanish_ratio`` of that peak.

    * its peak stands out from its usual content by five times the count
      noise (``peak >= median + 5 * sqrt(median + 1)``).

    The last two tests are what separate a real phase from the flickering rim
    of a broad persistent phase: bin by bin the rim comes and goes, but its
    total barely changes over time. Islands are grown into neighbouring bins
    that are not persistently occupied, so a phase's tails stay with it.

    Returns one boolean bin mask per island, largest first.
    """
    from scipy import ndimage

    stack = np.stack([np.asarray(h, dtype=np.float64) for h in histograms])
    if len(stack) < 2:
        return []
    smoothed = np.stack([
        ndimage.gaussian_filter(h, smoothing) if smoothing > 0 else h
        for h in stack
    ])
    occupied = smoothed >= occupied_level
    share = occupied.mean(axis=0)
    candidate = (share > 0) & (share < max_share)
    # The tails of a phase fall below the occupancy level; grow the
    # candidates into neighbouring bins that are not persistently occupied,
    # so those voxels join the phase instead of the nearest persistent
    # cluster, and fragments of one phase join up before being grouped.
    growable = share < max_share
    grow_steps = int(np.ceil(3 * smoothing)) + 1
    grown = ndimage.binary_dilation(
        candidate, structure=np.ones((3, 3)), iterations=grow_steps,
        mask=growable,
    ) | candidate
    labelled, count = ndimage.label(grown, structure=np.ones((3, 3)))
    islands = []
    for index in range(1, count + 1):
        mask = labelled == index
        per_time = stack[:, mask].sum(axis=1)
        peak = per_time.max()
        typical = float(np.median(per_time))
        if peak < min_voxels:
            continue
        # Well above what these bins usually hold (count noise ~ sqrt(n))...
        if peak < typical + 5.0 * np.sqrt(typical + 1.0):
            continue
        # ...and nearly gone at some timepoint
        if per_time.min() > vanish_ratio * peak:
            continue
        islands.append((peak, mask))
    islands.sort(key=lambda item: -item[0])
    return [mask for _peak, mask in islands]


def cluster_series(histograms, n_clusters: int, emphasise_small: bool = False,
                   find_transient: bool = True,
                   presence_fraction: float = DEFAULT_PRESENCE_FRACTION,
                   timepoints: Optional[Sequence[int]] = None,
                   random_state: int = 42,
                   **transient_options) -> SeriesClustering:
    """Cluster the histogram plane of a whole series.

    *histograms* are ``HistogramData`` objects on one shared bin grid (the
    local histograms of each timepoint, which the histogram engine always
    builds on the global grid).

    Two stages: *n_clusters* persistent phases by K-means on the pooled
    histogram, then (with *find_transient*) one extra cluster for every
    island of the histogram that is occupied only at some timepoints — see
    :func:`find_transient_islands`. Transient clusters are numbered after
    the persistent ones.
    """
    histograms = list(histograms)
    if not histograms:
        raise ValueError("No histograms to cluster")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    x_edges = np.asarray(histograms[0].x_edges, dtype=np.float64)
    y_edges = np.asarray(histograms[0].y_edges, dtype=np.float64)
    for hist in histograms[1:]:
        if (len(hist.x_edges) != len(x_edges)
                or not np.allclose(hist.x_edges, x_edges)
                or not np.allclose(hist.y_edges, y_edges)):
            raise ValueError("All histograms must share one bin grid")

    counts_stack = [np.asarray(h.histogram, dtype=np.float64) for h in histograms]
    islands = (find_transient_islands(counts_stack, **transient_options)
               if find_transient else [])
    in_island = np.zeros(counts_stack[0].shape, dtype=bool)
    for mask in islands:
        in_island |= mask

    weights = series_bin_weights(counts_stack, emphasise_small)
    occupied = np.stack(counts_stack).max(axis=0) > 0
    fit_bins = (weights > 0) & ~in_island
    if np.count_nonzero(fit_bins) < n_clusters:
        raise ValueError(
            f"Only {np.count_nonzero(fit_bins)} histogram bin(s) hold data; "
            f"cannot find {n_clusters} clusters"
        )

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)   # [y_bin, x_bin]
    all_bins = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    fit_points = all_bins[fit_bins.ravel()]
    fit_weights = weights.ravel()[fit_bins.ravel()]

    scaler = StandardScaler().fit(fit_points, sample_weight=fit_weights)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(scaler.transform(fit_points), sample_weight=fit_weights)

    # Every bin gets a cluster (the whole plane is partitioned), so any voxel
    # can be labelled, including values that only appear later.
    bin_labels = model.predict(scaler.transform(all_bins)).reshape(grid_x.shape)
    centers = list(scaler.inverse_transform(model.cluster_centers_))

    # Transient phases take over their islands
    peak_counts = np.stack(counts_stack).max(axis=0)
    for offset, mask in enumerate(islands):
        bin_labels[mask] = n_clusters + offset
        mass = peak_counts[mask]
        centers.append(np.array([
            np.average(grid_x[mask], weights=mass),
            np.average(grid_y[mask], weights=mass),
        ]))
    centers = np.array(centers)
    total_clusters = n_clusters + len(islands)

    counts = np.stack([
        np.bincount(bin_labels.ravel(), weights=hist.ravel(),
                    minlength=total_clusters)
        for hist in counts_stack
    ])
    totals = np.array([hist.sum() for hist in counts_stack])
    if timepoints is None:
        timepoints = list(range(len(histograms)))
    return SeriesClustering(
        x_edges=x_edges, y_edges=y_edges,
        bin_labels=bin_labels.astype(np.int32),
        centers=centers, counts=counts, totals=totals, occupied=occupied,
        presence_fraction=presence_fraction, emphasise_small=emphasise_small,
        n_persistent=n_clusters,
        timepoints=list(int(t) for t in timepoints),
    )


def run_series_clustering(dataset, histogram_engine, n_clusters: int,
                          emphasise_small: bool = False,
                          find_transient: bool = True,
                          presence_fraction: float = DEFAULT_PRESENCE_FRACTION,
                          progress_callback: Optional[Callable] = None,
                          cancel_check: Optional[Callable] = None):
    """Cluster a whole dataset and label every timepoint.

    Returns ``(SeriesClustering, {timepoint: label_volume})``. Local
    histograms come from the engine's cache when available.
    """
    count = dataset.num_timepoints
    histograms = []
    for timepoint in range(count):
        if cancel_check:
            cancel_check()
        hist = histogram_engine.get_cached_local_histogram(timepoint)
        if hist is None:
            neutron, xray = dataset.get_volume_at_time(timepoint)
            hist = histogram_engine.compute_local_histogram(
                neutron, xray, timepoint, cancel_check=cancel_check
            )
        histograms.append(hist)
        if progress_callback:
            progress_callback(int(40 * (timepoint + 1) / count),
                              f"Histogram {timepoint + 1}/{count}")

    result = cluster_series(histograms, n_clusters,
                            emphasise_small=emphasise_small,
                            find_transient=find_transient,
                            presence_fraction=presence_fraction)
    if progress_callback:
        progress_callback(45, "Clustered the histogram plane of the series")

    labels = {}
    for timepoint in range(count):
        if cancel_check:
            cancel_check()
        neutron, xray = dataset.get_volume_at_time(timepoint)
        labels[timepoint] = result.label_volume(neutron, xray)
        if progress_callback:
            progress_callback(45 + int(55 * (timepoint + 1) / count),
                              f"Labelled timepoint {timepoint + 1}/{count}")
    return result, labels
