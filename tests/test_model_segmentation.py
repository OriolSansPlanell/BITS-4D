"""The v17 model: validity, histogram cache, drift, mixture, MRF, segmenter.

The claim these tests exist to check is a specific one: a boundary fixed at
T0 loses track of the classes when the histogram drifts, and anchoring the
same selection as a *prior* instead of a constraint keeps track of them —
without inventing change that is not there.
"""

import numpy as np
import pytest

from model import (
    DriftTracker,
    MixelComponent,
    ROIAnchoredMixture,
    ROIDerivedMRF,
    SequentialSegmenter,
    ValidityPolicy,
    anchor_strength_to_kappa,
    build_histogram_cache,
    build_valid_mask,
    detect_mixing_lines,
    estimate_process_noise,
    find_acquisition_steps,
    moments_from_mask,
    validity_report,
)
from model.partial_volume import (
    alignment_angle,
    build_mixel_ladder,
    elongation,
    fraction_per_bin,
    verify_mixels,
)
from model.spatial_prior import UnaryScores, potts_cost
from model.temporal import DriftTransition, StaticTransition


# ── synthetic data ───────────────────────────────────────────────────────────

PHASES = {"Air": (400.0, 400.0), "Aluminium": (700.0, 700.0),
          "Lithium": (1000.0, 1000.0)}
SIGMA = 80.0
# The histogram grid every test shares. Values are clipped inside it so the
# fixture never produces out-of-range voxels by accident — an out-of-range
# voxel is a thing these tests assert about deliberately.
RANGE = (0.0, 2200.0)
_CLIP = (RANGE[0] + 20.0, RANGE[1] - 20.0)


def _volume(shift=0.0, lithium_width=6, seed=0, pad=False):
    """One timepoint: three slabs, optionally drifted, optionally zero-padded."""
    rng = np.random.default_rng(seed)
    shape = (5, 20, 24)
    neutron = np.zeros(shape)
    xray = np.zeros(shape)
    masks = {}
    for name, (mean_n, mean_x) in PHASES.items():
        mask = np.zeros(shape, dtype=bool)
        if name == "Air":
            mask[:, :, :8] = True
            if not pad:
                mask[:, :, 16 + lithium_width:] = True
        elif name == "Aluminium":
            mask[:, :, 8:16] = True
        else:
            mask[:, :, 16:16 + lithium_width] = True
        count = int(mask.sum())
        neutron[mask] = np.clip(
            rng.normal(mean_n + shift, SIGMA, count), *_CLIP
        )
        xray[mask] = np.clip(rng.normal(mean_x + shift, SIGMA, count), *_CLIP)
        masks[name] = mask
    return neutron, xray, masks


def _edges(bins=128, low=RANGE[0], high=RANGE[1]):
    edges = np.linspace(low, high, bins + 1)
    return edges, edges.copy()


def _cache(neutron, xray, valid=None, **kwargs):
    neutron_edges, xray_edges = _edges()
    return build_histogram_cache(
        neutron, xray, neutron_edges, xray_edges, valid_mask=valid, **kwargs
    )


class _FakeDataset:
    def __init__(self, neutron_4d, xray_4d):
        self.neutron_data = np.asarray(neutron_4d)
        self.xray_data = np.asarray(xray_4d)
        self.num_timepoints = self.neutron_data.shape[0]

    def get_volume_at_time(self, timepoint):
        return self.neutron_data[timepoint], self.xray_data[timepoint]


# ── validity ─────────────────────────────────────────────────────────────────

def test_zero_padding_is_excluded_but_real_data_is_not():
    neutron, xray, masks = _volume(pad=True)
    padding = ~(masks["Air"] | masks["Aluminium"] | masks["Lithium"])
    assert padding.any()

    valid = build_valid_mask(neutron, xray)
    assert not valid[padding].any()
    assert valid[masks["Aluminium"]].all()

    report = validity_report(neutron, xray)
    assert report["sentinel_voxels"] == int(padding.sum())
    assert report["valid_voxels"] == int((~padding).sum())


def test_non_finite_voxels_are_rejected():
    neutron, xray, _ = _volume()
    neutron[0, 0, 0] = np.nan
    xray[0, 0, 1] = np.inf
    valid = build_valid_mask(neutron, xray)
    assert not valid[0, 0, 0] and not valid[0, 0, 1]


def test_no_hard_intensity_floor_by_default():
    """A floor tuned on one dataset deletes a real phase in the next one."""
    policy = ValidityPolicy()
    assert policy.neutron_floor is None and policy.xray_floor is None

    neutron, xray, masks = _volume()
    neutron[masks["Air"]] = 5.0        # a genuinely low-attenuation phase
    xray[masks["Air"]] = 5.0
    assert build_valid_mask(neutron, xray)[masks["Air"]].all()

    # ...but a floor is available when the user has one to state
    strict = ValidityPolicy(neutron_floor=100.0)
    assert not build_valid_mask(neutron, xray, strict)[masks["Air"]].any()


def test_a_step_in_rejected_fraction_is_reported():
    reports = [
        {"rejected_fraction": 0.10},
        {"rejected_fraction": 0.11},
        {"rejected_fraction": 0.35},      # field of view changed here
        {"rejected_fraction": 0.35},
    ]
    steps = find_acquisition_steps(reports)
    assert [step[0] for step in steps] == [2]


# ── histogram cache ──────────────────────────────────────────────────────────

def test_cache_moments_match_the_voxels_exactly():
    """The point of storing moments: the M-step is not a bin-centre estimate."""
    neutron, xray, _ = _volume()
    valid = build_valid_mask(neutron, xray)
    cache = _cache(neutron, xray, valid)

    values = np.stack([neutron[valid], xray[valid]], axis=1)
    count, sums, scatter = cache.totals()
    assert count == pytest.approx(values.shape[0])
    np.testing.assert_allclose(sums, values.sum(axis=0), rtol=1e-10)
    np.testing.assert_allclose(scatter, values.T @ values, rtol=1e-10)


def test_cache_counts_match_a_plain_histogram():
    neutron, xray, _ = _volume()
    valid = build_valid_mask(neutron, xray)
    cache = _cache(neutron, xray, valid)
    neutron_edges, xray_edges = _edges()

    reference, _, _ = np.histogram2d(
        xray[valid], neutron[valid], bins=[xray_edges, neutron_edges]
    )
    np.testing.assert_array_equal(cache.to_image(), reference)


def test_out_of_range_voxels_are_dropped_not_clipped():
    """Clipping would pile foreign mass onto the edge bins."""
    neutron, xray, _ = _volume()
    neutron[0, 0, 0] = 5000.0        # outside the edge range
    xray[0, 0, 0] = 5000.0
    cache = _cache(neutron, xray, build_valid_mask(neutron, xray))
    assert cache.out_of_range == 1
    top_bin = cache.bin_ids.max()
    assert top_bin != cache.bins * cache.bins - 1 or cache.counts[-1] < 5


def test_bin_index_round_trips_a_per_bin_quantity():
    neutron, xray, _ = _volume()
    valid = build_valid_mask(neutron, xray)
    cache = _cache(neutron, xray, valid, store_bin_index=True)

    marker = np.arange(cache.num_bins, dtype=np.float32)
    volume = cache.expand_to_voxels(marker, fill=np.float32(-1))
    assert volume.shape == neutron.shape
    assert (volume[~valid] == -1).all()
    # Every valid voxel maps to a bin whose mean is close to its own value
    rows = cache.row_index_volume()
    assert (rows[valid] >= 0).all()
    means = cache.means[rows[valid]]
    assert np.abs(means[:, 0] - neutron[valid]).max() < 20


def test_moments_from_mask_matches_numpy():
    neutron, xray, masks = _volume()
    moments = moments_from_mask(neutron, xray, masks["Aluminium"])
    values = np.stack(
        [neutron[masks["Aluminium"]], xray[masks["Aluminium"]]], axis=1
    )
    np.testing.assert_allclose(moments["mean"], values.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(
        moments["covariance"], np.cov(values.T, bias=False), rtol=1e-10
    )


# ── mixture ──────────────────────────────────────────────────────────────────

def _fit(neutron, xray, masks, anchor_strength=0.5, **kwargs):
    valid = build_valid_mask(neutron, xray)
    cache = _cache(neutron, xray, valid, store_bin_index=True)
    moments = {
        name: moments_from_mask(neutron, xray, mask & valid)
        for name, mask in masks.items()
    }
    prior = ROIAnchoredMixture.prior_from_moments(
        moments, anchor_strength=anchor_strength
    )
    mixture = ROIAnchoredMixture(**kwargs)
    return mixture.fit(cache, prior), cache, moments


def test_mixture_recovers_the_class_moments():
    neutron, xray, masks = _volume()
    result, _, moments = _fit(neutron, xray, masks)

    assert result.converged
    for index, name in enumerate(result.names):
        np.testing.assert_allclose(
            result.means[index], moments[name]["mean"], atol=8.0
        )
        np.testing.assert_allclose(
            result.covariances[index], moments[name]["covariance"],
            rtol=0.25, atol=200.0,
        )


def test_anchor_strength_is_dimensionless_and_spans_both_limits():
    assert anchor_strength_to_kappa(0.0, 5000) == 0.0
    assert anchor_strength_to_kappa(0.5, 5000) == pytest.approx(5000)
    assert anchor_strength_to_kappa(0.9, 5000) == pytest.approx(45000)
    assert anchor_strength_to_kappa(1.0, 5000) > 1e11
    # The same strength means the same thing at any class size
    assert (
        anchor_strength_to_kappa(0.5, 100) / 100
        == pytest.approx(anchor_strength_to_kappa(0.5, 1_000_000) / 1_000_000)
    )


def test_anchor_strength_interpolates_between_frozen_and_free():
    """The fixed ROI and a free mixture are the two limits of one model.

    Note what is *not* asserted: that the free end tracks the drift
    correctly. Released from its anchor a component follows whatever mass is
    nearest, which on a drifted histogram may well be a neighbour's — losing
    the class identity that makes a time series a time series. Movement is
    what the strength controls; being right is what the anchor buys.
    """
    neutron, xray, masks = _volume()
    _, _, moments = _fit(neutron, xray, masks)

    # A drifted timepoint, fitted under the *undrifted* prior
    drifted_n, drifted_x, _ = _volume(shift=200.0, seed=1)
    valid = build_valid_mask(drifted_n, drifted_x)
    cache = _cache(drifted_n, drifted_x, valid)
    mixture = ROIAnchoredMixture(outlier_component=False)
    anchor = moments["Aluminium"]["mean"]

    movement = {}
    for strength in (1.0, 0.9, 0.5, 0.0):
        result = mixture.fit(
            cache,
            ROIAnchoredMixture.prior_from_moments(
                moments, anchor_strength=strength
            ),
        )
        index = result.names.index("Aluminium")
        movement[strength] = float(np.hypot(*(result.means[index] - anchor)))

    assert movement[1.0] < 1.0                       # pinned at T0
    assert movement[0.0] > 100.0                     # unconstrained
    # Monotone in between: the strength is a dial, not a switch
    assert movement[1.0] < movement[0.9] < movement[0.5] < movement[0.0]


def test_outlier_component_absorbs_an_unmodelled_material():
    neutron, xray, masks = _volume()
    # A material nobody drew an ROI around
    intruder = np.zeros(neutron.shape, dtype=bool)
    intruder[0, :4, :4] = True
    neutron[intruder] = 1600.0
    xray[intruder] = 500.0
    for name in masks:
        masks[name] = masks[name] & ~intruder

    with_outlier, cache, moments = _fit(
        neutron, xray, masks, outlier_component=True
    )
    prior = ROIAnchoredMixture.prior_from_moments(moments, anchor_strength=0.5)
    without = ROIAnchoredMixture(outlier_component=False).fit(cache, prior)

    # Without an outlier the nearest real class has to swallow the intruder,
    # which widens it; with one, it does not.
    widened = max(
        np.trace(without.covariances[i]) / np.trace(with_outlier.covariances[i])
        for i in range(with_outlier.n_components)
    )
    assert with_outlier.outlier_weight > 1e-4
    assert widened > 1.05


def test_bic_and_icl_use_the_voxel_count_not_the_bin_count():
    """Otherwise the criterion would depend on the histogram resolution."""
    neutron, xray, masks = _volume()
    result, cache, _ = _fit(neutron, xray, masks)
    assert result.num_voxels == cache.num_voxels
    assert cache.num_voxels > cache.num_bins
    # ICL penalises overlap on top of BIC, so it is never smaller
    assert result.icl() >= result.bic()
    assert result.entropy() >= 0


def test_moved_sigma_reports_how_far_the_prior_let_a_class_go():
    neutron, xray, masks = _volume()
    result, _, _ = _fit(neutron, xray, masks, anchor_strength=1.0)
    assert all(value < 0.05 for value in result.moved_sigma().values())


def test_well_separated_classes_barely_overlap():
    neutron, xray, masks = _volume()
    result, _, _ = _fit(neutron, xray, masks)
    for pair, value in result.overlap().items():
        assert value < 0.2, pair


def test_reject_margin_leaves_ambiguous_bins_unassigned():
    neutron, xray, masks = _volume()
    mixture = ROIAnchoredMixture(reject_margin=0.999, outlier_component=False)
    valid = build_valid_mask(neutron, xray)
    cache = _cache(neutron, xray, valid)
    moments = {
        name: moments_from_mask(neutron, xray, mask & valid)
        for name, mask in masks.items()
    }
    result = mixture.fit(
        cache, ROIAnchoredMixture.prior_from_moments(moments)
    )
    labels = mixture.label_bins(result)
    assert (labels == -1).any()

    permissive = ROIAnchoredMixture(reject_margin=None, outlier_component=False)
    assert (permissive.label_bins(result) >= 0).all()


# ── drift ────────────────────────────────────────────────────────────────────

def test_drift_is_measured_without_any_labels_at_the_later_timepoint():
    neutron, xray, masks = _volume()
    moments = {
        name: moments_from_mask(neutron, xray, mask) for name, mask in masks.items()
    }
    tracker = DriftTracker(anchor_classes=("Air", "Aluminium"))
    tracker.fit_reference(moments)

    drifted_n, drifted_x, _ = _volume(shift=60.0, seed=2)
    estimate = tracker.estimate(
        _cache(drifted_n, drifted_x, build_valid_mask(drifted_n, drifted_x))
    )
    np.testing.assert_allclose(estimate.shift, [60.0, 60.0], atol=20.0)
    assert not estimate.rejected_anchors


def test_anchors_that_land_on_the_same_mode_are_not_both_believed():
    """Half the class spacing is enough for one anchor to capture another.

    Neither anchor moves implausibly far on its own, so the distance guard
    sees nothing wrong — but two anchors sitting on one mode means the drift
    is being estimated from that mode counted twice.
    """
    neutron, xray, masks = _volume()
    moments = {
        name: moments_from_mask(neutron, xray, mask) for name, mask in masks.items()
    }
    tracker = DriftTracker(anchor_classes=("Air", "Aluminium"))
    tracker.fit_reference(moments)

    # A single 150-unit jump: Aluminium's search starts almost equidistant
    # between its own drifted mode and Air's, and Air is the larger class.
    drifted_n, drifted_x, _ = _volume(shift=150.0, seed=2)
    cache = _cache(drifted_n, drifted_x, build_valid_mask(drifted_n, drifted_x))
    estimate = tracker.estimate(cache)

    assert sorted(estimate.rejected_anchors) == ["Air", "Aluminium"]
    # Nothing survived, so no drift is claimed rather than a wrong one
    np.testing.assert_array_equal(estimate.shift, [0.0, 0.0])

    # Stepping through the same total drift resolves it cleanly
    previous = None
    for step in (50.0, 100.0, 150.0):
        stepped_n, stepped_x, _ = _volume(shift=step, seed=2)
        previous = tracker.estimate(
            _cache(stepped_n, stepped_x, build_valid_mask(stepped_n, stepped_x)),
            previous=previous,
        )
    assert not previous.rejected_anchors
    np.testing.assert_allclose(previous.shift, [150.0, 150.0], atol=25.0)


def test_drift_tracking_is_cumulative_across_a_long_series():
    """Total drift beyond a few sigma must not make every anchor implausible."""
    neutron, xray, masks = _volume()
    moments = {
        name: moments_from_mask(neutron, xray, mask) for name, mask in masks.items()
    }
    tracker = DriftTracker(anchor_classes=("Air", "Aluminium"))
    tracker.fit_reference(moments)

    previous = None
    shifts = []
    for step in range(6):
        shift = 100.0 * step                     # reaches 5 sigma
        drifted_n, drifted_x, _ = _volume(shift=shift, seed=step)
        estimate = tracker.estimate(
            _cache(drifted_n, drifted_x, build_valid_mask(drifted_n, drifted_x)),
            timepoint=step, previous=previous,
        )
        assert not estimate.rejected_anchors, (step, shift)
        shifts.append(estimate.shift[0])
        previous = estimate

    np.testing.assert_allclose(shifts, [100.0 * s for s in range(6)], atol=30.0)


def test_drift_is_applied_to_the_model_not_to_the_data():
    """Volumes stay in native units; the prior moves instead."""
    neutron, xray, masks = _volume()
    moments = {
        name: moments_from_mask(neutron, xray, mask) for name, mask in masks.items()
    }
    tracker = DriftTracker(anchor_classes=("Aluminium",))
    tracker.fit_reference(moments)
    drifted_n, drifted_x, _ = _volume(shift=120.0, seed=3)
    original = drifted_n.copy()

    estimate = tracker.estimate(
        _cache(drifted_n, drifted_x, build_valid_mask(drifted_n, drifted_x))
    )
    np.testing.assert_array_equal(drifted_n, original)

    prior = ROIAnchoredMixture.prior_from_moments(moments)
    moved = prior.scaled(estimate)
    for before, after in zip(prior.components, moved.components):
        np.testing.assert_allclose(
            after.mean, before.mean + estimate.shift, rtol=1e-9
        )


def test_reactive_classes_are_never_used_as_anchors():
    neutron, xray, masks = _volume()
    moments = {
        name: moments_from_mask(neutron, xray, mask) for name, mask in masks.items()
    }
    tracker = DriftTracker(anchor_classes=("Aluminium",))
    tracker.fit_reference(moments)
    assert set(tracker.reference) == {"Aluminium"}

    with pytest.raises(ValueError):
        DriftTracker(anchor_classes=("Nonexistent",)).fit_reference(moments)


def test_process_noise_is_the_residual_anchor_movement():
    from model.drift_tracker import DriftEstimate

    estimates = [
        DriftEstimate(
            timepoint=t,
            shift=np.array([10.0 * t, 0.0]),
            per_anchor={
                "A": np.array([10.0 * t + 1.0, 0.5]),
                "B": np.array([10.0 * t - 1.0, -0.5]),
            },
        )
        for t in range(4)
    ]
    noise = estimate_process_noise(estimates)
    assert noise.shape == (2,)
    assert noise[0] == pytest.approx(1.0, rel=0.3)


# ── spatial prior ────────────────────────────────────────────────────────────

def test_pairwise_costs_are_learned_from_the_t0_adjacencies():
    """Boundaries that occur at T0 must be cheaper than ones that never do."""
    labels = np.zeros((4, 12, 12), dtype=np.int32)
    labels[:, :, 4:8] = 1        # class 1 sits between 0 and 2
    labels[:, :, 8:] = 2

    mrf = ROIDerivedMRF()
    cost = mrf.fit_pairwise_from_labels(labels, n_classes=3)

    assert cost.shape == (3, 3)
    np.testing.assert_allclose(np.diag(cost), 0.0, atol=1e-12)
    np.testing.assert_allclose(cost, cost.T)
    # 0 never touches 2, but both touch 1
    assert cost[0, 2] > cost[0, 1]
    assert cost[0, 2] > cost[1, 2]


def test_forbidding_a_pair_makes_it_effectively_impossible():
    mrf = ROIDerivedMRF()
    mrf.pairwise = potts_cost(4)
    mrf.allow_only(3, [0, 1])
    assert mrf.pairwise[3, 2] > 100
    assert mrf.pairwise[3, 0] == 1.0
    assert mrf.pairwise[3, 3] == 0.0


@pytest.mark.parametrize("method", ["mean_field", "icm"])
def test_the_mrf_removes_speckle(method):
    shape = (4, 12, 12)
    truth = np.zeros(shape, dtype=np.int32)
    truth[:, :, 6:] = 1

    rng = np.random.default_rng(0)
    scores = np.zeros(shape + (2,), dtype=np.float32)
    scores[..., 0] = np.where(truth == 0, 1.0, -1.0)
    scores[..., 1] = -scores[..., 0]
    # 20% of voxels get their evidence flipped
    flip = rng.random(shape) < 0.2
    scores[flip] = scores[flip][:, ::-1]

    raw = np.argmax(scores, axis=3)
    mrf = ROIDerivedMRF(beta=2.0, n_sweeps=6, contrast_sigma=0)
    mrf.pairwise = potts_cost(2)
    labels, diagnostics = mrf.refine(scores, method=method)

    assert diagnostics.method == method
    assert np.mean(labels == truth) > np.mean(raw == truth)
    assert np.mean(labels == truth) > 0.95


def test_beta_zero_reproduces_the_raw_mixture_labels():
    scores = np.random.default_rng(0).normal(size=(3, 5, 5, 3)).astype(np.float32)
    mrf = ROIDerivedMRF(beta=0.0)
    mrf.pairwise = potts_cost(3)
    labels, diagnostics = mrf.refine(scores)
    np.testing.assert_array_equal(labels, np.argmax(scores, axis=3))
    assert diagnostics.method == "none"


def test_lazy_unary_matches_the_dense_array():
    """ICM never materialises [Z, Y, X, K]; it must still agree with it."""
    rng = np.random.default_rng(1)
    table = rng.normal(size=(7, 3)).astype(np.float32)
    rows = rng.integers(-1, 7, size=(3, 4, 5)).astype(np.int32)
    scores = UnaryScores(table, rows)

    dense = scores.dense()
    assert dense.shape == (3, 4, 5, 3)
    for k in range(3):
        np.testing.assert_array_equal(scores.column(k), dense[..., k])
    assert (dense[rows < 0] < -1e29).all()


def test_memory_estimate_drives_the_solver_choice():
    scores = np.zeros((2, 4, 4, 3), dtype=np.float32)

    frugal = ROIDerivedMRF(memory_budget_gb=0.0)
    frugal.pairwise = potts_cost(3)
    _, diagnostics = frugal.refine(scores, method="auto")
    assert diagnostics.method == "icm"

    generous = ROIDerivedMRF(memory_budget_gb=10.0)
    generous.pairwise = potts_cost(3)
    _, diagnostics = generous.refine(scores, method="auto")
    assert diagnostics.method == "mean_field"

    # A 38 M-voxel volume with 9 classes is what forces the choice in practice
    assert frugal.estimate_memory_gb(38_000_000, 9) > 2.0


# ── partial volume ───────────────────────────────────────────────────────────

def test_mixel_ladder_interpolates_between_its_parents():
    mixel = MixelComponent(name="Rim", phase_a="Steel", phase_b="Air", n_alpha=5)
    ladder = build_mixel_ladder(
        mixel, [1000.0, 1000.0], np.eye(2) * 100.0,
        [200.0, 200.0], np.eye(2) * 100.0,
    )
    assert ladder.means.shape == (5, 2)
    assert ladder.means[0, 0] < ladder.means[-1, 0]
    assert 200.0 < ladder.means[0, 0] < 1000.0
    assert 200.0 < ladder.means[-1, 0] < 1000.0
    # Everything lies on the line joining the parents
    for point in ladder.means:
        assert point[0] == pytest.approx(point[1])


def test_expected_fraction_moves_from_zero_to_one_along_the_line():
    mixel = MixelComponent(name="Rim", phase_a="A", phase_b="B", n_alpha=11)
    ladder = build_mixel_ladder(
        mixel, [1000.0, 1000.0], np.eye(2) * 400.0,
        [0.0, 0.0], np.eye(2) * 400.0,
    )
    points = np.stack([np.linspace(0, 1000, 11)] * 2, axis=1)
    expected, _ = fraction_per_bin(ladder, points)
    assert expected[0] < 0.3           # at phase B
    assert expected[-1] > 0.7          # at phase A
    assert np.all(np.diff(expected) > -1e-9)


def test_alignment_angle_flags_a_component_pointing_the_wrong_way():
    along = np.array([[400.0, 390.0], [390.0, 400.0]])      # long axis on y=x
    assert alignment_angle(along, [1000.0, 1000.0], [0.0, 0.0]) < 5.0
    assert alignment_angle(along, [1000.0, 0.0], [0.0, 1000.0]) > 80.0
    assert np.isnan(alignment_angle(np.eye(2) * 100, [1.0, 1.0], [0.0, 0.0]))


def test_elongation_matches_the_axis_ratio():
    assert elongation(np.diag([100.0, 100.0])) == pytest.approx(1.0)
    assert elongation(np.diag([100.0, 400.0])) == pytest.approx(2.0)


def test_mixing_lines_are_detected_only_when_geometry_agrees():
    class _Fit:
        names = ["A", "B", "Rim", "Blob"]
        means = np.array([
            [1000.0, 1000.0], [0.0, 0.0], [500.0, 500.0], [1000.0, 0.0],
        ])
        covariances = np.array([
            np.eye(2) * 100.0,
            np.eye(2) * 100.0,
            np.array([[40000.0, 39000.0], [39000.0, 40000.0]]),  # along A-B
            np.diag([40000.0, 100.0]),                           # elongated, but
        ])                                                       # not along A-B
        n_components = 4

    found = {m.name: (m.phase_a, m.phase_b) for m in detect_mixing_lines(_Fit())}
    assert "Rim" in found
    assert set(found["Rim"]) == {"A", "B"}
    assert "Blob" not in found


def test_verify_rejects_a_misdeclared_pair():
    class _Fit:
        names = ["A", "B", "Rim"]
        means = np.array([[1000.0, 1000.0], [0.0, 0.0], [500.0, 500.0]])
        covariances = np.array([
            np.eye(2) * 100.0, np.eye(2) * 100.0,
            np.diag([40000.0, 100.0]),         # horizontal, not along A-B
        ])
        n_components = 3

    report = verify_mixels(
        _Fit(), [MixelComponent(name="Rim", phase_a="A", phase_b="B")]
    )
    assert report["Rim"]["accepted"] is False
    assert "off the line" in report["Rim"]["reason"]


# ── the sequential loop ──────────────────────────────────────────────────────

def _series(n_timepoints=5, drift_per_step=90.0):
    """A drifting series in which Lithium genuinely shrinks."""
    neutron, xray, masks = [], [], []
    for step in range(n_timepoints):
        width = max(6 - step, 1)
        volume_n, volume_x, mask = _volume(
            shift=drift_per_step * step, lithium_width=width, seed=step
        )
        neutron.append(volume_n)
        xray.append(volume_x)
        masks.append(mask)
    return (
        _FakeDataset(np.stack(neutron), np.stack(xray)),
        masks,
        np.stack(neutron),
        np.stack(xray),
    )


def _run(dataset, masks, neutron, xray, tracker, temporal, beta=1.0):
    neutron_edges, xray_edges = _edges()
    segmenter = SequentialSegmenter(
        mixture=ROIAnchoredMixture(outlier_component=True),
        mrf=ROIDerivedMRF(beta=beta, n_sweeps=3) if beta > 0 else None,
        temporal=temporal,
        drift_tracker=tracker,
    )
    segmenter.prepare(
        neutron[0], xray[0], masks[0], neutron_edges, xray_edges,
        anchor_strength=0.5,
    )
    return segmenter.run(dataset)


def _total_error(outcome, masks):
    return sum(
        abs(entry.voxel_counts.get(name, 0) - int(masks[entry.timepoint][name].sum()))
        for entry in outcome.timepoints
        for name in masks[entry.timepoint]
    )


def test_drift_tracking_beats_a_frozen_boundary_on_a_drifting_series():
    """The claim the whole design rests on."""
    dataset, masks, neutron, xray = _series()

    frozen = _run(dataset, masks, neutron, xray, None, StaticTransition())
    tracked = _run(
        dataset, masks, neutron, xray,
        DriftTracker(anchor_classes=("Air", "Aluminium")),
        DriftTransition(memory=0.5),
    )

    frozen_error = _total_error(frozen, masks)
    tracked_error = _total_error(tracked, masks)
    assert tracked_error < frozen_error / 5, (tracked_error, frozen_error)


def test_the_model_does_not_invent_change_when_there_is_none():
    """A static series must produce flat curves, drift tracking or not."""
    static = [_volume(shift=0.0, seed=t) for t in range(4)]
    dataset = _FakeDataset(
        np.stack([entry[0] for entry in static]),
        np.stack([entry[1] for entry in static]),
    )
    masks = [entry[2] for entry in static]

    outcome = _run(
        dataset, masks, dataset.neutron_data, dataset.xray_data,
        DriftTracker(anchor_classes=("Air", "Aluminium")),
        DriftTransition(memory=0.5),
    )
    for name in PHASES:
        counts = [entry.voxel_counts[name] for entry in outcome.timepoints]
        assert max(counts) - min(counts) < 0.05 * max(counts), (name, counts)


def test_every_timepoint_emits_a_full_parameter_set():
    dataset, masks, neutron, xray = _series(n_timepoints=4)
    outcome = _run(
        dataset, masks, neutron, xray,
        DriftTracker(anchor_classes=("Air", "Aluminium")),
        DriftTransition(memory=0.5),
    )
    trajectories = outcome.parameter_trajectories()
    assert set(trajectories) == set(PHASES)
    for name, entry in trajectories.items():
        assert entry["timepoint"] == [0, 1, 2, 3]
        assert all(np.isfinite(entry["centroid_n"]))
        assert all(value > 0 for value in entry["sigma_n"])

    # Real physics survives: Lithium shrinks monotonically
    lithium = trajectories["Lithium"]["voxels"]
    assert lithium[0] > lithium[-1]


def test_padding_never_reaches_a_class():
    padded = [_volume(shift=60.0 * t, seed=t, pad=True) for t in range(3)]
    dataset = _FakeDataset(
        np.stack([entry[0] for entry in padded]),
        np.stack([entry[1] for entry in padded]),
    )
    masks = [entry[2] for entry in padded]

    outcome = _run(
        dataset, masks, dataset.neutron_data, dataset.xray_data,
        DriftTracker(anchor_classes=("Aluminium",)),
        DriftTransition(memory=0.5),
    )
    for entry in outcome.timepoints:
        neutron, xray = dataset.get_volume_at_time(entry.timepoint)
        padding = (neutron == 0) & (xray == 0)
        assert padding.any()
        assert (entry.labels[padding] == -1).all()


def test_results_expose_per_class_masks_and_counts():
    dataset, masks, neutron, xray = _series(n_timepoints=3)
    outcome = _run(
        dataset, masks, neutron, xray,
        DriftTracker(anchor_classes=("Air", "Aluminium")),
        DriftTransition(memory=0.5),
    )
    first = outcome.timepoints[0]
    mask = first.mask_for("Aluminium")
    assert mask.shape == neutron[0].shape
    assert int(mask.sum()) == first.voxel_counts["Aluminium"]
    with pytest.raises(KeyError):
        first.mask_for("Nonexistent")

    table = outcome.counts_table()
    assert set(table) == {0, 1, 2}


def test_an_implausible_jump_is_clipped_and_recorded():
    from model.mixture import FitResult

    def _result(mean):
        return FitResult(
            names=["A"], means=np.array([mean], dtype=float),
            covariances=np.array([np.eye(2)]), weights=np.array([1.0]),
            outlier_weight=0.0, responsibilities=np.zeros((1, 1)),
            log_density=np.zeros((1, 1)), counts=np.array([10.0]),
            log_likelihood=0.0, n_iter=1, converged=True, num_voxels=10,
        )

    previous = _result([0.0, 0.0])
    current = _result([1000.0, 0.0])
    temporal = DriftTransition(
        process_noise=np.array([1.0, 1.0]), step_limit_sigma=5.0
    )
    diagnostics = temporal.post_fit(1, current, previous)
    assert "A" in diagnostics.clipped
    assert current.means[0][0] == pytest.approx(5.0)


def test_a_collapsed_class_goes_dormant_instead_of_being_deleted():
    """A phase that vanishes and returns must keep its identity."""
    from model.mixture import FitResult

    def _result(weight):
        return FitResult(
            names=["Water"], means=np.array([[5.0, 6.0]]),
            covariances=np.array([np.eye(2)]), weights=np.array([weight]),
            outlier_weight=0.0, responsibilities=np.zeros((1, 1)),
            log_density=np.zeros((1, 1)), counts=np.array([1.0]),
            log_likelihood=0.0, n_iter=1, converged=True, num_voxels=10,
        )

    temporal = DriftTransition(min_weight=0.01)
    assert temporal.post_fit(0, _result(0.5)).dormant == []

    faded = _result(0.001)
    assert temporal.post_fit(1, faded).dormant == ["Water"]
    assert temporal.dormant_classes == ["Water"]

    returned = _result(0.4)
    diagnostics = temporal.post_fit(2, returned)
    assert diagnostics.resurrected == ["Water"]
    assert temporal.dormant_classes == []
