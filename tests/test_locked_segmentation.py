"""Locked-mode segmentation: fixed classes, guards, and the health check.

The design claim under test: with class definitions fixed, three failure
modes an adaptive fit can have are *structurally* impossible rather than
guarded against — classes cannot merge, identities cannot permute, and
timepoints cannot influence one another. What remains possible is
over-smoothing and a genuinely missing material, and both must be refused
rather than reported.
"""

import numpy as np
import pytest

from model import (
    UNCLASSIFIED,
    ClassLibrary,
    LockedSegmenter,
    MaterialClass,
    SegmentationRefused,
    Status,
    ValidityPolicy,
    build_histogram_cache,
    build_valid_mask,
    channel_coverage,
    auto_floor,
    match_table,
    run_health_check,
    validity_report,
)
from model.spatial_prior import ROIDerivedMRF, potts_cost

PHASES = {"Air": (400.0, 400.0), "Steel": (900.0, 900.0),
          "Lithium": (1400.0, 1400.0)}
SIGMA = 80.0
RANGE = (0.0, 2200.0)


def _volume(lithium_width=6, shift=0.0, seed=0, xray_gap=0):
    """One timepoint. *xray_gap* blanks the X-ray channel over N columns."""
    rng = np.random.default_rng(seed)
    shape = (4, 16, 24)
    neutron = np.zeros(shape)
    xray = np.zeros(shape)
    masks = {}
    for name, (mean_n, mean_x) in PHASES.items():
        mask = np.zeros(shape, dtype=bool)
        if name == "Air":
            mask[:, :, :8] = True
            mask[:, :, 16 + lithium_width:] = True
        elif name == "Steel":
            mask[:, :, 8:16] = True
        else:
            mask[:, :, 16:16 + lithium_width] = True
        count = int(mask.sum())
        neutron[mask] = np.clip(
            rng.normal(mean_n + shift, SIGMA, count), 20, 2180
        )
        xray[mask] = np.clip(rng.normal(mean_x + shift, SIGMA, count), 20, 2180)
        masks[name] = mask
    if xray_gap:
        xray[:, :, :xray_gap] = 0.0
    return neutron, xray, masks


def _edges(bins=128):
    edges = np.linspace(RANGE[0], RANGE[1], bins + 1)
    return edges, edges.copy()


class _FakeDataset:
    def __init__(self, neutron, xray):
        self.neutron_data = np.asarray(neutron)
        self.xray_data = np.asarray(xray)
        self.num_timepoints = self.neutron_data.shape[0]

    def get_volume_at_time(self, timepoint):
        return self.neutron_data[timepoint], self.xray_data[timepoint]


def _series(n_timepoints=4, shift_per_step=0.0, xray_gap=0):
    neutron, xray, masks = [], [], []
    for step in range(n_timepoints):
        volume_n, volume_x, mask = _volume(
            lithium_width=max(6 - step, 1),
            shift=shift_per_step * step, seed=step, xray_gap=xray_gap,
        )
        neutron.append(volume_n)
        xray.append(volume_x)
        masks.append(mask)
    return _FakeDataset(np.stack(neutron), np.stack(xray)), masks


def _segmenter(masks, neutron, xray, beta=1.0, inert=("Steel",)):
    valid = build_valid_mask(neutron, xray)
    library = ClassLibrary.from_masks(
        neutron, xray, masks, valid_mask=valid, inert=inert
    )
    prior = ROIDerivedMRF(beta=beta, n_sweeps=4) if beta > 0 else None
    segmenter = LockedSegmenter(library, prior=prior)
    segmenter.set_grid(*_edges())
    if prior is not None:
        reference = np.zeros(valid.shape, dtype=np.int32)
        for index, name in enumerate(library.names):
            reference[masks[name]] = index
        segmenter.learn_boundaries(reference, valid_mask=valid)
    return segmenter, library


# ── validity: both channels ──────────────────────────────────────────────────

def test_a_voxel_needs_data_in_both_channels():
    """The failure that made a 22% neutron-only region into a class."""
    neutron = np.full((4, 8, 8), 500.0)
    xray = np.full((4, 8, 8), 500.0)
    xray[:, :, :4] = 0.0                     # X-ray does not cover this half

    valid = build_valid_mask(neutron, xray)
    assert not valid[:, :, :4].any()
    assert valid[:, :, 4:].all()

    coverage = channel_coverage(neutron, xray)
    assert coverage["neutron_only"] == 4 * 8 * 4
    assert coverage["xray_only"] == 0
    assert coverage["overlap_fraction"] == pytest.approx(0.5)


def test_either_channel_missing_is_enough_to_reject():
    neutron = np.full((2, 4, 4), 500.0)
    xray = np.full((2, 4, 4), 500.0)
    neutron[0, 0, 0] = 0.0
    xray[0, 0, 1] = 0.0
    valid = build_valid_mask(neutron, xray)
    assert not valid[0, 0, 0] and not valid[0, 0, 1]
    assert valid.sum() == neutron.size - 2


def test_single_channel_mode_is_available_but_not_the_default():
    neutron = np.full((2, 4, 4), 500.0)
    xray = np.full((2, 4, 4), 500.0)
    xray[:, :, :2] = 0.0

    assert not build_valid_mask(neutron, xray)[:, :, :2].any()
    lenient = ValidityPolicy(require_both_channels=False)
    assert build_valid_mask(neutron, xray, lenient)[:, :, :2].all()


def test_floors_are_derived_not_assumed():
    neutron, xray, _ = _volume()
    policy = ValidityPolicy.from_data(neutron, xray)
    assert policy.neutron_floor is not None
    assert policy.neutron_floor < 400.0        # below the darkest real peak
    # ...and the default policy still has none
    assert ValidityPolicy().neutron_floor is None


def test_validity_report_names_the_coverage_problem():
    neutron, xray, _ = _volume(xray_gap=6)
    report = validity_report(neutron, xray)
    assert report["neutron_only"] > 0
    assert report["overlap_fraction"] < 1.0
    assert "either channel" in report["policy"]


# ── class library ────────────────────────────────────────────────────────────

def test_classes_are_the_rois_in_roi_order():
    """No matching step, so there is nothing to get wrong."""
    neutron, xray, masks = _volume()
    library = ClassLibrary.from_masks(neutron, xray, masks)

    assert library.names == list(masks)
    assert library.label_values() == {"Air": 1, "Steel": 2, "Lithium": 3}
    assert library.index_of("Steel") == 1


def test_class_moments_come_from_the_selected_voxels():
    neutron, xray, masks = _volume()
    library = ClassLibrary.from_masks(neutron, xray, masks)
    steel = library[library.index_of("Steel")]

    values = np.stack([neutron[masks["Steel"]], xray[masks["Steel"]]])
    np.testing.assert_allclose(steel.mu, values.mean(axis=1), rtol=1e-9)
    np.testing.assert_allclose(
        steel.sigma, np.cov(values, bias=False), rtol=1e-9
    )
    assert steel.source == "roi"


def test_control_materials_are_recorded():
    neutron, xray, masks = _volume()
    library = ClassLibrary.from_masks(
        neutron, xray, masks, inert=["Air", "Steel"]
    )
    assert library.inert_names == ["Air", "Steel"]
    library.mark_inert(["Steel"])
    assert library.inert_names == ["Steel"]


def test_physics_placed_classes_are_available():
    library = ClassLibrary.from_physics(
        loci={"Air": (400.0, 400.0), "Steel": (900.0, 900.0)},
        spreads={"Air": (80.0, 80.0), "Steel": (80.0, 80.0)},
        inert=["Steel"],
    )
    assert library.names == ["Air", "Steel"]
    assert all(material.source == "physics" for material in library)
    assert library[1].inert


def test_duplicate_class_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        ClassLibrary([
            MaterialClass("A", [1, 1], np.eye(2)),
            MaterialClass("A", [2, 2], np.eye(2)),
        ])


# ── match table ──────────────────────────────────────────────────────────────

def test_the_match_table_is_computed_once_per_bin_not_per_voxel():
    neutron, xray, masks = _volume()
    valid = build_valid_mask(neutron, xray)
    cache = build_histogram_cache(neutron, xray, *_edges(), valid_mask=valid)
    library = ClassLibrary.from_masks(neutron, xray, masks, valid_mask=valid)

    table = match_table(library, cache)
    assert table.scores.shape == (cache.num_bins, len(library))
    assert cache.num_bins < cache.num_voxels        # that is the whole point
    assert table.n_classes == 3


def test_each_material_matches_its_own_locus_best():
    neutron, xray, masks = _volume()
    valid = build_valid_mask(neutron, xray)
    cache = build_histogram_cache(neutron, xray, *_edges(), valid_mask=valid)
    library = ClassLibrary.from_masks(neutron, xray, masks, valid_mask=valid)
    table = match_table(library, cache)

    for index, material in enumerate(library):
        distances = np.linalg.norm(cache.means - material.mu, axis=1)
        nearest = int(np.argmin(distances))
        assert int(np.argmax(table.scores[nearest])) == index, material.name


def test_something_no_material_explains_lands_in_unclassified():
    neutron, xray, masks = _volume()
    valid = build_valid_mask(neutron, xray)
    library = ClassLibrary.from_masks(neutron, xray, masks, valid_mask=valid)

    # A bin far from every class, in an otherwise normal cache
    intruder_n = neutron.copy()
    intruder_x = xray.copy()
    intruder_n[0, 0, :4] = 300.0
    intruder_x[0, 0, :4] = 2000.0
    cache = build_histogram_cache(
        intruder_n, intruder_x, *_edges(),
        valid_mask=build_valid_mask(intruder_n, intruder_x),
    )
    table = match_table(library, cache)
    distances = np.linalg.norm(cache.means - np.array([300.0, 2000.0]), axis=1)
    odd_bin = int(np.argmin(distances))
    assert table.best_class_per_bin()[odd_bin] == -1


# ── locked segmentation ──────────────────────────────────────────────────────

def test_locked_mode_recovers_the_materials():
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, library = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)

    assert len(outcome) == dataset.num_timepoints
    assert outcome.class_names == list(PHASES)
    for entry in outcome.timepoints:
        for name in PHASES:
            truth = int(masks[entry.timepoint][name].sum())
            assert abs(entry.voxel_counts[name] - truth) < 0.1 * truth, name


def test_a_real_change_survives_and_is_not_smoothed_away():
    """Lithium genuinely shrinks; that is the measurement."""
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)

    curve = outcome.volume_curve("Lithium")
    truth = [int(mask["Lithium"].sum()) for mask in masks]
    assert curve == sorted(curve, reverse=True)
    assert curve[0] > curve[-1]
    # The measured shrinkage matches the real one, not a smoothed version
    for measured, actual in zip(curve, truth):
        assert abs(measured - actual) < 0.1 * actual


def test_timepoints_are_independent():
    """Order-independent, reproducible, and safe to parallelise."""
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)

    order = list(range(dataset.num_timepoints))
    forwards = segmenter.segment_series(dataset, timepoints=order)
    backwards = segmenter.segment_series(dataset, timepoints=order[::-1])
    single = segmenter.segment_series(dataset, timepoints=[2])

    for entry in forwards.timepoints:
        other = next(
            e for e in backwards.timepoints if e.timepoint == entry.timepoint
        )
        np.testing.assert_array_equal(entry.labels, other.labels)
    np.testing.assert_array_equal(
        single.timepoints[0].labels, forwards.timepoints[2].labels
    )


def test_every_voxel_is_counted_exactly_once():
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)

    for entry in outcome.timepoints:
        assert entry.budget_closes()
        assert (
            sum(entry.voxel_counts.values())
            + entry.unclassified_voxels + entry.excluded_voxels
            == entry.total_voxels
        )


def test_unmeasured_voxels_are_not_counted_as_unmatched():
    """'We looked and found nothing' is not 'there was nothing to look at'."""
    dataset, masks = _series(xray_gap=6)
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)

    entry = outcome.timepoints[0]
    assert entry.excluded_voxels > 0
    assert entry.unclassified_fraction < 0.02
    assert (entry.labels[:, :, :6] == UNCLASSIFIED).all()


def test_labels_use_the_roi_order_with_zero_reserved():
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, library = _segmenter(masks[0], neutron, xray)
    entry = segmenter.segment_timepoint(neutron, xray)

    for name, value in library.label_values().items():
        assert value >= 1
        np.testing.assert_array_equal(entry.mask_for(name), entry.labels == value)
    assert UNCLASSIFIED == 0


# ── smoothing ────────────────────────────────────────────────────────────────

def test_auto_smoothing_returns_a_sweep_you_can_inspect():
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)

    strength, sweep = segmenter.auto_smoothing(neutron, xray)
    assert sweep[0]["smoothing"] == 0.0
    assert all("retention" in row and "acceptable" in row for row in sweep)
    assert strength in [row["smoothing"] for row in sweep]
    assert all(row["worst_retention"] <= 1.001 for row in sweep)


def _dispersed_case(seed=7, fraction=0.03, separation=1.2):
    """A finely dispersed minority phase — isolated voxels, not a structure.

    This is the configuration over-smoothing actually destroys. A coherent
    sheet is reinforced by its own neighbours; scattered voxels have none to
    lean on, so they go first, and losing them is invisible afterwards.
    """
    rng = np.random.default_rng(seed)
    shape = (6, 20, 20)
    neutron = np.zeros(shape)
    xray = np.zeros(shape)
    dispersed = rng.random(shape) < fraction
    bulk = ~dispersed
    for mask, centre in (
        (bulk, 900.0), (dispersed, 900.0 + separation * SIGMA)
    ):
        count = int(mask.sum())
        neutron[mask] = np.clip(rng.normal(centre, SIGMA, count), 20, 2180)
        xray[mask] = np.clip(rng.normal(centre, SIGMA, count), 20, 2180)
    return neutron, xray, {"Bulk": bulk, "Plating": dispersed}



def test_smoothing_eats_into_a_dispersed_phase_and_it_is_measured():
    """The mechanism the automatic search exists to detect."""
    neutron, xray, masks = _dispersed_case()
    segmenter, _ = _segmenter(masks, neutron, xray, inert=("Bulk",))

    _, sweep = segmenter.auto_smoothing(
        neutron, xray, grid=(0.0, 1.0, 2.0, 4.0, 8.0)
    )
    retained = [row["retention"]["Plating"] for row in sweep]
    assert retained[0] == pytest.approx(1.0)
    assert retained[-1] < retained[0]
    assert all("Plating" in row["retention"] for row in sweep)


def test_a_coherent_structure_is_protected_by_the_learned_boundaries():
    """Why the boundary costs are learned rather than assumed.

    A minority material that genuinely borders its neighbour at the
    reference timepoint has a cheap boundary, so smoothing has no incentive
    to remove it even though it is small. A uniform cost cannot express
    that, and would trade the material away for a smoother picture.
    """
    neutron, xray, masks = _volume(lithium_width=1)
    segmenter, _ = _segmenter(masks, neutron, xray)

    _, sweep = segmenter.auto_smoothing(
        neutron, xray, grid=(0.0, 1.0, 4.0, 16.0)
    )
    assert all(row["retention"]["Lithium"] > 0.85 for row in sweep)
    assert all(row["acceptable"] for row in sweep)



def test_smoothing_removes_speckle_without_moving_the_boundary():
    neutron, xray, masks = _volume()
    rng = np.random.default_rng(3)
    noisy_n = neutron + rng.normal(0, 140, neutron.shape)
    noisy_x = xray + rng.normal(0, 140, xray.shape)

    segmenter, _ = _segmenter(masks, neutron, xray)
    rough = segmenter.segment_timepoint(noisy_n, noisy_x, beta=0.0)
    smooth = segmenter.segment_timepoint(noisy_n, noisy_x, beta=2.0)

    from utils.metrics_spatial import class_spatial_metrics

    rough_pieces = class_spatial_metrics(rough.mask_for("Steel"))["n_components_k"]
    smooth_pieces = class_spatial_metrics(smooth.mask_for("Steel"))["n_components_k"]
    assert smooth_pieces < rough_pieces
    truth = int(masks["Steel"].sum())
    assert abs(smooth.voxel_counts["Steel"] - truth) < abs(
        rough.voxel_counts["Steel"] - truth
    )


def test_the_refinement_reports_whether_it_settled():
    neutron, xray, masks = _volume()
    segmenter, _ = _segmenter(masks, neutron, xray, beta=1.0)
    entry = segmenter.segment_timepoint(neutron, xray, beta=1.0)

    assert entry.refinement is not None
    assert entry.refinement.energy_trace
    assert entry.refinement.monotone


def test_damping_is_on_so_a_two_cycle_cannot_settle_in():
    prior = ROIDerivedMRF(beta=1.0)
    assert 0.0 < prior.damping < 1.0


# ── boundary costs ───────────────────────────────────────────────────────────

def test_boundary_costs_come_from_the_reference_labels():
    labels = np.zeros((4, 12, 12), dtype=np.int32)
    labels[:, :, 4:8] = 1
    labels[:, :, 8:] = 2

    prior = ROIDerivedMRF()
    cost = prior.fit_pairwise_from_labels(labels, n_classes=3)
    np.testing.assert_allclose(np.diag(cost), 0.0, atol=1e-12)
    np.testing.assert_allclose(cost, cost.T)
    assert cost[0, 2] > cost[0, 1]          # 0 and 2 never touch


def test_the_raw_form_penalises_a_class_for_existing():
    """Why the diagonal is removed by default.

    A class whose own voxels are less reliably adjacent — a thin or scattered
    one — carries a standing cost in the raw form, on top of the cost of
    bordering anything. That is a bias against exactly the classes most at
    risk of being smoothed away.
    """
    labels = np.zeros((4, 12, 12), dtype=np.int32)
    labels[:, :, 5:6] = 1                    # one voxel thick
    labels[:, :, 6:] = 2

    prior = ROIDerivedMRF()
    raw = prior.fit_pairwise_from_labels(
        labels, n_classes=3, zero_diagonal=False
    )
    zeroed = prior.fit_pairwise_from_labels(labels, n_classes=3)

    assert raw[1, 1] > raw[0, 0]             # the thin class pays more
    assert zeroed[1, 1] == pytest.approx(0.0)


def test_unclassified_gets_a_boundary_cost_of_its_own():
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, library = _segmenter(masks[0], neutron, xray)

    cost = segmenter.prior.pairwise
    assert cost.shape == (len(library) + 1, len(library) + 1)
    assert cost[-1, -1] == 0.0
    assert cost[-1, 0] > 0                   # neither free nor prohibitive
    assert np.isfinite(cost).all()


# ── guards ───────────────────────────────────────────────────────────────────

def _fabricated(timepoint, counts, unsmoothed, valid=1000, total=1000):
    """A result with known numbers, to exercise a decision rule directly."""
    from model.locked import TimepointSegmentation

    assigned = sum(counts.values())
    return TimepointSegmentation(
        timepoint=timepoint,
        labels=np.zeros((1, 1, 1), dtype=np.int32),
        class_names=list(counts),
        voxel_counts=dict(counts),
        unsmoothed_counts=dict(unsmoothed),
        unclassified_voxels=valid - assigned,
        excluded_voxels=total - valid,
        valid_voxels=valid,
        total_voxels=total,
    )


def test_over_smoothing_is_refused_rather_than_reported():
    """The rule itself: a class smoothing removed must stop the run.

    Tested on constructed numbers rather than by hunting for a physical
    configuration that triggers it — the rule is what has to be right, and
    the empirical tests above already show the mechanism is live.
    """
    from model.locked import SeriesSegmentation

    neutron, xray, masks = _volume()
    segmenter, library = _segmenter(masks, neutron, xray)
    outcome = SeriesSegmentation(library=library)
    outcome.timepoints = [
        _fabricated(0, {"Air": 500, "Steel": 480, "Lithium": 20},
                    {"Air": 500, "Steel": 400, "Lithium": 100}),
    ]

    findings = segmenter.check_guards(outcome)
    joined = " ".join(findings)
    assert "Smoothing removed 80% of the class 'Lithium'" in joined
    assert "Reduce the smoothing strength" in joined
    assert SegmentationRefused(findings).findings == findings


def test_a_class_that_shrinks_without_smoothing_is_not_refused():
    """The same rule must not fire on a real change."""
    from model.locked import SeriesSegmentation

    neutron, xray, masks = _volume()
    segmenter, library = _segmenter(masks, neutron, xray)
    outcome = SeriesSegmentation(library=library)
    outcome.timepoints = [
        # Lithium is tiny, but smoothing is not what made it tiny
        _fabricated(0, {"Air": 900, "Steel": 80, "Lithium": 20},
                    {"Air": 900, "Steel": 80, "Lithium": 20}),
    ]
    assert segmenter.check_guards(outcome) == []



def test_a_real_shrinkage_is_never_refused():
    """The guard must not fire on the thing the software exists to measure."""
    dataset, masks = _series(n_timepoints=5)
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)

    outcome = segmenter.segment_series(dataset)        # no exception
    curve = outcome.volume_curve("Lithium")
    assert curve[-1] < 0.4 * curve[0]                  # a large real change


def test_a_missing_material_is_refused_with_advice():
    neutron, xray, masks = _volume()
    partial = {"Air": masks["Air"]}                    # two materials undefined
    dataset = _FakeDataset(neutron[None], xray[None])
    segmenter, _ = _segmenter(partial, neutron, xray, beta=0.0, inert=())

    with pytest.raises(SegmentationRefused) as raised:
        segmenter.segment_series(dataset)
    joined = " ".join(raised.value.findings)
    assert "did not match any material" in joined
    assert "missing a material" in joined


def test_drift_is_distinguished_from_a_missing_material():
    """Two failures that look alike need opposite responses."""
    dataset, masks = _series(n_timepoints=5, shift_per_step=140.0)
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)

    with pytest.raises(SegmentationRefused) as raised:
        segmenter.segment_series(dataset)
    joined = " ".join(raised.value.findings)
    assert "drifted" in joined
    assert "Check Instrument Stability" in joined


def test_guards_can_be_deferred_for_inspection():
    dataset, masks = _series(n_timepoints=5, shift_per_step=140.0)
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)

    outcome = segmenter.segment_series(dataset, enforce_guards=False)
    assert len(outcome) == 5
    assert segmenter.check_guards(outcome)          # still reports the problem


# ── health check ─────────────────────────────────────────────────────────────

def test_a_clean_run_passes_every_check():
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, library = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)

    report = run_health_check(outcome, control_materials=library.inert_names)
    assert report.status is Status.PASS
    assert report.passed
    assert "All checks passed" in report.headline()


def test_a_moving_control_material_fails_the_check():
    """The null control doing its job."""
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)

    # Air grows as Lithium shrinks — declaring it unchanging is a claim the
    # data does not support, and the check must say so.
    report = run_health_check(outcome, control_materials=["Air"])
    assert report.status is Status.FAIL
    problems = " ".join(f.message for f in report.problems())
    assert "'Air' changed by" in problems
    assert "should not change" in problems


def test_no_control_material_is_a_warning_not_a_pass():
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)

    report = run_health_check(outcome, control_materials=[])
    assert report.status is Status.WARN
    assert any("No control materials" in f.message for f in report.warnings())


def test_a_field_of_view_mismatch_is_surfaced():
    dataset, masks = _series(xray_gap=6)
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, library = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)

    report = run_health_check(outcome, control_materials=library.inert_names)
    text = report.describe()
    assert "only one of the two instruments" in text
    assert "excluded" in text


def test_every_finding_names_something_and_suggests_an_action():
    dataset, masks = _series()
    neutron, xray = dataset.get_volume_at_time(0)
    segmenter, _ = _segmenter(masks[0], neutron, xray)
    outcome = segmenter.segment_series(dataset)
    report = run_health_check(outcome, control_materials=["Air"])

    for finding in report.findings:
        assert finding.message.endswith((".", "%")) or ":" in finding.message
        if finding.status is not Status.PASS:
            assert finding.detail, finding.message


def test_an_empty_run_is_reported_not_crashed():
    class _Empty:
        timepoints = []
        class_names = []

    report = run_health_check(_Empty())
    assert report.status is Status.FAIL
    assert "Nothing was segmented" in report.describe()
