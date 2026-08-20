"""Ground-truth-free histogram/class metrics and the segmentation report."""

import csv
import os

import numpy as np
import pytest

from histograms.histogram_engine_4d import HistogramEngine4D
from utils.histogram_metrics import (
    METRIC_INFO,
    PER_CLASS_METRICS,
    SCALAR_METRICS,
    MetricsRow,
    compute_class_metrics,
    compute_shape_metrics,
    plot_metric_evolution,
    write_metrics_csv,
)
from utils.segmentation_report import build_segmentation_report


# ── fixtures ─────────────────────────────────────────────────────────────────

def _two_material_volumes(shift=0.0):
    """Two well-separated materials; *shift* moves both along neutron."""
    neutron = np.full((4, 16, 16), 500.0 + shift)
    xray = np.full((4, 16, 16), 500.0)
    blob_a = np.zeros(neutron.shape, dtype=bool)
    blob_a[:, :6, :6] = True
    blob_b = np.zeros(neutron.shape, dtype=bool)
    blob_b[:, 10:, 10:] = True
    neutron[blob_a] = 100.0 + shift
    xray[blob_a] = 900.0
    neutron[blob_b] = 900.0 + shift
    xray[blob_b] = 100.0
    return neutron, xray, {"A": blob_a, "B": blob_b}


def _histogram(neutron_3d, xray_3d, engine=None):
    engine = engine or HistogramEngine4D(bins=32, use_gpu=False)
    return engine.compute_global_histogram(
        neutron_3d[None, ...], xray_3d[None, ...]
    )


# ── shape metrics ────────────────────────────────────────────────────────────

def test_shape_metrics_are_finite_for_a_real_histogram():
    neutron, xray, _ = _two_material_volumes()
    metrics = compute_shape_metrics(_histogram(neutron, xray))

    for name in ("S_h", "S_v", "S_d", "A_x"):
        assert metrics[name] is not None
        assert np.isfinite(metrics[name]), name
    assert metrics["Delta_n"] is None      # no reference given


def test_empty_histogram_yields_no_shape_metrics():
    hist = _histogram(*_two_material_volumes()[:2])
    hist.histogram[:] = 0
    assert all(value is None for value in compute_shape_metrics(hist).values())


def test_diagonal_smear_is_signed_correlation():
    """S_d is a Pearson rho, so anticorrelated materials give a negative value."""
    # A: low neutron / high X-ray, B: high neutron / low X-ray -> anticorrelated
    neutron, xray, _ = _two_material_volumes()
    assert compute_shape_metrics(_histogram(neutron, xray))["S_d"] < 0

    # Make them correlated instead and the sign flips
    correlated_x = xray.copy()
    correlated_x[neutron < 300] = 100.0
    correlated_x[neutron > 700] = 900.0
    assert compute_shape_metrics(_histogram(neutron, correlated_x))["S_d"] > 0


def test_delta_n_tracks_a_neutron_shift():
    engine = HistogramEngine4D(bins=32, use_gpu=False)
    # One shared grid so both histograms are comparable
    neutron_a, xray_a, _ = _two_material_volumes()
    neutron_b, xray_b, _ = _two_material_volumes(shift=50.0)
    engine.compute_global_histogram(
        np.stack([neutron_a, neutron_b]), np.stack([xray_a, xray_b])
    )
    hist_a = engine.compute_local_histogram(neutron_a, xray_a, 0)
    hist_b = engine.compute_local_histogram(neutron_b, xray_b, 1)

    assert abs(compute_shape_metrics(hist_a, reference=hist_a)["Delta_n"]) < 1e-9
    shifted = compute_shape_metrics(hist_b, reference=hist_a)["Delta_n"]
    assert shifted == pytest.approx(50.0, abs=engine.bins and 40.0)
    assert shifted > 10.0


# ── class metrics ────────────────────────────────────────────────────────────

def test_class_metrics_describe_each_class():
    neutron, xray, masks = _two_material_volumes()
    scalars, per_class, centroids = compute_class_metrics(neutron, xray, masks)

    assert scalars["n_classes"] == 2
    assert per_class["voxels_k"]["A"] == float(masks["A"].sum())
    assert per_class["centroid_n_k"]["A"] == pytest.approx(100.0)
    assert per_class["centroid_x_k"]["A"] == pytest.approx(900.0)
    assert per_class["centroid_n_k"]["B"] == pytest.approx(900.0)
    # Uniform materials have no spread, so elongation is the isotropic 1.0
    assert per_class["sigma_n_k"]["A"] == pytest.approx(0.0)
    assert per_class["E_k"]["A"] == pytest.approx(1.0)
    assert centroids["A"] == pytest.approx((100.0, 900.0))
    assert not per_class["drift_k"]           # no reference centroids given


def test_elongation_grows_with_anisotropy():
    """E_k is the ratio of the larger spread to the smaller one."""
    neutron, xray, masks = _two_material_volumes()
    rng = np.random.default_rng(2)
    # Spread class A ten times further along neutron than along X-ray
    neutron[masks["A"]] += rng.normal(0, 100.0, int(masks["A"].sum()))
    xray[masks["A"]] += rng.normal(0, 10.0, int(masks["A"].sum()))

    _, per_class, _ = compute_class_metrics(neutron, xray, masks)
    assert per_class["E_k"]["A"] == pytest.approx(10.0, rel=0.3)
    assert per_class["E_k"]["B"] == pytest.approx(1.0)   # still uniform


def test_davies_bouldin_rewards_separated_classes():
    """Tight, distant classes score lower than broad, overlapping ones."""
    neutron, xray, masks = _two_material_volumes()
    tight, _, _ = compute_class_metrics(neutron, xray, masks)

    rng = np.random.default_rng(0)
    noisy_n = neutron + rng.normal(0, 200.0, neutron.shape)
    noisy_x = xray + rng.normal(0, 200.0, xray.shape)
    broad, _, _ = compute_class_metrics(noisy_n, noisy_x, masks)

    assert tight["DB"] < broad["DB"]


def test_davies_bouldin_needs_two_classes():
    neutron, xray, masks = _two_material_volumes()
    scalars, _, _ = compute_class_metrics(neutron, xray, {"A": masks["A"]})
    assert scalars["DB"] is None
    assert scalars["n_classes"] == 1


def test_class_drift_is_measured_against_the_reference_centroids():
    neutron, xray, masks = _two_material_volumes()
    _, _, reference = compute_class_metrics(neutron, xray, masks)

    moved_n, moved_x, moved_masks = _two_material_volumes(shift=30.0)
    scalars, per_class, _ = compute_class_metrics(
        moved_n, moved_x, moved_masks, reference_centroids=reference
    )
    assert per_class["drift_k"]["A"] == pytest.approx(30.0)
    assert per_class["drift_k"]["B"] == pytest.approx(30.0)
    assert scalars["CD"] == pytest.approx(30.0)


def test_empty_and_mismatched_masks_are_skipped():
    neutron, xray, masks = _two_material_volumes()
    masks["empty"] = np.zeros(neutron.shape, dtype=bool)
    masks["wrong_shape"] = np.ones((2, 2, 2), dtype=bool)

    scalars, per_class, centroids = compute_class_metrics(neutron, xray, masks)
    assert scalars["n_classes"] == 2
    assert set(centroids) == {"A", "B"}
    assert "empty" not in per_class["voxels_k"]
    assert "wrong_shape" not in per_class["voxels_k"]


def test_non_finite_voxels_do_not_poison_the_statistics():
    neutron, xray, masks = _two_material_volumes()
    neutron[0, 0, 0] = np.nan          # inside blob A
    _, per_class, _ = compute_class_metrics(neutron, xray, masks)
    assert np.isfinite(per_class["centroid_n_k"]["A"])
    assert per_class["voxels_k"]["A"] == float(masks["A"].sum()) - 1


def test_subsampling_preserves_the_class_statistics():
    neutron, xray, masks = _two_material_volumes()
    rng = np.random.default_rng(1)
    noisy_n = neutron + rng.normal(0, 20.0, neutron.shape)
    full, full_per_class, _ = compute_class_metrics(noisy_n, xray, masks)
    sampled, sampled_per_class, _ = compute_class_metrics(
        noisy_n, xray, masks, max_samples=50
    )
    assert sampled_per_class["centroid_n_k"]["A"] == pytest.approx(
        full_per_class["centroid_n_k"]["A"], abs=15.0
    )
    # voxels_k always reports the true size, not the sample size
    assert sampled_per_class["voxels_k"]["A"] == full_per_class["voxels_k"]["A"]


def test_ground_truth_metrics_are_not_offered():
    """CE / eps_k / O_ab need a phantom and must not be silently invented."""
    for name in ("CE", "eps_k", "O_ab"):
        assert name not in METRIC_INFO
        assert name not in SCALAR_METRICS
        assert name not in PER_CLASS_METRICS


# ── CSV output ───────────────────────────────────────────────────────────────

def _rows_for_two_timepoints():
    neutron, xray, masks = _two_material_volumes()
    global_row = MetricsRow(scope="global")
    global_row.scalars.update(compute_shape_metrics(_histogram(neutron, xray)))
    scalars, per_class, reference = compute_class_metrics(neutron, xray, masks)
    global_row.scalars.update(scalars)
    global_row.per_class = per_class

    rows = [global_row]
    for timepoint, shift in enumerate((0.0, 30.0)):
        n_t, x_t, m_t = _two_material_volumes(shift=shift)
        row = MetricsRow(scope="timepoint", timepoint=timepoint)
        row.scalars.update(compute_shape_metrics(_histogram(n_t, x_t)))
        scalars, per_class, _ = compute_class_metrics(
            n_t, x_t, m_t, reference_centroids=reference
        )
        row.scalars.update(scalars)
        row.per_class = per_class
        rows.append(row)
    return rows


def test_csv_holds_every_scalar_and_per_class_value(tmp_path):
    rows = _rows_for_two_timepoints()
    path = write_metrics_csv(rows, tmp_path / "metrics.csv")

    with open(path, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    assert records
    scopes = {record["scope"] for record in records}
    assert scopes == {"global", "timepoint"}

    written = {record["metric"] for record in records}
    assert {"S_h", "S_v", "S_d", "A_x", "DB", "n_classes"} <= written
    assert {"voxels_k", "centroid_n_k", "sigma_n_k", "E_k", "drift_k"} <= written

    # Every row is self-describing
    for record in records:
        assert record["label"] and record["unit"] and record["meaning"]
        assert record["metric"] in METRIC_INFO

    # Per-class rows name their class, scalar rows do not
    for record in records:
        if record["metric"] in PER_CLASS_METRICS:
            assert record["class"] in {"A", "B"}
        else:
            assert record["class"] == ""


def test_csv_round_trips_the_numbers(tmp_path):
    rows = _rows_for_two_timepoints()
    path = write_metrics_csv(rows, tmp_path / "metrics.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    voxels = {
        record["class"]: float(record["value"])
        for record in records
        if record["metric"] == "voxels_k" and record["timepoint"] == "0"
    }
    _, _, masks = _two_material_volumes()
    assert voxels["A"] == float(masks["A"].sum())
    assert voxels["B"] == float(masks["B"].sum())

    drift = next(
        record for record in records
        if record["metric"] == "drift_k"
        and record["timepoint"] == "1" and record["class"] == "A"
    )
    assert float(drift["value"]) == pytest.approx(30.0)


def test_missing_values_are_written_as_blanks(tmp_path):
    row = MetricsRow(scope="global", scalars={"Delta_n": None, "DB": None})
    path = write_metrics_csv([row], tmp_path / "metrics.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert {record["value"] for record in records} == {""}


# ── evolution plot ───────────────────────────────────────────────────────────

def test_evolution_plot_is_written(tmp_path):
    rows = _rows_for_two_timepoints()
    saved = plot_metric_evolution(rows, tmp_path / "evolution.png")
    assert saved is not None
    assert os.path.getsize(saved) > 5000


def test_evolution_plot_needs_at_least_two_timepoints(tmp_path):
    rows = _rows_for_two_timepoints()[:2]          # global + one timepoint
    assert plot_metric_evolution(rows, tmp_path / "evolution.png") is None
    assert not (tmp_path / "evolution.png").exists()


def test_evolution_plot_survives_missing_values(tmp_path):
    rows = [
        MetricsRow(scope="timepoint", timepoint=0,
                   scalars={"S_h": 1.0, "DB": None}),
        MetricsRow(scope="timepoint", timepoint=1,
                   scalars={"S_h": None, "DB": 2.0}),
        MetricsRow(scope="timepoint", timepoint=2,
                   scalars={"S_h": 3.0, "DB": 4.0}),
    ]
    assert plot_metric_evolution(rows, tmp_path / "evolution.png") is not None


def test_evolution_plot_skipped_when_nothing_is_plottable(tmp_path):
    rows = [
        MetricsRow(scope="timepoint", timepoint=0, scalars={"n_classes": 2.0}),
        MetricsRow(scope="timepoint", timepoint=1, scalars={"n_classes": 2.0}),
    ]
    assert plot_metric_evolution(rows, tmp_path / "evolution.png") is None


# ── segmentation report ──────────────────────────────────────────────────────

def test_report_lists_names_values_and_per_timepoint_counts():
    text = build_segmentation_report(
        class_names=["Lithium", "Electrolyte"],
        label_values={"Lithium": 1, "Electrolyte": 2},
        voxels_per_timepoint={
            0: {"Lithium": 144, "Electrolyte": 96},
            1: {"Lithium": 150, "Electrolyte": 90},
        },
        volume_shape=(4, 16, 16),
        dataset_info={"neutron_file": "/data/neutron.tif"},
        roi_info={"Lithium": "polygon ROI, class id 1"},
        settings={"histogram bins": "32"},
        notes=["Counts are full resolution."],
    )

    assert "Lithium" in text and "Electrolyte" in text
    assert "class name" in text and "value" in text
    # Class legend: value, name and the all-timepoint total
    assert "294" in text        # 144 + 150
    assert "186" in text        # 96 + 90
    # Per-timepoint table
    assert "Voxels per class and timepoint" in text
    assert "144" in text and "90" in text
    # Contextual sections
    assert "/data/neutron.tif" in text
    assert "polygon ROI, class id 1" in text
    assert "histogram bins" in text
    assert "Counts are full resolution." in text
    assert "(4, 16, 16)" in text


def test_report_volume_fractions_match_the_counts():
    text = build_segmentation_report(
        class_names=["Lithium"],
        label_values={"Lithium": 1},
        voxels_per_timepoint={0: {"Lithium": 512}},
        volume_shape=(4, 16, 16),
    )
    assert "Volume fraction per class and timepoint" in text
    assert "50.000" in text       # 512 / 1024


def test_report_handles_a_class_absent_from_a_timepoint():
    text = build_segmentation_report(
        class_names=["Lithium", "Electrolyte"],
        label_values={"Lithium": 1, "Electrolyte": 2},
        voxels_per_timepoint={0: {"Lithium": 10}, 1: {"Electrolyte": 20}},
        volume_shape=(2, 4, 4),
    )
    assert "Lithium" in text and "Electrolyte" in text
    lines = [line for line in text.splitlines() if line.strip().startswith("0 ")]
    assert lines and "0" in lines[0]     # missing class reported as zero


def test_report_without_segmented_timepoints():
    text = build_segmentation_report(
        class_names=[],
        label_values={},
        voxels_per_timepoint={},
        volume_shape=(2, 4, 4),
    )
    assert "(no segmented timepoints)" in text


def test_report_file_is_written(tmp_path):
    from utils.segmentation_report import write_segmentation_report

    path = write_segmentation_report(
        tmp_path / "segmentation_report.txt",
        class_names=["Lithium"],
        label_values={"Lithium": 1},
        voxels_per_timepoint={0: {"Lithium": 10}},
        volume_shape=(2, 4, 4),
    )
    assert (tmp_path / "segmentation_report.txt").exists()
    assert "Lithium" in open(path, encoding="utf-8").read()
