"""Spatial metrics, block cross-validation, and the composable feature spec."""

import csv

import numpy as np
import pytest

from segmentation.features import (
    LEGACY_SPECS,
    PRESETS,
    FeatureSpec,
    extract_features_at_indices,
    gradient_coherence,
    local_moments,
    resolve_spec,
    structure_descriptors,
)
from utils.histogram_metrics import write_metrics_csv
from utils.metrics_spatial import (
    class_spatial_metrics,
    combined_registry,
    comparison_rows,
    disagreement_topology,
    interface_area,
    spatial_metrics_rows,
    surface_area,
)
from utils.validation import (
    anchoring_index,
    block_cross_validation,
    block_ids_for_volume,
    bootstrap_bands,
    cohen_kappa,
    confusion_matrix,
    difference_within_band,
    per_class_iou,
    permutation_anchoring_index,
    staleness_half_life,
    temporal_generalisation_matrix,
)


# ── spatial descriptors ──────────────────────────────────────────────────────

def _cube(shape=(10, 10, 10), start=(2, 2, 2), size=4):
    mask = np.zeros(shape, dtype=bool)
    stop = tuple(a + size for a in start)
    mask[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = True
    return mask


def test_centre_of_mass_and_gyration_of_a_cube():
    mask = _cube()
    metrics = class_spatial_metrics(mask)
    for axis in "zyx":
        assert metrics[f"com_{axis}_k"] == pytest.approx(3.5)
    assert metrics["n_components_k"] == 1
    assert metrics["largest_frac_k"] == pytest.approx(1.0)
    assert metrics["rg_k"] > 0


def test_surface_area_of_a_cube_is_its_faces():
    mask = _cube(size=4)
    # A 4x4x4 cube fully inside the volume: 6 faces of 16 voxels
    assert surface_area(mask) == 6 * 16


def test_speckle_shows_up_as_components_and_roughness():
    solid = _cube(shape=(12, 12, 12), size=6)
    rng = np.random.default_rng(0)
    speckled = rng.random((12, 12, 12)) < (solid.sum() / 12 ** 3)

    solid_metrics = class_spatial_metrics(solid)
    speckled_metrics = class_spatial_metrics(speckled)

    assert speckled_metrics["n_components_k"] > 20 * solid_metrics["n_components_k"]
    assert speckled_metrics["sa_vol_k"] > 3 * solid_metrics["sa_vol_k"]
    assert speckled_metrics["largest_frac_k"] < 0.5
    # ...and is invisible to a count: both classes have a similar volume
    assert abs(speckled.sum() - solid.sum()) < 0.25 * solid.sum()


def test_interface_area_counts_the_shared_faces():
    shape = (4, 6, 6)
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    left[:, :, :3] = True
    right[:, :, 3:] = True
    # One shared plane of 4 x 6 voxel faces
    assert interface_area(left, right) == 4 * 6
    assert interface_area(left, left) == 0     # a class does not touch itself


def test_com_drift_measures_movement_across_time():
    masks = {
        timepoint: {"Blob": _cube(start=(2, 2, 2 + timepoint))}
        for timepoint in range(4)
    }
    rows = spatial_metrics_rows(masks, interfaces=False)
    drifts = [row.per_class["com_drift_k"]["Blob"] for row in rows]
    assert drifts[0] == pytest.approx(0.0)
    assert drifts == pytest.approx([0.0, 1.0, 2.0, 3.0])


# ── rind versus blob ─────────────────────────────────────────────────────────

def test_a_boundary_shell_is_recognised_as_a_rind():
    from scipy.ndimage import binary_dilation

    core = _cube(shape=(16, 16, 16), start=(4, 4, 4), size=8)
    grown = binary_dilation(core)          # differs by a one-voxel shell

    topology = disagreement_topology(core, grown)
    assert topology["disagreement_voxels"] > 0
    assert topology["f_rind"] > 0.99
    assert topology["n_interior_components"] == 0


def test_a_displaced_blob_is_recognised_as_a_real_disagreement():
    first = _cube(shape=(40, 40, 40), start=(2, 2, 2), size=12)
    second = _cube(shape=(40, 40, 40), start=(20, 20, 20), size=12)

    topology = disagreement_topology(first, second)
    assert topology["f_rind"] < 0.8
    assert topology["n_interior_components"] >= 2


def test_f_rind_is_scale_dependent_but_the_component_count_is_not():
    """A small displaced object erodes away like a shell does.

    Both signals have to be read together: f_rind alone would call this a
    boundary effect, which it is not.
    """
    first = _cube(shape=(20, 20, 20), start=(2, 2, 2), size=6)
    second = _cube(shape=(20, 20, 20), start=(12, 12, 12), size=6)

    topology = disagreement_topology(first, second)
    assert topology["f_rind"] > 0.9                     # looks like a rind
    assert topology["n_interior_components"] >= 2       # but is not one


def test_identical_masks_disagree_about_nothing():
    mask = _cube()
    topology = disagreement_topology(mask, mask)
    assert topology["disagreement_voxels"] == 0
    assert topology["f_rind"] == 1.0
    assert topology["n_interior_components"] == 0


def test_comparison_rows_aggregate_over_classes():
    from scipy.ndimage import binary_dilation

    core = _cube(shape=(16, 16, 16), size=8)
    masks_a = {0: {"Blob": core}}
    masks_b = {0: {"Blob": binary_dilation(core)}}

    rows = comparison_rows(masks_a, masks_b)
    assert len(rows) == 1
    assert rows[0].scalars["f_rind"] > 0.99
    assert rows[0].per_class["disagreement_voxels"]["Blob"] > 0


# ── CSV interoperability ─────────────────────────────────────────────────────

def test_spatial_rows_write_through_the_shared_csv_schema(tmp_path):
    masks = {
        timepoint: {"Blob": _cube(start=(2, 2, 2 + timepoint)),
                    "Other": _cube(start=(6, 6, 6))}
        for timepoint in range(3)
    }
    rows = spatial_metrics_rows(masks)
    info, scalars, per_class = combined_registry()
    path = write_metrics_csv(
        rows, tmp_path / "spatial.csv", metric_info=info,
        scalar_metrics=scalars, per_class_metrics=per_class,
    )

    with open(path, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    metrics = {record["metric"] for record in records}
    assert {"com_z_k", "com_drift_k", "rg_k", "n_components_k"} <= metrics
    # Same nine columns as the histogram metrics, so the files concatenate
    assert set(records[0]) == {
        "scope", "timepoint", "metric", "class", "value",
        "label", "unit", "meaning", "better_when",
    }
    for record in records:
        assert record["label"] and record["unit"] and record["meaning"]


def test_combined_registry_keeps_both_metric_families():
    info, scalars, per_class = combined_registry()
    assert "S_h" in info and "com_drift_k" in info      # histogram + spatial
    assert "DB" in scalars and "f_rind" in scalars
    assert "voxels_k" in per_class and "rg_k" in per_class


# ── block cross-validation ───────────────────────────────────────────────────

def test_blocks_are_contiguous_and_cover_the_volume():
    blocks = block_ids_for_volume((10, 10, 10), grid=(2, 2, 2))
    assert blocks.shape == (10, 10, 10)
    assert len(np.unique(blocks)) == 8
    # Each block is a solid box, not scattered voxels
    for block in np.unique(blocks):
        coordinates = np.array(np.nonzero(blocks == block))
        extent = coordinates.max(axis=1) - coordinates.min(axis=1) + 1
        assert int(np.prod(extent)) == int((blocks == block).sum())


def test_grid_larger_than_the_volume_is_clamped():
    blocks = block_ids_for_volume((2, 3, 4), grid=(5, 5, 5))
    assert blocks.shape == (2, 3, 4)
    assert len(np.unique(blocks)) == 2 * 3 * 4


def test_confusion_iou_and_kappa():
    truth = np.array([0, 0, 1, 1, 2, 2])
    perfect = truth.copy()
    matrix = confusion_matrix(truth, perfect, 3)
    assert np.array_equal(matrix, np.diag([2, 2, 2]))
    assert cohen_kappa(matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(per_class_iou(matrix), [1.0, 1.0, 1.0])

    # Chance-level agreement scores near zero, unlike accuracy
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 3000)
    guesses = rng.integers(0, 3, 3000)
    assert abs(cohen_kappa(confusion_matrix(labels, guesses, 3))) < 0.05


def test_block_cv_is_harder_than_scoring_on_the_training_voxels():
    """Neighbouring voxels are near-duplicates, so in-sample looks perfect."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    shape = (10, 10, 10)
    blocks = block_ids_for_volume(shape, grid=(2, 2, 2))
    # A label that depends on position, with a feature that is mostly noise
    labels = (blocks % 2).reshape(-1)
    coordinates = np.array(
        np.meshgrid(*[np.arange(size) for size in shape], indexing="ij")
    ).reshape(3, -1).T.astype(np.float32)
    features = coordinates + rng.normal(0, 0.3, coordinates.shape)

    def factory():
        return RandomForestClassifier(n_estimators=20, random_state=0)

    result = block_cross_validation(
        features, labels, blocks.reshape(-1), factory
    )
    in_sample = factory().fit(features, labels)
    in_sample_accuracy = float(np.mean(in_sample.predict(features) == labels))

    assert in_sample_accuracy > 0.95
    assert result.accuracy < in_sample_accuracy
    assert result.n_folds == 8
    assert "block CV" in result.describe()


def test_block_cv_needs_more_than_one_block():
    features = np.zeros((10, 2))
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        block_cross_validation(
            features, labels, np.zeros(10, dtype=int), lambda: None
        )


# ── anchoring ────────────────────────────────────────────────────────────────

def test_anchoring_index_is_the_share_carried_by_frozen_features():
    names = ["neutron", "xray", "coord_z", "coord_y", "coord_x"]
    importances = [0.3, 0.3, 0.2, 0.1, 0.1]
    assert anchoring_index(importances, names, ["coord_z", "coord_y", "coord_x"]) \
        == pytest.approx(0.4)
    assert anchoring_index(importances, names, []) == 0.0


def test_permutation_anchoring_index_uses_held_out_data():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    # The label depends only on the coordinate, never on the intensity
    coordinate = rng.random(600)
    intensity = rng.random(600)
    features = np.column_stack([intensity, coordinate])
    labels = (coordinate > 0.5).astype(int)

    model = RandomForestClassifier(n_estimators=30, random_state=0)
    model.fit(features, labels)
    index = permutation_anchoring_index(
        model, features, labels, ["neutron", "coord_z"], ["coord_z"],
        n_repeats=3,
    )
    assert index > 0.9


def test_a_time_invariant_spec_has_nothing_to_anchor_on():
    assert PRESETS["texture"].anchored_features() == []
    assert PRESETS["texture"].is_time_invariant
    assert PRESETS["anchored"].anchored_features() == [
        "coord_z", "coord_y", "coord_x"
    ]
    assert not PRESETS["anchored"].is_time_invariant


# ── temporal generalisation ──────────────────────────────────────────────────

def test_generalisation_matrix_shows_decay_with_distance():
    def fit_at(train):
        return train

    def score_at(model, predict):
        return max(0.0, 1.0 - 0.2 * abs(predict - model))

    matrix, trains, predicts = temporal_generalisation_matrix(
        [0, 4], [0, 2, 4, 6], fit_at, score_at
    )
    assert matrix.shape == (2, 4)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 3] < matrix[0, 0]

    half_lives = staleness_half_life(matrix, trains, [0, 2, 4, 6])
    assert half_lives[0] == pytest.approx(4.0)


# ── bootstrap ────────────────────────────────────────────────────────────────

def test_bootstrap_bands_bracket_the_truth():
    rng = np.random.default_rng(0)

    def estimate(seed):
        local = np.random.default_rng(seed)
        return {"Lithium": 1000.0 + local.normal(0, 50)}

    bands = bootstrap_bands(estimate, n_resamples=40)
    band = bands["Lithium"]
    assert band["low"] < 1000.0 < band["high"]
    assert band["n"] == 40
    # A 3% difference is inside a band this wide, so it is not evidence
    assert difference_within_band(30.0, band)
    assert not difference_within_band(5000.0, band)


# ── feature spec ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", ["basic", "advanced", "expert"])
def test_legacy_levels_reproduce_the_old_columns_exactly(level):
    """Existing models and notebooks must keep working bit-for-bit."""
    from segmentation.random_forest_4d import _extract_features_at_indices

    rng = np.random.default_rng(0)
    neutron = rng.normal(500, 80, (5, 9, 11)).astype(np.float32)
    xray = rng.normal(400, 60, (5, 9, 11)).astype(np.float32)
    indices = rng.choice(neutron.size, 200, replace=False)

    through_rf = _extract_features_at_indices(neutron, xray, indices, level)
    direct = extract_features_at_indices(neutron, xray, indices, level)
    np.testing.assert_array_equal(through_rf, direct)
    assert through_rf.shape[1] == LEGACY_SPECS[level].n_features


def test_texture_is_reachable_without_frozen_geometry():
    """The coupling that made the original experiment uninterpretable."""
    spec = FeatureSpec(texture_scales=(1, 2))
    assert spec.geometry == "none"
    assert spec.anchored_features() == []
    assert any("std" in name for name in spec.feature_names())
    # ...whereas the legacy ladder forced them together
    assert LEGACY_SPECS["expert"].geometry == "absolute"
    assert LEGACY_SPECS["expert"].texture_scales == (1,)


def test_feature_names_match_the_extracted_columns():
    rng = np.random.default_rng(1)
    neutron = rng.normal(500, 80, (6, 8, 8)).astype(np.float32)
    xray = rng.normal(400, 60, (6, 8, 8)).astype(np.float32)
    indices = np.arange(64, dtype=np.int64)

    for spec in list(PRESETS.values()) + [
        FeatureSpec(texture_scales=(1, 2, 4), gradient=True, structure=True,
                    laplacian=True, geometry="relative"),
    ]:
        matrix = extract_features_at_indices(neutron, xray, indices, spec)
        assert matrix.shape == (64, spec.n_features), spec.describe()
        assert len(spec.feature_names()) == len(set(spec.feature_names()))
        assert np.all(np.isfinite(matrix))


def test_relative_geometry_moves_with_the_sample():
    """An absolute coordinate pins the model to T0; a relative one does not."""
    shape = (6, 12, 12)
    neutron = np.zeros(shape, dtype=np.float32)
    xray = np.zeros(shape, dtype=np.float32)
    sample = np.zeros(shape, dtype=bool)
    sample[:, 2:6, 2:6] = True
    neutron[sample] = 800.0

    shifted_sample = np.zeros(shape, dtype=bool)
    shifted_sample[:, 6:10, 6:10] = True
    shifted_neutron = np.zeros(shape, dtype=np.float32)
    shifted_neutron[shifted_sample] = 800.0

    spec = FeatureSpec(intensity=False, projections=False, geometry="relative")
    first = extract_features_at_indices(
        neutron, xray, np.flatnonzero(sample.reshape(-1)), spec,
        sample_mask=sample,
    )
    second = extract_features_at_indices(
        shifted_neutron, xray, np.flatnonzero(shifted_sample.reshape(-1)), spec,
        sample_mask=shifted_sample,
    )
    # The same voxel of the same object gets the same description
    np.testing.assert_allclose(first, second, atol=1e-6)

    absolute = FeatureSpec(intensity=False, projections=False, geometry="absolute")
    first_absolute = extract_features_at_indices(
        neutron, xray, np.flatnonzero(sample.reshape(-1)), absolute
    )
    second_absolute = extract_features_at_indices(
        shifted_neutron, xray, np.flatnonzero(shifted_sample.reshape(-1)), absolute
    )
    assert not np.allclose(first_absolute, second_absolute)


def test_local_moments_match_a_direct_neighbourhood_computation():
    rng = np.random.default_rng(2)
    volume = rng.normal(100, 20, (7, 7, 7)).astype(np.float32)
    mean, deviation = local_moments(volume, radius=1)

    window = volume[2:5, 2:5, 2:5]
    assert mean[3, 3, 3] == pytest.approx(window.mean(), rel=1e-5)
    assert deviation[3, 3, 3] == pytest.approx(window.std(), rel=1e-4)


def test_gradient_coherence_separates_shared_edges_from_single_ones():
    shape = (5, 12, 12)
    step = np.zeros(shape, dtype=np.float32)
    step[:, :, 6:] = 100.0

    shared = gradient_coherence(step, step.copy())
    edge = np.zeros(shape, dtype=bool)
    edge[:, :, 5:7] = True
    assert shared[edge].mean() > 0.95

    # An edge only the neutron channel sees — an artifact, not an interface
    flat = np.zeros(shape, dtype=np.float32)
    one_sided = gradient_coherence(step, flat)
    assert one_sided[edge].mean() < 0.05


def test_structure_descriptors_distinguish_a_plane_from_a_blob():
    shape = (12, 12, 12)
    plane = np.zeros(shape, dtype=np.float32)
    plane[:, :, 6:] = 100.0
    anisotropy, planarity = structure_descriptors(plane, radius=1)
    edge = np.zeros(shape, dtype=bool)
    edge[2:10, 2:10, 5:7] = True
    assert anisotropy[edge].mean() > 0.8

    uniform = np.full(shape, 50.0, dtype=np.float32)
    flat_anisotropy, _ = structure_descriptors(uniform, radius=1)
    assert flat_anisotropy.max() < 1e-3


def test_unknown_spec_names_are_rejected_clearly():
    with pytest.raises(ValueError, match="Unknown feature spec"):
        resolve_spec("wizard")
    with pytest.raises(TypeError):
        resolve_spec(42)
    with pytest.raises(ValueError, match="geometry"):
        FeatureSpec(geometry="global")
