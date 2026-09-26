"""Review item 7: the spatial refinement must scale past memory.

Refining in z-slabs with a halo of ``n_sweeps + 1`` slices gives exactly the
labels of a whole-volume run, so a volume too large for mean-field no longer
has to fall back to the greedier ICM solver.
"""

import numpy as np
import pytest

from model.spatial_prior import ROIDerivedMRF, UnaryScores, potts_cost


def _problem(seed=0, shape=(23, 14, 15), k=4):
    rng = np.random.default_rng(seed)
    truth = (np.arange(shape[2])[None, None, :] * k // shape[2]
             + np.zeros(shape, int))
    table = rng.normal(size=(60, k)).astype(np.float32)
    rows = rng.integers(0, 60, size=shape)
    # Evidence mostly agrees with the truth; some voxels unmeasured
    table_rows = np.where(rng.random(shape) < 0.7,
                          truth * 15 + rng.integers(0, 15, shape), rows)
    table[np.arange(60), np.arange(60) // 15] += 2.0
    table_rows[rng.random(shape) < 0.05] = -1
    neutron = truth * 100.0 + rng.normal(0, 30, shape)
    xray = truth * 50.0 + rng.normal(0, 30, shape)
    return UnaryScores(table, table_rows), neutron, xray


@pytest.mark.parametrize("method", ["mean_field", "icm"])
@pytest.mark.parametrize("thickness", [1, 4, 9])
def test_slabs_reproduce_the_whole_volume(method, thickness):
    scores, neutron, xray = _problem()
    mrf = ROIDerivedMRF(beta=1.5, n_sweeps=4)
    mrf.pairwise = potts_cost(scores.n_classes)
    whole, whole_info = mrf.refine(scores, neutron, xray,
                                   valid_mask=scores.valid_mask(), method=method)
    chunked, info = mrf.refine(scores, neutron, xray,
                               valid_mask=scores.valid_mask(), method=method,
                               slab_thickness=thickness)
    np.testing.assert_array_equal(chunked, whole)
    assert info.chunks == -(-scores.shape[0] // thickness)
    assert info.changed_fraction == pytest.approx(whole_info.changed_fraction)
    np.testing.assert_allclose(info.energy_trace, whole_info.energy_trace,
                               rtol=1e-5)


def test_dense_scores_can_be_chunked_too():
    scores, neutron, xray = _problem(seed=3)
    dense = scores.dense()
    mrf = ROIDerivedMRF(beta=2.0, n_sweeps=3)
    mrf.pairwise = potts_cost(scores.n_classes)
    whole, _ = mrf.refine(dense, neutron, xray, method="mean_field")
    chunked, _ = mrf.refine(dense, neutron, xray, method="mean_field",
                            slab_thickness=5)
    np.testing.assert_array_equal(chunked, whole)


def test_budget_picks_slabs_before_giving_up_mean_field():
    mrf = ROIDerivedMRF(n_sweeps=5)
    # 1000³ voxels × 9 classes does not fit 2 GiB whole; slabs do
    shape = (1000, 300, 300)
    method, thickness = mrf.plan_slabs(shape, 9)
    assert method == "mean_field" and thickness is not None
    slab_bytes = ((thickness + 2 * mrf.halo) * 9e4
                  * mrf.mean_field_bytes_per_voxel(9) + 4 * 9e7)
    assert slab_bytes <= 2 * 1024 ** 3
    # A small volume is refined in one piece
    assert mrf.plan_slabs((50, 100, 100), 9) == ("mean_field", None)


def test_auto_reports_its_slabs():
    scores, neutron, xray = _problem()
    per_slice = 14 * 15
    budget_for_six_slices = (
        ROIDerivedMRF.mean_field_bytes_per_voxel(4) * per_slice * (6 + 2 * 5)
        + 4 * 23 * per_slice) / 1024 ** 3
    mrf = ROIDerivedMRF(beta=1.0, n_sweeps=4,
                        memory_budget_gb=budget_for_six_slices)
    mrf.pairwise = potts_cost(scores.n_classes)
    labels, info = mrf.refine(scores, neutron, xray,
                              valid_mask=scores.valid_mask())
    assert info.method == "mean_field" and info.chunks == 4
    assert "slabs" in info.describe()
    reference = ROIDerivedMRF(beta=1.0, n_sweeps=4)
    reference.pairwise = mrf.pairwise
    whole, _ = reference.refine(scores, neutron, xray,
                                valid_mask=scores.valid_mask())
    np.testing.assert_array_equal(labels, whole)
