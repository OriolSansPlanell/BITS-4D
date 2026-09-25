"""Review item 5: one Gaussian per class cannot describe every cloud.

A class may now be a small Gaussian mixture, with the number of components
chosen by BIC from its own voxels. It helps where a single Gaussian's
envelope would cover a neighbouring class — a material in two states, a
cloud bent by an artefact — and changes nothing for a class that is one
blob.
"""

import numpy as np

from model import ClassLibrary, LockedSegmenter
from model.histogram_cache import build_histogram_cache
from model.likelihood import match_table
from validation import methods
from validation.phantom import make_phantom
from validation.scoring import score_series, summarise


def _two_lobes_with_a_class_between(seed=0):
    """Class A has lobes at (0,0) and (1000,1000); class B sits between."""
    rng = np.random.default_rng(seed)
    shape = (6, 30, 30)
    neutron = np.empty(shape)
    xray = np.empty(shape)
    a = np.zeros(shape, bool)
    a[:, :10] = True
    a[:, 20:] = True
    b = ~a
    lobe_low = np.zeros(shape, bool)
    lobe_low[:, :10] = True
    for mask, centre in ((lobe_low, (0.0, 0.0)), (a & ~lobe_low, (1000.0, 1000.0)),
                         (b, (500.0, 500.0))):
        neutron[mask] = rng.normal(centre[0], 40.0, mask.sum())
        xray[mask] = rng.normal(centre[1], 40.0, mask.sum())
    return neutron, xray, {"A": a, "B": b}


def test_components_are_chosen_by_bic():
    neutron, xray, masks = _two_lobes_with_a_class_between()
    library = ClassLibrary.from_masks(neutron, xray, masks, max_components=3)
    assert library[library.index_of("A")].n_components == 2
    assert library[library.index_of("B")].n_components == 1
    # The overall moments are still there for everything that needs a centre
    np.testing.assert_allclose(library[0].mu, [500, 500], atol=30)


def test_default_is_one_gaussian():
    neutron, xray, masks = _two_lobes_with_a_class_between()
    library = ClassLibrary.from_masks(neutron, xray, masks)
    assert all(material.components is None for material in library)


def test_mixture_class_stops_swallowing_its_neighbour():
    neutron, xray, masks = _two_lobes_with_a_class_between()
    edges = np.linspace(-300, 1300, 129)
    results = {}
    for components in (1, 3):
        library = ClassLibrary.from_masks(neutron, xray, masks,
                                          max_components=components)
        segmenter = LockedSegmenter(library, prior=None)
        segmenter.set_grid(edges, edges)
        labels = segmenter.segment_timepoint(neutron, xray).labels
        results[components] = np.mean(labels[masks["B"]]
                                      == library.index_of("B") + 1)
    # One broad Gaussian over both lobes of A covers B's position and takes
    # B's outer voxels; with A as two lobes, B keeps all of them
    assert results[1] < 0.98
    assert results[3] > 0.999


def test_mixture_scores_are_a_proper_density():
    """log Σ w N integrates to one over the plane (up to the grid)."""
    neutron, xray, masks = _two_lobes_with_a_class_between()
    library = ClassLibrary.from_masks(neutron, xray, {"A": masks["A"]},
                                      max_components=3)
    library[0].weight = 1.0
    grid = np.linspace(-400, 1400, 181)
    n, x = np.meshgrid(grid, grid)
    cache = build_histogram_cache(n[None], x[None], np.linspace(-405, 1405, 182),
                                  np.linspace(-405, 1405, 182))
    table = match_table(library, cache)
    cell = (grid[1] - grid[0]) ** 2
    assert abs(np.exp(table.scores[:, 0]).sum() * cell - 1.0) < 0.02


def test_two_state_material_on_the_phantom():
    phantom = make_phantom(two_state_lithium=(1500.0, 1300.0))
    scores = {}
    for components in (1, 3):
        labels = methods.run_bits(phantom, max_components=components)
        scores[components] = summarise(score_series(labels, phantom.labels,
                                                    phantom.classes))
    assert scores[3]["Product"] > scores[1]["Product"] + 0.02
    assert scores[3]["mean"] > scores[1]["mean"]
