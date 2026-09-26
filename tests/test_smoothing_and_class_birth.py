"""Review items 3 and 4.

3. The automatic smoothing strength had only a lower guard (no class may
   lose volume), so on data where every grid value passes it returned the
   top of the grid. It now also protects thin structures, stops once more
   smoothing changes nothing, checks several timepoints, and reports when
   it ends at the top of the range unsettled.
4. A class whose region was empty at the reference timepoint (a thin
   product layer eroded away at T0) was dropped silently. It is now
   reported, and can be defined at the timepoint where it exists.
"""

import warnings

import numpy as np
import pytest

from model import ClassDefinitionWarning, ClassLibrary, run_health_check
from model.locked import (
    LockedSegmenter, smoothing_check_timepoints, thin_structure_mask,
)
from validation import methods
from validation.phantom import make_phantom
from validation.scoring import dice


@pytest.fixture(scope="module")
def phantom():
    return make_phantom(shape=(10, 56, 56), timepoints=5)


# ── 4: phases that appear ────────────────────────────────────────────────────

def test_empty_region_is_reported_not_silently_dropped(phantom):
    masks = phantom.roi_masks(0)
    assert not masks["Product"].any()          # the eroded T0 layer
    with pytest.warns(ClassDefinitionWarning, match="Product"):
        library = ClassLibrary.from_masks(phantom.neutron[0], phantom.xray[0],
                                          masks, timepoint=0)
    assert "Product" not in library.names
    assert [name for name, _reason in library.dropped] == ["Product"]
    assert "timepoint 0" in library.dropped[0][1]


def test_dropped_class_fails_the_health_check(phantom):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels, details = methods.run_bits(phantom, smoothing=0.0, birth=False,
                                           return_details=True)
    report = run_health_check(details["outcome"])
    problems = [f for f in report.problems() if f.check == "classes defined"]
    assert problems and "Product" in problems[0].message


def test_a_phase_that_appears_is_defined_where_it_exists(phantom):
    labels, details = methods.run_bits(phantom, smoothing=0.0,
                                       return_details=True)
    library = details["library"]
    product = library[library.index_of("Product")]
    assert product.defined_at > 0 and not library.dropped
    k = phantom.classes.index("Product") + 1
    # Found once it is thick enough to see, and (nearly) absent before
    assert dice(labels[-1] == k, phantom.mask("Product", phantom.num_timepoints - 1)) > 0.9
    assert np.mean(labels[0] == k) < 0.005
    # Its absence before it was defined is expected, not a failure
    report = run_health_check(details["outcome"])
    assert not [f for f in report.problems() if "Product" in f.message]
    assert [f for f in report.findings if f.check == "phases that appear"]


def test_classes_defined_from_few_voxels_are_noted(phantom):
    masks = {"Air": phantom.roi_masks(0)["Air"],
             "Aluminium": np.zeros(phantom.labels.shape[1:], bool)}
    masks["Aluminium"][5, 5, 5] = True
    library = ClassLibrary.from_masks(phantom.neutron[0], phantom.xray[0], masks)
    assert [name for name, _note in library.notes] == ["Aluminium"]


# ── 3: choosing the smoothing strength ───────────────────────────────────────

def test_rule_stops_once_the_labels_settle():
    grid = [0.0, 0.5, 1.0, 2.0, 4.0]
    ok = [True] * 5
    changes = [0.0, 0.01, 0.004, 0.001, 0.0005]
    strength, reason, converged = LockedSegmenter._choose_smoothing(
        grid, ok, changes, 0.002)
    assert strength == 1.0 and converged and "settled" in reason


def test_rule_skips_weak_failures_and_stops_at_the_next_failure():
    grid = [0.0, 0.5, 1.0, 2.0, 4.0]
    ok = [False, True, True, False, True]
    changes = [0.0, 0.01, 0.01, 0.01, 0.01]
    strength, reason, converged = LockedSegmenter._choose_smoothing(
        grid, ok, changes, 0.002)
    assert strength == 1.0 and not converged and "guard failed at 2" in reason


def test_rule_reports_the_ceiling():
    grid = [0.0, 1.0, 2.0]
    strength, reason, converged = LockedSegmenter._choose_smoothing(
        grid, [True] * 3, [0.0, 0.01, 0.01], 0.002)
    assert strength == 2.0 and not converged and "top" in reason


def test_thin_structures_are_sheets_not_speckle():
    mask = np.zeros((8, 20, 20), bool)
    mask[:, 5, 2:18] = True                   # a 1-voxel sheet: 128 voxels
    mask[1, 15, 15] = True                    # an isolated noise voxel
    mask[2:7, 10:16, 2:8] = True              # a thick block
    thin = thin_structure_mask(mask)
    assert thin[:, 5, 2:18].all()
    assert not thin[1, 15, 15] and not thin[2:7, 10:16, 2:8].any()


def test_thin_structure_guard_limits_smoothing_on_a_thin_layer():
    """At higher noise, smoothing erodes a 1-voxel layer long before any
    class loses volume; the guard stops the search where that starts."""
    noisy = make_phantom(noise=(90.0, 90.0))
    _labels, details = methods.run_bits(noisy, smoothing=0.0,
                                        return_details=True)
    segmenter = details["segmenter"]
    strength, sweep = segmenter.auto_smoothing(
        noisy.neutron[0], noisy.xray[0],
        extra_volumes=[(noisy.neutron[1], noisy.xray[1], 1)],
    )
    at_t1 = [row for row in sweep if row["timepoint"] == 1]
    eroded = [row["smoothing"] for row in at_t1
              if row["worst_thin_retention"] < 0.8]
    assert eroded, "the layer should erode at some strength"
    assert strength < min(eroded)
    # And that is where accuracy on the layer is best
    k = segmenter.library.names.index("Product") + 1
    truth = noisy.mask("Product", 1)
    chosen = segmenter.segment_timepoint(noisy.neutron[1], noisy.xray[1],
                                         beta=strength)
    strongest = segmenter.segment_timepoint(noisy.neutron[1], noisy.xray[1],
                                            beta=8.0)
    assert dice(chosen.labels == k, truth) > dice(strongest.labels == k, truth)


def test_report_is_attached_and_checked(phantom):
    _labels, details = methods.run_bits(phantom, return_details=True)
    report = details["outcome"].smoothing_report
    assert report["strength"] == details["smoothing"]
    assert set(report) >= {"converged", "at_ceiling", "sensitivity",
                           "timepoints", "reason"}
    assert len(report["timepoints"]) > 1
    findings = [f for f in run_health_check(details["outcome"]).findings
                if f.check == "smoothing strength"]
    assert findings


def test_ceiling_is_a_warning_in_the_health_check(phantom):
    _labels, details = methods.run_bits(phantom, smoothing=0.0,
                                        return_details=True)
    outcome = details["outcome"]
    outcome.smoothing_report = {"strength": 8.0, "at_ceiling": True,
                                "converged": False, "reason": "top",
                                "timepoints": [0]}
    findings = [f for f in run_health_check(outcome).findings
                if f.check == "smoothing strength"]
    assert findings[0].status.value == "warn"


def test_checked_timepoints():
    assert smoothing_check_timepoints(10, 0, [0, 4]) == [0, 4, 5, 9]
    assert smoothing_check_timepoints(1, 0) == [0]
    assert len(smoothing_check_timepoints(50, 0, range(20))) == 5
