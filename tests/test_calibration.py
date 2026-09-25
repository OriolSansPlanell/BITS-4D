"""Review item 8: classes placed from attenuation coefficients.

Two drawn materials with known coefficients calibrate each modality's grey
scale; a material with no region to draw is then placed from its
coefficients, merged with the drawn classes and tracked.
"""

import os
import warnings

import numpy as np
import pytest

from model import run_health_check
from model.calibration import (
    ReferenceMaterial, fit_calibration, merge_libraries, predicted_classes,
    reference_from_region,
)
from model.likelihood import ClassLibrary
from validation import methods
from validation.phantom import make_phantom
from validation.scoring import score_series, summarise


def test_two_references_fix_gain_and_offset():
    references = [
        ReferenceMaterial("A", (0.0, 1.0), (100.0, 250.0), (10.0, 20.0)),
        ReferenceMaterial("B", (2.0, 3.0), (500.0, 650.0), (30.0, 20.0)),
    ]
    calibration = fit_calibration(references)
    np.testing.assert_allclose(calibration.gain, (200.0, 200.0))
    np.testing.assert_allclose(calibration.offset, (100.0, 50.0))
    np.testing.assert_allclose(calibration.predict((1.0, 2.0)), (300.0, 450.0))
    # The spread is the references' (an instrument property)
    np.testing.assert_allclose(calibration.spread, (np.sqrt(500.0), 20.0))
    assert "A, B" in calibration.describe()


def test_three_references_report_their_misfit():
    references = [
        ReferenceMaterial("A", (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
        ReferenceMaterial("B", (1.0, 1.0), (10.0, 10.0), (1.0, 1.0)),
        ReferenceMaterial("C", (2.0, 2.0), (22.0, 20.0), (1.0, 1.0)),
    ]
    calibration = fit_calibration(references)
    assert calibration.residual[0] > 0.4 and calibration.residual[1] < 1e-9
    assert "misfit" in calibration.describe()


def test_calibration_needs_two_distinct_references():
    one = [ReferenceMaterial("A", (1.0, 1.0), (1.0, 1.0), (1.0, 1.0))]
    with pytest.raises(ValueError):
        fit_calibration(one)
    same = one + [ReferenceMaterial("B", (1.0, 2.0), (2.0, 2.0), (1.0, 1.0))]
    with pytest.raises(ValueError, match="neutron"):
        fit_calibration(same)


def test_drawn_class_wins_over_prediction():
    rng = np.random.default_rng(0)
    neutron, xray = rng.normal(size=(2, 4, 8, 8)) * 10 + 100
    mask = np.ones((4, 8, 8), bool)
    drawn = ClassLibrary.from_masks(neutron, xray, {"A": mask})
    calibration = fit_calibration([
        reference_from_region("A", (1.0, 1.0), neutron, xray, mask),
        ReferenceMaterial("B", (2.0, 2.0), (200.0, 200.0), (10.0, 10.0)),
    ])
    predicted = predicted_classes({"A": (5.0, 5.0), "C": (3.0, 3.0)}, calibration)
    merged = merge_libraries(drawn, predicted)
    assert merged.names == ["A", "C"]
    assert merged[0].source != "physics" and merged[1].source == "physics"
    np.testing.assert_allclose(merged[1].mu, (300.0, 300.0), rtol=0.05)


def test_a_phase_placed_from_coefficients_is_tracked():
    phantom = make_phantom()
    scores = {}
    for label, kwargs in (("none", {"birth": False}),
                          ("coefficients", {"predict": ("Product",)})):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels, details = methods.run_bits(phantom, return_details=True,
                                               **kwargs)
        scores[label] = summarise(score_series(labels, phantom.labels,
                                               phantom.classes))
    assert scores["none"]["Product"] < 0.2
    assert scores["coefficients"]["Product"] > 0.7
    # Predicted, so its absence before it forms is not a failure
    report = run_health_check(details["outcome"])
    assert not [f for f in report.problems() if "Product" in f.message]


def test_a_prediction_never_found_is_a_warning():
    phantom = make_phantom(shape=(8, 40, 40), timepoints=3)
    labels, details = methods.run_bits(phantom, smoothing=0.0,
                                       return_details=True)
    library = details["library"]
    outcome = details["outcome"]
    calibration = fit_calibration([
        ReferenceMaterial("Air", (0.0, 0.0), (400.0, 300.0), (60.0, 60.0)),
        ReferenceMaterial("Aluminium", (1.0, 1.0), (700.0, 1500.0), (60.0, 60.0)),
    ])
    ghost = predicted_classes({"Ghost": (9.0, 9.0)}, calibration)
    outcome.library = merge_libraries(library, ghost)
    findings = [f for f in run_health_check(outcome).findings
                if f.check == "predicted classes"]
    assert findings and findings[0].status.value == "warn"


def test_physics_dialog_and_tracking(monkeypatch):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication, QDialog
    from gui.physics_dialog import PhysicsMaterialsDialog

    app = QApplication.instance() or QApplication([])
    dialog = PhysicsMaterialsDialog(["Air", "Aluminium", "Lithium"])
    _r, _t, problems = dialog.values()
    assert problems                              # nothing chosen yet
    dialog.set_reference("Air", 0.3125, 0.2083)
    dialog.set_reference("Aluminium", 0.6875, 1.2083)
    dialog.add_target("Product", "1.5", "0.7917")
    dialog.add_target("Air", "1", "1")           # already drawn
    references, targets, problems = dialog.values()
    assert set(references) == {"Air", "Aluminium"}
    assert targets == {"Product": (1.5, 0.7917)}
    assert any("already drawn" in p for p in problems)
    dialog.target_table.setCurrentCell(1, 0)
    dialog._remove_target()
    assert not dialog.values()[2]
    dialog.deleteLater()
    app.processEvents()
