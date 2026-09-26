"""Review item 6: the modalities must be co-registered.

Measures the neutron/X-ray offset (maximum mutual information over
whole-voxel shifts, refined to sub-voxel), corrects it, and shows what an offset costs.
"""

import os

import numpy as np
import pytest

from model.registration import (
    apply_shift, estimate_alignment, mutual_information, resample_to_shape,
)
from validation.phantom import make_phantom


@pytest.mark.parametrize("offset", [(0, 1, 0), (0, -2, 1.5), (0, 3, -3)])
def test_in_plane_offsets_are_measured(offset):
    phantom = make_phantom(misregistration=offset, timepoints=1)
    alignment = estimate_alignment(phantom.neutron[0], phantom.xray[0])
    # The shift that undoes the offset
    np.testing.assert_allclose(alignment.shift[1:], [-offset[1], -offset[2]],
                               atol=0.2)
    assert alignment.worth_correcting


def test_an_offset_along_z_is_found_too():
    # Needs structure along z, which the default (extruded) phantom lacks
    phantom = make_phantom(misregistration=(2, 0, 0), timepoints=1,
                           closed_ends=True)
    alignment = estimate_alignment(phantom.neutron[0], phantom.xray[0])
    np.testing.assert_allclose(alignment.shift, [-2, 0, 0], atol=0.3)


def test_half_voxel_offsets_are_found():
    phantom = make_phantom(misregistration=(0, 0.5, 0.5), timepoints=1)
    alignment = estimate_alignment(phantom.neutron[0], phantom.xray[0])
    np.testing.assert_allclose(alignment.shift[1:], [-0.5, -0.5], atol=0.2)
    # Measured, but left alone: interpolating it away narrows the X-ray
    # noise and costs more accuracy than the half-voxel offset does
    assert alignment.correction == (0, 0, 0)
    assert not alignment.worth_correcting


def test_correction_is_whole_voxels():
    phantom = make_phantom(misregistration=(0, -2, 1.5), timepoints=1)
    alignment = estimate_alignment(phantom.neutron[0], phantom.xray[0])
    assert alignment.correction == (0, 2, -2)
    corrected = apply_shift(phantom.xray[0], alignment.correction)
    # A whole-voxel shift copies values; nothing is interpolated
    np.testing.assert_array_equal(corrected[:, 2:, :-2],
                                  phantom.xray[0][:, :-2, 2:])


def test_axis_without_structure_is_not_guessed():
    # The default phantom is identical along z: no z offset can be measured
    phantom = make_phantom(misregistration=(0, 3, -3), timepoints=1)
    alignment = estimate_alignment(phantom.neutron[0], phantom.xray[0])
    assert not alignment.supported[0] and alignment.shift[0] == 0.0


def test_aligned_data_needs_no_correction():
    phantom = make_phantom(timepoints=1)
    alignment = estimate_alignment(phantom.neutron[0], phantom.xray[0])
    assert alignment.magnitude < 0.25
    assert not alignment.worth_correcting


def test_correction_restores_mutual_information():
    aligned = make_phantom(timepoints=1)
    offset = make_phantom(misregistration=(0, 2, -2), timepoints=1)
    alignment = estimate_alignment(offset.neutron[0], offset.xray[0])
    corrected = apply_shift(offset.xray[0], alignment.correction)
    common = np.isfinite(corrected)
    reference = mutual_information(aligned.neutron[0], aligned.xray[0],
                                   valid=common)
    after = mutual_information(offset.neutron[0], corrected, valid=common)
    before = mutual_information(offset.neutron[0], offset.xray[0], valid=common)
    assert before < 0.8 * reference
    assert after > 0.95 * reference


def test_shift_leaves_unmeasured_voxels_empty():
    volume = np.ones((4, 8, 8), np.float32)
    shifted = apply_shift(volume, (0, 2, 0))
    assert np.isnan(shifted[:, :2]).all() and np.isfinite(shifted[:, 2:]).all()


def test_resample_to_a_common_grid():
    volume = np.arange(4 * 6 * 6, dtype=np.float32).reshape(4, 6, 6)
    assert resample_to_shape(volume, (8, 12, 12)).shape == (8, 12, 12)


def test_segmentation_suffers_from_an_offset_and_recovers():
    """What misregistration costs, and that correcting it recovers it.

    Scored on the voxels that have both measurements after correction (the
    edge band the shift leaves empty cannot be classified by any method).
    """
    from validation import methods
    from validation.scoring import score_series, summarise

    def mean_dice(phantom, region):
        labels = methods.run_bits(phantom)
        return summarise(score_series(labels, phantom.labels, phantom.classes,
                                      region=region))["mean"]

    aligned = make_phantom()
    offset = make_phantom(misregistration=(0, 1, 1))
    alignment = estimate_alignment(offset.neutron[0], offset.xray[0])
    corrected_xray = np.stack([apply_shift(v, alignment.correction)
                               for v in offset.xray])
    region = np.isfinite(corrected_xray)
    base = mean_dice(aligned, region)
    shifted = mean_dice(offset, region)
    offset.xray = corrected_xray
    recovered = mean_dice(offset, region)
    assert shifted < base - 0.03
    assert recovered > base - 0.02


def test_check_alignment_action(monkeypatch):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from gui import BiTS4DMainWindow
    from data import Dataset4D
    from histograms import HistogramEngine4D

    app = QApplication.instance() or QApplication([])
    phantom = make_phantom(timepoints=2, misregistration=(0, 2, 0))
    window = BiTS4DMainWindow()
    window.dataset = Dataset4D(phantom.neutron, phantom.xray)
    window.histogram_engine = HistogramEngine4D(bins=64, use_gpu=False)
    hist = window.histogram_engine.compute_global_histogram(phantom.neutron,
                                                            phantom.xray)
    window.global_histogram = hist
    window.dual_histogram.set_global_histogram(hist)
    window._update_current_timepoint(0)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    window._on_check_alignment()
    applied = window.alignment_applied
    assert len(applied) == 2
    np.testing.assert_allclose(applied[0][1], -2.0, atol=0.2)
    after = estimate_alignment(*window.dataset.get_volume_at_time(0))
    assert after.magnitude < 0.25
    assert window.global_histogram is not None
    app.processEvents()
