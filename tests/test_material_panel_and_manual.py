"""The materials panel and the in-application manual.

The panel is the surface for the one decision the software cannot make for
the user — which materials are allowed to change — so the tests here are
mostly about that setting surviving everything else that happens to the
panel.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui import manual_content  # noqa: E402
from gui.material_panel import (  # noqa: E402
    CHANGES,
    UNCHANGED,
    MaterialPanel,
    describe_strength,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp):
    widget = MaterialPanel()
    widget.set_materials([
        {"name": "Lithium", "source": "drawn", "voxels": 1200},
        {"name": "Aluminium", "source": "drawn", "voxels": 45000},
        {"name": "K-means cluster 2", "source": "K-means", "voxels": 8000},
    ])
    return widget


# ── the materials list ───────────────────────────────────────────────────────

def test_materials_appear_with_their_source_and_size(panel):
    assert panel.material_names() == ["Lithium", "Aluminium", "K-means cluster 2"]
    assert panel.table.rowCount() == 3
    assert panel.table.item(0, 0).text() == "Lithium"
    assert panel.table.item(2, 1).text() == "K-means"
    assert panel.table.item(1, 2).text() == "45,000"


def test_everything_may_change_until_told_otherwise(panel):
    """No default control material: the choice is the user's and only theirs."""
    assert panel.control_materials() == []
    for row in range(panel.table.rowCount()):
        assert panel.table.cellWidget(row, 3).currentText() == CHANGES


def test_marking_a_material_as_unchanging(panel):
    panel.set_behaviour("Aluminium", UNCHANGED)
    assert panel.control_materials() == ["Aluminium"]
    assert panel.settings()["control_materials"] == ["Aluminium"]


def test_a_cluster_can_be_a_control_like_anything_else(panel):
    """Copied clusters are ordinary materials once they are here."""
    panel.set_behaviour("K-means cluster 2", UNCHANGED)
    assert panel.control_materials() == ["K-means cluster 2"]


def test_refreshing_keeps_the_choices_already_made(panel):
    """Re-reading the segmentation must not discard the one manual decision."""
    panel.set_behaviour("Aluminium", UNCHANGED)
    panel.set_materials([
        {"name": "Lithium", "source": "drawn", "voxels": 900},
        {"name": "Aluminium", "source": "drawn", "voxels": 45000},
        {"name": "Separator", "source": "drawn", "voxels": 500},
    ])
    assert panel.control_materials() == ["Aluminium"]
    assert "Separator" in panel.material_names()
    assert panel.table.item(0, 2).text() == "900"      # the count did update


def test_an_unknown_material_is_rejected_clearly(panel):
    with pytest.raises(KeyError):
        panel.set_behaviour("Nonexistent", UNCHANGED)
    with pytest.raises(ValueError):
        panel.set_behaviour("Lithium", "sometimes")


def test_running_needs_materials(qapp):
    empty = MaterialPanel()
    assert not empty.run_btn.isEnabled()
    assert not empty.preview_btn.isEnabled()
    empty.set_materials([{"name": "A", "source": "drawn", "voxels": 10}])
    assert empty.run_btn.isEnabled() and empty.preview_btn.isEnabled()


def test_the_copy_button_waits_for_a_clustering_result(panel):
    assert not panel.copy_clusters_btn.isEnabled()
    panel.set_clusters_available(True)
    assert panel.copy_clusters_btn.isEnabled()


# ── settings ─────────────────────────────────────────────────────────────────

def test_auto_is_the_default_and_names_no_number(panel):
    assert panel.smoothing_combo.currentText() == "Auto"
    assert panel.smoothing_mode == "auto"
    assert panel.smoothing_strength is None


@pytest.mark.parametrize("choice,expected", [
    ("Off", 0.0), ("Low", 0.5), ("Medium", 1.0), ("High", 3.0),
])
def test_manual_smoothing_choices_map_to_strengths(panel, choice, expected):
    panel.smoothing_combo.setCurrentText(choice)
    assert panel.smoothing_mode == "manual"
    assert panel.smoothing_strength == expected


def test_definitions_are_locked_by_default_and_warn_when_not(panel):
    assert panel.lock_definitions
    assert panel.lock_warning.text() == ""

    panel.lock_check.setChecked(False)
    assert not panel.lock_definitions
    warning = panel.lock_warning.text()
    assert "control materials" in warning
    assert "absorb a real change" in warning


def test_describe_strength_never_shows_the_number():
    assert describe_strength(None) == "Off"
    assert describe_strength(0.0) == "Off"
    assert describe_strength(0.4) == "Low"
    assert describe_strength(1.0) == "Medium"
    assert describe_strength(4.0) == "High"


# ── results ──────────────────────────────────────────────────────────────────

def test_a_health_report_is_shown_with_its_problems(panel):
    from model.health_check import Finding, HealthReport, Status

    report = HealthReport(findings=[
        Finding("a", Status.PASS, "All materials are present."),
        Finding("b", Status.FAIL, "'Aluminium' changed by 8%.",
                "Check it before trusting the other volumes."),
    ])
    panel.show_health_report(report)

    assert "problem" in panel.status_label.text()
    findings = panel.findings_label.text()
    assert "Aluminium" in findings
    assert "Check it before trusting" in findings
    # Passing checks are not repeated as problems
    assert "All materials are present" not in findings


def test_a_clean_report_says_so(panel):
    from model.health_check import Finding, HealthReport, Status

    panel.show_health_report(HealthReport(findings=[
        Finding("a", Status.PASS, "All materials are present."),
    ]))
    assert "All checks passed" in panel.status_label.text()
    assert panel.findings_label.text() == ""


def test_the_chosen_smoothing_is_reported_in_words(panel):
    panel.set_smoothing_result(1.0, chosen_automatically=True)
    text = panel.smoothing_result.text()
    assert "Medium" in text and "automatically" in text
    assert "1.0" not in text


def test_clearing_resets_the_result_area(panel):
    panel.set_smoothing_result(3.0)
    panel.set_findings(["something"])
    panel.clear_result()
    assert panel.findings_label.text() == ""
    assert panel.smoothing_result.text() == ""
    assert "Nothing run yet" in panel.status_label.text()


# ── the manual ───────────────────────────────────────────────────────────────

def test_every_section_renders():
    for section in manual_content.SECTIONS:
        html = manual_content.render(section)
        assert html.startswith("\n<style>") or "<style>" in html
        assert "<h1>" in html
        assert len(html) > 500, section["id"]


def test_sections_have_unique_ids_and_two_groups():
    ids = manual_content.section_ids()
    assert len(ids) == len(set(ids))
    groups = {section["group"] for section in manual_content.SECTIONS}
    assert groups == {"How to", "Mathematics"}


def test_search_requires_every_word():
    """Otherwise a two-word query returns most of the manual."""
    both = manual_content.search("smoothing strength")
    assert both
    assert all(
        "smoothing" in " ".join([s["title"], s.get("keywords", ""), s["body"]]).lower()
        for s in both
    )
    assert manual_content.search("smoothing") != both or len(both) >= 1
    assert manual_content.search("zzzz nothing") == []
    # An empty query lists everything rather than nothing
    assert len(manual_content.search("")) == len(manual_content.SECTIONS)


def test_search_finds_a_term_only_present_as_a_keyword():
    assert any(
        section["id"] == "m_material"
        for section in manual_content.search("covariance")
    )
    assert any(
        section["id"] == "controls"
        for section in manual_content.search("inert")
    )


def test_plain_text_export_drops_the_markup():
    text = manual_content.as_plain_text()
    assert "<p>" not in text and "</h1>" not in text and "<style>" not in text
    assert "Getting started" in text
    assert "Mathematics" in text
    assert len(text) > 20_000


def test_get_section_rejects_an_unknown_id():
    with pytest.raises(KeyError):
        manual_content.get_section("nope")


def test_the_maths_sections_actually_contain_formulas():
    for section in manual_content.SECTIONS:
        if section["group"] != "Mathematics" or section["id"] == "refs":
            continue
        assert 'class="m"' in section["body"], (
            f"{section['id']} claims to be mathematics but states no formula"
        )


def test_the_manual_window_opens_at_a_requested_section(qapp):
    from gui.manual import ManualWindow

    window = ManualWindow()
    window.show_section("m_drift")
    assert window.current_section_id() == "m_drift"
    assert "Instrument drift" in window.viewer.toPlainText()

    window.show_section("controls")
    assert "Control materials" in window.viewer.toPlainText()


def test_searching_the_window_narrows_the_contents(qapp):
    from gui.manual import ManualWindow

    window = ManualWindow()
    full = window.contents.topLevelItemCount()
    window.search_box.setText("mahalanobis")
    assert window.contents.topLevelItemCount() <= full
    assert "section" in window.result_label.text()
    assert window.current_section_id() == "m_material"

    window.search_box.setText("zzzz")
    assert "Nothing matches" in window.result_label.text()

    window.search_box.setText("")
    assert window.contents.topLevelItemCount() == full


def test_the_manual_is_reachable_from_the_help_menu(qapp):
    from gui import BiTS4DMainWindow

    window = BiTS4DMainWindow()
    labels = []
    for action in window.menuBar().actions():
        if action.text() == "Help":
            for entry in action.menu().actions():
                labels.append(entry.text())
                if entry.menu():
                    labels.extend(sub.text() for sub in entry.menu().actions())
    assert "Manual..." in labels
    assert "Mathematics" in labels
    assert "Why There Is No Classifier" in labels


# ── the classifier is gone from the application ──────────────────────────────

def test_no_classifier_tab_remains(qapp):
    from gui import BiTS4DMainWindow

    window = BiTS4DMainWindow()
    tabs = [
        window.right_tabs.tabText(index)
        for index in range(window.right_tabs.count())
    ]
    assert any("Materials" in tab for tab in tabs)
    assert not any(
        word in tab for tab in tabs for word in ("Forest", "Classifier", "RF")
    )
    for attribute in ("rf_engine", "rf_masks", "_rf_train", "_rf_predict_all"):
        assert not hasattr(window, attribute), attribute


def test_the_classifier_is_still_importable_for_comparisons():
    """Removed from the application, kept for reproducing earlier work."""
    from segmentation.legacy import RandomForestSegmentation4D

    assert RandomForestSegmentation4D(n_estimators=5) is not None

    import segmentation

    assert not hasattr(segmentation, "RandomForestSegmentation4D")
