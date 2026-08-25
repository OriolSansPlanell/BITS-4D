"""The interface must not require knowing how the method works.

Someone who knows segmentation — thresholding, regions, classes, smoothing —
should be able to run this end to end without meeting a term from the
statistics of it. That is a property of the software, not a style preference,
so it is checked rather than trusted: this test parses every user-facing
string literal and fails on the vocabulary that belongs in the derivation.

Scope. Docstrings, comments and identifiers are exempt — they are for whoever
is reading the code, who does need those words. What is checked is the text a
user can actually see: widget labels and tooltips in ``gui/``, and the
findings the health check and the segmentation guards produce, which are
quoted verbatim into dialogs.
"""

import ast
import pathlib
import re

import pytest

#: Terms that belong in the method documentation and nowhere a user can see.
FORBIDDEN_TERMS = [
    "MRF", "Markov random field", "Potts", "prior", "posterior", "likelihood",
    "unary", "pairwise", "mean-field", "mean field", "ICM", "beta", "kappa",
    "EM", "mixture model", "mixture", "Gaussian", "Bayesian", "energy",
    "responsibility", "mixel",
]

#: Files whose string literals reach the user.
USER_FACING = [
    "gui/main_window.py",
    "gui/material_panel.py",
    "gui/dual_histogram_widget.py",
    "gui/statistics_panel.py",
    "gui/selection_manager.py",
    "gui/time_navigation_widget.py",
    "model/health_check.py",
    "model/locked.py",
]

#: The manual is the one place the vocabulary belongs: it exists to explain
#: the method to someone who has chosen to read about it. It is checked for
#: the opposite property instead — see test_the_manual_explains_the_terms.
MANUAL = "gui/manual_content.py"

_PATTERN = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in FORBIDDEN_TERMS) + r")(?![A-Za-z])",
    re.IGNORECASE,
)


def _docstring_ids(tree):
    """Nodes that are docstrings, which are for readers of the code."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                found.add(id(body[0].value))
    return found


def _user_strings(path):
    """Every non-docstring string literal, with its line number."""
    source = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


@pytest.mark.parametrize("path", USER_FACING)
def test_no_method_vocabulary_reaches_the_user(path):
    offences = []
    for line, text in _user_strings(path):
        found = _PATTERN.findall(text)
        if found:
            offences.append(
                f"{path}:{line}: {sorted(set(t.lower() for t in found))} "
                f"in {text[:70]!r}"
            )
    assert not offences, (
        "These strings use vocabulary from the method rather than from "
        "segmentation:\n  " + "\n  ".join(offences)
    )


def test_the_check_would_catch_a_regression():
    """The guard is only worth having if it actually fires."""
    samples = [
        "Set the beta parameter",
        "Adjust kappa for the prior",
        "Uses a Gaussian mixture model",
        "mean-field refinement",
        "the MRF energy",
        "posterior responsibility per voxel",
    ]
    for text in samples:
        assert _PATTERN.search(text), text


def test_ordinary_segmentation_words_are_not_flagged():
    """It must not fire on the vocabulary the user is expected to have."""
    samples = [
        "Smoothing strength",
        "Uses neighbouring voxels to clean up noisy assignments.",
        "Voxels that don't match any material you defined.",
        "Control materials",
        "Mixing fraction",
        "Lock material definitions",
        "How much of each material is in this voxel (0-100 %).",
        "Draw a region on the histogram",
        "Segment the current timepoint",
    ]
    for text in samples:
        assert not _PATTERN.search(text), text


def test_smoothing_strength_is_described_in_words_not_numbers():
    pytest.importorskip("PyQt5")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from gui.material_panel import describe_strength as describe
    assert describe(None) == "Off"
    assert describe(0.0) == "Off"
    assert describe(0.25) == "Low"
    assert describe(1.0) == "Medium"
    assert describe(8.0) == "High"
    # The user-facing value is never the raw number
    for value in (0.0, 0.5, 1.0, 4.0):
        assert not any(char.isdigit() for char in describe(value))


# ── the manual is the exception, and has to earn it ──────────────────────────

def test_the_manual_explains_the_terms_the_interface_hides():
    """The vocabulary has to live somewhere, and this is where.

    A user who wants to know what "smoothing" actually does should be able to
    find the answer without leaving the application.
    """
    from gui import manual_content

    text = manual_content.as_plain_text().lower()
    for term in ("mahalanobis", "winding rule", "covariance", "eigenvector",
                 "marginal", "bayes error", "davies", "kappa"):
        assert term in text, f"the manual never explains {term!r}"


def test_the_manual_covers_every_operation_the_menus_offer():
    from gui import manual_content

    text = manual_content.as_plain_text().lower()
    for topic in ("check data", "control material", "smoothing",
                  "instrument stability", "mixed boundaries", "health check",
                  "export", "spatial metrics"):
        assert topic in text, f"the manual never mentions {topic!r}"


def test_the_manual_says_why_the_classifier_was_removed():
    from gui import manual_content

    section = manual_content.get_section("m_why")
    body = section["body"].lower()
    assert "random forest" in body
    assert "constants" in body
    assert "legacy" in body
