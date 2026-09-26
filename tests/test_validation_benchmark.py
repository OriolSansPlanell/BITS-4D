"""Review items 1 and 2: the validation is reproducible.

A smoke run of ``python -m validation.run`` on reduced settings, plus the
properties the documented numbers rest on: the scoring is right, and the
same seed gives the same phantom and the same score.
"""

import csv

import numpy as np

from validation import run
from validation.phantom import make_phantom
from validation.scoring import dice, oracle_match, score_series, summarise


def test_dice_and_empty_classes():
    a = np.array([1, 1, 0, 0], bool)
    assert dice(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    assert dice(np.zeros(4, bool), np.zeros(4, bool)) == 1.0


def test_oracle_matching_renames_clusters():
    truth = np.array([1, 1, 2, 2, 3, 3])
    clusters = np.array([7, 7, 4, 4, 9, 9])
    np.testing.assert_array_equal(oracle_match(clusters, truth, 3), truth)


def test_phantom_is_deterministic():
    first, second = make_phantom(timepoints=2), make_phantom(timepoints=2)
    np.testing.assert_array_equal(first.neutron, second.neutron)
    np.testing.assert_array_equal(first.labels, second.labels)
    # The product is absent at T0 and grows afterwards
    fractions = first.volume_fractions()["Product"]
    assert fractions[0] == 0.0 and fractions[1] > 0.0


def test_perfect_labels_score_one():
    phantom = make_phantom(timepoints=2)
    summary = summarise(score_series(phantom.labels, phantom.labels,
                                     phantom.classes))
    assert summary["mean"] == 1.0


def test_benchmark_runs_and_writes_its_tables(tmp_path, capsys):
    assert run.main(["--quick", "--only", "class_birth", "coefficients",
                     "--out", str(tmp_path)]) == 0
    with open(tmp_path / "class_birth.csv") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["method"]: float(row["Product"]) for row in rows}
    # Defining the product where it exists is what makes it trackable
    assert by_method["defined where it exists"] > \
        by_method["all classes from T0"] + 0.4
    assert (tmp_path / "coefficients.csv").exists()
    summary = (tmp_path / "summary.md").read_text()
    assert "### class_birth" in summary and "| setting |" in summary
    assert "class_birth" in capsys.readouterr().out
