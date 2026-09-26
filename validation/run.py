"""Run the synthetic validation and write its tables.

    python -m validation.run                  # everything, into validation/results/
    python -m validation.run --quick          # smaller phantom, fewer scenarios
    python -m validation.run --only baselines drift --out /tmp/results

Every number in ``docs/validation.md`` comes from this script with its
default arguments (seed 0). Each scenario writes a CSV with one row per
(setting, method, class) and the script prints a Markdown summary, also
saved as ``summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from typing import Callable, Dict, List

import numpy as np

from model.registration import apply_shift, estimate_alignment
from validation import methods
from validation.phantom import make_phantom
from validation.scoring import score_series, summarise


def _score(labels, phantom, region=None) -> Dict[str, float]:
    summary = summarise(score_series(labels, phantom.labels, phantom.classes,
                                     region=region))
    # Voxels left without a class (BiTS: matched no class; unsupervised
    # baselines: a cluster the oracle matching left unnamed)
    labels = np.asarray(labels)
    summary["unclassified"] = float(np.mean(
        (labels == 0) if region is None else (labels[region] == 0)))
    return summary


def _row(scenario, setting, method, summary, seconds, **extra):
    row = {"scenario": scenario, "setting": setting, "method": method,
           "seconds": round(seconds, 2)}
    row.update({name: round(value, 4) for name, value in summary.items()})
    row.update(extra)
    return row


def _timed(function: Callable, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


# ── scenarios ────────────────────────────────────────────────────────────────

def baselines(quick=False) -> List[dict]:
    """Every method on the default phantom."""
    phantom = make_phantom(shape=(8, 48, 48) if quick else (12, 64, 64))
    rows = []
    (labels, details), seconds = _timed(methods.run_bits, phantom,
                                        return_details=True)
    rows.append(_row("baselines", "default", "BiTS (locked, auto smoothing)",
                     _score(labels, phantom), seconds,
                     smoothing=details["smoothing"]))
    for name, run in methods.BASELINES.items():
        labels, seconds = _timed(run, phantom)
        rows.append(_row("baselines", "default", name, _score(labels, phantom),
                         seconds))
    return rows


def seeds(quick=False) -> List[dict]:
    """The three strongest methods over several noise realisations."""
    chosen = {"BiTS (locked, auto smoothing)": lambda p: methods.run_bits(p),
              "Random walker": methods.run_random_walker,
              "Pixel classifier (RF, ilastik features)":
                  methods.run_pixel_classifier}
    rows = []
    for seed in range(2 if quick else 5):
        phantom = make_phantom(seed=seed)
        for name, run in chosen.items():
            labels, seconds = _timed(run, phantom)
            rows.append(_row("seeds", f"seed={seed}", name,
                             _score(labels, phantom), seconds))
    for name in chosen:
        means = [row["mean"] for row in rows if row["method"] == name]
        products = [row["Product"] for row in rows if row["method"] == name]
        rows.append({"scenario": "seeds", "setting": "mean ± sd",
                     "method": name,
                     "mean": f"{np.mean(means):.3f} ± {np.std(means):.3f}",
                     "Product": f"{np.mean(products):.3f} ± {np.std(products):.3f}"})
    return rows


def noise(quick=False) -> List[dict]:
    """Accuracy as the noise grows (signal separations are fixed)."""
    levels = (60.0, 120.0) if quick else (40.0, 60.0, 90.0, 120.0)
    chosen = {"BiTS (locked, auto smoothing)": lambda p: methods.run_bits(p),
              "Random walker": methods.run_random_walker,
              "Pixel classifier (RF, ilastik features)":
                  methods.run_pixel_classifier}
    rows = []
    for level in levels:
        phantom = make_phantom(noise=(level, level))
        for name, run in chosen.items():
            labels, seconds = _timed(run, phantom)
            rows.append(_row("noise", f"sigma={level:g}", name,
                             _score(labels, phantom), seconds))
    return rows


def smoothing(quick=False) -> List[dict]:
    """Accuracy against the smoothing strength, and what auto chooses."""
    grid = (0.0, 1.0, 4.0, 16.0) if quick else (0.0, 0.5, 1.0, 2.0, 4.0, 8.0,
                                                 16.0, 32.0)
    rows = []
    for level in (60.0, 90.0):
        phantom = make_phantom(noise=(level, level))
        for beta in grid:
            labels, seconds = _timed(methods.run_bits, phantom, smoothing=beta)
            rows.append(_row("smoothing", f"sigma={level:g}", f"beta={beta:g}",
                             _score(labels, phantom), seconds))
        (labels, details), seconds = _timed(methods.run_bits, phantom,
                                            return_details=True)
        report = details["outcome"].smoothing_report or {}
        rows.append(_row("smoothing", f"sigma={level:g}",
                         f"auto (beta={details['smoothing']:g})",
                         _score(labels, phantom), seconds,
                         reason=report.get("reason", "")))
    return rows


def class_birth(quick=False) -> List[dict]:
    """A phase absent at T0: defined at its own timepoint, or not at all."""
    phantom = make_phantom()
    rows = []
    for birth in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels, seconds = _timed(methods.run_bits, phantom, birth=birth)
        rows.append(_row("class birth", "product absent at T0",
                         "defined where it exists" if birth
                         else "all classes from T0", _score(labels, phantom),
                         seconds))
    return rows


def drift(quick=False) -> List[dict]:
    """X-ray gain drifting per timepoint: locked vs. the adaptive modes."""
    levels = (0.0, 0.06) if quick else (0.0, 0.03, 0.06)
    modes = {
        "BiTS locked": lambda p: methods.run_bits(p),
        "adaptive, frozen boundaries":
            lambda p: methods.run_bits_adaptive(p, track_drift=False),
        "adaptive, anchored drift":
            lambda p: methods.run_bits_adaptive(p, track_drift=True),
    }
    rows = []
    for level in levels:
        phantom = make_phantom(timepoints=8, gain_drift=level)
        for name, run in modes.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels, seconds = _timed(run, phantom)
            scores = score_series(labels, phantom.labels, phantom.classes)
            worst = min(min(entry["dice"]) for name_, entry in scores.items()
                        if name_ != "Product")
            rows.append(_row("drift", f"gain {level:+.0%}/step", name,
                             summarise(scores), seconds,
                             worst_fixed_class_dice=round(worst, 4)))
    return rows


def misregistration(quick=False) -> List[dict]:
    """What an X-ray/neutron offset costs, and what the correction recovers.

    Scored on the voxels both volumes still cover after correction.
    """
    offsets = [((0, 0, 0), False), ((0, 1, 1), False), ((0, 2, 2), False)]
    if not quick:
        offsets[1:1] = [((0, 0.5, 0.5), False)]
        offsets.append(((2, 0, 0), True))
    rows = []
    for offset, closed in offsets:
        phantom = make_phantom(misregistration=offset, closed_ends=closed)
        alignment = estimate_alignment(phantom.neutron[0], phantom.xray[0])
        shift = alignment.correction if alignment.worth_correcting else (0, 0, 0)
        corrected = np.stack([apply_shift(v, shift) for v in phantom.xray])
        region = np.isfinite(corrected)
        setting = (f"offset {tuple(float(v) for v in offset)}"
                   + (" (closed ends)" if closed else ""))
        labels, seconds = _timed(methods.run_bits, phantom)
        rows.append(_row("misregistration", setting, "uncorrected",
                         _score(labels, phantom, region), seconds))
        original = phantom.xray
        phantom.xray = corrected
        labels, seconds = _timed(methods.run_bits, phantom)
        phantom.xray = original
        rows.append(_row("misregistration", setting, "after Check Alignment",
                         _score(labels, phantom, region), seconds,
                         estimated_shift=" ".join(f"{v:+.2f}" for v in alignment.shift),
                         applied_shift=" ".join(f"{v:+d}" for v in shift)))
    return rows


def class_shapes(quick=False) -> List[dict]:
    """Single Gaussian vs. BIC-chosen mixture per class."""
    settings = {"two-state lithium": dict(two_state_lithium=(1500.0, 1300.0)),
                "cupping 0.3": dict(cupping=0.3)}
    if quick:
        settings.pop("cupping 0.3")
    rows = []
    for setting, kwargs in settings.items():
        phantom = make_phantom(**kwargs)
        for components in (1, 3):
            (labels, details), seconds = _timed(
                methods.run_bits, phantom, max_components=components,
                return_details=True)
            used = {m.name: m.n_components for m in details["library"]}
            rows.append(_row("class shapes", setting,
                             "one Gaussian" if components == 1
                             else "mixture (BIC, up to 3)",
                             _score(labels, phantom), seconds,
                             components=" ".join(f"{k}:{v}" for k, v in used.items())))
    return rows


def coefficients(quick=False) -> List[dict]:
    """Product placed from attenuation coefficients instead of a region."""
    phantom = make_phantom()
    variants = [("region (defined where it exists)", {}),
                ("no definition (from T0 only)", {"birth": False}),
                ("coefficients, exact", {"predict": ("Product",)}),
                ("coefficients, +5% error", {"predict": ("Product",),
                                             "coefficient_error": 0.05}),
                ("coefficients, +15% error", {"predict": ("Product",),
                                              "coefficient_error": 0.15})]
    if quick:
        variants = variants[:3]
    rows = []
    for name, kwargs in variants:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels, seconds = _timed(methods.run_bits, phantom, **kwargs)
        rows.append(_row("coefficients", "Product", name,
                         _score(labels, phantom), seconds))
    return rows


SCENARIOS = {
    "baselines": baselines,
    "seeds": seeds,
    "noise": noise,
    "smoothing": smoothing,
    "class_birth": class_birth,
    "drift": drift,
    "misregistration": misregistration,
    "class_shapes": class_shapes,
    "coefficients": coefficients,
}


# ── output ───────────────────────────────────────────────────────────────────

def write_csv(rows: List[dict], path: str):
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: List[dict], classes) -> str:
    extra = []
    for row in rows:
        extra += [key for key in row if key not in extra and key not in (
            "scenario", "setting", "method", "seconds", "mean", *classes)]
    header = ["setting", "method", *classes, "mean", "seconds", *extra]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for row in rows:
        cells = []
        for key in header:
            value = row.get(key, "")
            cells.append(f"{value:.3f}" if isinstance(value, float)
                         and key not in ("seconds",) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="*", choices=sorted(SCENARIOS))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    classes = make_phantom(shape=(4, 16, 16), timepoints=1).classes

    sections = []
    for name in args.only or SCENARIOS:
        start = time.perf_counter()
        rows = SCENARIOS[name](quick=args.quick)
        write_csv(rows, os.path.join(args.out, f"{name}.csv"))
        section = f"### {name}\n\n{markdown(rows, classes)}\n"
        print(section, flush=True)
        print(f"({time.perf_counter() - start:.0f} s)\n", flush=True)
        sections.append(section)
    with open(os.path.join(args.out, "summary.md"), "w") as handle:
        handle.write("Mean Dice per class over the series (1 = perfect).\n\n")
        handle.write("\n".join(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
