"""Automatic checks that run before a result is presented as an answer.

The point is not to produce a score. It is that a run which has gone wrong in
a way the user cannot see should not reach a results screen at all — and when
it does fail, the message should name the class involved and say what to do,
not report a number and leave the interpretation to the reader.

Every finding here is written to be read by someone who knows segmentation
and has no reason to know anything about how this is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np


class Status(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

    @property
    def symbol(self) -> str:
        return {"pass": "OK", "warn": "CHECK", "fail": "PROBLEM"}[self.value]


@dataclass
class Finding:
    """One check's outcome, in language the user can act on."""

    check: str
    status: Status
    message: str
    detail: str = ""

    def __str__(self) -> str:
        line = f"[{self.status.symbol}] {self.message}"
        return f"{line}\n        {self.detail}" if self.detail else line


@dataclass
class HealthReport:
    findings: List[Finding] = field(default_factory=list)

    @property
    def status(self) -> Status:
        if any(f.status is Status.FAIL for f in self.findings):
            return Status.FAIL
        if any(f.status is Status.WARN for f in self.findings):
            return Status.WARN
        return Status.PASS

    @property
    def passed(self) -> bool:
        return self.status is not Status.FAIL

    def problems(self) -> List[Finding]:
        return [f for f in self.findings if f.status is Status.FAIL]

    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.status is Status.WARN]

    def headline(self) -> str:
        if self.status is Status.PASS:
            return "All checks passed."
        if self.status is Status.WARN:
            count = len(self.warnings())
            return f"{count} thing{'s' if count != 1 else ''} worth checking."
        count = len(self.problems())
        return (
            f"{count} problem{'s' if count != 1 else ''} found — "
            f"the results are probably not reliable."
        )

    def describe(self) -> str:
        return "\n".join([self.headline(), ""] + [str(f) for f in self.findings])


def _volume_cv(values: Sequence[float]) -> float:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if array.size < 2 or array.mean() == 0:
        return 0.0
    return float(array.std() / abs(array.mean()))


def run_health_check(
    outcome,
    control_materials: Sequence[str] = (),
    max_unclassified: float = 0.05,
    control_tolerance: float = 0.02,
    coverage_step: float = 0.02,
    mixing_report: Optional[Dict[str, dict]] = None,
) -> HealthReport:
    """Check a finished series before its numbers are used.

    *outcome* is a :class:`~model.locked.SeriesSegmentation`.
    *control_materials* are the classes the user marked as not changing;
    if one of them moves, the segmentation is wrong, not the sample.
    """
    report = HealthReport()
    entries = list(getattr(outcome, "timepoints", []))
    if not entries:
        report.findings.append(Finding(
            "any results", Status.FAIL,
            "Nothing was segmented.",
        ))
        return report

    names = list(outcome.class_names)
    controls = [name for name in control_materials if name in names]
    library = getattr(outcome, "library", None)

    # ── classes that could not be defined at all ─────────────────────────
    for name, reason in getattr(library, "dropped", []) or []:
        report.findings.append(Finding(
            "classes defined", Status.FAIL,
            f"The material '{name}' could not be defined and is missing from "
            f"these results.",
            reason[:1].upper() + reason[1:] + ".",
        ))
    for name, note in getattr(library, "notes", []) or []:
        report.findings.append(Finding(
            "classes defined", Status.WARN,
            f"The material '{name}' was {note}.",
        ))

    # Where each class was defined: a phase defined at a later timepoint
    # (it appears during the experiment) is expected to be absent before.
    born_at: Dict[str, int] = {}
    for material in getattr(library, "classes", []) or []:
        if getattr(material, "defined_at", None) is not None:
            born_at[material.name] = int(material.defined_at)
    first = min(entry.timepoint for entry in entries)
    # A class predicted from attenuation coefficients has no region: it may
    # appear at any point, and is checked from where it is first found.
    predicted = set()
    for material in getattr(library, "classes", []) or []:
        if getattr(material, "source", "") != "physics":
            continue
        found = [e.timepoint for e in entries
                 if e.voxel_counts.get(material.name, 0) > 0]
        predicted.add(material.name)
        if found:
            born_at[material.name] = min(found)
        else:
            born_at[material.name] = max(e.timepoint for e in entries) + 1
            report.findings.append(Finding(
                "predicted classes", Status.WARN,
                f"The material '{material.name}', placed from its attenuation "
                "coefficients, was not found at any timepoint.",
                "Either it is not in the sample, or the calibration or the "
                "coefficients put it in the wrong place. Check the prediction "
                "against the histogram.",
            ))

    # ── how the smoothing strength was chosen ────────────────────────────
    choice = getattr(outcome, "smoothing_report", None)
    if choice:
        checked = ", ".join(f"T{t}" for t in choice.get("timepoints", []))
        if choice.get("at_ceiling"):
            report.findings.append(Finding(
                "smoothing strength", Status.WARN,
                f"The smoothing strength reached the top of the tested range "
                f"({choice['strength']:g}) without the result settling.",
                "More smoothing was still changing the result, so this is the "
                "limit of the search rather than a measured optimum. Compare "
                "with a weaker setting (Advanced) before relying on fine "
                "structures.",
            ))
        elif choice.get("converged"):
            report.findings.append(Finding(
                "smoothing strength", Status.PASS,
                f"Smoothing {choice['strength']:g}: {choice['reason']} "
                f"(checked at {checked}).",
            ))
        else:
            report.findings.append(Finding(
                "smoothing strength", Status.PASS,
                f"Smoothing {choice['strength']:g}: {choice['reason']} — "
                f"thin structures and every class's volume were protected "
                f"(checked at {checked}).",
            ))

    # ── every class present at every timepoint ───────────────────────────
    missing: Dict[str, int] = {}
    for name in names:
        start = born_at.get(name, first)
        for entry in entries:
            if entry.timepoint < start:
                continue
            if entry.voxel_counts.get(name, 0) == 0:
                missing.setdefault(name, entry.timepoint)
                break
    for name, start in sorted(born_at.items(), key=lambda item: item[1]):
        if start > first and name in names:
            before = [e for e in entries if e.timepoint < start]
            if name in predicted:
                if len(before) < len(entries):
                    report.findings.append(Finding(
                        "phases that appear", Status.PASS,
                        f"'{name}' (predicted from its coefficients) is first "
                        f"found at timepoint {start}.",
                    ))
                continue
            absent = sum(1 for e in before if e.voxel_counts.get(name, 0) == 0)
            report.findings.append(Finding(
                "phases that appear", Status.PASS,
                f"'{name}' is defined at timepoint {start}, where it exists; "
                f"it is absent at {absent} of the {len(before)} earlier "
                f"timepoint(s).",
            ))
    if missing:
        for name, timepoint in missing.items():
            entry = next(e for e in entries if e.timepoint == timepoint)
            kept = entry.smoothing_retention().get(name, 1.0)
            if kept < 0.5:
                detail = (
                    "Smoothing is what removed it — without smoothing the "
                    "class is still there at that timepoint. Reduce the "
                    "smoothing strength."
                )
            else:
                detail = (
                    "This is not caused by smoothing: the class is absent "
                    "with or without it. Either the material really is gone, "
                    "or its region on the histogram no longer covers it."
                )
            report.findings.append(Finding(
                "classes present", Status.FAIL,
                f"The class '{name}' disappeared after timepoint "
                f"{max(timepoint - 1, 0)}.",
                detail,
            ))
    else:
        report.findings.append(Finding(
            "classes present", Status.PASS,
            f"All {len(names)} materials are present at every timepoint.",
        ))

    # ── control materials stable ─────────────────────────────────────────
    if controls:
        unstable = []
        for name in controls:
            curve = [entry.voxel_counts.get(name, 0) for entry in entries]
            variation = _volume_cv(curve)
            if variation > control_tolerance:
                unstable.append((name, variation))
        if unstable:
            for name, variation in unstable:
                report.findings.append(Finding(
                    "control materials", Status.FAIL,
                    f"'{name}' changed by {100 * variation:.0f}%, but it was "
                    f"marked as a material that should not change.",
                    "Either the segmentation is unreliable, or this material "
                    "is not as stable as expected. Check it before trusting "
                    "the other volumes.",
                ))
        else:
            report.findings.append(Finding(
                "control materials", Status.PASS,
                "The control materials stayed within "
                f"{100 * control_tolerance:.0f}% across the series.",
            ))
    else:
        report.findings.append(Finding(
            "control materials", Status.WARN,
            "No control materials were marked.",
            "Ticking one material that should not change during the "
            "experiment gives an independent check on every other result.",
        ))

    # ── unclassified fraction ────────────────────────────────────────────
    worst_share, worst_timepoint = 0.0, entries[0].timepoint
    for entry in entries:
        share = entry.unclassified_fraction
        if share > worst_share:
            worst_share, worst_timepoint = share, entry.timepoint
    if worst_share > max_unclassified:
        report.findings.append(Finding(
            "unclassified", Status.FAIL,
            f"{100 * worst_share:.0f}% of the measured voxels at timepoint "
            f"{worst_timepoint} did not match any material you defined.",
            "You may be missing a class. Look at where those voxels sit on "
            "the histogram and add a region for them.",
        ))
    else:
        report.findings.append(Finding(
            "unclassified", Status.PASS,
            f"At most {100 * worst_share:.1f}% of measured voxels went "
            f"unmatched.",
        ))

    # ── voxel budget ─────────────────────────────────────────────────────
    leaking = [entry.timepoint for entry in entries if not entry.budget_closes()]
    if leaking:
        report.findings.append(Finding(
            "voxel budget", Status.FAIL,
            f"The voxel count does not add up at timepoint {leaking[0]} — "
            f"some voxels were counted more than once.",
            "This is a bug, please report it.",
        ))
    else:
        report.findings.append(Finding(
            "voxel budget", Status.PASS,
            "Every voxel is accounted for exactly once at every timepoint.",
        ))

    # ── usable data stable ───────────────────────────────────────────────
    fractions = [
        entry.valid_voxels / max(entry.total_voxels, 1) for entry in entries
    ]
    step_at = None
    for index in range(1, len(fractions)):
        if abs(fractions[index] - fractions[index - 1]) > coverage_step:
            step_at = entries[index].timepoint
            break
    if step_at is not None:
        report.findings.append(Finding(
            "usable data", Status.WARN,
            f"The amount of usable data changed at timepoint {step_at}.",
            "Check whether the acquisition changed there. Comparing volumes "
            "across that point may not be meaningful.",
        ))
    else:
        report.findings.append(Finding(
            "usable data", Status.PASS,
            f"The amount of usable data is stable "
            f"({100 * float(np.mean(fractions)):.0f}% of the array).",
        ))

    # ── field-of-view overlap ────────────────────────────────────────────
    overlap = [
        entry.validity.get("neutron_only_fraction", 0.0)
        + entry.validity.get("xray_only_fraction", 0.0)
        for entry in entries if entry.validity
    ]
    if overlap and max(overlap) > 0.01:
        report.findings.append(Finding(
            "field of view", Status.WARN,
            f"{100 * max(overlap):.0f}% of the array has data from only one "
            f"of the two instruments.",
            "Those voxels are excluded, because a material can only be "
            "identified where both measurements exist.",
        ))

    # ── mixed boundaries sit between their materials ─────────────────────
    if mixing_report:
        for name, entry in mixing_report.items():
            if not entry.get("accepted", True):
                report.findings.append(Finding(
                    "mixed boundaries", Status.WARN,
                    f"'{name}' does not sit between the two materials it was "
                    f"linked to.",
                    "The class may be mislabelled, or it may be a material in "
                    "its own right rather than a boundary between two others.",
                ))

    return report
