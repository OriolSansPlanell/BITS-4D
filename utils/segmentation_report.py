"""Human-readable report describing an exported segmentation.

Records what each class is called, the integer value it carries in the
exported label volumes, how many voxels it holds at every timepoint, and
the settings the segmentation was produced with — so an exported dataset
can be understood months later without the session that made it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Sequence

import numpy as np

_RULE = "=" * 78
_THIN = "-" * 78


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list:
    """Left-aligned fixed-width table as a list of lines."""
    columns = len(headers)
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index in range(columns):
            widths[index] = max(widths[index], len(str(row[index])))

    def line(values):
        return "  ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    lines = [line(headers), "  ".join("-" * width for width in widths)]
    lines.extend(line(row) for row in rows)
    return lines


def build_segmentation_report(
    class_names: Sequence[str],
    label_values: Dict[str, int],
    voxels_per_timepoint: Dict[int, Dict[str, int]],
    volume_shape,
    *,
    dataset_info: Optional[Dict[str, str]] = None,
    roi_info: Optional[Dict[str, str]] = None,
    settings: Optional[Dict[str, str]] = None,
    notes: Optional[Sequence[str]] = None,
) -> str:
    """Compose the report text.

    Parameters
    ----------
    class_names
        Class names in the order they were exported.
    label_values
        Class name -> integer value it takes in the label volumes.
    voxels_per_timepoint
        timepoint -> {class name -> voxel count}.
    volume_shape
        Shape of one volume, used to turn counts into volume fractions.
    """
    total_voxels = int(np.prod(list(volume_shape))) if len(volume_shape) else 0
    timepoints = sorted(voxels_per_timepoint)

    lines = [
        _RULE,
        " BiTS 4D — segmentation export report",
        _RULE,
        f" Written           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f" Volume shape      : {tuple(int(v) for v in volume_shape)} (Z, Y, X)",
        f" Voxels per volume : {total_voxels:,}",
        f" Timepoints        : {len(timepoints)}"
        + (f"  ({timepoints[0]} … {timepoints[-1]})" if timepoints else ""),
        f" Classes           : {len(class_names)}",
    ]

    if dataset_info:
        lines.append("")
        lines.append(" Dataset")
        lines.append(_THIN)
        for key, value in dataset_info.items():
            lines.append(f"   {key:<24}: {value}")

    # ── Class legend ────────────────────────────────────────────────────────
    lines.append("")
    lines.append(" Class legend  (value = integer in the exported label volume)")
    lines.append(_THIN)
    legend_rows = []
    for name in class_names:
        counts = [
            voxels_per_timepoint[t].get(name, 0) for t in timepoints
        ]
        total = sum(counts)
        mean_fraction = (
            100.0 * total / (total_voxels * len(timepoints))
            if total_voxels and timepoints else 0.0
        )
        legend_rows.append([
            str(label_values.get(name, "-")),
            name,
            f"{total:,}",
            f"{mean_fraction:.3f}%",
        ])
    lines.extend(
        "   " + line for line in _format_table(
            ["value", "class name", "voxels (all T)", "mean vol%"], legend_rows
        )
    )

    # ── Per-timepoint counts ────────────────────────────────────────────────
    lines.append("")
    lines.append(" Voxels per class and timepoint")
    lines.append(_THIN)
    headers = ["timepoint"] + list(class_names) + ["total"]
    count_rows = []
    for timepoint in timepoints:
        per_class = voxels_per_timepoint[timepoint]
        values = [per_class.get(name, 0) for name in class_names]
        count_rows.append(
            [str(timepoint)] + [f"{value:,}" for value in values]
            + [f"{sum(values):,}"]
        )
    if count_rows:
        lines.extend("   " + line for line in _format_table(headers, count_rows))
    else:
        lines.append("   (no segmented timepoints)")

    # ── Volume fraction per timepoint ───────────────────────────────────────
    if total_voxels:
        lines.append("")
        lines.append(" Volume fraction per class and timepoint  (% of the volume)")
        lines.append(_THIN)
        fraction_rows = []
        for timepoint in timepoints:
            per_class = voxels_per_timepoint[timepoint]
            fraction_rows.append(
                [str(timepoint)]
                + [
                    f"{100.0 * per_class.get(name, 0) / total_voxels:.3f}"
                    for name in class_names
                ]
            )
        lines.extend(
            "   " + line for line in _format_table(
                ["timepoint"] + list(class_names), fraction_rows
            )
        )

    # ── How the segmentation was made ───────────────────────────────────────
    if roi_info:
        lines.append("")
        lines.append(" Histogram selections used")
        lines.append(_THIN)
        for key, value in roi_info.items():
            lines.append(f"   {key:<24}: {value}")

    if settings:
        lines.append("")
        lines.append(" Segmentation settings")
        lines.append(_THIN)
        for key, value in settings.items():
            lines.append(f"   {key:<24}: {value}")

    if notes:
        lines.append("")
        lines.append(" Notes")
        lines.append(_THIN)
        for note in notes:
            lines.append(f"   • {note}")

    lines.append(_RULE)
    return "\n".join(lines) + "\n"


def write_segmentation_report(output_path, **kwargs) -> str:
    """Build the report and write it to *output_path*."""
    text = build_segmentation_report(**kwargs)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return str(output_path)
