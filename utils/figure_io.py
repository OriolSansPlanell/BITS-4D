"""
figure_io.py - Saving analysis figures and their data

Every analysis figure in BiTS is saved through :func:`save_figure`, so they
all behave the same way: the format follows the extension, SVG keeps its
text as text (editable in Inkscape or Illustrator), and raster formats use
the resolution asked for. :func:`write_csv` writes the numbers a figure was
drawn from, so a plot can be redrawn or checked elsewhere.

No Qt here; the format pop-up is ``gui/figure_save_dialog.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence

#: (extension, label) of every format a figure can be saved as, default first
FIGURE_FORMATS = (
    ("svg", "SVG — vector, editable text (recommended)"),
    ("pdf", "PDF — vector"),
    ("png", "PNG — image"),
    ("tif", "TIFF — image"),
)
RASTER_FORMATS = {"png", "tif", "tiff", "jpg", "jpeg"}
DEFAULT_FORMAT = "svg"


def figure_path(base, fmt: str) -> Path:
    """*base* (with or without an extension) with the extension of *fmt*."""
    base = Path(base)
    known = {"." + ext for ext, _label in FIGURE_FORMATS} | {
        ".tiff", ".jpg", ".jpeg", ".csv",
    }
    if base.suffix.lower() in known:
        base = base.with_suffix("")
    return base.with_name(base.name + "." + fmt.lstrip("."))


def save_figure(figure, path, dpi: Optional[int] = None) -> str:
    """Save a matplotlib *figure*; the format follows the extension.

    SVG keeps text as text. *dpi* sets the resolution of raster formats and
    of any images embedded in a vector one (a histogram drawn with imshow).
    """
    import matplotlib

    path = str(path)
    kwargs = {"bbox_inches": "tight"}
    if dpi:
        kwargs["dpi"] = int(dpi)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        figure.savefig(path, **kwargs)
    return path


def write_csv(path, header: Sequence[str], rows: Iterable[Sequence]) -> str:
    """Write *rows* under *header* to a CSV file; returns the path."""
    path = str(path)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return path
