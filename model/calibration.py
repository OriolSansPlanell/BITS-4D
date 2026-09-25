"""Placing classes from attenuation coefficients.

A material that has no region to draw (not yet formed, too thin, or hidden)
can still be placed on the bivariate histogram if its attenuation
coefficients are known: each modality's grey value is, to first order, a
linear function of the attenuation coefficient,

    grey = gain · μ + offset,

with a gain and offset per modality that depend on the reconstruction, not
on the material. Two materials that are present in the scan and whose
coefficients are known fix both numbers (more than two give a least-squares
fit and a residual to judge it by). Every other material is then predicted
from its coefficients alone.

The spread of a predicted class is an instrument property (noise, partial
volume), so it is taken from the reference materials, not guessed.

Caveat: X-ray attenuation depends strongly on the spectrum; the coefficients
entered must be the effective values for the scan's energy, and the
prediction is only as good as that value. Treat a predicted class as a
hypothesis the health check and the ROIs then test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from .likelihood import ClassLibrary

Pair = Tuple[float, float]


@dataclass
class ReferenceMaterial:
    """A material measured in the scan whose coefficients are known."""

    name: str
    coefficients: Pair          # (neutron μ, X-ray μ), any consistent unit
    measured: Pair              # mean grey value in its region
    spread: Pair                # grey-value σ in its region


@dataclass
class Calibration:
    """``grey = gain · μ + offset`` for each modality."""

    gain: Pair
    offset: Pair
    residual: Pair = (0.0, 0.0)          # RMS misfit of the references
    references: Sequence[str] = field(default_factory=tuple)
    spread: Pair = (0.0, 0.0)

    def predict(self, coefficients: Pair) -> Pair:
        return tuple(float(g * c + o) for g, c, o in
                     zip(self.gain, coefficients, self.offset))

    def describe(self) -> str:
        lines = [f"Calibrated from {', '.join(self.references)}:"]
        for axis, label in enumerate(("neutron", "X-ray")):
            text = (f"  {label}: grey = {self.gain[axis]:.4g} · μ "
                    f"{'+' if self.offset[axis] >= 0 else '−'} "
                    f"{abs(self.offset[axis]):.4g}")
            if len(self.references) > 2:
                text += f"  (RMS misfit {self.residual[axis]:.3g})"
            lines.append(text)
        return "\n".join(lines)


def fit_calibration(references: Sequence[ReferenceMaterial]) -> Calibration:
    """Least-squares gain and offset per modality from the references."""
    references = list(references)
    if len(references) < 2:
        raise ValueError("Calibration needs at least two reference materials")
    coefficients = np.array([r.coefficients for r in references], dtype=float)
    measured = np.array([r.measured for r in references], dtype=float)
    gain, offset, residual = [], [], []
    for axis, label in enumerate(("neutron", "X-ray")):
        x = coefficients[:, axis]
        if np.ptp(x) <= 0:
            raise ValueError(
                f"The reference materials have the same {label} coefficient; "
                "choose two that differ in both modalities")
        design = np.column_stack([x, np.ones_like(x)])
        solution, *_ = np.linalg.lstsq(design, measured[:, axis], rcond=None)
        gain.append(float(solution[0]))
        offset.append(float(solution[1]))
        misfit = measured[:, axis] - design @ solution
        residual.append(float(np.sqrt(np.mean(misfit ** 2))))
    spreads = np.array([r.spread for r in references], dtype=float)
    return Calibration(
        gain=tuple(gain), offset=tuple(offset), residual=tuple(residual),
        references=tuple(r.name for r in references),
        spread=tuple(float(v) for v in np.sqrt(np.mean(spreads ** 2, axis=0))),
    )


def predicted_classes(targets: Dict[str, Pair], calibration: Calibration,
                      spread: Pair = None) -> ClassLibrary:
    """Classes at the calibrated positions of *targets* ``{name: (μn, μx)}``.

    Uses :meth:`ClassLibrary.from_physics`; *spread* defaults to the RMS
    spread of the reference regions.
    """
    spread = calibration.spread if spread is None else spread
    loci = {name: calibration.predict(coefficients)
            for name, coefficients in targets.items()}
    return ClassLibrary.from_physics(loci, {name: spread for name in loci})


def merge_libraries(measured: ClassLibrary, predicted: ClassLibrary) -> ClassLibrary:
    """Measured classes first, then predicted ones not already measured.

    A class drawn in the scan always wins over a prediction of the same
    name: the measurement is the better estimate of where it sits.
    """
    materials = list(measured)
    names = set(measured.names)
    for material in predicted:
        if material.name not in names:
            materials.append(material)
    merged = ClassLibrary(materials)
    merged.dropped = [entry for entry in getattr(measured, "dropped", [])
                      if entry[0] not in set(predicted.names)]
    merged.notes = list(getattr(measured, "notes", []))
    return merged


def reference_from_region(name: str, coefficients: Pair, neutron, xray,
                          mask) -> ReferenceMaterial:
    """Measure a reference material's mean and spread in its region."""
    mask = np.asarray(mask, dtype=bool)
    n = np.asarray(neutron, dtype=np.float64)[mask]
    x = np.asarray(xray, dtype=np.float64)[mask]
    keep = np.isfinite(n) & np.isfinite(x)
    if keep.sum() < 2:
        raise ValueError(f"The region of {name} has no measured voxels")
    n, x = n[keep], x[keep]
    return ReferenceMaterial(
        name=name, coefficients=tuple(float(c) for c in coefficients),
        measured=(float(n.mean()), float(x.mean())),
        spread=(float(n.std()), float(x.std())),
    )


__all__ = ["ReferenceMaterial", "Calibration", "fit_calibration",
           "predicted_classes", "merge_libraries", "reference_from_region"]
