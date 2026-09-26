"""Reproducible validation of BiTS 4D against known ground truth.

* :mod:`validation.phantom` — synthetic 4-D paired neutron/X-ray series with
  exact labels at every timepoint, including a phase that appears during
  the experiment, partial volume, noise and optional artefacts.
* :mod:`validation.baselines` — the methods BiTS 4D is compared against.
* :mod:`validation.scoring` — Dice, volume error, and oracle class matching.
* :mod:`validation.run` — the benchmark runner (``python -m validation.run``).

Everything is seeded, so every number in ``docs/validation.md`` can be
regenerated exactly from this package.
"""
