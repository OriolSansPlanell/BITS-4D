"""How a mixture component is allowed to move from one timepoint to the next.

Two extremes bracket the useful range. Re-anchoring every timepoint to T0
(:class:`StaticTransition`) is the fixed-ROI behaviour: perfectly stable and
blind to real change. Fitting each timepoint independently is free to follow
anything, including noise, and loses class identity the moment two
components swap. :class:`DriftTransition` sits between them by making each
timepoint's prior a precision-weighted blend of the T0 anchor and the
previous timepoint's posterior — both Gaussian, so the blend is closed-form
and costs nothing.

Deliberately not implemented
────────────────────────────
Birth-by-novelty and the fuel-cell path from the source plan are absent. For
fuel cells the plan's own advice is the right one and does not need this
machinery: where a dry reference scan exists, water thickness follows
directly from Beer–Lambert, which is quantitative, needs no training, and
handles sub-resolution saturation natively — segmenting first would throw
that away. Component *birth* is also the one part of the design that cannot
be validated without data containing a genuinely new phase, and an untested
birth rule that fires on noise is worse than no birth rule at all.

What is here instead is **dormancy**: a component whose weight collapses is
frozen rather than deleted, so when the phase returns it is the *same*
component, and its time series stays continuous across the gap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from model.mixture import ComponentPrior, MixturePrior


@dataclass
class TransitionDiagnostics:
    timepoint: int
    dormant: List[str] = field(default_factory=list)
    resurrected: List[str] = field(default_factory=list)
    clipped: Dict[str, float] = field(default_factory=dict)


class TemporalModel(ABC):
    """Interface between consecutive timepoints."""

    def __init__(self, min_weight: float = 0.0) -> None:
        self.min_weight = float(min_weight)
        self._dormant: Dict[str, dict] = {}

    @abstractmethod
    def prior_for(self, timepoint: int, base_prior: MixturePrior,
                  previous_result=None, drift=None) -> MixturePrior:
        """The prior to fit *timepoint* under."""

    def post_fit(self, timepoint: int, result,
                 previous_result=None) -> TransitionDiagnostics:
        """Freeze collapsed components and revive returning ones.

        A dormant component keeps the parameters it had when it faded, so a
        phase that disappears and comes back keeps its identity — deleting
        and re-creating it would break every time series drawn from it.
        """
        diagnostics = TransitionDiagnostics(timepoint=timepoint)
        if self.min_weight <= 0:
            return diagnostics

        for index, name in enumerate(result.names):
            weight = float(result.weights[index])
            if weight < self.min_weight:
                if name not in self._dormant:
                    diagnostics.dormant.append(name)
                    self._dormant[name] = {
                        "mean": result.means[index].copy(),
                        "covariance": result.covariances[index].copy(),
                        "timepoint": timepoint,
                    }
                # Hold the parameters it had when it faded, so noise in an
                # empty component cannot wander them off
                stored = self._dormant[name]
                result.means[index] = stored["mean"]
                result.covariances[index] = stored["covariance"]
            elif name in self._dormant:
                diagnostics.resurrected.append(name)
                del self._dormant[name]
        return diagnostics

    @property
    def dormant_classes(self) -> List[str]:
        return sorted(self._dormant)


class StaticTransition(TemporalModel):
    """Re-anchor every timepoint to T0 — the fixed-ROI limit.

    Useful as the control arm: with the drift tracker switched off as well,
    it reproduces the behaviour of segmenting every timepoint with the same
    frozen histogram partition.
    """

    def prior_for(self, timepoint, base_prior, previous_result=None,
                  drift=None) -> MixturePrior:
        return base_prior.scaled(drift)


class DriftTransition(TemporalModel):
    """Fixed component count, components allowed to drift between timepoints.

    Parameters
    ----------
    memory
        How much of the prior comes from the previous timepoint rather than
        from T0. ``0`` re-anchors to T0 every time; ``1`` is a pure random
        walk that never looks back at the manual work; ``0.5`` weighs them
        equally. The T0 term is what stops a slow drift from compounding into
        a component that has walked somewhere else entirely.
    process_noise
        Per-axis variance of the step a component may take between
        timepoints, as returned by
        :func:`model.drift_tracker.estimate_process_noise`. This is the
        *instrumental* noise floor measured on the anchors.
    step_limit_sigma
        A component that moves further than this many process-noise σ in one
        timepoint has almost certainly latched onto a different mode. Its
        movement is clipped back to the limit and recorded, rather than being
        accepted silently.
    reactive_factor
        Multiplier on the step limit for classes that are genuinely expected
        to change. Reactive phases *should* move faster than the noise floor;
        this makes by how much an explicit choice.
    """

    def __init__(
        self,
        memory: float = 0.5,
        process_noise: Optional[Sequence[float]] = None,
        step_limit_sigma: float = 6.0,
        reactive_factor: float = 4.0,
        reactive_classes: Sequence[str] = (),
        min_weight: float = 0.0,
    ) -> None:
        super().__init__(min_weight=min_weight)
        self.memory = float(np.clip(memory, 0.0, 1.0))
        self.process_noise = (
            None if process_noise is None
            else np.asarray(process_noise, dtype=np.float64)
        )
        self.step_limit_sigma = float(step_limit_sigma)
        self.reactive_factor = float(reactive_factor)
        self.reactive_classes = set(reactive_classes)

    def prior_for(self, timepoint, base_prior, previous_result=None,
                  drift=None) -> MixturePrior:
        anchored = base_prior.scaled(drift)
        if previous_result is None or self.memory <= 0:
            return anchored

        previous = {
            name: index for index, name in enumerate(previous_result.names)
        }
        components = []
        for component in anchored.components:
            index = previous.get(component.name)
            if index is None:
                components.append(component)
                continue

            kappa_anchor = component.kappa * (1.0 - self.memory)
            kappa_previous = component.kappa * self.memory
            total_kappa = kappa_anchor + kappa_previous
            if total_kappa <= 0:
                components.append(component)
                continue

            # Two Gaussian pulls combine into one, precision-weighted
            mean = (
                kappa_anchor * component.mean
                + kappa_previous * previous_result.means[index]
            ) / total_kappa
            nu_anchor = component.nu * (1.0 - self.memory)
            nu_previous = component.nu * self.memory
            total_nu = nu_anchor + nu_previous
            covariance = (
                nu_anchor * component.covariance
                + nu_previous * previous_result.covariances[index]
            ) / max(total_nu, 1e-12)

            components.append(
                ComponentPrior(
                    name=component.name,
                    mean=mean,
                    covariance=covariance,
                    kappa=total_kappa,
                    nu=total_nu if total_nu > 0 else component.nu,
                    weight=float(previous_result.weights[index]),
                    count=component.count,
                    fixed=component.fixed,
                    class_id=component.class_id,
                )
            )
        return MixturePrior(
            components=components,
            dirichlet_strength=anchored.dirichlet_strength,
            outlier_weight=anchored.outlier_weight,
        )

    def post_fit(self, timepoint, result,
                 previous_result=None) -> TransitionDiagnostics:
        diagnostics = super().post_fit(timepoint, result, previous_result)
        if previous_result is None or self.process_noise is None:
            return diagnostics

        sigma = np.sqrt(np.maximum(self.process_noise, 1e-30))
        previous = {
            name: index for index, name in enumerate(previous_result.names)
        }
        for index, name in enumerate(result.names):
            source = previous.get(name)
            if source is None:
                continue
            limit = self.step_limit_sigma * (
                self.reactive_factor if name in self.reactive_classes else 1.0
            )
            step = result.means[index] - previous_result.means[source]
            distance = float(np.hypot(*(step / sigma)))
            if distance > limit and distance > 0:
                result.means[index] = (
                    previous_result.means[source] + step * (limit / distance)
                )
                diagnostics.clipped[name] = distance
        return diagnostics
