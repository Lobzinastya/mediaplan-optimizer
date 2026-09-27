"""Compact adaptive policies for sequential media-budget allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .optimizer import optimize_type_a
from .schemas import ChannelConfig, ObjectiveMetric
from .simulator import ChunkObservation


@dataclass(frozen=True)
class DecisionContext:
    channel: str
    day: int
    vector: np.ndarray
    proposed_spend: float
    expected_kpi: float
    marginal_expected_kpi_per_rub: float
    daily_spend: float
    cumulative_spend: float


class StaticPolicy:
    """Execute the original Type A plan without learning."""

    name = "static"

    def __init__(
        self,
        planning_catalog: Mapping[str, ChannelConfig] | None = None,
        name: str | None = None,
    ) -> None:
        self._planning_catalog = dict(planning_catalog) if planning_catalog else None
        if name is not None:
            self.name = name

    def reset(
        self,
        catalog: Mapping[str, ChannelConfig],
        budget: float,
        horizon_days: int,
        objective: ObjectiveMetric,
        seed: int,
    ) -> None:
        del seed
        self.catalog = dict(self._planning_catalog or catalog)
        self.horizon_days = horizon_days
        outcome = optimize_type_a(self.catalog, horizon_days, objective, budget)
        self.daily_quota = {
            name: spend / horizon_days for name, spend in outcome.channel_spend.items()
        }

    def channel_belief(self, channel: str) -> ChannelConfig:
        return self.catalog[channel]

    def allocation_limit(self, context: DecisionContext) -> float:
        return max(0.0, self.daily_quota.get(context.channel, 0.0) - context.daily_spend)

    def select_channel(self, contexts: list[DecisionContext]) -> str:
        return max(
            contexts,
            key=lambda item: self.allocation_limit(item),
        ).channel

    def update(self, context: DecisionContext, observation: ChunkObservation) -> None:
        del context, observation


class PeriodicReoptimizationPolicy:
    """Update observed rates and re-run the static planner every N days."""

    name = "periodic_reoptimization"

    def __init__(
        self,
        interval_days: int = 3,
        prior_impressions: float = 20_000.0,
        prior_clicks: float = 500.0,
    ) -> None:
        if interval_days <= 0:
            raise ValueError("interval_days must be positive")
        self.interval_days = interval_days
        self.prior_impressions = prior_impressions
        self.prior_clicks = prior_clicks

    def reset(
        self,
        catalog: Mapping[str, ChannelConfig],
        budget: float,
        horizon_days: int,
        objective: ObjectiveMetric,
        seed: int,
    ) -> None:
        del seed
        self.base_catalog = dict(catalog)
        self.catalog = dict(catalog)
        self.objective = objective
        self.horizon_days = horizon_days
        self.daily_quota = {name: 0.0 for name in catalog}
        self.totals = {
            name: {"impressions": 0, "clicks": 0, "conversions": 0}
            for name in catalog
        }

    def _update_beliefs(self) -> None:
        updated: dict[str, ChannelConfig] = {}
        for name, base in self.base_catalog.items():
            totals = self.totals[name]
            ctr = (
                base.base_ctr * self.prior_impressions + totals["clicks"]
            ) / (self.prior_impressions + totals["impressions"])
            cr = (
                base.base_cr * self.prior_clicks + totals["conversions"]
            ) / (self.prior_clicks + totals["clicks"])
            updated[name] = base.model_copy(
                update={"base_ctr": float(ctr), "base_cr": float(cr)}
            )
        self.catalog = updated

    def on_day_start(
        self, day: int, remaining_budget: float, remaining_days: int
    ) -> None:
        if day != 1 and (day - 1) % self.interval_days != 0:
            return
        if day != 1:
            self._update_beliefs()
        outcome = optimize_type_a(
            self.catalog,
            remaining_days,
            self.objective,
            remaining_budget,
        )
        self.daily_quota = {
            name: spend / remaining_days for name, spend in outcome.channel_spend.items()
        }

    def channel_belief(self, channel: str) -> ChannelConfig:
        return self.catalog[channel]

    def allocation_limit(self, context: DecisionContext) -> float:
        return max(0.0, self.daily_quota.get(context.channel, 0.0) - context.daily_spend)

    def select_channel(self, contexts: list[DecisionContext]) -> str:
        return max(contexts, key=lambda item: self.allocation_limit(item)).channel

    def update(self, context: DecisionContext, observation: ChunkObservation) -> None:
        totals = self.totals[context.channel]
        totals["impressions"] += observation.impressions
        totals["clicks"] += observation.clicks
        totals["conversions"] += observation.conversions


class ThompsonSamplingPolicy:
    """Beta-Bernoulli performance learning combined with marginal economics."""

    name = "thompson_sampling"

    def __init__(self, prior_strength: float = 5_000.0) -> None:
        if not np.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        self.prior_strength = prior_strength

    def reset(
        self,
        catalog: Mapping[str, ChannelConfig],
        budget: float,
        horizon_days: int,
        objective: ObjectiveMetric,
        seed: int,
    ) -> None:
        del budget, horizon_days
        self.catalog = dict(catalog)
        self.objective = objective
        self.rng = np.random.default_rng(seed)
        self.base_probability: dict[str, float] = {}
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}
        for name, channel in catalog.items():
            probability = channel.base_ctr
            if objective == ObjectiveMetric.CONVERSIONS:
                probability *= channel.base_cr
            # Proper Beta parameters even for valid boundary-rate catalogs.
            probability = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
            self.base_probability[name] = probability
            self.alpha[name] = probability * self.prior_strength
            self.beta[name] = (1.0 - probability) * self.prior_strength

    def channel_belief(self, channel: str) -> ChannelConfig:
        return self.catalog[channel]

    def allocation_limit(self, context: DecisionContext) -> float:
        del context
        return float("inf")

    def select_channel(self, contexts: list[DecisionContext]) -> str:
        scores: dict[str, float] = {}
        for context in contexts:
            sampled_probability = self.rng.beta(
                self.alpha[context.channel], self.beta[context.channel]
            )
            multiplier = sampled_probability / self.base_probability[context.channel]
            scores[context.channel] = (
                multiplier * context.marginal_expected_kpi_per_rub
            )
        return max(contexts, key=lambda item: scores[item.channel]).channel

    def update(self, context: DecisionContext, observation: ChunkObservation) -> None:
        successes = (
            observation.clicks
            if self.objective == ObjectiveMetric.CLICKS
            else observation.conversions
        )
        failures = max(0, observation.impressions - successes)
        self.alpha[context.channel] += successes
        self.beta[context.channel] += failures


class LinUCBPolicy:
    """Disjoint LinUCB over dimensionless performance multipliers."""

    name = "linucb"

    def __init__(self, alpha: float = 0.7, regularization: float = 1.0) -> None:
        if not np.isfinite([alpha, regularization]).all() or alpha < 0 or regularization <= 0:
            raise ValueError("invalid LinUCB hyperparameters")
        self.exploration_alpha = alpha
        self.regularization = regularization

    def reset(
        self,
        catalog: Mapping[str, ChannelConfig],
        budget: float,
        horizon_days: int,
        objective: ObjectiveMetric,
        seed: int,
    ) -> None:
        del budget, horizon_days, seed
        self.catalog = dict(catalog)
        self.objective = objective
        self.dimension = 7
        self.A = {
            name: np.eye(self.dimension, dtype=float) * self.regularization
            for name in catalog
        }
        self.b = {name: np.zeros(self.dimension, dtype=float) for name in catalog}

    def channel_belief(self, channel: str) -> ChannelConfig:
        return self.catalog[channel]

    def allocation_limit(self, context: DecisionContext) -> float:
        del context
        return float("inf")

    def select_channel(self, contexts: list[DecisionContext]) -> str:
        scores: dict[str, float] = {}
        for context in contexts:
            matrix = self.A[context.channel]
            theta = np.linalg.solve(matrix, self.b[context.channel])
            solved_context = np.linalg.solve(matrix, context.vector)
            uncertainty = float(
                np.sqrt(max(0.0, context.vector @ solved_context))
            )
            multiplier_score = max(0.0, float(theta @ context.vector))
            multiplier_score += self.exploration_alpha * uncertainty
            scores[context.channel] = (
                multiplier_score * context.marginal_expected_kpi_per_rub
            )
        return max(contexts, key=lambda item: scores[item.channel]).channel

    def update(self, context: DecisionContext, observation: ChunkObservation) -> None:
        observed_kpi = (
            observation.clicks
            if self.objective == ObjectiveMetric.CLICKS
            else observation.conversions
        )
        if context.expected_kpi <= 1e-12:
            return
        reward_multiplier = observed_kpi / context.expected_kpi
        vector = context.vector
        self.A[context.channel] += np.outer(vector, vector)
        self.b[context.channel] += reward_multiplier * vector
