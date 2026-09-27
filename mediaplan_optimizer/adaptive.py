"""Sequential campaign runner and stable experimental result tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from .bandits import DecisionContext
from .curves import metric_response
from .schemas import ChannelConfig, ObjectiveMetric
from .simulator import CampaignSimulator, ChunkObservation


DAILY_COLUMNS = [
    "policy", "seed", "day", "spend", "impressions", "clicks", "conversions",
    "cumulative_spend", "cumulative_clicks", "cumulative_conversions",
]
CHANNEL_COLUMNS = ["policy", "seed", "channel", "spend", "impressions", "clicks", "conversions"]
ACTION_COLUMNS = ["policy", "seed", "day", "channel", "spend", "impressions", "clicks", "conversions"]


class AdaptivePolicy(Protocol):
    name: str

    def reset(
        self,
        catalog: Mapping[str, ChannelConfig],
        budget: float,
        horizon_days: int,
        objective: ObjectiveMetric,
        seed: int,
    ) -> None: ...

    def channel_belief(self, channel: str) -> ChannelConfig: ...
    def allocation_limit(self, context: DecisionContext) -> float: ...
    def select_channel(self, contexts: list[DecisionContext]) -> str: ...
    def update(self, context: DecisionContext, observation: ChunkObservation) -> None: ...


@dataclass(frozen=True)
class CampaignRunConfig:
    budget: float = 1_200_000.0
    horizon_days: int = 21
    objective: ObjectiveMetric = ObjectiveMetric.CONVERSIONS
    quantum_rub: float = 10_000.0
    seed: int = 42

    def __post_init__(self) -> None:
        if (not np.isfinite([self.budget, self.horizon_days, self.quantum_rub]).all()
                or self.budget <= 0 or self.horizon_days <= 0 or self.quantum_rub <= 0):
            raise ValueError("budget, horizon_days, and quantum_rub must be positive")
        if self.objective not in {ObjectiveMetric.CLICKS, ObjectiveMetric.CONVERSIONS}:
            raise ValueError("adaptive objective must be clicks or conversions")


@dataclass
class AdaptiveRunResult:
    policy: str
    seed: int
    objective: ObjectiveMetric
    daily: pd.DataFrame
    channels: pd.DataFrame
    actions: pd.DataFrame
    total_spend: float
    total_clicks: int
    total_conversions: int
    total_kpi: float
    cpc: float | None
    cpa: float | None
    oracle_kpi: float | None = None
    regret_to_oracle: float | None = None


def _incremental_expected_kpi(
    channel: ChannelConfig,
    objective: ObjectiveMetric,
    spend_before: float,
    increment: float,
) -> float:
    before = float(metric_response(spend_before, channel, objective))
    after = float(metric_response(spend_before + increment, channel, objective))
    return max(0.0, after - before)


def _context_vector(
    day: int,
    horizon_days: int,
    channel: ChannelConfig,
    daily_spend: float,
    cumulative_spend: float,
    remaining_budget: float,
    total_budget: float,
) -> np.ndarray:
    return np.array(
        [
            1.0,
            (day - 1) / max(1, horizon_days - 1),
            float((day - 1) % 7 in (5, 6)),
            daily_spend / channel.max_daily_spend,
            cumulative_spend / (channel.max_daily_spend * horizon_days),
            remaining_budget / total_budget,
            (horizon_days - day + 1) / horizon_days,
        ],
        dtype=float,
    )


def run_adaptive_campaign(
    policy: AdaptivePolicy,
    catalog: Mapping[str, ChannelConfig],
    config: CampaignRunConfig,
    simulator: CampaignSimulator | None = None,
) -> AdaptiveRunResult:
    """Execute one seeded campaign without exposing simulator truth to policy."""
    catalog = dict(catalog)
    environment = simulator or CampaignSimulator(
        catalog, config.horizon_days, config.seed
    )
    policy.reset(
        catalog,
        config.budget,
        config.horizon_days,
        config.objective,
        config.seed + 7919,
    )

    cumulative_spend = {name: 0.0 for name in catalog}
    action_rows: list[dict[str, float | int | str]] = []
    daily_rows: list[dict[str, float | int | str]] = []
    spent = 0.0
    total_clicks = 0
    total_conversions = 0

    for day in range(1, config.horizon_days + 1):
        remaining_days = config.horizon_days - day + 1
        remaining_budget = max(0.0, config.budget - spent)
        daily_available = remaining_budget / remaining_days
        if hasattr(policy, "on_day_start"):
            policy.on_day_start(day, remaining_budget, remaining_days)  # type: ignore[attr-defined]
        daily_spend = {name: 0.0 for name in catalog}
        daily_clicks = daily_conversions = daily_impressions = 0
        daily_total = 0.0

        while daily_available - daily_total > 1e-8:
            available_contexts: list[DecisionContext] = []
            for name, channel in catalog.items():
                cap_remaining = channel.max_daily_spend - daily_spend[name]
                base_proposal = min(
                    config.quantum_rub,
                    daily_available - daily_total,
                    cap_remaining,
                )
                if base_proposal <= 1e-8:
                    continue
                belief = policy.channel_belief(name)
                vector = _context_vector(
                    day,
                    config.horizon_days,
                    channel,
                    daily_spend[name],
                    cumulative_spend[name],
                    max(0.0, config.budget - spent),
                    config.budget,
                )
                provisional = DecisionContext(
                    channel=name,
                    day=day,
                    vector=vector,
                    proposed_spend=base_proposal,
                    expected_kpi=0.0,
                    marginal_expected_kpi_per_rub=0.0,
                    daily_spend=daily_spend[name],
                    cumulative_spend=cumulative_spend[name],
                )
                proposal = min(base_proposal, policy.allocation_limit(provisional))
                if proposal <= 1e-8:
                    continue
                expected = _incremental_expected_kpi(
                    belief, config.objective, daily_spend[name], proposal
                )
                available_contexts.append(
                    DecisionContext(
                        channel=name,
                        day=day,
                        vector=vector,
                        proposed_spend=proposal,
                        expected_kpi=expected,
                        marginal_expected_kpi_per_rub=expected / proposal,
                        daily_spend=daily_spend[name],
                        cumulative_spend=cumulative_spend[name],
                    )
                )

            if not available_contexts:
                break
            chosen_name = policy.select_channel(available_contexts)
            context = next(
                item for item in available_contexts if item.channel == chosen_name
            )
            observation = environment.simulate_chunk(
                chosen_name,
                day,
                context.proposed_spend,
                daily_spend[chosen_name],
                cumulative_spend[chosen_name],
            )
            policy.update(context, observation)
            daily_spend[chosen_name] += observation.spend
            cumulative_spend[chosen_name] += observation.spend
            daily_total += observation.spend
            spent += observation.spend
            daily_impressions += observation.impressions
            daily_clicks += observation.clicks
            daily_conversions += observation.conversions
            total_clicks += observation.clicks
            total_conversions += observation.conversions
            action_rows.append(
                {
                    "policy": policy.name,
                    "seed": config.seed,
                    "day": day,
                    "channel": chosen_name,
                    "spend": observation.spend,
                    "impressions": observation.impressions,
                    "clicks": observation.clicks,
                    "conversions": observation.conversions,
                }
            )

        daily_rows.append(
            {
                "policy": policy.name,
                "seed": config.seed,
                "day": day,
                "spend": daily_total,
                "impressions": daily_impressions,
                "clicks": daily_clicks,
                "conversions": daily_conversions,
                "cumulative_spend": spent,
                "cumulative_clicks": total_clicks,
                "cumulative_conversions": total_conversions,
            }
        )

    actions = pd.DataFrame(action_rows, columns=ACTION_COLUMNS)
    daily = pd.DataFrame(daily_rows, columns=DAILY_COLUMNS)
    if actions.empty:
        channels = pd.DataFrame(columns=CHANNEL_COLUMNS)
    else:
        channels = (
            actions.groupby(["policy", "seed", "channel"], as_index=False)[
                ["spend", "impressions", "clicks", "conversions"]
            ]
            .sum()
            .reindex(columns=CHANNEL_COLUMNS)
        )
    total_kpi = (
        float(total_clicks)
        if config.objective == ObjectiveMetric.CLICKS
        else float(total_conversions)
    )
    return AdaptiveRunResult(
        policy=policy.name,
        seed=config.seed,
        objective=config.objective,
        daily=daily,
        channels=channels,
        actions=actions,
        total_spend=spent,
        total_clicks=total_clicks,
        total_conversions=total_conversions,
        total_kpi=total_kpi,
        cpc=spent / total_clicks if total_clicks else None,
        cpa=spent / total_conversions if total_conversions else None,
    )
