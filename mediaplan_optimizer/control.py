"""Campaign replanning, policy diagnostics, and sequential simulated execution."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .adaptive import _context_vector, _incremental_expected_kpi
from .bandits import (
    DecisionContext,
    LinUCBPolicy,
    PeriodicReoptimizationPolicy,
    StaticPolicy,
    ThompsonSamplingPolicy,
)
from .campaign import (
    CampaignObservation,
    CampaignState,
    add_observations,
    add_plan_version,
    future_spend_scale,
    record_forecast,
)
from .calendar import effective_channel
from .calendar_optimizer import optimize_type_a_calendar
from .curves import channel_response, marginal_metric_response
from .metrics import build_calendar_allocations, build_daily_allocations
from .monitoring import build_monitoring, estimate_channel_beliefs
from .optimizer import optimize_type_a
from .schemas import ChannelConfig, MediaPlanResult, ObjectiveMetric, PlanningMode
from .simulator import CampaignSimulator, ChunkObservation


PolicyRuntime = (
    StaticPolicy
    | PeriodicReoptimizationPolicy
    | ThompsonSamplingPolicy
    | LinUCBPolicy
)


def create_policy_runtime(state: CampaignState, *, seed: int = 42) -> PolicyRuntime:
    """Initialize the selected policy and replay only observed feedback."""
    if state.objective == ObjectiveMetric.REACH and state.current_policy in {
        "thompson_sampling",
        "linucb",
    }:
        raise ValueError("adaptive policies support clicks or conversions, not reach")
    if state.current_policy == "static":
        policy: PolicyRuntime = StaticPolicy()
    elif state.current_policy == "periodic_reoptimization":
        policy = PeriodicReoptimizationPolicy(
            interval_days=state.periodic_interval_days
        )
    elif state.current_policy == "thompson_sampling":
        policy = ThompsonSamplingPolicy()
    elif state.current_policy == "linucb":
        policy = LinUCBPolicy()
    else:
        raise ValueError(f"unknown policy: {state.current_policy}")
    policy.reset(
        state.planning_catalog,
        state.original_budget,
        state.horizon_days,
        state.objective,
        seed + 7919,
    )
    _replay_observations(policy, state)
    return policy


def _replay_observations(policy: PolicyRuntime, state: CampaignState) -> None:
    cumulative = {name: 0.0 for name in state.planning_catalog}
    daily = {name: 0.0 for name in state.planning_catalog}
    current_day = 0
    total_spend = 0.0
    for observation in state.observations:
        if observation.day != current_day:
            current_day = observation.day
            daily = {name: 0.0 for name in state.planning_catalog}
            if hasattr(policy, "on_day_start"):
                policy.on_day_start(  # type: ignore[attr-defined]
                    current_day,
                    max(0.0, state.original_budget - total_spend),
                    state.horizon_days - current_day + 1,
                )
        channel = state.planning_catalog[observation.channel]
        belief = policy.channel_belief(observation.channel)
        expected = _incremental_expected_kpi(
            belief,
            state.objective,
            daily[observation.channel],
            observation.spend,
        )
        context = DecisionContext(
            channel=observation.channel,
            day=observation.day,
            vector=_context_vector(
                observation.day,
                state.horizon_days,
                channel,
                daily[observation.channel],
                cumulative[observation.channel],
                max(0.0, state.original_budget - total_spend),
                state.original_budget,
            ),
            proposed_spend=observation.spend,
            expected_kpi=expected,
            marginal_expected_kpi_per_rub=(
                expected / observation.spend if observation.spend > 0 else 0.0
            ),
            daily_spend=daily[observation.channel],
            cumulative_spend=cumulative[observation.channel],
        )
        policy.update(
            context,
            ChunkObservation(
                day=observation.day,
                channel=observation.channel,
                spend=observation.spend,
                impressions=observation.impressions,  # type: ignore[arg-type]
                clicks=observation.clicks,  # type: ignore[arg-type]
                conversions=observation.conversions,  # type: ignore[arg-type]
            ),
        )
        daily[observation.channel] += observation.spend
        cumulative[observation.channel] += observation.spend
        total_spend += observation.spend


def _next_contexts(
    state: CampaignState,
    policy: PolicyRuntime,
    *,
    day: int,
    quantum_rub: float,
    daily_spend: Mapping[str, float] | None = None,
) -> list[DecisionContext]:
    spent_today = dict(daily_spend or {name: 0.0 for name in state.planning_catalog})
    cumulative = {
        name: sum(item.spend for item in state.observations if item.channel == name)
        for name in state.planning_catalog
    }
    contexts: list[DecisionContext] = []
    for name, channel in state.planning_catalog.items():
        proposal = min(quantum_rub, state.day_cap(day, name) - spent_today[name])
        if proposal <= 1e-8:
            continue
        belief = policy.channel_belief(name)
        expected = _incremental_expected_kpi(
            belief, state.objective, spent_today[name], proposal
        )
        contexts.append(
            DecisionContext(
                channel=name,
                day=day,
                vector=_context_vector(
                    day,
                    state.horizon_days,
                    channel,
                    spent_today[name],
                    cumulative[name] + spent_today[name],
                    max(0.0, state.remaining_budget - sum(spent_today.values())),
                    state.original_budget,
                ),
                proposed_spend=proposal,
                expected_kpi=expected,
                marginal_expected_kpi_per_rub=expected / proposal,
                daily_spend=spent_today[name],
                cumulative_spend=cumulative[name] + spent_today[name],
            )
        )
    return contexts


def policy_diagnostics(
    state: CampaignState,
    policy: PolicyRuntime,
    *,
    quantum_rub: float = 10_000.0,
) -> pd.DataFrame:
    day = min(state.horizon_days, state.current_day + 1)
    contexts = _next_contexts(state, policy, day=day, quantum_rub=quantum_rub)
    rows: list[dict[str, float | int | str | bool | None]] = []
    for context in contexts:
        posterior_mean: float | None = None
        uncertainty = 0.0
        context_contribution: float | None = None
        exploration_bonus = 0.0
        if isinstance(policy, ThompsonSamplingPolicy):
            alpha = policy.alpha[context.channel]
            beta = policy.beta[context.channel]
            posterior_mean = alpha / (alpha + beta)
            uncertainty = float(
                np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
            )
            multiplier = posterior_mean / policy.base_probability[context.channel]
            score = multiplier * context.marginal_expected_kpi_per_rub
        elif isinstance(policy, LinUCBPolicy):
            matrix = policy.A[context.channel]
            theta = np.linalg.solve(matrix, policy.b[context.channel])
            context_contribution = float(theta @ context.vector)
            uncertainty = float(
                np.sqrt(
                    max(
                        0.0,
                        context.vector
                        @ np.linalg.solve(matrix, context.vector),
                    )
                )
            )
            exploration_bonus = policy.exploration_alpha * uncertainty
            score = (
                max(0.0, context_contribution) + exploration_bonus
            ) * context.marginal_expected_kpi_per_rub
        else:
            belief = policy.channel_belief(context.channel)
            posterior_mean = (
                belief.base_ctr
                if state.objective == ObjectiveMetric.CLICKS
                else belief.base_ctr * belief.base_cr
            )
            score = context.marginal_expected_kpi_per_rub
        rows.append(
            {
                "channel": context.channel,
                "posterior_mean": posterior_mean,
                "uncertainty": uncertainty,
                "context_contribution": context_contribution,
                "exploration_bonus": exploration_bonus,
                "marginal_kpi_per_rub": context.marginal_expected_kpi_per_rub,
                "score": score,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["rank"] = result["score"].rank(method="first", ascending=False).astype(int)
        result["recommended"] = result["rank"] == 1
        result = result.sort_values("rank").reset_index(drop=True)
    return result


def _policy_adjusted_catalog(
    state: CampaignState, policy: PolicyRuntime
) -> dict[str, ChannelConfig]:
    if isinstance(policy, PeriodicReoptimizationPolicy):
        return estimate_channel_beliefs(state)
    if isinstance(policy, StaticPolicy):
        return dict(state.planning_catalog)
    diagnostics = policy_diagnostics(state, policy)
    by_channel = diagnostics.set_index("channel") if not diagnostics.empty else None
    adjusted: dict[str, ChannelConfig] = {}
    for name, channel in state.planning_catalog.items():
        multiplier = 1.0
        if by_channel is not None and name in by_channel.index:
            if isinstance(policy, ThompsonSamplingPolicy):
                probability = float(by_channel.loc[name, "posterior_mean"])
                multiplier = probability / policy.base_probability[name]
            elif isinstance(policy, LinUCBPolicy):
                mean = float(by_channel.loc[name, "context_contribution"])
                bonus = float(by_channel.loc[name, "exploration_bonus"])
                multiplier = max(0.05, mean + bonus)
        multiplier = float(np.clip(multiplier, 0.25, 4.0))
        if state.objective == ObjectiveMetric.CLICKS:
            adjusted[name] = channel.model_copy(
                update={"base_ctr": min(1.0, channel.base_ctr * multiplier)}
            )
        elif state.objective == ObjectiveMetric.CONVERSIONS:
            adjusted[name] = channel.model_copy(
                update={"base_cr": min(1.0, channel.base_cr * multiplier)}
            )
        else:
            adjusted[name] = channel
    return adjusted


def recalculate_remaining_plan(
    state: CampaignState,
    policy: PolicyRuntime | None = None,
    *,
    reason: str = "Manual remaining-plan recalculation",
) -> CampaignState:
    """Optimize only future days and append a new immutable plan version."""
    if state.remaining_days <= 0:
        raise ValueError("completed campaign cannot be replanned")
    runtime = policy or create_policy_runtime(state)
    catalog = _policy_adjusted_catalog(state, runtime)
    if state.remaining_budget <= 1e-8:
        future = ()
    elif state.original_request.planning_mode == PlanningMode.CALENDAR_AWARE:
        future_profiles = tuple(
            profile for profile in state.temporal_profiles if profile.day > state.current_day
        )
        outcome = optimize_type_a_calendar(
            catalog, future_profiles, state.objective, state.remaining_budget,
        )
        future = tuple(build_calendar_allocations(
            outcome.cell_spend, future_profiles, catalog, state.objective,
        ))
    else:
        outcome = optimize_type_a(
            catalog,
            state.remaining_days,
            state.objective,
            state.remaining_budget,
        )
        local = build_daily_allocations(
            outcome.channel_spend,
            state.remaining_days,
            catalog,
            state.objective,
        )
        future = tuple(
            allocation.model_copy(
                update={"day": allocation.day + state.current_day}
            )
            for allocation in local
        )
    revised = add_plan_version(
        state,
        future,
        reason=reason,
        policy=state.current_policy,
    )
    if revised is state:
        return state
    monitoring = build_monitoring(revised)
    progress = monitoring.progress
    return record_forecast(
        revised,
        reason=reason,
        actual_kpi=progress.actual_kpi_to_date,
        projected_final_kpi=progress.projected_final_kpi,
        projected_cost=progress.projected_unit_cost,
    )


def ingest_and_record(
    state: CampaignState, observations: Iterable[CampaignObservation], reason: str
) -> CampaignState:
    """Append fact and record the resulting forecast."""
    updated = add_observations(state, observations)
    progress = build_monitoring(updated).progress
    return record_forecast(
        updated,
        reason=reason,
        actual_kpi=progress.actual_kpi_to_date,
        projected_final_kpi=progress.projected_final_kpi,
        projected_cost=progress.projected_unit_cost,
    )


def _simulate_planned_day(
    state: CampaignState,
    simulator: CampaignSimulator,
    day: int,
) -> tuple[CampaignObservation, ...]:
    cumulative = {
        name: sum(item.spend for item in state.observations if item.channel == name)
        for name in state.planning_catalog
    }
    rows = [row for row in state.latest_plan.future_allocations if row.day == day]
    scale = future_spend_scale(state)
    observations: list[CampaignObservation] = []
    for allocation in rows:
        if allocation.spend <= 1e-8:
            continue
        chunk = simulator.simulate_chunk(
            allocation.channel,
            day,
            allocation.spend * scale,
            0.0,
            cumulative[allocation.channel],
            daily_cap_override=state.day_cap(day, allocation.channel),
        )
        observations.append(
            CampaignObservation(
                day=day,
                date=allocation.date,
                channel=chunk.channel,
                spend=chunk.spend,
                impressions=chunk.impressions,
                clicks=chunk.clicks,
                conversions=chunk.conversions,
                source="simulator",
            )
        )
    return tuple(observations)


def _simulate_adaptive_day(
    state: CampaignState,
    simulator: CampaignSimulator,
    policy: PolicyRuntime,
    day: int,
    quantum_rub: float,
) -> tuple[CampaignObservation, ...]:
    remaining_days = state.horizon_days - day + 1
    daily_available = state.remaining_budget / remaining_days
    daily_spend = {name: 0.0 for name in state.planning_catalog}
    cumulative = {
        name: sum(item.spend for item in state.observations if item.channel == name)
        for name in state.planning_catalog
    }
    totals = {
        name: {"spend": 0.0, "impressions": 0.0, "clicks": 0.0, "conversions": 0.0}
        for name in state.planning_catalog
    }
    spent = 0.0
    while daily_available - spent > 1e-8:
        contexts = _next_contexts(
            state,
            policy,
            day=day,
            quantum_rub=min(quantum_rub, daily_available - spent),
            daily_spend=daily_spend,
        )
        contexts = [
            context
            for context in contexts
            if context.proposed_spend <= daily_available - spent + 1e-8
        ]
        if not contexts:
            break
        chosen = policy.select_channel(contexts)
        context = next(item for item in contexts if item.channel == chosen)
        chunk = simulator.simulate_chunk(
            chosen,
            day,
            context.proposed_spend,
            daily_spend[chosen],
            cumulative[chosen] + daily_spend[chosen],
            daily_cap_override=state.day_cap(day, chosen),
        )
        policy.update(context, chunk)
        daily_spend[chosen] += chunk.spend
        spent += chunk.spend
        totals[chosen]["spend"] += chunk.spend
        totals[chosen]["impressions"] += chunk.impressions
        totals[chosen]["clicks"] += chunk.clicks
        totals[chosen]["conversions"] += chunk.conversions
    return tuple(
        CampaignObservation(
            day=day,
            date=state.day_date(day),
            channel=name,
            spend=values["spend"],
            impressions=values["impressions"],
            clicks=values["clicks"],
            conversions=values["conversions"],
            source="simulator",
        )
        for name, values in totals.items()
        if values["spend"] > 0
    )


def simulate_next_days(
    state: CampaignState,
    simulator: CampaignSimulator,
    policy: PolicyRuntime,
    *,
    days: int,
    quantum_rub: float = 10_000.0,
) -> tuple[CampaignState, PolicyRuntime]:
    """Advance sequentially without duplicate days or historical mutation."""
    if days <= 0 or not np.isfinite(quantum_rub) or quantum_rub <= 0:
        raise ValueError("days and quantum_rub must be positive")
    if state.objective == ObjectiveMetric.REACH:
        raise ValueError("simulator does not generate reach; upload observed reach instead")
    if getattr(policy, "name", None) != state.current_policy:
        policy = create_policy_runtime(state, seed=simulator.seed)
    updated = state
    for _ in range(min(days, updated.remaining_days)):
        if updated.remaining_budget <= 1e-8:
            break
        day = updated.current_day + 1
        if (
            updated.current_policy == "periodic_reoptimization"
            and day != 1
            and (day - 1) % updated.periodic_interval_days == 0
        ):
            updated = recalculate_remaining_plan(
                updated,
                policy,
                reason=f"Scheduled replan before day {day}",
            )
            policy = create_policy_runtime(updated, seed=simulator.seed)
        if updated.current_policy in {"thompson_sampling", "linucb"}:
            observations = _simulate_adaptive_day(
                updated, simulator, policy, day, quantum_rub
            )
        else:
            observations = _simulate_planned_day(updated, simulator, day)
        if not observations:
            break
        updated = add_observations(updated, observations)
        progress = build_monitoring(updated).progress
        updated = record_forecast(
            updated,
            reason=f"Observed day {day}",
            actual_kpi=progress.actual_kpi_to_date,
            projected_final_kpi=progress.projected_final_kpi,
            projected_cost=progress.projected_unit_cost,
        )
    return updated, policy


def allocation_explanations(
    result: MediaPlanResult,
    catalog: Mapping[str, ChannelConfig],
) -> pd.DataFrame:
    """Derive deterministic allocation reasons from actual response economics."""
    horizon = result.request.horizon_days
    metric = result.request.objective_metric
    by_channel = {item.channel: item for item in result.channel_metrics}
    planned = {(row.day, row.channel): row for row in result.allocations}
    rows: list[dict[str, float | str]] = []
    for name, channel in catalog.items():
        metrics = by_channel.get(name)
        total_spend = 0.0 if metrics is None else metrics.spend
        if result.request.planning_mode == PlanningMode.CALENDAR_AWARE and metrics is not None:
            profiles = [p for p in result.temporal_profiles if p.channel == name]
            effective = [effective_channel(channel, p) for p in profiles]
            spends = [planned[(p.day, name)].spend for p in profiles]
            capacity = sum(item.daily_capacity for item in effective)
            impressions = sum(float(channel_response(spend, item).impressions)
                              for spend, item in zip(spends, effective, strict=True))
            saturation = impressions / capacity if capacity > 0 else 0.0
            marginal = max(marginal_metric_response(spend, item, metric)
                           for spend, item in zip(spends, effective, strict=True))
            cap = sum(item.max_daily_spend for item in effective)
        else:
            daily_spend = total_spend / horizon
            response = channel_response(daily_spend, channel)
            saturation = float(response.impressions) / channel.daily_capacity
            marginal = marginal_metric_response(daily_spend, channel, metric)
            cap = channel.max_daily_spend * horizon
        rows.append(
            {
                "channel": name,
                "planned_spend": total_spend,
                "inventory_saturation": saturation,
                "marginal_kpi_per_rub": marginal,
                "horizon_spend_cap": cap,
                "eligible": name in by_channel,
            }
        )
    result_frame = pd.DataFrame(rows)
    eligible = result_frame["eligible"]
    best_marginal = max(float(result_frame.loc[eligible, "marginal_kpi_per_rub"].max()), 1e-15)
    result_frame["relative_marginal_efficiency"] = (
        result_frame["marginal_kpi_per_rub"] / best_marginal
    ).where(eligible)
    result_frame["efficiency_rank"] = (
        result_frame["marginal_kpi_per_rub"]
        .where(eligible)
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    def explain(row: pd.Series) -> str:
        if not row["eligible"]:
            return "Excluded by the request's include/exclude channel selection."
        tolerance = max(0.01, float(row["horizon_spend_cap"]) * 1e-8)
        relative = float(row["relative_marginal_efficiency"])
        if float(row["planned_spend"]) >= float(row["horizon_spend_cap"]) - tolerance:
            return "Spend is capped by the configured daily inventory limit."
        if float(row["planned_spend"]) <= 1e-8:
            return (
                "Not funded: initial marginal efficiency is "
                f"{relative:.0%} of the strongest current alternative."
            )
        if float(row["marginal_kpi_per_rub"]) > 0:
            return (
                "Funded while marginal KPI per RUB remains competitive "
                f"(current efficiency rank {int(row['efficiency_rank'])})."
            )
        return "No additional KPI is expected at this saturation level."

    result_frame["explanation"] = result_frame.apply(explain, axis=1)
    return result_frame.sort_values("planned_spend", ascending=False)
