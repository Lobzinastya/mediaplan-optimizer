"""Feasibility checks and counterfactuals for KPI-driven requests."""

from __future__ import annotations

import math
from datetime import date as Date
from typing import Mapping

from .calendar import generate_day_profiles
from .calendar_optimizer import calendar_maximum, optimize_type_b_calendar
from .curves import metric_response
from .optimizer import optimize_type_b
from .schemas import (
    ChannelConfig,
    FeasibilityReason,
    FeasibilityResult,
    ObjectiveMetric,
    PlanningMode,
    DEFAULT_CALENDAR_START,
)


def maximum_achievable(
    catalog: Mapping[str, ChannelConfig],
    horizon_days: int,
    metric: ObjectiveMetric,
    *,
    planning_mode: PlanningMode = PlanningMode.UNIFORM,
    start_date: Date | None = None,
    calendar_profile_seed: int = 42,
) -> float:
    """Practical maximum at configured daily spend limits."""
    if planning_mode == PlanningMode.CALENDAR_AWARE:
        profiles = generate_day_profiles(
            catalog, start_date or DEFAULT_CALENDAR_START, horizon_days,
            calendar_profile_seed,
        )
        return calendar_maximum(catalog, profiles, metric)
    return float(
        horizon_days
        * sum(metric_response(c.max_daily_spend, c, metric) for c in catalog.values())
    )


def _safe_recommended_target(maximum: float) -> float | None:
    """Round down to three significant digits below practical capacity."""
    if maximum <= 0:
        return None
    safe = math.nextafter(maximum * (1.0 - 1e-6), 0.0)
    step = 10.0 ** (math.floor(math.log10(safe)) - 2)
    rounded = math.floor(safe / step) * step
    return rounded if rounded > 0 else safe


def _first_feasible_horizon(
    catalog: Mapping[str, ChannelConfig],
    current_horizon: int,
    metric: ObjectiveMetric,
    target: float,
    max_horizon: int = 365,
    planning_mode: PlanningMode = PlanningMode.UNIFORM,
    start_date: Date | None = None,
    calendar_profile_seed: int = 42,
) -> int | None:
    """Bounded integer search; each candidate uses the actual capacity model."""
    if current_horizon >= max_horizon:
        return None
    def capacity(horizon: int) -> float:
        return maximum_achievable(
            catalog, horizon, metric, planning_mode=planning_mode,
            start_date=start_date, calendar_profile_seed=calendar_profile_seed,
        )

    if capacity(max_horizon) < target:
        return None
    low, high = current_horizon + 1, max_horizon
    while low < high:
        middle = (low + high) // 2
        if capacity(middle) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def _estimated_budget(
    catalog: Mapping[str, ChannelConfig],
    horizon_days: int,
    metric: ObjectiveMetric,
    target_value: float,
    planning_mode: PlanningMode = PlanningMode.UNIFORM,
    start_date: Date | None = None,
    calendar_profile_seed: int = 42,
) -> float | None:
    """Use the existing Type B solver, never a separate budget approximation."""
    try:
        if planning_mode == PlanningMode.CALENDAR_AWARE:
            profiles = generate_day_profiles(
                catalog, start_date or DEFAULT_CALENDAR_START, horizon_days,
                calendar_profile_seed,
            )
            outcome = optimize_type_b_calendar(catalog, profiles, metric, target_value)
            return float(sum(outcome.cell_spend.values()))
        outcome = optimize_type_b(catalog, horizon_days, metric, target_value)
    except (RuntimeError, ValueError):
        return None
    return float(sum(outcome.channel_spend.values()))


def analyze_feasibility(
    selected_catalog: Mapping[str, ChannelConfig],
    full_catalog: Mapping[str, ChannelConfig],
    horizon_days: int,
    metric: ObjectiveMetric,
    target_value: float,
    *,
    estimate_budgets: bool = False,
    planning_mode: PlanningMode = PlanningMode.UNIFORM,
    start_date: Date | None = None,
    calendar_profile_seed: int = 42,
) -> FeasibilityResult:
    """Check capacity, then classify infeasibility by single-relaxation precedence.

    CHANNEL_SET_TOO_NARROW takes priority if all channels meet the target in
    the same horizon. Otherwise, HORIZON_TOO_SHORT applies if the selected
    channels meet it at a longer horizon within the 365-day search bound.
    MARKET_CAPACITY means neither check resolves the request, not absolute
    impossibility: enabling all channels AND extending the horizon is not tested.
    """
    selected_maximum = maximum_achievable(
        selected_catalog, horizon_days, metric, planning_mode=planning_mode,
        start_date=start_date, calendar_profile_seed=calendar_profile_seed,
    )
    full_maximum = maximum_achievable(
        full_catalog, horizon_days, metric, planning_mode=planning_mode,
        start_date=start_date, calendar_profile_seed=calendar_profile_seed,
    )
    gap = max(0.0, target_value - selected_maximum)
    tolerance = max(1e-6, target_value * 1e-9)
    restricted = set(selected_catalog) != set(full_catalog)

    if gap <= tolerance:
        return FeasibilityResult(
            feasible=True,
            requested_value=target_value,
            maximum_achievable=selected_maximum,
            target_gap=0.0,
            explanation="The selected channels can achieve the requested KPI within the horizon.",
            maximum_with_all_channels=full_maximum,
        )

    recommended_target = _safe_recommended_target(selected_maximum)
    minimum_horizon = _first_feasible_horizon(
        selected_catalog, horizon_days, metric, target_value,
        planning_mode=planning_mode, start_date=start_date,
        calendar_profile_seed=calendar_profile_seed,
    )
    additional_days = minimum_horizon - horizon_days if minimum_horizon is not None else None
    extra_capacity = max(0.0, full_maximum - selected_maximum) if restricted else None
    all_channels_feasible = restricted and full_maximum >= target_value

    if all_channels_feasible:
        reason = FeasibilityReason.CHANNEL_SET_TOO_NARROW
    elif minimum_horizon is not None:
        reason = FeasibilityReason.HORIZON_TOO_SHORT
    else:
        reason = FeasibilityReason.MARKET_CAPACITY

    recommended_budget = None
    horizon_budget = None
    if estimate_budgets:
        if recommended_target is not None:
            recommended_budget = _estimated_budget(
                selected_catalog, horizon_days, metric, recommended_target,
                planning_mode=planning_mode, start_date=start_date,
                calendar_profile_seed=calendar_profile_seed,
            )
        if minimum_horizon is not None:
            horizon_budget = _estimated_budget(
                selected_catalog, minimum_horizon, metric, target_value,
                planning_mode=planning_mode, start_date=start_date,
                calendar_profile_seed=calendar_profile_seed,
            )

    recommendations: list[str] = []
    if all_channels_feasible:
        recommendations.append(
            f"Enable all channels: modeled capacity at {horizon_days} days rises "
            f"from {selected_maximum:,.2f} to {full_maximum:,.2f} {metric.value}, "
            f"enough for the {target_value:,.2f} target."
        )
    if minimum_horizon is not None:
        horizon_option = (
            f"Extend the campaign from {horizon_days} to {minimum_horizon} days "
            f"(+{additional_days}) with the selected channels."
        )
        if horizon_budget is not None:
            horizon_option += f" Estimated minimum budget: RUB {horizon_budget:,.2f}."
        recommendations.append(horizon_option)
    if recommended_target is not None:
        target_option = (
            f"At {horizon_days} days with the selected channels, reduce the target "
            f"to about {recommended_target:,.2f} {metric.value}."
        )
        if recommended_budget is not None:
            target_option += f" Estimated minimum budget: RUB {recommended_budget:,.2f}."
        recommendations.append(target_option)

    return FeasibilityResult(
        feasible=False,
        requested_value=target_value,
        maximum_achievable=selected_maximum,
        target_gap=gap,
        recommended_target=recommended_target,
        reason=reason,
        explanation=(
            f"Requested {target_value:,.2f} {metric.value}, but the practical maximum "
            f"is {selected_maximum:,.2f} for the selected channels and horizon."
        ),
        recommendations=recommendations,
        minimum_feasible_horizon=minimum_horizon,
        additional_days_needed=additional_days,
        maximum_with_all_channels=full_maximum,
        additional_capacity_with_all_channels=extra_capacity,
        estimated_budget_for_recommended_target=recommended_budget,
        estimated_budget_at_minimum_horizon=horizon_budget,
    )
