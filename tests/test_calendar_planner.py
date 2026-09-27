"""Calendar-aware optimization is genuinely day-level and preserves V1 when neutral."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
import pytest
from scipy.optimize import minimize

from mediaplan_optimizer.calendar import effective_channel, generate_day_profiles
from mediaplan_optimizer.calendar_optimizer import (
    calendar_maximum, optimize_type_a_calendar, optimize_type_b_calendar,
)
from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.curves import (
    channel_response, effective_performance_rates, marginal_metric_response,
    metric_response,
)
from mediaplan_optimizer.feasibility import analyze_feasibility, maximum_achievable
from mediaplan_optimizer.metrics import aggregate_weekly_metrics
from mediaplan_optimizer.optimizer import optimize_type_a
from mediaplan_optimizer.schemas import ObjectiveMetric, OptimizationRequest, PlanningMode, ResultStatus
from mediaplan_optimizer.service import optimize_media_plan


START = date(2026, 9, 21)
FACTOR_FIELDS = ("supply_factor", "cpm_factor", "ctr_factor", "cr_factor", "max_spend_factor")


def test_profiles_are_reproducible_bounded_weekly_normalized_and_date_anchored():
    catalog = generate_catalog(42)
    first = generate_day_profiles(catalog, START, 21, 42)
    assert first == generate_day_profiles(catalog, START, 21, 42)
    assert first != generate_day_profiles(catalog, START, 21, 43)
    assert first[0].date == START and first[0].weekday == 0
    assert first[-1].date == date(2026, 10, 11)
    assert len(first) == 21 * len(catalog)
    for channel in catalog:
        for week in range(3):
            rows = [p for p in first if p.channel == channel and (p.day - 1) // 7 == week]
            assert len(rows) == 7
            for field in FACTOR_FIELDS:
                values = [getattr(p, field) for p in rows]
                assert min(values) >= 0.8 and max(values) <= 1.2
                assert np.mean(values) == pytest.approx(1.0, abs=1e-12)
    assert len({p.ctr_factor for p in first if p.channel == "Marketplace 2"}) > 1
    extended = generate_day_profiles(catalog, START, 24, 42)
    assert extended[:len(first)] == first


def test_day_specific_curves_keep_rates_bounds_monotonicity_and_concavity():
    catalog = generate_catalog(42)
    profiles = generate_day_profiles(catalog, START, 7, 42)
    for profile in profiles:
        channel = effective_channel(catalog[profile.channel], profile)
        assert channel.daily_capacity > 0 and channel.max_daily_spend > 0
        spend = np.linspace(0.0, channel.max_daily_spend, 41)
        ctr, cr = effective_performance_rates(spend, channel)
        assert np.all((0 <= ctr) & (ctr <= 1))
        assert np.all((0 <= cr) & (cr <= 1))
        for metric in ObjectiveMetric:
            values = np.asarray(metric_response(spend, channel, metric))
            derivatives = np.array([marginal_metric_response(float(s), channel, metric)
                                    for s in spend])
            assert values[0] == pytest.approx(0.0)
            assert np.all(np.diff(values) >= -1e-7)
            assert np.all(np.diff(derivatives) <= 1e-9)
            assert np.all(derivatives >= -1e-10)


def test_all_one_factors_reproduce_uniform_optimum():
    catalog = generate_catalog(42)
    profiles = tuple(p.model_copy(update={field: 1.0 for field in FACTOR_FIELDS})
                     for p in generate_day_profiles(catalog, START, 21, 42))
    uniform = optimize_type_a(catalog, 21, ObjectiveMetric.CONVERSIONS, 1_200_000)
    calendar = optimize_type_a_calendar(catalog, profiles, ObjectiveMetric.CONVERSIONS, 1_200_000)
    assert calendar.achieved_value == pytest.approx(uniform.achieved_value, abs=1e-6)
    for name in catalog:
        total = sum(spend for (day, channel), spend in calendar.cell_spend.items()
                    if channel == name)
        assert total == pytest.approx(uniform.channel_spend[name], abs=0.02)


def test_calendar_type_a_is_nonflat_respects_caps_exclusions_and_marginal_equality():
    catalog = generate_catalog(42)
    request = OptimizationRequest(
        task_type="A", objective_metric="conversions", budget=1_200_000,
        horizon_days=21, planning_mode="calendar_aware", start_date=START,
    )
    result = optimize_media_plan(request, catalog)
    assert result.status == ResultStatus.OPTIMAL
    assert result.summary.spend == pytest.approx(1_200_000, abs=0.01)
    assert np.isfinite(result.summary.conversions)
    assert len(result.allocations) == 168
    profiles = {(p.day, p.channel): p for p in result.temporal_profiles}
    daily = defaultdict(float)
    interior_marginals = []
    for row in result.allocations:
        channel = effective_channel(catalog[row.channel], profiles[(row.day, row.channel)])
        assert 0 <= row.spend <= channel.max_daily_spend + 1e-6
        assert row.date == profiles[(row.day, row.channel)].date
        daily[row.day] += row.spend
        if 1 < row.spend < channel.max_daily_spend - 1:
            interior_marginals.append(marginal_metric_response(
                row.spend, channel, ObjectiveMetric.CONVERSIONS))
    assert max(daily.values()) - min(daily.values()) > 100
    assert max(interior_marginals) / min(interior_marginals) < 1.001
    weeks = aggregate_weekly_metrics(result.allocations)
    assert sum(week["spend"] for week in weeks) == pytest.approx(result.summary.spend)
    assert len({round(week["spend"], 2) for week in weeks}) > 1
    restricted = request.model_copy(update={"excluded_channels": ["SMS"]})
    without_sms = optimize_media_plan(restricted, catalog)
    assert "SMS" not in {row.channel for row in without_sms.allocations}


def test_small_calendar_waterfill_matches_independent_slsqp():
    full = generate_catalog(42)
    catalog = {name: full[name] for name in ("Social Network 1", "Marketplace 2")}
    profiles = generate_day_profiles(catalog, START, 3, 42)
    channels = [effective_channel(catalog[p.channel], p) for p in profiles]
    caps = np.array([channel.max_daily_spend for channel in channels])
    budget = 40_000.0
    objective = ObjectiveMetric.CONVERSIONS
    solved = optimize_type_a_calendar(catalog, profiles, objective, budget)
    numerical = minimize(
        lambda x: -sum(float(metric_response(float(spend), channel, objective))
                       for spend, channel in zip(x, channels, strict=True)),
        x0=np.full(len(channels), budget / len(channels)),
        jac=lambda x: -np.array([marginal_metric_response(float(spend), channel, objective)
                                 for spend, channel in zip(x, channels, strict=True)]),
        method="SLSQP", bounds=[(0, cap) for cap in caps],
        constraints={"type": "eq", "fun": lambda x: float(sum(x)) - budget},
        options={"ftol": 1e-11, "maxiter": 1000},
    )
    assert numerical.success
    assert solved.achieved_value == pytest.approx(-numerical.fun, rel=1e-6)


def test_calendar_type_b_frontier_and_start_date_effect():
    catalog = generate_catalog(42)
    monday = generate_day_profiles(catalog, START, 3, 42)
    thursday = generate_day_profiles(catalog, date(2026, 9, 24), 3, 42)
    low = optimize_type_b_calendar(catalog, monday, ObjectiveMetric.CLICKS, 10_000)
    high = optimize_type_b_calendar(catalog, monday, ObjectiveMetric.CLICKS, 12_000)
    shifted = optimize_type_b_calendar(catalog, thursday, ObjectiveMetric.CLICKS, 10_000)
    low_budget = sum(low.cell_spend.values())
    assert low.achieved_value >= 10_000 * (1 - 1e-7)
    assert high.achieved_value >= 12_000 * (1 - 1e-7)
    assert sum(high.cell_spend.values()) >= low_budget
    assert abs(sum(shifted.cell_spend.values()) - low_budget) > 1
    with pytest.raises(ValueError, match="infeasible"):
        optimize_type_b_calendar(catalog, monday, ObjectiveMetric.CLICKS, 1e12)


def test_calendar_service_type_b_feasible_and_impossible_are_structured():
    catalog = generate_catalog(42)
    feasible = optimize_media_plan(OptimizationRequest(
        task_type="B", objective_metric="clicks", target_value=10_000,
        horizon_days=14, planning_mode="calendar_aware", start_date=START,
    ), catalog)
    assert feasible.status == ResultStatus.OPTIMAL
    assert feasible.summary.spend == pytest.approx(103_937.97140206024, abs=0.02)
    assert feasible.summary.clicks >= 10_000 * (1 - 1e-7)
    assert all(row.date is not None for row in feasible.allocations)
    impossible = optimize_media_plan(OptimizationRequest(
        task_type="B", objective_metric="conversions", target_value=1e12,
        horizon_days=1, planning_mode="calendar_aware", start_date=START,
    ), catalog)
    assert impossible.status == ResultStatus.INFEASIBLE
    assert impossible.feasibility.maximum_achievable == pytest.approx(3955.2328030809704)
    assert impossible.feasibility.reason.value == "MARKET_CAPACITY"


def test_calendar_feasibility_uses_actual_caps_and_extends_same_sequence():
    catalog = generate_catalog(42)
    selected = {"SMS": catalog["SMS"]}
    profiles = generate_day_profiles(selected, START, 3, 42)
    explicit = sum(float(metric_response(
        effective_channel(selected[p.channel], p).max_daily_spend,
        effective_channel(selected[p.channel], p), ObjectiveMetric.CLICKS))
        for p in profiles)
    assert calendar_maximum(selected, profiles, ObjectiveMetric.CLICKS) == pytest.approx(explicit)
    assert maximum_achievable(selected, 3, ObjectiveMetric.CLICKS,
                              planning_mode=PlanningMode.CALENDAR_AWARE,
                              start_date=START) == pytest.approx(explicit)
    two_days = calendar_maximum(selected, profiles[:2], ObjectiveMetric.CLICKS)
    target = (two_days + explicit) / 2
    result = analyze_feasibility(
        selected, catalog, 1, ObjectiveMetric.CLICKS, target,
        planning_mode=PlanningMode.CALENDAR_AWARE, start_date=START,
    )
    assert result.minimum_feasible_horizon == 3
    assert result.additional_days_needed == 2
    assert two_days < target <= explicit
    assert result.maximum_with_all_channels == pytest.approx(
        maximum_achievable(catalog, 1, ObjectiveMetric.CLICKS,
                           planning_mode=PlanningMode.CALENDAR_AWARE, start_date=START))


def test_calendar_counterfactual_budget_uses_calendar_solver():
    catalog = generate_catalog(42)
    selected = {"SMS": catalog["SMS"]}
    one_day = maximum_achievable(selected, 1, ObjectiveMetric.CLICKS,
                                 planning_mode=PlanningMode.CALENDAR_AWARE, start_date=START)
    result = analyze_feasibility(
        selected, catalog, 1, ObjectiveMetric.CLICKS, one_day * 1.5,
        planning_mode=PlanningMode.CALENDAR_AWARE, start_date=START,
        estimate_budgets=True,
    )
    assert result.estimated_budget_for_recommended_target is not None
    assert result.estimated_budget_at_minimum_horizon is not None
    profiles = generate_day_profiles(selected, START, result.minimum_feasible_horizon)
    planned = optimize_type_a_calendar(
        selected, profiles, ObjectiveMetric.CLICKS, result.estimated_budget_at_minimum_horizon)
    assert planned.achieved_value >= result.requested_value * (1 - 1e-7)
