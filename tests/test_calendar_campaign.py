"""Approved calendar dates and caps survive campaign execution and replanning."""

from __future__ import annotations

from datetime import timedelta

import pytest

from mediaplan_optimizer.campaign import add_observations, start_campaign, switch_policy
from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.control import (
    create_policy_runtime, recalculate_remaining_plan, simulate_next_days,
)
from mediaplan_optimizer.ingestion import parse_fact_csv
from mediaplan_optimizer.schemas import OptimizationRequest
from mediaplan_optimizer.service import optimize_media_plan
from mediaplan_optimizer.simulator import CampaignSimulator


def _campaign():
    catalog = generate_catalog(42)
    request = OptimizationRequest(
        task_type="A", objective_metric="conversions", budget=1_200_000,
        horizon_days=21, planning_mode="calendar_aware",
    )
    result = optimize_media_plan(request, catalog)
    return start_campaign("calendar", request, result, catalog, campaign_id="calendar-test")


def test_replan_preserves_approved_profiles_past_dates_budget_and_future_caps():
    state = _campaign()
    approved_profiles = state.temporal_profiles
    initial_plan = state.plan_versions[0]
    assert approved_profiles == tuple(state.original_result.temporal_profiles)
    assert state.original_request.start_date.isoformat() == "2026-09-21"
    simulator = CampaignSimulator(state.planning_catalog, 21, 42)
    runtime = create_policy_runtime(state)
    observed, runtime = simulate_next_days(state, simulator, runtime, days=3)
    assert observed.current_day == 3
    assert observed.observations[0].date == state.original_request.start_date
    fact_snapshot = observed.observations
    remaining = observed.remaining_budget
    revised = recalculate_remaining_plan(observed, runtime)
    assert revised.temporal_profiles == approved_profiles
    assert revised.plan_versions[0] == initial_plan
    assert revised.observations == fact_snapshot
    assert revised.remaining_budget == pytest.approx(remaining)
    assert revised.latest_plan.version_id == "v2"
    first = revised.latest_plan.future_allocations[0]
    assert first.day == 4
    assert first.date == state.original_request.start_date + timedelta(days=3)
    assert first.weekday == 3
    assert all(row.day > 3 and row.date == state.day_date(row.day)
               and row.spend <= state.day_cap(row.day, row.channel) + 1e-6
               for row in revised.latest_plan.future_allocations)
    assert sum(row.spend for row in revised.latest_plan.future_allocations) <= remaining + 0.01


def test_calendar_product_flow_completes_without_overspend_or_duplicate_fact():
    state = _campaign()
    simulator = CampaignSimulator(state.planning_catalog, 21, 42)
    runtime = create_policy_runtime(state)
    state, runtime = simulate_next_days(state, simulator, runtime, days=3)
    state = recalculate_remaining_plan(state, runtime)
    state, runtime = simulate_next_days(state, simulator, runtime, days=18)
    assert state.current_day == 21
    assert state.actual_spend == pytest.approx(1_200_000, abs=0.01)
    assert state.remaining_budget == pytest.approx(0.0, abs=0.01)
    assert len({(row.day, row.channel) for row in state.observations}) == len(state.observations)
    assert state.observations[-1].date.isoformat() == "2026-10-11"
    assert state.plan_versions[0].future_allocations[0].date.isoformat() == "2026-09-21"
    assert all(row.spend <= state.day_cap(row.day, row.channel) + 1e-6
               for row in state.observations)


def test_calendar_csv_accepts_approved_cap_and_date_anchor():
    state = _campaign()
    profile = next(p for p in state.temporal_profiles
                   if p.channel == "SMS" and p.max_spend_factor > 1.001)
    base = state.planning_catalog["SMS"].max_daily_spend
    spend = (base + state.day_cap(profile.day, "SMS")) / 2
    csv = ("date,channel,spend,impressions,clicks,conversions\n"
           f"{profile.date.isoformat()},SMS,{spend},1000,10,1\n")
    caps = {(p.day, p.channel): state.day_cap(p.day, p.channel)
            for p in state.temporal_profiles}
    with pytest.raises(ValueError, match="daily spend cap"):
        parse_fact_csv(csv, state.planning_catalog, horizon_days=21,
                       campaign_start_date=state.original_request.start_date)
    parsed = parse_fact_csv(csv, state.planning_catalog, horizon_days=21,
                            campaign_start_date=state.original_request.start_date,
                            daily_caps=caps)
    updated = add_observations(state, parsed.observations)
    assert updated.observations[0].day == profile.day
    assert updated.observations[0].date == profile.date
    day_only_csv = ("day,channel,spend,impressions,clicks,conversions\n"
                    f"{profile.day},SMS,{spend},1000,10,1\n")
    day_only = parse_fact_csv(
        day_only_csv, state.planning_catalog, horizon_days=21,
        campaign_start_date=state.original_request.start_date, daily_caps=caps,
    )
    assert day_only.observations[0].date == profile.date


def test_product_adaptive_policy_respects_calendar_day_caps():
    state = switch_policy(_campaign(), "thompson_sampling")
    simulator = CampaignSimulator(state.planning_catalog, 21, 42)
    runtime = create_policy_runtime(state)
    updated, _ = simulate_next_days(state, simulator, runtime, days=1)
    assert updated.current_day == 1
    assert all(row.spend <= updated.day_cap(row.day, row.channel) + 1e-6
               for row in updated.observations)
