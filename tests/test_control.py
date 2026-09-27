from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mediaplan_optimizer.campaign import (
    CampaignObservation, add_observations, future_spend_scale, switch_policy,
)
from mediaplan_optimizer.control import (
    _next_contexts,
    allocation_explanations,
    create_policy_runtime,
    policy_diagnostics,
    recalculate_remaining_plan,
    simulate_next_days,
)
from mediaplan_optimizer.ingestion import parse_fact_csv
from mediaplan_optimizer.monitoring import build_monitoring
from mediaplan_optimizer.service import optimize_media_plan
from mediaplan_optimizer.simulator import CampaignSimulator


def test_simulation_progression_is_normalized_and_not_duplicated(product_state):
    simulator = CampaignSimulator(product_state.planning_catalog, 21, 17)
    policy = create_policy_runtime(product_state, seed=17)
    day_two, policy = simulate_next_days(
        product_state, simulator, policy, days=2
    )

    assert day_two.current_day == 2
    assert {item.day for item in day_two.observations} == {1, 2}
    assert len({(item.day, item.channel) for item in day_two.observations}) == len(
        day_two.observations
    )
    day_three, _ = simulate_next_days(day_two, simulator, policy, days=1)
    assert day_three.current_day == 3
    assert day_three.observations[: len(day_two.observations)] == day_two.observations


def test_replan_preserves_fact_and_replaces_future_only(product_state):
    simulator = CampaignSimulator(product_state.planning_catalog, 21, 19)
    policy = create_policy_runtime(product_state, seed=19)
    observed, policy = simulate_next_days(product_state, simulator, policy, days=3)
    revised = recalculate_remaining_plan(observed, policy)

    assert revised.observations == observed.observations
    assert len(revised.plan_versions) == 2
    assert revised.latest_plan.created_at_day == 3
    assert revised.latest_plan.remaining_horizon == 18
    assert all(row.day > 3 for row in revised.latest_plan.future_allocations)
    assert sum(row.spend for row in revised.latest_plan.future_allocations) <= (
        revised.remaining_budget + 0.1
    )


def test_policy_switch_replays_history_and_produces_diagnostics(product_state):
    simulator = CampaignSimulator(product_state.planning_catalog, 21, 23)
    static = create_policy_runtime(product_state, seed=23)
    observed, _ = simulate_next_days(product_state, simulator, static, days=2)
    switched = switch_policy(observed, "thompson_sampling")
    thompson = create_policy_runtime(switched, seed=23)
    diagnostics = policy_diagnostics(switched, thompson)

    assert switched.observations == observed.observations
    assert switched.actual_spend == observed.actual_spend
    assert not diagnostics.empty
    assert diagnostics["score"].notna().all()
    assert diagnostics["recommended"].sum() == 1


def test_future_budget_guard_preserves_history_and_prevents_overspend(product_state):
    channel = max(product_state.planning_catalog.values(), key=lambda c: c.max_daily_spend)
    state = add_observations(product_state, [CampaignObservation(day=1, channel=channel.name, spend=1_190_000,
                                       impressions=1000, clicks=10, conversions=1)])
    history = state.plan_versions
    assert future_spend_scale(state) < 1
    progress = build_monitoring(state).progress
    assert np.isfinite(progress.projected_final_kpi)
    assert progress.projected_unit_cost * progress.projected_final_kpi == pytest.approx(
        state.original_budget
    )
    simulator = CampaignSimulator(state.planning_catalog, 21, 42)
    completed, _ = simulate_next_days(state, simulator, create_policy_runtime(state), days=20)
    assert completed.current_day == 21
    assert completed.actual_spend == pytest.approx(1_200_000)
    assert completed.remaining_budget >= 0
    assert completed.plan_versions == history
    assert completed.observations[:1] == state.observations
    assert build_monitoring(completed).progress.projected_final_kpi == completed.actual_kpi


def test_repeated_identical_replan_is_idempotent(product_state):
    state = add_observations(product_state, [CampaignObservation(day=1, channel="Programmatic", spend=100,
                                       impressions=1000, clicks=10, conversions=1)])
    revised = recalculate_remaining_plan(state)
    repeated = recalculate_remaining_plan(revised)
    assert repeated is revised
    assert len(repeated.forecast_history) == len(revised.forecast_history)


def test_excluded_channels_are_not_explained_as_inefficient(product_catalog, product_request):
    request = product_request.model_copy(update={"included_channels": ["SMS"]})
    result = optimize_media_plan(request, product_catalog)
    explanations = allocation_explanations(result, product_catalog).set_index("channel")
    assert explanations.loc["SMS", "eligible"]
    assert explanations.loc["SMS", "efficiency_rank"] == 1
    assert "Excluded" in explanations.loc["Programmatic", "explanation"]
    assert pd.isna(explanations.loc["Programmatic", "efficiency_rank"])


def test_online_context_counts_spend_already_made_today(product_state):
    policy = create_policy_runtime(switch_policy(product_state, "linucb"))
    daily = {name: 0.0 for name in product_state.planning_catalog}
    daily["Programmatic"] = 10_000
    contexts = _next_contexts(product_state, policy, day=1, quantum_rub=1000, daily_spend=daily)
    context = next(c for c in contexts if c.channel == "Programmatic")
    assert context.cumulative_spend == 10_000
    assert context.vector[5] == pytest.approx(1_190_000 / 1_200_000)
    assert context.vector[4] > 0


def test_uploaded_fact_can_follow_simulation_but_cannot_be_ingested_twice(
    product_state,
):
    simulator = CampaignSimulator(product_state.planning_catalog, 21, 101)
    runtime = create_policy_runtime(product_state, seed=101)
    simulated, _ = simulate_next_days(
        product_state,
        simulator,
        runtime,
        days=1,
    )
    channel = next(iter(simulated.planning_catalog))
    upload = parse_fact_csv(
        ("day,channel,spend,impressions,clicks,conversions\n"
         f"2,{channel},100,1000,10,1\n"),
        simulated.planning_catalog,
        horizon_days=simulated.horizon_days,
    )

    mixed = add_observations(simulated, upload.observations)

    assert {item.source for item in mixed.observations} == {"simulator", "upload"}
    assert mixed.current_day == 2
    assert mixed.observations[: len(simulated.observations)] == simulated.observations
    with pytest.raises(ValueError, match="already ingested"):
        add_observations(mixed, upload.observations)


def test_replan_is_rejected_after_final_day(product_state):
    channel = next(iter(product_state.planning_catalog))
    completed = add_observations(
        product_state,
        [
            CampaignObservation(
                day=product_state.horizon_days,
                channel=channel,
                spend=100.0,
                impressions=1_000,
                clicks=10,
                conversions=1,
                source="upload",
            )
        ],
    )

    with pytest.raises(ValueError, match="completed campaign cannot be replanned"):
        recalculate_remaining_plan(completed)


def test_fully_spent_campaign_gets_empty_future_plan_and_does_not_advance(
    product_state,
):
    channel = max(
        product_state.planning_catalog.values(),
        key=lambda item: item.max_daily_spend,
    )
    assert channel.max_daily_spend >= product_state.original_budget
    spent = add_observations(
        product_state,
        [
            CampaignObservation(
                day=1,
                channel=channel.name,
                spend=product_state.original_budget,
                impressions=1_000_000,
                clicks=10_000,
                conversions=500,
                source="upload",
            )
        ],
    )
    runtime = create_policy_runtime(spent, seed=103)

    replanned = recalculate_remaining_plan(spent, runtime)

    assert replanned.remaining_budget == 0.0
    assert len(replanned.plan_versions) == len(spent.plan_versions) + 1
    assert replanned.latest_plan.future_allocations == ()
    simulator = CampaignSimulator(replanned.planning_catalog, 21, 103)
    unchanged, _ = simulate_next_days(replanned, simulator, runtime, days=1)
    assert unchanged.current_day == 1
    assert unchanged.observations == replanned.observations
