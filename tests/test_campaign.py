from __future__ import annotations

import pytest

from mediaplan_optimizer.campaign import (
    CampaignObservation,
    add_observations,
    add_plan_version,
    start_campaign,
    switch_policy,
)


def observation(day: int = 1, channel: str = "Social Network 1") -> CampaignObservation:
    return CampaignObservation(
        day=day,
        channel=channel,
        spend=1_000,
        impressions=5_000,
        clicks=50,
        conversions=2,
        source="upload",
    )


def test_campaign_initialization_and_serialization(product_state):
    state = product_state
    assert state.current_day == 0
    assert state.remaining_days == 21
    assert state.remaining_budget == 1_200_000
    assert state.current_policy == "static"
    assert state.latest_plan.version_id == "v1"
    assert state.latest_plan.created_at_day == 0
    assert all(row.day > 0 for row in state.latest_plan.future_allocations)
    restored = type(state).model_validate_json(state.model_dump_json())
    assert restored == state


def test_observations_append_without_mutating_history(product_state):
    original = product_state
    updated = add_observations(original, [observation()])
    later = add_observations(updated, [observation(day=2)])

    assert original.observations == ()
    assert updated.current_day == 1
    assert later.current_day == 2
    assert later.observations[:1] == updated.observations
    assert later.actual_spend == 2_000
    assert later.remaining_budget == 1_198_000


def test_duplicate_fact_and_overspend_are_rejected(product_state):
    updated = add_observations(product_state, [observation()])
    with pytest.raises(ValueError, match="already ingested"):
        add_observations(updated, [observation()])
    with pytest.raises(ValueError, match="exceeds campaign budget"):
        add_observations(
            product_state,
            [observation().model_copy(update={"spend": 1_300_000})],
        )


def test_policy_switch_preserves_all_history(product_state):
    observed = add_observations(product_state, [observation()])
    switched = switch_policy(observed, "thompson_sampling")

    assert switched.current_policy == "thompson_sampling"
    assert switched.observations == observed.observations
    assert switched.plan_versions == observed.plan_versions
    assert switched.actual_spend == observed.actual_spend


def test_invalid_observation_funnel_is_rejected():
    with pytest.raises(ValueError, match="clicks cannot exceed"):
        CampaignObservation(**{**observation().model_dump(), "clicks": 6_000})


def test_campaign_snapshot_does_not_alias_draft(product_catalog, product_request, product_plan):
    draft = product_plan.model_copy(deep=True)
    request = product_request.model_copy(deep=True)
    state = start_campaign("snapshot", request, draft, product_catalog)
    original_spend = state.latest_plan.future_allocations[0].spend
    draft.allocations[0].spend = 123
    request.budget = 42
    assert state.latest_plan.future_allocations[0].spend == original_spend
    assert state.original_result.allocations[0].spend == original_spend
    assert state.original_budget == 1_200_000
    with pytest.raises(ValueError, match="must match"):
        start_campaign("mismatch", request, draft, product_catalog)


def test_direct_observation_obeys_daily_cap(product_state):
    channel = min(product_state.planning_catalog.values(), key=lambda c: c.max_daily_spend)
    with pytest.raises(ValueError, match="daily spend cap"):
        add_observations(product_state, [CampaignObservation(day=1, channel=channel.name, spend=channel.max_daily_spend + 1,
                                       impressions=1000, clicks=10, conversions=1)])


def test_invalid_future_plan_is_rejected(product_state):
    row = product_state.latest_plan.future_allocations[0]
    with pytest.raises(ValueError, match="duplicate"):
        add_plan_version(product_state, [row, row], reason="invalid")
    with pytest.raises(ValueError, match="outside"):
        add_plan_version(product_state, [row.model_copy(update={"day": 22})], reason="invalid")
    with pytest.raises(ValueError, match="positive"):
        switch_policy(product_state, "static", periodic_interval_days=0)


def test_late_policy_switch_preserves_fact_spend_and_plan_versions(product_state):
    channel = next(iter(product_state.planning_catalog))
    late = add_observations(
        product_state,
        [
            CampaignObservation(
                day=20,
                channel=channel,
                spend=1_000.0,
                impressions=10_000,
                clicks=100,
                conversions=5,
                source="upload",
            )
        ],
    )

    switched = switch_policy(late, "linucb")

    assert switched.current_day == 20
    assert switched.current_policy == "linucb"
    assert switched.observations == late.observations
    assert switched.actual_spend == late.actual_spend
    assert switched.remaining_budget == late.remaining_budget
    assert switched.plan_versions == late.plan_versions
