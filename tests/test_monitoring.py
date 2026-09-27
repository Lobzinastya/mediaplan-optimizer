from __future__ import annotations

import math

import pytest

from mediaplan_optimizer.campaign import (
    CampaignObservation,
    add_observations,
    add_plan_version,
)
from mediaplan_optimizer.monitoring import build_monitoring, effective_plan_frame


def _day_fact(product_state, day: int, multiplier: float = 1.0):
    rows = [row for row in product_state.original_result.allocations if row.day == day]
    return [
        CampaignObservation(
            day=day,
            channel=row.channel,
            spend=row.spend,
            impressions=row.impressions,
            clicks=min(row.impressions, row.clicks * multiplier),
            conversions=min(row.clicks * multiplier, row.conversions * multiplier),
            source="upload",
        )
        for row in rows
        if row.spend > 0
    ]


def test_monitoring_pacing_and_variance_are_transparent(product_state):
    state = add_observations(product_state, _day_fact(product_state, 1, 0.5))
    result = build_monitoring(state)
    progress = result.progress

    assert progress.expected_progress_fraction == pytest.approx(1 / 21)
    assert progress.budget_pace == pytest.approx(1.0)
    assert progress.kpi_pace == pytest.approx(0.5)
    assert progress.plan_vs_fact_variance == pytest.approx(-0.5)
    assert progress.projected_final_kpi < progress.planned_final_kpi


def test_channel_status_detects_under_and_outperformance(product_state):
    weak = add_observations(product_state, _day_fact(product_state, 1, 0.5))
    strong = add_observations(product_state, _day_fact(product_state, 1, 1.5))
    weak_status = set(build_monitoring(weak).channels.query("actual_spend > 0")["status"])
    strong_status = set(build_monitoring(strong).channels.query("actual_spend > 0")["status"])

    assert "Underperforming" in weak_status
    assert "Outperforming" in strong_status


def test_forecast_and_costs_remain_finite(product_state):
    state = add_observations(product_state, _day_fact(product_state, 1))
    progress = build_monitoring(state).progress
    assert math.isfinite(progress.projected_final_kpi)
    assert progress.projected_unit_cost is not None
    assert math.isfinite(progress.projected_unit_cost)


def test_monitoring_uses_plan_version_in_force_for_each_elapsed_day(product_state):
    after_day_one = add_observations(product_state, _day_fact(product_state, 1))
    revised_future = tuple(
        row.model_copy(
            update={
                "spend": row.spend * 0.5,
                "impressions": row.impressions * 0.5,
                "reach": row.reach * 0.5,
                "clicks": row.clicks * 0.5,
                "conversions": row.conversions * 0.5,
                "video_views": (
                    None if row.video_views is None else row.video_views * 0.5
                ),
                "cumulative_target_kpi": row.cumulative_target_kpi * 0.5,
            }
        )
        for row in product_state.original_result.allocations
        if row.day > 1
    )
    revised = add_plan_version(
        after_day_one,
        revised_future,
        reason="Test revision",
    )
    state = add_observations(revised, _day_fact(product_state, 2, 0.5))

    effective = effective_plan_frame(state)
    day_two = effective[effective["day"] == 2]
    assert set(day_two["plan_version"]) == {"v2"}

    progress = build_monitoring(state).progress
    original_day_one = sum(
        row.spend for row in product_state.original_result.allocations if row.day == 1
    )
    revised_day_two = sum(row.spend for row in revised_future if row.day == 2)
    assert progress.planned_spend_to_date == pytest.approx(
        original_day_one + revised_day_two
    )


def test_empirical_projection_interval_requires_three_observed_days(product_state):
    state = add_observations(
        product_state,
        _day_fact(product_state, 1) + _day_fact(product_state, 2, 0.9),
    )
    two_day_progress = build_monitoring(state).progress
    assert two_day_progress.projection_interval_lower is None
    assert two_day_progress.projection_interval_upper is None
    assert two_day_progress.projection_interval_observation_days == 2

    state = add_observations(state, _day_fact(product_state, 3, 1.1))
    progress = build_monitoring(state).progress
    assert progress.projection_interval_observation_days == 3
    assert progress.projection_interval_lower is not None
    assert progress.projection_interval_upper is not None
    assert progress.projection_interval_lower <= progress.projected_final_kpi
    assert progress.projected_final_kpi <= progress.projection_interval_upper
