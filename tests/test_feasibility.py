import pytest

from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.feasibility import analyze_feasibility, maximum_achievable
from mediaplan_optimizer.optimizer import optimize_type_a
from mediaplan_optimizer.schemas import FeasibilityReason, ObjectiveMetric, OptimizationRequest, ResultStatus
from mediaplan_optimizer.service import optimize_media_plan


def test_impossible_request_is_structured_and_has_counterfactuals():
    catalog = generate_catalog(seed=42)
    selected = {"SMS": catalog["SMS"]}
    maximum = maximum_achievable(selected, 2, ObjectiveMetric.CONVERSIONS)
    result = analyze_feasibility(
        selected, catalog, 2, ObjectiveMetric.CONVERSIONS, maximum * 2
    )
    assert not result.feasible
    assert result.maximum_achievable == maximum
    assert result.target_gap > 0
    assert result.reason in set(FeasibilityReason)
    assert result.recommendations
    assert result.maximum_with_all_channels >= maximum


def test_feasible_request_is_detected():
    catalog = generate_catalog(seed=42)
    maximum = maximum_achievable(catalog, 14, ObjectiveMetric.CLICKS)
    result = analyze_feasibility(
        catalog, catalog, 14, ObjectiveMetric.CLICKS, maximum * 0.5
    )
    assert result.feasible
    assert result.target_gap == 0


def test_longer_supported_horizon_counterfactual_is_reported():
    catalog = generate_catalog(seed=42)
    one_day = maximum_achievable(catalog, 1, ObjectiveMetric.CLICKS)
    result = analyze_feasibility(
        catalog, catalog, 1, ObjectiveMetric.CLICKS, one_day * 1.5
    )
    assert result.reason == FeasibilityReason.HORIZON_TOO_SHORT
    assert result.minimum_feasible_horizon == 2
    assert result.additional_days_needed == 1
    assert "approximate_minimum_horizon" not in result.model_dump(mode="json")
    assert "approximate_minimum_horizon" not in result.model_json_schema()["properties"]


def test_safe_target_is_below_capacity_and_service_can_plan_it():
    catalog = generate_catalog(seed=42)
    selected = {"SMS": catalog["SMS"]}
    maximum = maximum_achievable(selected, 1, ObjectiveMetric.CLICKS)
    result = analyze_feasibility(selected, catalog, 1, ObjectiveMetric.CLICKS, maximum * 1.5)
    assert 0 < result.recommended_target < maximum
    assert result.estimated_budget_for_recommended_target is None
    planned = optimize_media_plan(
        OptimizationRequest(task_type="B", horizon_days=1, objective_metric="clicks",
                            target_value=result.recommended_target, included_channels=["SMS"]),
        catalog,
    )
    assert planned.status == ResultStatus.OPTIMAL


def test_minimum_horizon_is_first_actually_feasible_integer_day():
    catalog = generate_catalog(seed=42)
    one_day = maximum_achievable(catalog, 1, ObjectiveMetric.CLICKS)
    target = one_day * 2.4
    result = analyze_feasibility(catalog, catalog, 1, ObjectiveMetric.CLICKS, target)
    assert result.reason == FeasibilityReason.HORIZON_TOO_SHORT
    assert result.minimum_feasible_horizon == 3
    assert result.additional_days_needed == 2
    assert maximum_achievable(catalog, result.minimum_feasible_horizon, ObjectiveMetric.CLICKS) >= target
    assert maximum_achievable(catalog, result.minimum_feasible_horizon - 1, ObjectiveMetric.CLICKS) < target
    assert not any("Enable all channels" in item for item in result.recommendations)


def test_all_channel_counterfactual_is_calculated_and_prioritized():
    catalog = generate_catalog(seed=42)
    selected = {"SMS": catalog["SMS"]}
    target = maximum_achievable(selected, 1, ObjectiveMetric.CLICKS) * 1.5
    result = analyze_feasibility(selected, catalog, 1, ObjectiveMetric.CLICKS, target)
    assert result.reason == FeasibilityReason.CHANNEL_SET_TOO_NARROW
    assert result.minimum_feasible_horizon == 2  # Both relaxations work; channels take priority.
    assert result.maximum_with_all_channels == pytest.approx(maximum_achievable(catalog, 1, ObjectiveMetric.CLICKS))
    assert result.additional_capacity_with_all_channels == pytest.approx(
        result.maximum_with_all_channels - result.maximum_achievable
    )
    assert result.recommendations[0].startswith("Enable all channels")


def test_true_market_capacity_has_no_impossible_action():
    catalog = generate_catalog(seed=42)
    result = analyze_feasibility(catalog, catalog, 1, ObjectiveMetric.CONVERSIONS, 1e12)
    assert result.reason == FeasibilityReason.MARKET_CAPACITY
    assert result.minimum_feasible_horizon is None
    assert result.additional_days_needed is None
    assert result.additional_capacity_with_all_channels is None
    assert result.recommended_target <= result.maximum_achievable
    assert len(result.recommendations) == 1
    assert not any("budget" in item.lower() or "Enable" in item for item in result.recommendations)


def test_opt_in_budget_counterfactuals_use_feasible_type_b_targets():
    catalog = generate_catalog(seed=42)
    selected = {"SMS": catalog["SMS"]}
    target = maximum_achievable(selected, 1, ObjectiveMetric.CLICKS) * 1.5
    result = analyze_feasibility(
        selected, catalog, 1, ObjectiveMetric.CLICKS, target, estimate_budgets=True
    )
    assert result.estimated_budget_for_recommended_target is not None
    assert result.estimated_budget_at_minimum_horizon is not None
    first = optimize_type_a(
        selected, 1, ObjectiveMetric.CLICKS, result.estimated_budget_for_recommended_target
    )
    second = optimize_type_a(
        selected, result.minimum_feasible_horizon, ObjectiveMetric.CLICKS,
        result.estimated_budget_at_minimum_horizon,
    )
    assert first.achieved_value >= result.recommended_target - max(1e-6, result.recommended_target * 1e-7)
    assert second.achieved_value >= target - max(1e-6, target * 1e-7)
