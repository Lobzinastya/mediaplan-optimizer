import math

import pytest

from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.curves import marginal_metric_response, metric_response
from mediaplan_optimizer.optimizer import (
    optimize_type_a,
    optimize_type_b,
)
from mediaplan_optimizer.schemas import ObjectiveMetric


def test_type_a_respects_budget_caps_and_is_nonuniform():
    catalog = generate_catalog(seed=42)
    outcome = optimize_type_a(catalog, 14, ObjectiveMetric.CONVERSIONS, 800_000)
    assert all(value >= 0 for value in outcome.channel_spend.values())
    assert sum(outcome.channel_spend.values()) <= 800_000 + 0.1
    assert all(
        spend <= catalog[name].max_daily_spend * 14 + 0.1
        for name, spend in outcome.channel_spend.items()
    )
    assert outcome.achieved_value > 0
    assert len({round(value, 2) for value in outcome.channel_spend.values()}) > 1
    assert outcome.metadata.validation_passed


def test_type_a_beats_single_channel_and_proportional_baselines():
    catalog = generate_catalog(seed=42)
    horizon = 21
    budget = 5_000_000
    outcome = optimize_type_a(catalog, horizon, ObjectiveMetric.CONVERSIONS, budget)

    proportional = {
        name: budget * channel.max_daily_spend
        / sum(item.max_daily_spend for item in catalog.values())
        for name, channel in catalog.items()
    }
    proportional_value = sum(
        horizon
        * metric_response(
            proportional[name] / horizon, channel, ObjectiveMetric.CONVERSIONS
        )
        for name, channel in catalog.items()
    )
    best_single = max(
        horizon
        * metric_response(
            min(budget / horizon, channel.max_daily_spend),
            channel,
            ObjectiveMetric.CONVERSIONS,
        )
        for channel in catalog.values()
    )
    assert outcome.achieved_value >= proportional_value
    assert outcome.achieved_value >= best_single


def test_type_b_reaches_target_and_larger_target_costs_more():
    catalog = generate_catalog(seed=42)
    low = optimize_type_b(catalog, 14, ObjectiveMetric.CLICKS, 10_000)
    high = optimize_type_b(catalog, 14, ObjectiveMetric.CLICKS, 15_000)
    assert low.achieved_value >= 10_000 * (1 - 1e-7)
    assert high.achieved_value >= 15_000 * (1 - 1e-7)
    assert sum(high.channel_spend.values()) >= sum(low.channel_spend.values())
    assert math.isfinite(sum(high.channel_spend.values()))


def test_excluded_channel_is_absent_when_catalog_is_filtered():
    catalog = generate_catalog(seed=42)
    catalog.pop("SMS")
    outcome = optimize_type_a(catalog, 7, ObjectiveMetric.CLICKS, 300_000)
    assert "SMS" not in outcome.channel_spend


def test_type_b_rejects_impossible_target():
    catalog = generate_catalog(seed=42)
    with pytest.raises(ValueError, match="infeasible"):
        optimize_type_b(catalog, 1, ObjectiveMetric.CONVERSIONS, 1e12)


@pytest.mark.parametrize("budget", [1e-12, 1e-9, 1e-6])
def test_microscopic_positive_budget_is_not_lost_to_solver_precision(budget):
    catalog = generate_catalog(seed=42)
    outcome = optimize_type_a(
        catalog, 21, ObjectiveMetric.CONVERSIONS, budget
    )
    assert sum(outcome.channel_spend.values()) == pytest.approx(
        budget, rel=1e-9, abs=1e-18
    )
    assert outcome.achieved_value > 0


def test_type_a_allocation_satisfies_marginal_optimality_conditions():
    catalog = generate_catalog(seed=42)
    horizon = 21
    outcome = optimize_type_a(
        catalog, horizon, ObjectiveMetric.CONVERSIONS, 1_200_000
    )
    active_marginals = []
    inactive_marginals = []
    for name, total_spend in outcome.channel_spend.items():
        channel = catalog[name]
        marginal = marginal_metric_response(
            total_spend / horizon, channel, ObjectiveMetric.CONVERSIONS
        )
        if total_spend > 0.01:
            active_marginals.append(marginal)
        else:
            inactive_marginals.append(marginal)

    assert max(active_marginals) == pytest.approx(
        min(active_marginals), rel=1e-9
    )
    assert max(inactive_marginals) <= min(active_marginals) * (1 + 1e-9)
