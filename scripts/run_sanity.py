"""Run deterministic numerical sanity checks and print scenario summaries."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.feasibility import maximum_achievable
from mediaplan_optimizer.optimizer import optimize_type_a, optimize_type_b
from mediaplan_optimizer.schemas import ObjectiveMetric, OptimizationRequest
from mediaplan_optimizer.service import optimize_media_plan


def run_sanity_checks() -> dict[str, object]:
    catalog = generate_catalog(seed=42)
    metric = ObjectiveMetric.CONVERSIONS

    # 1. A larger usable budget cannot reduce the optimized KPI frontier.
    budgets = [100_000, 500_000, 1_200_000, 5_000_000]
    frontier = [
        optimize_type_a(catalog, 21, metric, budget).achieved_value
        for budget in budgets
    ]
    assert all(right + 1e-7 >= left for left, right in zip(frontier, frontier[1:]))

    # 2. Capacity grows linearly with exchangeable additional days.
    horizon_maxima = [maximum_achievable(catalog, h, metric) for h in (7, 14, 28)]
    assert horizon_maxima == sorted(horizon_maxima)

    # 3. A larger target cannot have a smaller minimum budget.
    low_target = optimize_type_b(catalog, 14, ObjectiveMetric.CLICKS, 10_000)
    high_target = optimize_type_b(catalog, 14, ObjectiveMetric.CLICKS, 20_000)
    assert sum(high_target.channel_spend.values()) >= sum(low_target.channel_spend.values())

    # 4. Removing inventory cannot improve theoretical maximum capacity.
    reduced = dict(catalog)
    reduced.pop("SMS")
    assert maximum_achievable(reduced, 14, metric) <= maximum_achievable(catalog, 14, metric)

    # 5. Catalog generation is deterministic.
    assert generate_catalog(seed=42) == generate_catalog(seed=42)

    # 6. Full service output is deterministic for a fixed request/catalog.
    request = OptimizationRequest(
        task_type="A",
        horizon_days=21,
        objective_metric="conversions",
        budget=1_200_000,
    )
    first = optimize_media_plan(request, catalog)
    second = optimize_media_plan(request, catalog)
    assert first.model_dump() == second.model_dump()

    # 7. Near total spend capacity, an incremental budget goes to an
    # unsaturated channel with the best remaining incremental return.
    total_cap = sum(c.max_daily_spend * 14 for c in catalog.values())
    base = optimize_type_a(catalog, 14, metric, total_cap - 100_000)
    richer = optimize_type_a(catalog, 14, metric, total_cap - 90_000)
    increments = {
        name: richer.channel_spend[name] - base.channel_spend[name] for name in catalog
    }
    recipient = max(increments, key=increments.get)
    assert increments[recipient] > 9_999
    assert base.channel_spend[recipient] < catalog[recipient].max_daily_spend * 14

    # 8. Heterogeneous channel parameters should yield a nonuniform allocation.
    diversified = optimize_type_a(catalog, 21, metric, 5_000_000)
    positive_spend = [
        value for value in diversified.channel_spend.values() if value > 1e-6
    ]
    assert len(positive_spend) >= 2
    assert len({round(value, 2) for value in positive_spend}) > 1

    # 9. Excess Type A budget remains unspent at inventory limits.
    huge_budget = total_cap * 2
    saturated = optimize_media_plan(
        OptimizationRequest(
            task_type="A",
            horizon_days=14,
            objective_metric="clicks",
            budget=huge_budget,
        ),
        catalog,
    )
    assert saturated.summary is not None
    assert saturated.summary.inventory_saturated
    assert saturated.summary.unspent_budget is not None
    assert saturated.summary.unspent_budget > 0

    # 10. Every numeric result exposed by the plan is finite.
    dump = saturated.model_dump()

    def assert_finite(value: object) -> None:
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for item in value.values():
                assert_finite(item)
        elif isinstance(value, list):
            for item in value:
                assert_finite(item)

    assert_finite(dump)

    infeasible = optimize_media_plan(
        OptimizationRequest(
            task_type="B",
            horizon_days=1,
            objective_metric="conversions",
            target_value=1e12,
        ),
        catalog,
    )
    return {
        "status": "passed",
        "type_a": {
            "budget": request.budget,
            "spend": first.summary.spend if first.summary else None,
            "conversions": first.summary.conversions if first.summary else None,
        },
        "type_b_feasible": {
            "target_clicks": 10_000,
            "spend": sum(low_target.channel_spend.values()),
            "clicks": low_target.achieved_value,
        },
        "type_b_infeasible": {
            "requested_conversions": 1e12,
            "maximum": infeasible.feasibility.maximum_achievable
            if infeasible.feasibility
            else None,
            "reason": infeasible.feasibility.reason.value
            if infeasible.feasibility and infeasible.feasibility.reason
            else None,
        },
        "budget_frontier": dict(zip(budgets, frontier, strict=True)),
    }


def main() -> None:
    print(json.dumps(run_sanity_checks(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
