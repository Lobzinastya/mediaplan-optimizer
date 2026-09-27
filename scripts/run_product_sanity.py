"""Exercise the complete product workflow through pure application services."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mediaplan_optimizer.campaign import start_campaign, switch_policy
from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.control import (
    create_policy_runtime,
    recalculate_remaining_plan,
    simulate_next_days,
)
from mediaplan_optimizer.monitoring import build_monitoring
from mediaplan_optimizer.schemas import OptimizationRequest
from mediaplan_optimizer.service import optimize_media_plan
from mediaplan_optimizer.simulator import CampaignSimulator


def run_product_sanity() -> dict[str, object]:
    catalog = generate_catalog(seed=42)
    request = OptimizationRequest(
        task_type="A",
        horizon_days=21,
        objective_metric="conversions",
        budget=1_200_000,
    )
    plan = optimize_media_plan(request, catalog)
    state = start_campaign(
        "End-to-end product sanity",
        request,
        plan,
        catalog,
        campaign_id="product-sanity",
    )
    simulator = CampaignSimulator(state.planning_catalog, 21, seed=42)
    runtime = create_policy_runtime(state, seed=42)

    state, runtime = simulate_next_days(state, simulator, runtime, days=3)
    first_history = state.observations
    first_monitoring = build_monitoring(state).progress
    assert state.current_day == 3
    assert math.isfinite(first_monitoring.projected_final_kpi)

    state = switch_policy(
        state, "periodic_reoptimization", periodic_interval_days=3
    )
    runtime = create_policy_runtime(state, seed=42)
    state = recalculate_remaining_plan(
        state, runtime, reason="Sanity periodic reoptimization after day 3"
    )
    assert state.observations == first_history
    assert len(state.plan_versions) == 2

    state = switch_policy(state, "thompson_sampling")
    runtime = create_policy_runtime(state, seed=42)
    state = recalculate_remaining_plan(
        state, runtime, reason="Sanity policy switch to Thompson sampling"
    )
    assert state.observations == first_history
    assert len(state.plan_versions) == 3

    state, runtime = simulate_next_days(state, simulator, runtime, days=4)
    assert state.current_day == 7
    assert state.observations[: len(first_history)] == first_history
    state, runtime = simulate_next_days(
        state, simulator, runtime, days=state.remaining_days
    )
    final = build_monitoring(state).progress

    keys = [(item.day, item.channel) for item in state.observations]
    assert len(keys) == len(set(keys))
    assert state.current_day == state.horizon_days
    assert state.actual_spend <= state.original_budget + 0.1
    assert state.remaining_budget <= 0.1
    assert math.isfinite(final.projected_final_kpi)
    assert final.projected_final_kpi == final.actual_kpi_to_date
    assert state.observations[: len(first_history)] == first_history

    return {
        "status": "passed",
        "campaign_id": state.campaign_id,
        "days_completed": state.current_day,
        "observations": len(state.observations),
        "plan_versions": len(state.plan_versions),
        "policies_exercised": [
            "static",
            "periodic_reoptimization",
            "thompson_sampling",
        ],
        "actual_spend": state.actual_spend,
        "remaining_budget": state.remaining_budget,
        "planned_conversions": plan.summary.conversions if plan.summary else None,
        "actual_conversions": state.actual_kpi,
        "final_projected_conversions": final.projected_final_kpi,
        "history_preserved": True,
        "duplicate_day_channels": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_product_sanity(), indent=2))
