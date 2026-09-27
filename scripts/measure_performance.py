"""Measure representative local runtimes without changing project state."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.compare_policies import run_policy_comparison
from mediaplan_optimizer.campaign import start_campaign, switch_policy
from mediaplan_optimizer.catalog import load_catalog
from mediaplan_optimizer.control import (
    create_policy_runtime,
    recalculate_remaining_plan,
    simulate_next_days,
)
from mediaplan_optimizer.schemas import ObjectiveMetric, OptimizationRequest
from mediaplan_optimizer.service import optimize_media_plan
from mediaplan_optimizer.simulator import CampaignSimulator


def _median_runtime(action, repeats: int = 3) -> float:
    durations = []
    for _ in range(repeats):
        started = perf_counter()
        action()
        durations.append(perf_counter() - started)
    return statistics.median(durations)


def main() -> None:
    catalog = load_catalog()
    type_a_request = OptimizationRequest(
        task_type="A",
        budget=1_200_000,
        horizon_days=21,
        objective_metric="conversions",
    )
    type_b_request = OptimizationRequest(
        task_type="B",
        target_value=10_000,
        horizon_days=14,
        objective_metric="clicks",
    )
    type_a_result = optimize_media_plan(type_a_request, catalog)
    campaign = start_campaign(
        "Performance check",
        type_a_request,
        type_a_result,
        catalog,
        campaign_id="performance-check",
    )
    simulator = CampaignSimulator(campaign.planning_catalog, 21, seed=42)
    runtime = create_policy_runtime(campaign, seed=42)
    campaign, _runtime = simulate_next_days(
        campaign, simulator, runtime, days=3, quantum_rub=10_000
    )
    campaign = switch_policy(campaign, "periodic_reoptimization")

    measurements = {
        "type_a_seconds_median_3": _median_runtime(
            lambda: optimize_media_plan(type_a_request, catalog)
        ),
        "type_b_seconds_median_3": _median_runtime(
            lambda: optimize_media_plan(type_b_request, catalog)
        ),
        "replan_seconds_median_3": _median_runtime(
            lambda: recalculate_remaining_plan(
                campaign,
                create_policy_runtime(campaign, seed=42),
                reason="Performance measurement",
            )
        ),
    }
    for seed_count in (5, 20):
        started = perf_counter()
        run_policy_comparison(catalog, range(seed_count))
        measurements[f"experiment_{seed_count}_seeds_seconds"] = (
            perf_counter() - started
        )
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
