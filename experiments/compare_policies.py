"""Compare static, periodic, Thompson, LinUCB, and oracle allocations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mediaplan_optimizer.adaptive import CampaignRunConfig, run_adaptive_campaign
from mediaplan_optimizer.bandits import (
    LinUCBPolicy,
    PeriodicReoptimizationPolicy,
    StaticPolicy,
    ThompsonSamplingPolicy,
)
from mediaplan_optimizer.catalog import load_catalog
from mediaplan_optimizer.schemas import ChannelConfig, ObjectiveMetric
from mediaplan_optimizer.simulator import CampaignSimulator


RUN_COLUMNS = [
    "policy", "seed", "total_spend", "total_clicks", "total_conversions",
    "total_kpi", "cpc", "cpa", "oracle_kpi", "regret_to_oracle",
    "kpi_delta_vs_static",
]
SUMMARY_COLUMNS = [
    "policy", "runs", "mean_kpi", "std_kpi", "median_kpi", "q25_kpi",
    "q75_kpi", "mean_cpc", "mean_cpa", "mean_regret_to_oracle",
    "mean_delta_vs_static",
]


def run_policy_comparison(
    catalog: dict[str, ChannelConfig],
    seeds: Iterable[int],
    *,
    budget: float = 1_200_000.0,
    horizon_days: int = 21,
    objective: ObjectiveMetric = ObjectiveMetric.CONVERSIONS,
    quantum_rub: float = 10_000.0,
    reoptimization_interval: int = 3,
    thompson_prior_strength: float = 5_000.0,
    linucb_alpha: float = 0.7,
    parameter_deviation: float = 0.20,
    context_strength: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return summary, run, daily, and channel tables for paired seeds."""
    results = []
    for seed in seeds:
        config = CampaignRunConfig(
            budget=budget,
            horizon_days=horizon_days,
            objective=objective,
            quantum_rub=quantum_rub,
            seed=int(seed),
        )
        truth_environment = CampaignSimulator(
            catalog,
            horizon_days,
            int(seed),
            parameter_deviation=parameter_deviation,
            context_strength=context_strength,
        )
        policies = [
            StaticPolicy(),
            PeriodicReoptimizationPolicy(interval_days=reoptimization_interval),
            ThompsonSamplingPolicy(prior_strength=thompson_prior_strength),
            LinUCBPolicy(alpha=linucb_alpha),
            StaticPolicy(
                planning_catalog=truth_environment.oracle_catalog(), name="oracle"
            ),
        ]
        seed_results = [
            run_adaptive_campaign(
                policy,
                catalog,
                config,
                CampaignSimulator(
                    catalog,
                    horizon_days,
                    int(seed),
                    parameter_deviation=parameter_deviation,
                    context_strength=context_strength,
                ),
            )
            for policy in policies
        ]
        oracle_kpi = next(item.total_kpi for item in seed_results if item.policy == "oracle")
        static_kpi = next(item.total_kpi for item in seed_results if item.policy == "static")
        for result in seed_results:
            result.oracle_kpi = oracle_kpi
            result.regret_to_oracle = oracle_kpi - result.total_kpi
            results.append((result, result.total_kpi - static_kpi))

    run_rows = [
        {
            "policy": result.policy,
            "seed": result.seed,
            "total_spend": result.total_spend,
            "total_clicks": result.total_clicks,
            "total_conversions": result.total_conversions,
            "total_kpi": result.total_kpi,
            "cpc": result.cpc,
            "cpa": result.cpa,
            "oracle_kpi": result.oracle_kpi,
            "regret_to_oracle": result.regret_to_oracle,
            "kpi_delta_vs_static": delta,
        }
        for result, delta in results
    ]
    runs = pd.DataFrame(run_rows, columns=RUN_COLUMNS)
    daily = pd.concat([result.daily for result, _ in results], ignore_index=True)
    channels = pd.concat(
        [result.channels for result, _ in results], ignore_index=True
    )

    summary_rows = []
    for policy, group in runs.groupby("policy", sort=False):
        summary_rows.append(
            {
                "policy": policy,
                "runs": len(group),
                "mean_kpi": group["total_kpi"].mean(),
                "std_kpi": group["total_kpi"].std(ddof=1),
                "median_kpi": group["total_kpi"].median(),
                "q25_kpi": group["total_kpi"].quantile(0.25),
                "q75_kpi": group["total_kpi"].quantile(0.75),
                "mean_cpc": group["cpc"].mean(),
                "mean_cpa": group["cpa"].mean(),
                "mean_regret_to_oracle": group["regret_to_oracle"].mean(),
                "mean_delta_vs_static": group["kpi_delta_vs_static"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    return summary, runs, daily, channels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20, help="number of seeds")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=float, default=1_200_000.0)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument(
        "--objective", choices=("clicks", "conversions"), default="conversions"
    )
    parser.add_argument("--quantum", type=float, default=10_000.0)
    parser.add_argument("--reopt-interval", type=int, default=3)
    parser.add_argument("--ts-prior-strength", type=float, default=5_000.0)
    parser.add_argument("--linucb-alpha", type=float, default=0.7)
    parser.add_argument("--parameter-deviation", type=float, default=0.20)
    parser.add_argument("--context-strength", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments") / "results"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    objective = ObjectiveMetric(args.objective)
    catalog = load_catalog()
    configuration = {
        "planning_catalog": {name: channel.model_dump(mode="json") for name, channel in catalog.items()},
        "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        "budget": args.budget,
        "horizon_days": args.horizon,
        "objective": objective.value,
        "quantum_rub": args.quantum,
        "reoptimization_interval_days": args.reopt_interval,
        "thompson_prior_strength": args.ts_prior_strength,
        "linucb_alpha": args.linucb_alpha,
        "parameter_deviation": args.parameter_deviation,
        "context_strength": args.context_strength,
    }
    summary, runs, daily, channels = run_policy_comparison(
        catalog,
        configuration["seeds"],
        budget=args.budget,
        horizon_days=args.horizon,
        objective=objective,
        quantum_rub=args.quantum,
        reoptimization_interval=args.reopt_interval,
        thompson_prior_strength=args.ts_prior_strength,
        linucb_alpha=args.linucb_alpha,
        parameter_deviation=args.parameter_deviation,
        context_strength=args.context_strength,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    daily.to_csv(args.output_dir / "daily_curves.csv", index=False)
    channels.to_csv(args.output_dir / "channel_totals.csv", index=False)
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(configuration, indent=2))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


if __name__ == "__main__":
    main()
