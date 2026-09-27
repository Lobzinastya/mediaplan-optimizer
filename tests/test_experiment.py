from experiments.compare_policies import RUN_COLUMNS, SUMMARY_COLUMNS, run_policy_comparison
from mediaplan_optimizer.catalog import load_catalog
from mediaplan_optimizer.schemas import ObjectiveMetric


def test_comparison_tables_have_stable_schema_and_all_policies():
    summary, runs, daily, channels = run_policy_comparison(
        load_catalog(),
        [3, 4],
        budget=60_000,
        horizon_days=3,
        objective=ObjectiveMetric.CLICKS,
        quantum_rub=5_000,
    )
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert list(runs.columns) == RUN_COLUMNS
    assert set(summary["policy"]) == {
        "static",
        "periodic_reoptimization",
        "thompson_sampling",
        "linucb",
        "oracle",
    }
    assert len(runs) == 10
    assert not daily.empty
    assert not channels.empty
