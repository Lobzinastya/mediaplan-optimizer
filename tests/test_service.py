import pytest

from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.schemas import OptimizationRequest, ResultStatus
from mediaplan_optimizer.service import optimize_media_plan


def test_type_a_end_to_end_and_exclusion():
    catalog = generate_catalog(seed=42)
    request = OptimizationRequest(
        task_type="A",
        horizon_days=21,
        objective_metric="conversions",
        budget=1_200_000,
        excluded_channels=["SMS"],
    )
    result = optimize_media_plan(request, catalog)
    assert result.status == ResultStatus.OPTIMAL
    assert result.summary is not None
    assert result.summary.spend <= request.budget + 0.1
    assert result.summary.conversions > 0
    assert "SMS" not in {row.channel for row in result.allocations}
    assert len(result.allocations) == 7 * 21


def test_type_b_feasible_end_to_end():
    catalog = generate_catalog(seed=42)
    request = OptimizationRequest(
        task_type="B", horizon_days=14, objective_metric="clicks", target_value=10_000
    )
    result = optimize_media_plan(request, catalog)
    assert result.status == ResultStatus.OPTIMAL
    assert result.summary is not None
    assert result.summary.achieved_target >= request.target_value * (1 - 1e-7)
    assert result.feasibility is not None and result.feasibility.feasible


def test_type_b_infeasible_skips_optimizer():
    catalog = generate_catalog(seed=42)
    request = OptimizationRequest(
        task_type="B",
        horizon_days=1,
        objective_metric="conversions",
        target_value=1e12,
    )
    result = optimize_media_plan(request, catalog)
    assert result.status == ResultStatus.INFEASIBLE
    assert not result.allocations
    assert result.optimizer_metadata.solver == "not-run"
    assert result.feasibility is not None
    assert result.feasibility.target_gap > 0


def test_unknown_and_empty_effective_channel_selection_rejected():
    catalog = generate_catalog(seed=42)
    with pytest.raises(ValueError, match="Unknown channels"):
        optimize_media_plan(
            OptimizationRequest(
                task_type="A",
                horizon_days=7,
                objective_metric="clicks",
                budget=100_000,
                included_channels=["Unknown"],
            ),
            catalog,
        )
    with pytest.raises(ValueError, match="At least one"):
        optimize_media_plan(
            OptimizationRequest(
                task_type="A",
                horizon_days=7,
                objective_metric="clicks",
                budget=100_000,
                included_channels=["SMS"],
                excluded_channels=["SMS"],
            ),
            catalog,
        )
