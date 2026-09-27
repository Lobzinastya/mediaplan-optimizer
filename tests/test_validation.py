"""Public validation contract tests for optimization inputs."""

import pytest
from pydantic import ValidationError

from mediaplan_optimizer.schemas import ChannelConfig, OptimizationRequest


def valid_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "task_type": "A",
        "horizon_days": 30,
        "objective_metric": "clicks",
        "budget": 100_000,
    }
    request.update(overrides)
    return request


def valid_channel(**overrides: object) -> dict[str, object]:
    channel: dict[str, object] = {
        "name": "Display",
        "daily_capacity": 1_000_000,
        "daily_reach_capacity": 500_000,
        "response_scale_rub": 100_000,
        "max_daily_spend": 50_000,
        "average_frequency": 2,
        "base_ctr": 0.02,
        "base_cr": 0.05,
        "base_vtr": 0.4,
    }
    channel.update(overrides)
    return channel


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_type_a_requires_positive_budget(budget: float) -> None:
    with pytest.raises(ValidationError):
        OptimizationRequest(**valid_request(budget=budget))


@pytest.mark.parametrize("target_value", [0, -1, float("inf"), float("nan")])
def test_type_b_requires_positive_target(target_value: float) -> None:
    with pytest.raises(ValidationError):
        OptimizationRequest(
            **valid_request(
                task_type="B", budget=None, target_value=target_value
            )
        )


@pytest.mark.parametrize("horizon_days", [0, -1, 366])
def test_horizon_must_be_between_one_and_365_days(horizon_days: int) -> None:
    with pytest.raises(ValidationError):
        OptimizationRequest(**valid_request(horizon_days=horizon_days))


@pytest.mark.parametrize("selection", ["included_channels", "excluded_channels"])
def test_explicit_channel_selection_cannot_be_empty(selection: str) -> None:
    with pytest.raises(ValidationError, match=f"{selection} cannot be empty"):
        OptimizationRequest(**valid_request(**{selection: []}))


def test_unknown_objective_metric_is_rejected() -> None:
    with pytest.raises(ValidationError) as error:
        OptimizationRequest(**valid_request(objective_metric="revenue"))

    assert error.value.errors()[0]["loc"] == ("objective_metric",)


def test_type_a_rejects_reach_objective() -> None:
    with pytest.raises(
        ValidationError,
        match="Type A objective_metric must be clicks or conversions",
    ):
        OptimizationRequest(**valid_request(objective_metric="reach"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_value": 10}, "Type A does not accept target_value"),
        (
            {"task_type": "B", "budget": 10, "target_value": 100},
            "Type B does not accept budget",
        ),
    ],
)
def test_task_specific_forbidden_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        OptimizationRequest(**valid_request(**overrides))


@pytest.mark.parametrize("objective_metric", ["clicks", "conversions"])
def test_valid_type_a_request(objective_metric: str) -> None:
    request = OptimizationRequest(
        **valid_request(objective_metric=objective_metric)
    )

    assert request.task_type.value == "A"
    assert request.objective_metric.value == objective_metric


@pytest.mark.parametrize("objective_metric", ["reach", "clicks", "conversions"])
def test_type_b_accepts_all_public_objective_metrics(
    objective_metric: str,
) -> None:
    request = OptimizationRequest(
        **valid_request(
            task_type="B",
            objective_metric=objective_metric,
            budget=None,
            target_value=1_000,
        )
    )

    assert request.objective_metric.value == objective_metric


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("name", ""),
        ("daily_capacity", 0),
        ("daily_reach_capacity", 0),
        ("response_scale_rub", 0),
        ("base_cpm_rub", 0),
        ("max_daily_spend", 0),
        ("average_frequency", 0.99),
        ("base_ctr", -0.01),
        ("base_ctr", 1.01),
        ("base_cr", -0.01),
        ("base_cr", 1.01),
        ("base_vtr", -0.01),
        ("base_vtr", 1.01),
        ("ctr_saturation_decay", -0.01),
        ("ctr_saturation_decay", 0.51),
        ("cr_saturation_decay", -0.01),
        ("cr_saturation_decay", 0.51),
        ("daily_capacity", float("inf")),
    ],
)
def test_channel_config_rejects_values_outside_public_bounds(
    field: str, invalid_value: object
) -> None:
    with pytest.raises(ValidationError) as error:
        ChannelConfig(**valid_channel(**{field: invalid_value}))

    assert error.value.errors()[0]["loc"] == (field,)


def test_channel_reach_capacity_cannot_exceed_total_capacity() -> None:
    with pytest.raises(
        ValidationError,
        match="daily_reach_capacity cannot exceed daily_capacity",
    ):
        ChannelConfig(
            **valid_channel(
                daily_capacity=100,
                daily_reach_capacity=101,
            )
        )


def test_valid_channel_config_accepts_boundary_rates() -> None:
    channel = ChannelConfig(
        **valid_channel(base_ctr=0, base_cr=1, base_vtr=None)
    )

    assert channel.base_ctr == 0
    assert channel.base_cr == 1
    assert channel.base_vtr is None


@pytest.mark.parametrize("decays", [(0.5, 0.5), (0.4, 0.4), (0.5, 0.1)])
def test_nonmonotone_joint_decay_is_rejected(product_catalog, decays):
    data = next(iter(product_catalog.values())).model_dump()
    data.update(ctr_saturation_decay=decays[0], cr_saturation_decay=decays[1])
    with pytest.raises(ValueError, match="joint CTR/CR"):
        ChannelConfig(**data)
