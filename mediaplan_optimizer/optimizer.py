"""Numerically validated optimizers for budget-capped and KPI-driven plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import minimize

from .curves import marginal_metric_response, metric_response
from .schemas import ChannelConfig, ObjectiveMetric, OptimizerMetadata


@dataclass(frozen=True)
class OptimizationOutcome:
    channel_spend: dict[str, float]
    achieved_value: float
    metadata: OptimizerMetadata


def _total_response(
    values: np.ndarray,
    channels: list[ChannelConfig],
    horizon_days: int,
    metric: ObjectiveMetric,
) -> float:
    return float(
        sum(
            horizon_days * metric_response(total / horizon_days, channel, metric)
            for total, channel in zip(values, channels, strict=True)
        )
    )


def _waterfill_fallback(
    channels: list[ChannelConfig],
    horizon_days: int,
    metric: ObjectiveMetric,
    spendable: float,
) -> np.ndarray:
    """Allocate spend by equalizing decreasing marginal KPI returns."""
    caps = np.array([c.max_daily_spend * horizon_days for c in channels], dtype=float)
    if spendable >= float(caps.sum()) - 1e-8:
        return caps

    def allocation_for_shadow_price(shadow_price: float) -> np.ndarray:
        allocation = np.zeros(len(channels), dtype=float)
        for idx, (channel, cap) in enumerate(zip(channels, caps, strict=True)):
            if shadow_price >= marginal_metric_response(0.0, channel, metric):
                continue
            if shadow_price <= marginal_metric_response(
                channel.max_daily_spend, channel, metric
            ):
                allocation[idx] = cap
                continue
            low_daily, high_daily = 0.0, channel.max_daily_spend
            for _ in range(60):
                middle = (low_daily + high_daily) / 2.0
                if marginal_metric_response(middle, channel, metric) > shadow_price:
                    low_daily = middle
                else:
                    high_daily = middle
            allocation[idx] = (low_daily + high_daily) / 2.0 * horizon_days
        return allocation

    low = 0.0
    high = max(marginal_metric_response(0.0, c, metric) for c in channels)
    result = np.zeros(len(channels), dtype=float)
    feasible_result = result.copy()
    for _ in range(80):
        shadow = (low + high) / 2.0
        result = allocation_for_shadow_price(shadow)
        if float(result.sum()) > spendable:
            low = shadow
        else:
            high = shadow
            feasible_result = result.copy()

    # Retain the last allocation known not to exceed the budget. Floating-point
    # shadow-price resolution can leave a residual, especially for microscopic
    # budgets, so assign every positive remainder by current marginal return.
    result = feasible_result
    remainder = spendable - float(result.sum())
    if remainder > 0.0:
        order = sorted(
            range(len(channels)),
            key=lambda i: marginal_metric_response(
                result[i] / horizon_days, channels[i], metric
            ),
            reverse=True,
        )
        for idx in order:
            addition = min(remainder, caps[idx] - result[idx])
            result[idx] += addition
            remainder -= addition
            if remainder <= 0.0:
                break
    return result


def _validate_allocation(
    values: np.ndarray,
    channels: list[ChannelConfig],
    horizon_days: int,
    budget: float,
) -> tuple[bool, str]:
    caps = np.array([c.max_daily_spend * horizon_days for c in channels], dtype=float)
    tolerance = max(1e-5, budget * 1e-7)
    if not np.all(np.isfinite(values)):
        return False, "allocation contains NaN or infinity"
    if np.any(values < -tolerance):
        return False, "allocation contains negative spend"
    if np.any(values - caps > tolerance):
        return False, "allocation exceeds the channel campaign cap implied by the daily spend limit"
    if float(values.sum()) - budget > tolerance:
        return False, "allocation exceeds available budget"
    return True, "allocation passed numerical validation"


def optimize_type_a(
    catalog: Mapping[str, ChannelConfig],
    horizon_days: int,
    metric: ObjectiveMetric,
    budget: float,
) -> OptimizationOutcome:
    """Maximize KPI under a budget using SLSQP with analytic fallback."""
    names = list(catalog)
    channels = [catalog[name] for name in names]
    caps = np.array([c.max_daily_spend * horizon_days for c in channels], dtype=float)
    spendable = min(float(budget), float(caps.sum()))
    if spendable <= 0 or not channels:
        raise ValueError("optimizer requires selected channels and positive budget")

    # The analytic concave solution is both a strong initial point and an
    # independent optimality benchmark for the numerical solver.
    reference_values = _waterfill_fallback(
        channels, horizon_days, metric, spendable
    )
    result = minimize(
        lambda values: -_total_response(values, channels, horizon_days, metric),
        x0=reference_values,
        method="SLSQP",
        bounds=[(0.0, float(cap)) for cap in caps],
        constraints={"type": "eq", "fun": lambda values: float(values.sum()) - spendable},
        options={"ftol": 1e-11, "maxiter": 500},
    )
    values = np.clip(np.asarray(result.x, dtype=float), 0.0, caps)
    valid, validation_message = _validate_allocation(
        values, channels, horizon_days, budget
    )
    numerical_value = _total_response(values, channels, horizon_days, metric)
    reference_value = _total_response(
        reference_values, channels, horizon_days, metric
    )
    optimality_tolerance = max(1e-7, reference_value * 1e-8)
    numerically_suboptimal = numerical_value + optimality_tolerance < reference_value

    solver = "scipy.optimize.SLSQP"
    message = str(result.message)
    status: int | str = int(result.status)
    iterations = int(result.nit)
    if (
        not result.success
        or not valid
        or numerically_suboptimal
        or abs(float(values.sum()) - spendable) > max(1e-4, spendable * 1e-7)
    ):
        values = reference_values
        valid, validation_message = _validate_allocation(
            values, channels, horizon_days, budget
        )
        solver = "analytic-marginal-waterfill"
        status = "fallback"
        iterations = None
        reason = "suboptimal result" if numerically_suboptimal else str(result.message)
        message = f"SLSQP result replaced by controlled fallback: {reason}"

    achieved = _total_response(values, channels, horizon_days, metric)
    valid = valid and np.isfinite(achieved)
    metadata = OptimizerMetadata(
        success=bool(valid),
        solver=solver,
        status=status,
        message=f"{message}; {validation_message}",
        objective=achieved,
        iterations=iterations,
        validation_passed=bool(valid),
    )
    if not valid:
        raise RuntimeError(metadata.message)
    return OptimizationOutcome(
        channel_spend={name: float(value) for name, value in zip(names, values, strict=True)},
        achieved_value=achieved,
        metadata=metadata,
    )


def optimize_type_b(
    catalog: Mapping[str, ChannelConfig],
    horizon_days: int,
    metric: ObjectiveMetric,
    target_value: float,
) -> OptimizationOutcome:
    """Find minimum budget by bisection on the monotone Type A frontier."""
    upper = float(sum(c.max_daily_spend * horizon_days for c in catalog.values()))
    maximum = optimize_type_a(catalog, horizon_days, metric, upper)
    target_tolerance = max(1e-6, target_value * 1e-7)
    if maximum.achieved_value + target_tolerance < target_value:
        raise ValueError("target is infeasible for the selected catalog and horizon")

    low, high = 0.0, upper
    best = maximum
    budget_tolerance = max(0.01, upper * 1e-9)
    iterations = 0
    while high - low > budget_tolerance and iterations < 70:
        middle = (low + high) / 2.0
        candidate = optimize_type_a(catalog, horizon_days, metric, middle)
        if candidate.achieved_value + target_tolerance >= target_value:
            high = middle
            best = candidate
        else:
            low = middle
        iterations += 1

    achieved = best.achieved_value
    valid = best.metadata.validation_passed and achieved + target_tolerance >= target_value
    metadata = OptimizerMetadata(
        success=valid,
        solver=f"budget-bisection + {best.metadata.solver}",
        status="converged" if valid else "target_not_reached",
        message=(
            f"Minimum-budget frontier converged in {iterations} iterations; "
            f"{best.metadata.message}"
        ),
        objective=float(sum(best.channel_spend.values())),
        iterations=iterations,
        validation_passed=valid,
    )
    if not valid:
        raise RuntimeError(metadata.message)
    return OptimizationOutcome(best.channel_spend, achieved, metadata)
