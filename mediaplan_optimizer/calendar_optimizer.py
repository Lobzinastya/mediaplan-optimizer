"""Vectorized separable concave solver for real channel-by-day decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .calendar import effective_channel
from .curves import buyout_scale_rub
from .schemas import ChannelConfig, ObjectiveMetric, OptimizerMetadata, TemporalProfile


@dataclass(frozen=True)
class CalendarOutcome:
    cell_spend: dict[tuple[int, str], float]
    achieved_value: float
    metadata: OptimizerMetadata


@dataclass(frozen=True)
class _Cells:
    profiles: tuple[TemporalProfile, ...]
    cap: np.ndarray
    capacity: np.ndarray
    reach_capacity: np.ndarray
    frequency: np.ndarray
    scale: np.ndarray
    ctr: np.ndarray
    cr: np.ndarray
    ctr_decay: np.ndarray
    cr_decay: np.ndarray


def _prepare(
    catalog: Mapping[str, ChannelConfig], profiles: Sequence[TemporalProfile]
) -> _Cells:
    selected = tuple(profile for profile in profiles if profile.channel in catalog)
    if not selected or len({(p.day, p.channel) for p in selected}) != len(selected):
        raise ValueError("calendar cells must be nonempty with unique day/channel keys")
    channels = [effective_channel(catalog[p.channel], p) for p in selected]
    return _Cells(
        profiles=selected,
        cap=np.array([c.max_daily_spend for c in channels]),
        capacity=np.array([c.daily_capacity for c in channels]),
        reach_capacity=np.array([c.daily_reach_capacity for c in channels]),
        frequency=np.array([c.average_frequency for c in channels]),
        scale=np.array([buyout_scale_rub(c) for c in channels]),
        ctr=np.array([c.base_ctr for c in channels]),
        cr=np.array([c.base_cr for c in channels]),
        ctr_decay=np.array([c.ctr_saturation_decay for c in channels]),
        cr_decay=np.array([c.cr_saturation_decay for c in channels]),
    )


def _response(spend: np.ndarray, cells: _Cells, metric: ObjectiveMetric) -> np.ndarray:
    saturation = -np.expm1(-spend / cells.scale)
    impressions = cells.capacity * saturation
    if metric == ObjectiveMetric.REACH:
        return cells.reach_capacity * -np.expm1(
            -impressions / (cells.reach_capacity * cells.frequency)
        )
    clicks = impressions * cells.ctr * (1.0 - cells.ctr_decay * saturation)
    if metric == ObjectiveMetric.CLICKS:
        return clicks
    return clicks * cells.cr * (1.0 - cells.cr_decay * saturation)


def _marginal(spend: np.ndarray, cells: _Cells, metric: ObjectiveMetric) -> np.ndarray:
    exp_term = np.exp(-spend / cells.scale)
    saturation = 1.0 - exp_term
    ds = exp_term / cells.scale
    if metric == ObjectiveMetric.REACH:
        impressions = cells.capacity * saturation
        return (np.exp(-impressions / (cells.reach_capacity * cells.frequency))
                * cells.capacity * ds / cells.frequency)
    if metric == ObjectiveMetric.CLICKS:
        return cells.capacity * cells.ctr * ds * (1.0 - 2.0 * cells.ctr_decay * saturation)
    polynomial = (1.0 - 2.0 * (cells.ctr_decay + cells.cr_decay) * saturation
                  + 3.0 * cells.ctr_decay * cells.cr_decay * saturation**2)
    return cells.capacity * cells.ctr * cells.cr * ds * polynomial


def _allocate(cells: _Cells, metric: ObjectiveMetric, budget: float) -> tuple[np.ndarray, float]:
    spendable = min(float(budget), float(cells.cap.sum()))
    if spendable <= 0 or not np.isfinite(spendable):
        raise ValueError("calendar budget must be finite and positive")
    if spendable >= float(cells.cap.sum()) - 1e-8:
        values = cells.cap.copy()
        return values, float(_response(values, cells, metric).sum())
    zero = np.zeros_like(cells.cap)
    marginal_zero = _marginal(zero, cells, metric)
    marginal_cap = _marginal(cells.cap, cells, metric)
    shadow_low, shadow_high = 0.0, float(marginal_zero.max())
    feasible = zero.copy()
    for _ in range(52):
        shadow = (shadow_low + shadow_high) / 2.0
        inactive = shadow >= marginal_zero
        capped = shadow <= marginal_cap
        interior = ~(inactive | capped)
        lower = zero.copy()
        upper = cells.cap.copy()
        for _ in range(38):
            middle = (lower + upper) / 2.0
            above = _marginal(middle, cells, metric) > shadow
            lower = np.where(interior & above, middle, lower)
            upper = np.where(interior & ~above, middle, upper)
        values = np.where(inactive, 0.0, np.where(capped, cells.cap, (lower + upper) / 2.0))
        if float(values.sum()) > spendable:
            shadow_low = shadow
        else:
            shadow_high = shadow
            feasible = values
        if spendable - float(feasible.sum()) <= max(1e-7, spendable * 1e-11):
            break
    remaining = spendable - float(feasible.sum())
    if remaining > 0:
        for index in np.argsort(-_marginal(feasible, cells, metric)):
            addition = min(remaining, cells.cap[index] - feasible[index])
            feasible[index] += addition
            remaining -= addition
            if remaining <= 1e-8:
                break
    if (np.any(~np.isfinite(feasible)) or np.any(feasible < -1e-8)
            or np.any(feasible > cells.cap + 1e-6)
            or float(feasible.sum()) > spendable + max(0.01, spendable * 1e-8)):
        raise RuntimeError("calendar allocation failed spend/cap validation")
    return feasible, float(_response(feasible, cells, metric).sum())


def _outcome(cells: _Cells, values: np.ndarray, achieved: float, iterations: int) -> CalendarOutcome:
    return CalendarOutcome(
        cell_spend={(profile.day, profile.channel): float(spend)
                    for profile, spend in zip(cells.profiles, values, strict=True)},
        achieved_value=achieved,
        metadata=OptimizerMetadata(
            success=True, solver="calendar-vectorized-waterfill", status="optimal",
            message="Separable concave marginal allocation passed cap/budget checks.",
            objective=achieved, iterations=iterations, validation_passed=True,
        ),
    )


def optimize_type_a_calendar(
    catalog: Mapping[str, ChannelConfig], profiles: Sequence[TemporalProfile],
    metric: ObjectiveMetric, budget: float,
) -> CalendarOutcome:
    cells = _prepare(catalog, profiles)
    values, achieved = _allocate(cells, metric, budget)
    return _outcome(cells, values, achieved, 52)


def optimize_type_b_calendar(
    catalog: Mapping[str, ChannelConfig], profiles: Sequence[TemporalProfile],
    metric: ObjectiveMetric, target_value: float,
) -> CalendarOutcome:
    cells = _prepare(catalog, profiles)
    upper = float(cells.cap.sum())
    maximum = float(_response(cells.cap, cells, metric).sum())
    tolerance = max(1e-6, target_value * 1e-7)
    if maximum + tolerance < target_value:
        raise ValueError("calendar target is infeasible")
    low, high = 0.0, upper
    best_values, best_achieved = cells.cap.copy(), maximum
    iterations = 0
    while high - low > max(0.01, upper * 1e-9) and iterations < 70:
        middle = (low + high) / 2.0
        values, achieved = _allocate(cells, metric, middle)
        if achieved + tolerance >= target_value:
            high = middle
            best_values, best_achieved = values, achieved
        else:
            low = middle
        iterations += 1
    return _outcome(cells, best_values, best_achieved, iterations)


def calendar_maximum(
    catalog: Mapping[str, ChannelConfig], profiles: Sequence[TemporalProfile],
    metric: ObjectiveMetric,
) -> float:
    cells = _prepare(catalog, profiles)
    return float(_response(cells.cap, cells, metric).sum())
