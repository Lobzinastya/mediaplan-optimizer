"""Reconstruct daily plans and aggregate mathematically correct metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .calendar import effective_channel
from .curves import channel_response
from .schemas import (
    ChannelConfig,
    ChannelMetrics,
    DailyAllocation,
    ObjectiveMetric,
    PlanSummary,
    TemporalProfile,
)


def _metric_name(metric: ObjectiveMetric | str) -> str:
    name = metric.value if isinstance(metric, ObjectiveMetric) else str(metric)
    if name not in {item.value for item in ObjectiveMetric}:
        raise ValueError(f"Unsupported objective metric: {metric!r}")
    return name


def _finite_nonnegative(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return number


def _safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _safe_cost(spend: float, volume: float, factor: float = 1.0) -> float | None:
    return spend / volume * factor if volume > 0 else None


def build_daily_allocations(
    channel_totals: Mapping[str, float],
    horizon_days: int,
    catalog: Mapping[str, ChannelConfig],
    objective_metric: ObjectiveMetric | str,
) -> list[DailyAllocation]:
    """Split Uniform channel totals evenly; repeat the end-of-day cumulative KPI."""
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")

    objective = _metric_name(objective_metric)
    daily_spend: dict[str, float] = {}
    for channel, total in channel_totals.items():
        if channel not in catalog:
            raise KeyError(f"Unknown channel: {channel}")
        daily_spend[channel] = _finite_nonnegative(total, f"spend for {channel}") / horizon_days

    allocations: list[DailyAllocation] = []
    cumulative = 0.0
    for day in range(1, horizon_days + 1):
        day_rows: list[tuple[str, float, dict[str, float], float | None]] = []
        for channel, spend in daily_spend.items():
            response = channel_response(spend, catalog[channel])
            values = {
                field: _finite_nonnegative(getattr(response, field), field)
                for field in ("impressions", "reach", "clicks", "conversions")
            }
            raw_video_views = response.video_views
            video_views = (
                None
                if raw_video_views is None
                else _finite_nonnegative(raw_video_views, "video_views")
            )
            day_rows.append((channel, spend, values, video_views))

        # The cumulative KPI is a plan-level end-of-day value.  Repeating it
        # across the day's rows avoids dependence on arbitrary channel order.
        cumulative += sum(values[objective] for _, _, values, _ in day_rows)
        for channel, spend, values, video_views in day_rows:
            allocations.append(
                DailyAllocation(
                    day=day,
                    channel=channel,
                    spend=spend,
                    video_views=video_views,
                    cumulative_target_kpi=cumulative,
                    **values,
                )
            )
    return allocations


def build_calendar_allocations(
    cell_spend: Mapping[tuple[int, str], float],
    profiles: Sequence[TemporalProfile],
    catalog: Mapping[str, ChannelConfig],
    objective_metric: ObjectiveMetric | str,
) -> list[DailyAllocation]:
    """Evaluate the actual optimized channel-day cells, never an equal split."""
    objective = _metric_name(objective_metric)
    rows: list[DailyAllocation] = []
    cumulative = 0.0
    for profile in sorted(profiles, key=lambda item: (item.day, item.channel)):
        if profile.channel not in catalog:
            continue
        channel = effective_channel(catalog[profile.channel], profile)
        spend = _finite_nonnegative(
            cell_spend.get((profile.day, profile.channel), 0.0), "calendar spend"
        )
        if spend > channel.max_daily_spend + 1e-6:
            raise ValueError("calendar allocation exceeds day-specific cap")
        response = channel_response(spend, channel)
        values = {
            field: _finite_nonnegative(getattr(response, field), field)
            for field in ("impressions", "reach", "clicks", "conversions")
        }
        raw_video = response.video_views
        video = None if raw_video is None else _finite_nonnegative(raw_video, "video_views")
        rows.append(DailyAllocation(
            day=profile.day, date=profile.date, weekday=profile.weekday,
            channel=profile.channel, spend=spend, video_views=video,
            cumulative_target_kpi=0.0, **values,
        ))
    by_day: dict[int, list[DailyAllocation]] = {}
    for row in rows:
        by_day.setdefault(row.day, []).append(row)
    completed: list[DailyAllocation] = []
    for day in sorted(by_day):
        cumulative += sum(getattr(row, objective) for row in by_day[day])
        completed.extend(row.model_copy(update={"cumulative_target_kpi": cumulative})
                         for row in by_day[day])
    return completed


def _totals(records: Sequence[DailyAllocation]) -> dict[str, float | None]:
    spend = sum(record.spend for record in records)
    impressions = sum(record.impressions for record in records)
    reach = sum(record.reach for record in records)
    clicks = sum(record.clicks for record in records)
    conversions = sum(record.conversions for record in records)
    video_records = [record for record in records if record.video_views is not None]
    video_views = (
        sum(record.video_views or 0.0 for record in video_records) if video_records else None
    )
    video_impressions = sum(record.impressions for record in video_records)
    return {
        "spend": spend,
        "impressions": impressions,
        "reach": reach,
        "clicks": clicks,
        "conversions": conversions,
        "video_views": video_views,
        "ctr": _safe_rate(clicks, impressions),
        "cr": _safe_rate(conversions, clicks),
        "vtr": None if video_views is None else _safe_rate(video_views, video_impressions),
        "cpm": _safe_cost(spend, impressions, 1000.0),
        "cpc": _safe_cost(spend, clicks),
        "cpa": _safe_cost(spend, conversions),
    }


def aggregate_channel_metrics(
    allocations: Sequence[DailyAllocation],
) -> list[ChannelMetrics]:
    """Aggregate records by channel using ratios of totals, never mean rates."""
    grouped: dict[str, list[DailyAllocation]] = {}
    for allocation in allocations:
        grouped.setdefault(allocation.channel, []).append(allocation)

    return [
        ChannelMetrics(channel=channel, **_totals(records))
        for channel, records in grouped.items()
    ]


def aggregate_plan_summary(
    allocations: Sequence[DailyAllocation],
    objective_metric: ObjectiveMetric | str,
    available_budget: float | None = None,
    requested_target: float | None = None,
    inventory_saturated: bool = False,
) -> PlanSummary:
    """Aggregate overall plan totals, rates, costs, and objective achievement."""
    objective = _metric_name(objective_metric)
    totals = _totals(allocations)

    if available_budget is not None:
        available_budget = _finite_nonnegative(available_budget, "available_budget")
    if requested_target is not None:
        requested_target = _finite_nonnegative(requested_target, "requested_target")

    spend = float(totals["spend"])
    summary_values = {key: value for key, value in totals.items() if key != "video_views"}
    unspent_budget = None
    if available_budget is not None:
        unspent_budget = max(0.0, available_budget - spend)
        if unspent_budget <= max(1e-9, available_budget * 1e-12):
            unspent_budget = 0.0
    return PlanSummary(
        **summary_values,
        available_budget=available_budget,
        unspent_budget=unspent_budget,
        requested_target=requested_target,
        achieved_target=float(totals[objective]),
        inventory_saturated=inventory_saturated,
    )


def aggregate_weekly_metrics(
    allocations: Sequence[DailyAllocation],
) -> list[dict[str, int | float]]:
    """Aggregate daily predictions into consecutive seven-day periods.

    Reach retains the stable contract: it is additive expected reach without
    cross-day or cross-channel audience deduplication.
    """
    weeks: dict[int, list[DailyAllocation]] = {}
    for allocation in allocations:
        week = (allocation.day - 1) // 7 + 1
        weeks.setdefault(week, []).append(allocation)

    result: list[dict[str, int | float]] = []
    for week, records in sorted(weeks.items()):
        totals = _totals(records)
        result.append(
            {
                "week": week,
                "start_day": min(record.day for record in records),
                "end_day": max(record.day for record in records),
                "spend": float(totals["spend"]),
                "impressions": float(totals["impressions"]),
                "non_deduplicated_reach": float(totals["reach"]),
                "clicks": float(totals["clicks"]),
                "conversions": float(totals["conversions"]),
                "cumulative_target_kpi": max(
                    record.cumulative_target_kpi for record in records
                ),
            }
        )
    return result
