"""Transparent plan-versus-fact monitoring, pacing, and reforecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .campaign import CampaignState, future_spend_scale, observations_frame
from .curves import channel_response
from .schemas import ChannelConfig, ObjectiveMetric


DEFAULT_STATUS_THRESHOLD = 0.10
MIN_PROJECTION_INTERVAL_DAYS = 3


@dataclass(frozen=True)
class CampaignProgress:
    current_day: int
    horizon_days: int
    expected_progress_fraction: float
    planned_budget: float
    actual_spend: float
    remaining_budget: float
    planned_spend_to_date: float
    budget_pace: float
    planned_final_kpi: float
    planned_kpi_to_date: float
    actual_kpi_to_date: float
    kpi_pace: float
    plan_vs_fact_variance: float
    projected_final_kpi: float
    projection_interval_lower: float | None
    projection_interval_upper: float | None
    projection_interval_observation_days: int
    forecast_delta: float
    current_unit_cost: float | None
    projected_unit_cost: float | None


@dataclass(frozen=True)
class MonitoringResult:
    progress: CampaignProgress
    daily: pd.DataFrame
    channels: pd.DataFrame
    beliefs: dict[str, ChannelConfig]


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator > 0 else default


def estimate_channel_beliefs(
    state: CampaignState,
    *,
    prior_impressions: float = 20_000.0,
    prior_clicks: float = 500.0,
) -> dict[str, ChannelConfig]:
    """Update CTR/CR with transparent prior-weighted observed rates."""
    frame = observations_frame(state)
    if frame.empty:
        return dict(state.planning_catalog)
    totals = frame.groupby("channel", as_index=True)[
        ["impressions", "clicks", "conversions"]
    ].sum()
    beliefs: dict[str, ChannelConfig] = {}
    for name, base in state.planning_catalog.items():
        if name in totals.index:
            observed = totals.loc[name]
            ctr = (
                base.base_ctr * prior_impressions + float(observed["clicks"])
            ) / (prior_impressions + float(observed["impressions"]))
            cr = (
                base.base_cr * prior_clicks + float(observed["conversions"])
            ) / (prior_clicks + float(observed["clicks"]))
            beliefs[name] = base.model_copy(
                update={"base_ctr": float(ctr), "base_cr": float(cr)}
            )
        else:
            beliefs[name] = base
    return beliefs


def effective_plan_frame(state: CampaignState) -> pd.DataFrame:
    """Return the plan version that was actually in force for each campaign day."""
    records: list[dict[str, object]] = []
    for day in range(1, state.horizon_days + 1):
        applicable = [
            version for version in state.plan_versions if version.created_at_day < day
        ]
        if not applicable:
            continue
        version = applicable[-1]
        for allocation in version.future_allocations:
            if allocation.day == day:
                row = allocation.model_dump()
                row["plan_version"] = version.version_id
                row["plan_policy"] = version.policy
                records.append(row)
    columns = [
        "day",
        "channel",
        "spend",
        "impressions",
        "reach",
        "clicks",
        "conversions",
        "video_views",
        "cumulative_target_kpi",
        "plan_version",
        "plan_policy",
    ]
    return pd.DataFrame(records, columns=columns)


def _daily_comparison(state: CampaignState) -> pd.DataFrame:
    objective = state.objective.value
    original_plan = pd.DataFrame([row.model_dump() for row in state.original_result.allocations])
    original = original_plan.groupby("day", as_index=False).agg(
        original_planned_spend=("spend", "sum"),
        original_planned_kpi=(objective, "sum"),
    )
    effective_plan = effective_plan_frame(state)
    planned = effective_plan.groupby("day", as_index=False).agg(
        planned_spend=("spend", "sum"),
        planned_kpi=(objective, "sum"),
    )
    facts = observations_frame(state)
    if facts.empty:
        actual = pd.DataFrame(columns=["day", "actual_spend", "actual_kpi"])
    else:
        actual = facts.groupby("day", as_index=False).agg(
            actual_spend=("spend", "sum"),
            actual_kpi=(objective, "sum"),
        )
    days = pd.DataFrame({"day": range(1, state.horizon_days + 1)})
    daily = (
        days.merge(original, on="day", how="left")
        .merge(planned, on="day", how="left")
        .merge(actual, on="day", how="left")
    )
    value_columns = (
        "original_planned_spend",
        "original_planned_kpi",
        "planned_spend",
        "planned_kpi",
        "actual_spend",
        "actual_kpi",
    )
    for column in value_columns:
        daily[column] = daily[column].fillna(0.0).astype(float)
    daily["original_planned_cumulative_spend"] = daily[
        "original_planned_spend"
    ].cumsum()
    daily["original_planned_cumulative_kpi"] = daily[
        "original_planned_kpi"
    ].cumsum()
    daily["planned_cumulative_spend"] = daily["planned_spend"].cumsum()
    daily["planned_cumulative_kpi"] = daily["planned_kpi"].cumsum()
    daily["actual_cumulative_spend"] = daily["actual_spend"].cumsum()
    daily["actual_cumulative_kpi"] = daily["actual_kpi"].cumsum()
    return daily


def _future_forecast(
    state: CampaignState, beliefs: dict[str, ChannelConfig]
) -> tuple[float, float]:
    objective = state.objective.value
    future_kpi = 0.0
    future_spend = 0.0
    scale = future_spend_scale(state)
    for allocation in state.latest_plan.future_allocations:
        if allocation.day <= state.current_day:
            continue
        response = channel_response(allocation.spend * scale, beliefs[allocation.channel])
        future_kpi += float(getattr(response, objective))
        future_spend += allocation.spend * scale
    return future_kpi, future_spend


def _empirical_projection_interval(
    state: CampaignState,
    beliefs: dict[str, ChannelConfig],
    *,
    future_kpi: float,
) -> tuple[float | None, float | None, int]:
    """Project observed day-level variation; this is not a confidence interval."""
    facts = observations_frame(state)
    if facts.empty:
        return None, None, 0
    objective = state.objective.value
    ratios: list[float] = []
    for _day, day_rows in facts.groupby("day"):
        actual = float(day_rows[objective].sum())
        expected = 0.0
        for row in day_rows.itertuples(index=False):
            response = channel_response(float(row.spend), beliefs[row.channel])
            expected += float(getattr(response, objective))
        if expected > 0:
            ratios.append(actual / expected)
    observation_days = len(ratios)
    if observation_days < MIN_PROJECTION_INTERVAL_DAYS:
        return None, None, observation_days
    lower_multiplier, upper_multiplier = np.quantile(ratios, [0.10, 0.90])
    point = state.actual_kpi + future_kpi
    lower = state.actual_kpi + future_kpi * max(0.0, float(lower_multiplier))
    upper = state.actual_kpi + future_kpi * max(0.0, float(upper_multiplier))
    return min(lower, point), max(upper, point), observation_days


def _channel_comparison(
    state: CampaignState,
    *,
    status_threshold: float,
) -> pd.DataFrame:
    if not 0 < status_threshold < 1:
        raise ValueError("status_threshold must be between 0 and 1")
    objective = state.objective.value
    plan = effective_plan_frame(state)
    elapsed = plan[plan["day"] <= state.current_day]
    planned = elapsed.groupby("channel", as_index=False).agg(
        planned_spend=("spend", "sum"),
        planned_impressions=("impressions", "sum"),
        planned_reach=("reach", "sum"),
        planned_clicks=("clicks", "sum"),
        planned_conversions=("conversions", "sum"),
    )
    facts = observations_frame(state)
    if facts.empty:
        actual = pd.DataFrame(
            columns=[
                "channel",
                "actual_spend",
                "actual_impressions",
                "actual_reach",
                "actual_clicks",
                "actual_conversions",
            ]
        )
    else:
        actual = facts.groupby("channel", as_index=False).agg(
            actual_spend=("spend", "sum"),
            actual_impressions=("impressions", "sum"),
            actual_reach=("reach", "sum"),
            actual_clicks=("clicks", "sum"),
            actual_conversions=("conversions", "sum"),
        )
    channels = pd.DataFrame({"channel": list(state.planning_catalog)})
    result = channels.merge(planned, on="channel", how="left").merge(
        actual, on="channel", how="left"
    )
    numeric = [column for column in result if column != "channel"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result["spend_pace"] = np.where(
        result["planned_spend"] > 0,
        result["actual_spend"] / result["planned_spend"],
        0.0,
    )
    result["spend_delta"] = result["actual_spend"] - result["planned_spend"]
    result["planned_ctr"] = np.where(
        result["planned_impressions"] > 0,
        result["planned_clicks"] / result["planned_impressions"],
        0.0,
    )
    result["actual_ctr"] = np.where(
        result["actual_impressions"] > 0,
        result["actual_clicks"] / result["actual_impressions"],
        0.0,
    )
    result["ctr_delta"] = result["actual_ctr"] - result["planned_ctr"]
    result["planned_cr"] = np.where(
        result["planned_clicks"] > 0,
        result["planned_conversions"] / result["planned_clicks"],
        0.0,
    )
    result["actual_cr"] = np.where(
        result["actual_clicks"] > 0,
        result["actual_conversions"] / result["actual_clicks"],
        0.0,
    )
    result["cr_delta"] = result["actual_cr"] - result["planned_cr"]
    result["planned_cpa"] = np.where(
        result["planned_conversions"] > 0,
        result["planned_spend"] / result["planned_conversions"],
        np.nan,
    )
    result["actual_cpa"] = np.where(
        result["actual_conversions"] > 0,
        result["actual_spend"] / result["actual_conversions"],
        np.nan,
    )
    result["cpa_delta"] = result["actual_cpa"] - result["planned_cpa"]
    planned_kpi = result[f"planned_{objective}"]
    actual_kpi = result[f"actual_{objective}"]
    result["planned_kpi"] = planned_kpi
    result["actual_kpi"] = actual_kpi
    result["kpi_delta"] = actual_kpi - planned_kpi
    planned_efficiency = np.where(
        result["planned_spend"] > 0, planned_kpi / result["planned_spend"], 0.0
    )
    actual_efficiency = np.where(
        result["actual_spend"] > 0, actual_kpi / result["actual_spend"], 0.0
    )
    result["planned_unit_cost"] = np.where(
        planned_kpi > 0, result["planned_spend"] / planned_kpi, np.nan
    )
    result["actual_unit_cost"] = np.where(
        actual_kpi > 0, result["actual_spend"] / actual_kpi, np.nan
    )
    result["unit_cost_delta"] = (
        result["actual_unit_cost"] - result["planned_unit_cost"]
    )
    ratio = np.divide(
        actual_efficiency,
        planned_efficiency,
        out=np.ones_like(actual_efficiency, dtype=float),
        where=planned_efficiency > 0,
    )
    result["status"] = np.select(
        [
            (result["actual_spend"] > 0) & (ratio > 1 + status_threshold),
            (result["actual_spend"] > 0) & (ratio < 1 - status_threshold),
        ],
        ["Outperforming", "Underperforming"],
        default="On track",
    )
    return result


def build_monitoring(
    state: CampaignState,
    *,
    status_threshold: float = DEFAULT_STATUS_THRESHOLD,
) -> MonitoringResult:
    """Build a deterministic PLAN / FACT / FORECAST monitoring snapshot."""
    summary = state.original_result.summary
    if summary is None:
        raise ValueError("campaign original plan has no summary")
    daily = _daily_comparison(state)
    beliefs = estimate_channel_beliefs(state)
    future_kpi, future_spend = _future_forecast(state, beliefs)
    actual_kpi = state.actual_kpi
    projected = actual_kpi + future_kpi
    interval_lower, interval_upper, interval_days = _empirical_projection_interval(
        state,
        beliefs,
        future_kpi=future_kpi,
    )
    projected_spend = state.actual_spend + future_spend
    planned_to_date = daily.loc[
        daily["day"] <= state.current_day, "planned_spend"
    ].sum()
    planned_kpi_to_date = daily.loc[
        daily["day"] <= state.current_day, "planned_kpi"
    ].sum()
    budget_pace = _safe_ratio(state.actual_spend, float(planned_to_date), 1.0)
    kpi_pace = _safe_ratio(actual_kpi, float(planned_kpi_to_date), 1.0)
    variance = _safe_ratio(
        actual_kpi - float(planned_kpi_to_date),
        float(planned_kpi_to_date),
        0.0,
    )
    original_forecast = float(summary.achieved_target)
    unit_cost = _safe_ratio(state.actual_spend, actual_kpi) if actual_kpi > 0 else None
    projected_cost = _safe_ratio(projected_spend, projected) if projected > 0 else None
    progress = CampaignProgress(
        current_day=state.current_day,
        horizon_days=state.horizon_days,
        expected_progress_fraction=state.current_day / state.horizon_days,
        planned_budget=state.original_budget,
        actual_spend=state.actual_spend,
        remaining_budget=state.remaining_budget,
        planned_spend_to_date=float(planned_to_date),
        budget_pace=budget_pace,
        planned_final_kpi=original_forecast,
        planned_kpi_to_date=float(planned_kpi_to_date),
        actual_kpi_to_date=actual_kpi,
        kpi_pace=kpi_pace,
        plan_vs_fact_variance=variance,
        projected_final_kpi=projected,
        projection_interval_lower=interval_lower,
        projection_interval_upper=interval_upper,
        projection_interval_observation_days=interval_days,
        forecast_delta=projected - original_forecast,
        current_unit_cost=unit_cost,
        projected_unit_cost=projected_cost,
    )
    channels = _channel_comparison(state, status_threshold=status_threshold)
    return MonitoringResult(
        progress=progress,
        daily=daily,
        channels=channels,
        beliefs=beliefs,
    )
