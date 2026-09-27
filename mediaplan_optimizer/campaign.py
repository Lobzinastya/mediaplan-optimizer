"""Application-level campaign state, immutable fact, and plan versioning."""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timedelta, timezone
from typing import Iterable
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    ChannelConfig,
    DailyAllocation,
    MediaPlanResult,
    ObjectiveMetric,
    OptimizationRequest,
    PlanningMode,
    ResultStatus,
    TemporalProfile,
    TaskType,
)


POLICY_NAMES = (
    "static",
    "periodic_reoptimization",
    "thompson_sampling",
    "linucb",
)


class CampaignObservation(BaseModel):
    """Normalized observed fact shared by simulator and CSV ingestion."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    day: int = Field(gt=0)
    date: Date | None = None
    channel: str = Field(min_length=1)
    spend: float = Field(ge=0)
    impressions: float = Field(ge=0)
    clicks: float = Field(ge=0)
    conversions: float = Field(ge=0)
    reach: float | None = Field(default=None, ge=0)
    video_views: float | None = Field(default=None, ge=0)
    source: str = Field(default="upload", pattern="^(upload|simulator)$")

    @model_validator(mode="after")
    def validate_funnel(self) -> "CampaignObservation":
        if self.clicks > self.impressions:
            raise ValueError("clicks cannot exceed impressions")
        if self.conversions > self.clicks:
            raise ValueError("conversions cannot exceed clicks")
        if self.reach is not None and self.reach > self.impressions:
            raise ValueError("reach cannot exceed impressions")
        if self.video_views is not None and self.video_views > self.impressions:
            raise ValueError("video_views cannot exceed impressions")
        return self


class PlanVersion(BaseModel):
    """A point-in-time future plan; historical fact is never included."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    version_id: str
    created_at_day: int = Field(ge=0)
    created_at: datetime
    policy: str
    reason: str
    remaining_budget: float = Field(ge=0)
    remaining_horizon: int = Field(ge=0)
    future_allocations: tuple[DailyAllocation, ...]

    @model_validator(mode="after")
    def validate_future_only(self) -> "PlanVersion":
        if any(row.day <= self.created_at_day for row in self.future_allocations):
            raise ValueError("plan versions may contain future allocations only")
        return self


class ForecastRecord(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    created_at_day: int = Field(ge=0)
    reason: str
    actual_kpi: float = Field(ge=0)
    projected_final_kpi: float = Field(ge=0)
    projected_cost: float | None = Field(default=None, ge=0)


class CampaignState(BaseModel):
    """Serializable campaign lifecycle state stored in Streamlit session state."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    campaign_id: str
    name: str
    created_at: datetime
    original_request: OptimizationRequest
    original_result: MediaPlanResult
    planning_catalog: dict[str, ChannelConfig]
    temporal_profiles: tuple[TemporalProfile, ...] = ()
    observations: tuple[CampaignObservation, ...] = ()
    plan_versions: tuple[PlanVersion, ...]
    forecast_history: tuple[ForecastRecord, ...] = ()
    current_day: int = Field(default=0, ge=0)
    current_policy: str = "static"
    periodic_interval_days: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def validate_state(self) -> "CampaignState":
        horizon = self.original_request.horizon_days
        if self.current_day > horizon:
            raise ValueError("current_day cannot exceed campaign horizon")
        if self.current_policy not in POLICY_NAMES:
            raise ValueError(f"unknown campaign policy: {self.current_policy}")
        if not self.plan_versions:
            raise ValueError("campaign requires at least one plan version")
        if self.original_request.planning_mode == PlanningMode.CALENDAR_AWARE:
            approved = {(p.day, p.channel) for p in self.temporal_profiles}
            planned = {(row.day, row.channel) for row in self.original_result.allocations}
            if not approved or approved != planned:
                raise ValueError("calendar campaign requires the approved day-profile snapshot")
        return self

    @property
    def horizon_days(self) -> int:
        return self.original_request.horizon_days

    @property
    def remaining_days(self) -> int:
        return max(0, self.horizon_days - self.current_day)

    @property
    def original_budget(self) -> float:
        if self.original_request.task_type == TaskType.A:
            return float(self.original_request.budget or 0.0)
        summary = self.original_result.summary
        return 0.0 if summary is None else float(summary.spend)

    @property
    def actual_spend(self) -> float:
        return float(sum(item.spend for item in self.observations))

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.original_budget - self.actual_spend)

    @property
    def objective(self) -> ObjectiveMetric:
        return self.original_request.objective_metric

    @property
    def actual_kpi(self) -> float:
        field = self.objective.value
        return float(sum((getattr(item, field) or 0.0) for item in self.observations))

    @property
    def latest_plan(self) -> PlanVersion:
        return self.plan_versions[-1]

    def day_cap(self, day: int, channel: str) -> float:
        base = self.planning_catalog[channel].max_daily_spend
        if self.original_request.planning_mode == PlanningMode.UNIFORM:
            return base
        profile = next(
            (item for item in self.temporal_profiles
             if item.day == day and item.channel == channel), None
        )
        if profile is None:
            raise ValueError("missing approved calendar day profile")
        return base * profile.max_spend_factor

    def day_date(self, day: int) -> Date | None:
        if self.original_request.planning_mode == PlanningMode.UNIFORM:
            return None
        assert self.original_request.start_date is not None
        return self.original_request.start_date + timedelta(days=day - 1)


def start_campaign(
    name: str,
    request: OptimizationRequest,
    result: MediaPlanResult,
    catalog: dict[str, ChannelConfig],
    *,
    policy: str = "static",
    campaign_id: str | None = None,
) -> CampaignState:
    """Create campaign state and immutable initial plan version v1."""
    if result.status != ResultStatus.OPTIMAL or result.summary is None:
        raise ValueError("only a successful media plan can start a campaign")
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown campaign policy: {policy}")
    if request != result.request:
        raise ValueError("campaign request must match the optimized result request")
    request = request.model_copy(deep=True)
    result = result.model_copy(deep=True)
    selected = {row.channel for row in result.allocations}
    planning_catalog = {key: value for key, value in catalog.items() if key in selected}
    if not planning_catalog:
        raise ValueError("campaign plan contains no selected channels")
    created_at = datetime.now(timezone.utc)
    initial = PlanVersion(
        version_id="v1",
        created_at_day=0,
        created_at=created_at,
        policy=policy,
        reason="Initial static media plan",
        remaining_budget=(
            float(request.budget)
            if request.task_type == TaskType.A and request.budget is not None
            else float(result.summary.spend)
        ),
        remaining_horizon=request.horizon_days,
        future_allocations=tuple(row.model_copy(deep=True) for row in result.allocations),
    )
    return CampaignState(
        campaign_id=campaign_id or uuid4().hex[:12],
        name=name.strip() or "Untitled campaign",
        created_at=created_at,
        original_request=request,
        original_result=result,
        planning_catalog=planning_catalog,
        temporal_profiles=tuple(profile.model_copy(deep=True) for profile in result.temporal_profiles),
        plan_versions=(initial,),
        current_policy=policy,
    )


def add_observations(
    state: CampaignState,
    observations: Iterable[CampaignObservation],
) -> CampaignState:
    """Append validated fact without changing or replacing existing history."""
    incoming = tuple(observations)
    if not incoming:
        raise ValueError("no observations supplied")
    existing_keys = {(item.day, item.channel) for item in state.observations}
    incoming_keys = [(item.day, item.channel) for item in incoming]
    if len(incoming_keys) != len(set(incoming_keys)):
        raise ValueError("duplicate day/channel rows in incoming observations")
    duplicates = existing_keys.intersection(incoming_keys)
    if duplicates:
        rendered = ", ".join(f"day {day} / {channel}" for day, channel in sorted(duplicates))
        raise ValueError(f"observations already ingested for {rendered}")
    unknown = sorted({item.channel for item in incoming} - set(state.planning_catalog))
    if unknown:
        raise ValueError(f"unknown campaign channels: {', '.join(unknown)}")
    if any(item.day > state.horizon_days for item in incoming):
        raise ValueError("observation day exceeds campaign horizon")
    combined = state.observations + incoming
    spend = sum(item.spend for item in combined)
    tolerance = max(0.01, state.original_budget * 1e-8)
    if spend > state.original_budget + tolerance:
        raise ValueError("observed spend exceeds campaign budget")
    if any(item.spend > state.day_cap(item.day, item.channel) + 1e-7 for item in incoming):
        raise ValueError("observation exceeds channel daily spend cap")
    if any(item.date is not None and state.day_date(item.day) is not None
           and item.date != state.day_date(item.day) for item in incoming):
        raise ValueError("observation date does not match approved campaign calendar")
    if state.objective == ObjectiveMetric.REACH and any(item.reach is None for item in incoming):
        raise ValueError("reach-target campaigns require observed reach for every fact row")
    ordered = tuple(sorted(combined, key=lambda item: (item.day, item.channel)))
    return state.model_copy(
        update={
            "observations": ordered,
            "current_day": max(item.day for item in ordered),
        }
    )


def add_plan_version(
    state: CampaignState,
    future_allocations: Iterable[DailyAllocation],
    *,
    reason: str,
    policy: str | None = None,
) -> CampaignState:
    """Append a future-only plan version while retaining all prior versions."""
    allocations = tuple(row.model_copy(deep=True) for row in future_allocations)
    if state.remaining_days <= 0:
        raise ValueError("completed campaign cannot be replanned")
    chosen_policy = policy or state.current_policy
    if chosen_policy not in POLICY_NAMES:
        raise ValueError(f"unknown campaign policy: {chosen_policy}")
    keys = [(row.day, row.channel) for row in allocations]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate day/channel rows in future plan")
    for row in allocations:
        if row.channel not in state.planning_catalog or row.day > state.horizon_days:
            raise ValueError("future plan channel/day is outside the campaign")
        if row.spend > state.day_cap(row.day, row.channel) + 1e-7:
            raise ValueError("future plan exceeds channel daily spend cap")
        if row.date is not None and state.day_date(row.day) is not None and row.date != state.day_date(row.day):
            raise ValueError("future plan date does not match approved campaign calendar")
    if sum(row.spend for row in allocations) > state.remaining_budget + 0.01:
        raise ValueError("future plan exceeds remaining campaign budget")
    if (state.latest_plan.created_at_day == state.current_day
            and state.latest_plan.policy == chosen_policy
            and state.latest_plan.future_allocations == allocations):
        return state
    version = PlanVersion(
        version_id=f"v{len(state.plan_versions) + 1}",
        created_at_day=state.current_day,
        created_at=datetime.now(timezone.utc),
        policy=chosen_policy,
        reason=reason,
        remaining_budget=state.remaining_budget,
        remaining_horizon=state.remaining_days,
        future_allocations=allocations,
    )
    return state.model_copy(
        update={
            "plan_versions": state.plan_versions + (version,),
            "current_policy": chosen_policy,
        }
    )


def switch_policy(
    state: CampaignState,
    policy: str,
    *,
    periodic_interval_days: int | None = None,
) -> CampaignState:
    """Switch future control policy without touching fact or plan history."""
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown campaign policy: {policy}")
    interval = state.periodic_interval_days if periodic_interval_days is None else periodic_interval_days
    if interval <= 0:
        raise ValueError("periodic interval must be positive")
    return state.model_copy(
        update={
            "current_policy": policy,
            "periodic_interval_days": interval,
        }
    )


def record_forecast(
    state: CampaignState,
    *,
    reason: str,
    actual_kpi: float,
    projected_final_kpi: float,
    projected_cost: float | None,
) -> CampaignState:
    record = ForecastRecord(
        created_at_day=state.current_day,
        reason=reason,
        actual_kpi=actual_kpi,
        projected_final_kpi=projected_final_kpi,
        projected_cost=projected_cost,
    )
    return state.model_copy(
        update={"forecast_history": state.forecast_history + (record,)}
    )


def observations_frame(state: CampaignState) -> pd.DataFrame:
    columns = [
        "day",
        "date",
        "channel",
        "spend",
        "impressions",
        "clicks",
        "conversions",
        "reach",
        "video_views",
        "source",
    ]
    return pd.DataFrame(
        [item.model_dump(mode="json") for item in state.observations],
        columns=columns,
    )


def future_spend_scale(state: CampaignState) -> float:
    """Budget guard for unchanged plans after fact spends ahead of plan.

    Scale execution/forecast only; never rewrite the version that was approved.
    """
    future_spend = sum(row.spend for row in state.latest_plan.future_allocations
                       if row.day > state.current_day)
    if future_spend <= state.remaining_budget + 1e-8:
        return 1.0
    return state.remaining_budget / future_spend
