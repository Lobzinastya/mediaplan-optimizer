"""Stable Pydantic contracts shared by the planner, UI, and notebook."""

from __future__ import annotations

from datetime import date as Date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskType(str, Enum):
    A = "A"
    B = "B"


class ObjectiveMetric(str, Enum):
    REACH = "reach"
    CLICKS = "clicks"
    CONVERSIONS = "conversions"


class ResultStatus(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    ERROR = "error"


class PlanningMode(str, Enum):
    UNIFORM = "uniform"
    CALENDAR_AWARE = "calendar_aware"


DEFAULT_CALENDAR_START = Date(2026, 9, 21)


class FeasibilityReason(str, Enum):
    MARKET_CAPACITY = "MARKET_CAPACITY"
    HORIZON_TOO_SHORT = "HORIZON_TOO_SHORT"
    CHANNEL_SET_TOO_NARROW = "CHANNEL_SET_TOO_NARROW"


class ChannelConfig(BaseModel):
    """Synthetic parameters for one advertising inventory."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    channel_type: Literal["media", "sms"] = "media"
    daily_capacity: float = Field(gt=0, description="Daily impressions or delivered SMS")
    daily_reach_capacity: float = Field(gt=0)
    base_cpm_rub: float | None = Field(
        default=None,
        gt=0,
        description="Low-spend marginal CPM; generated catalogs use it to price inventory buyout",
    )
    response_scale_rub: float = Field(gt=0)
    max_daily_spend: float = Field(gt=0)
    average_frequency: float = Field(ge=1)
    base_ctr: float = Field(ge=0, le=1)
    base_cr: float = Field(ge=0, le=1)
    base_vtr: float | None = Field(default=None, ge=0, le=1)
    ctr_saturation_decay: float = Field(default=0.0, ge=0, le=0.5)
    cr_saturation_decay: float = Field(default=0.0, ge=0, le=0.5)

    @model_validator(mode="after")
    def validate_capacities(self) -> "ChannelConfig":
        if self.daily_reach_capacity > self.daily_capacity:
            raise ValueError("daily_reach_capacity cannot exceed daily_capacity")
        a, b = self.ctr_saturation_decay, self.cr_saturation_decay
        if 1.0 - 2.0 * (a + b) + 3.0 * a * b < -1e-12:
            raise ValueError(
                "joint CTR/CR saturation decay must preserve monotone conversions: "
                "1 - 2*(d_ctr + d_cr) + 3*d_ctr*d_cr >= 0"
            )
        return self


class OptimizationRequest(BaseModel):
    """Validated request for either budget-capped or KPI-driven planning."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    task_type: TaskType
    horizon_days: int = Field(gt=0, le=365)
    objective_metric: ObjectiveMetric
    budget: float | None = None
    target_value: float | None = None
    included_channels: list[str] | None = None
    excluded_channels: list[str] | None = None
    planning_mode: PlanningMode = PlanningMode.UNIFORM
    start_date: Date | None = None
    calendar_profile_seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_task(self) -> "OptimizationRequest":
        if self.planning_mode == PlanningMode.CALENDAR_AWARE and self.start_date is None:
            self.start_date = DEFAULT_CALENDAR_START
        if self.task_type == TaskType.A:
            if self.budget is None or self.budget <= 0:
                raise ValueError("Type A requires budget > 0")
            if self.objective_metric not in {
                ObjectiveMetric.CLICKS,
                ObjectiveMetric.CONVERSIONS,
            }:
                raise ValueError("Type A objective_metric must be clicks or conversions")
            if self.target_value is not None:
                raise ValueError("Type A does not accept target_value")
        else:
            if self.target_value is None or self.target_value <= 0:
                raise ValueError("Type B requires target_value > 0")
            if self.budget is not None:
                raise ValueError("Type B does not accept budget")

        if self.included_channels is not None and not self.included_channels:
            raise ValueError("included_channels cannot be empty")
        if self.excluded_channels is not None and not self.excluded_channels:
            raise ValueError("excluded_channels cannot be empty")
        return self


class TemporalProfile(BaseModel):
    """Immutable synthetic factors for one approved channel-day."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    channel: str
    day: int = Field(gt=0)
    date: Date
    weekday: int = Field(ge=0, le=6)
    supply_factor: float = Field(ge=0.8, le=1.2)
    cpm_factor: float = Field(ge=0.8, le=1.2)
    ctr_factor: float = Field(ge=0.8, le=1.2)
    cr_factor: float = Field(ge=0.8, le=1.2)
    max_spend_factor: float = Field(ge=0.8, le=1.2)


class DailyAllocation(BaseModel):
    day: int = Field(gt=0)
    date: Date | None = None
    weekday: int | None = Field(default=None, ge=0, le=6)
    channel: str
    spend: float = Field(ge=0)
    impressions: float = Field(ge=0)
    reach: float = Field(ge=0)
    clicks: float = Field(ge=0)
    conversions: float = Field(ge=0)
    video_views: float | None = Field(default=None, ge=0)
    cumulative_target_kpi: float = Field(ge=0)


class ChannelMetrics(BaseModel):
    channel: str
    spend: float = Field(ge=0)
    impressions: float = Field(ge=0)
    reach: float = Field(ge=0)
    clicks: float = Field(ge=0)
    conversions: float = Field(ge=0)
    video_views: float | None = Field(default=None, ge=0)
    ctr: float = Field(ge=0, le=1)
    cr: float = Field(ge=0, le=1)
    vtr: float | None = Field(default=None, ge=0, le=1)
    cpm: float | None = Field(default=None, ge=0)
    cpc: float | None = Field(default=None, ge=0)
    cpa: float | None = Field(default=None, ge=0)


class PlanSummary(BaseModel):
    spend: float = Field(ge=0)
    available_budget: float | None = Field(default=None, ge=0)
    unspent_budget: float | None = Field(default=None, ge=0)
    requested_target: float | None = Field(default=None, ge=0)
    achieved_target: float = Field(ge=0)
    impressions: float = Field(ge=0)
    reach: float = Field(ge=0)
    clicks: float = Field(ge=0)
    conversions: float = Field(ge=0)
    ctr: float = Field(ge=0, le=1)
    cr: float = Field(ge=0, le=1)
    vtr: float | None = Field(default=None, ge=0, le=1)
    cpm: float | None = Field(default=None, ge=0)
    cpc: float | None = Field(default=None, ge=0)
    cpa: float | None = Field(default=None, ge=0)
    inventory_saturated: bool = False


class FeasibilityResult(BaseModel):
    feasible: bool
    requested_value: float = Field(ge=0)
    maximum_achievable: float = Field(ge=0)
    target_gap: float = Field(ge=0)
    recommended_target: float | None = Field(default=None, gt=0)
    reason: FeasibilityReason | None = None
    explanation: str
    recommendations: list[str] = Field(default_factory=list)
    minimum_feasible_horizon: int | None = Field(default=None, gt=0, le=365)
    additional_days_needed: int | None = Field(default=None, gt=0)
    maximum_with_all_channels: float | None = Field(default=None, ge=0)
    additional_capacity_with_all_channels: float | None = Field(default=None, ge=0)
    estimated_budget_for_recommended_target: float | None = Field(default=None, ge=0)
    estimated_budget_at_minimum_horizon: float | None = Field(default=None, ge=0)


class OptimizerMetadata(BaseModel):
    success: bool
    solver: str
    status: int | str
    message: str
    objective: float | None = None
    iterations: int | None = Field(default=None, ge=0)
    validation_passed: bool


class MediaPlanResult(BaseModel):
    status: ResultStatus
    request: OptimizationRequest
    allocations: list[DailyAllocation] = Field(default_factory=list)
    channel_metrics: list[ChannelMetrics] = Field(default_factory=list)
    summary: PlanSummary | None = None
    feasibility: FeasibilityResult | None = None
    optimizer_metadata: OptimizerMetadata
    messages: list[str] = Field(default_factory=list)
    temporal_profiles: tuple[TemporalProfile, ...] = ()
