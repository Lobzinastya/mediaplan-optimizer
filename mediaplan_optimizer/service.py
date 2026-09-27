"""Public application service for media-plan optimization."""

from __future__ import annotations

from typing import Mapping

from .calendar import effective_channel, generate_day_profiles
from .calendar_optimizer import optimize_type_a_calendar, optimize_type_b_calendar
from .catalog import load_catalog
from .feasibility import analyze_feasibility
from .metrics import (
    aggregate_channel_metrics,
    aggregate_plan_summary,
    build_daily_allocations,
    build_calendar_allocations,
)
from .optimizer import optimize_type_a, optimize_type_b
from .schemas import (
    ChannelConfig,
    FeasibilityResult,
    MediaPlanResult,
    OptimizationRequest,
    OptimizerMetadata,
    ResultStatus,
    PlanningMode,
    TaskType,
)


def _select_catalog(
    request: OptimizationRequest, catalog: Mapping[str, ChannelConfig]
) -> dict[str, ChannelConfig]:
    known = set(catalog)
    included = set(request.included_channels) if request.included_channels is not None else known
    excluded = set(request.excluded_channels or [])
    invalid = (included | excluded) - known
    if invalid:
        raise ValueError(f"Unknown channels: {', '.join(sorted(invalid))}")
    selected_names = [name for name in catalog if name in included and name not in excluded]
    if not selected_names:
        raise ValueError("At least one channel must remain selected")
    return {name: catalog[name] for name in selected_names}


def _infeasible_result(
    request: OptimizationRequest, feasibility: FeasibilityResult
) -> MediaPlanResult:
    return MediaPlanResult(
        status=ResultStatus.INFEASIBLE,
        request=request,
        feasibility=feasibility,
        optimizer_metadata=OptimizerMetadata(
            success=False,
            solver="not-run",
            status="infeasible",
            message="Optimization skipped after feasibility analysis.",
            validation_passed=True,
        ),
        messages=[feasibility.explanation, *feasibility.recommendations],
    )


def optimize_media_plan(
    request: OptimizationRequest,
    catalog: Mapping[str, ChannelConfig] | None = None,
) -> MediaPlanResult:
    """Validate, select inventory, optimize, calculate metrics, and explain."""
    full_catalog = dict(catalog or load_catalog())
    selected = _select_catalog(request, full_catalog)
    calendar_aware = request.planning_mode == PlanningMode.CALENDAR_AWARE
    profiles = (
        generate_day_profiles(
            selected, request.start_date, request.horizon_days,
            request.calendar_profile_seed,
        )
        if calendar_aware else ()
    )
    messages: list[str] = []
    feasibility: FeasibilityResult | None = None

    if request.task_type == TaskType.B:
        assert request.target_value is not None
        feasibility = analyze_feasibility(
            selected,
            full_catalog,
            request.horizon_days,
            request.objective_metric,
            request.target_value,
            planning_mode=request.planning_mode,
            start_date=request.start_date,
            calendar_profile_seed=request.calendar_profile_seed,
        )
        if not feasibility.feasible:
            return _infeasible_result(request, feasibility)
        outcome = (
            optimize_type_b_calendar(selected, profiles, request.objective_metric, request.target_value)
            if calendar_aware else optimize_type_b(
                selected, request.horizon_days, request.objective_metric, request.target_value,
            )
        )
        requested_target = request.target_value
        available_budget = None
        saturated = False
    else:
        assert request.budget is not None
        outcome = (
            optimize_type_a_calendar(selected, profiles, request.objective_metric, request.budget)
            if calendar_aware else optimize_type_a(
                selected, request.horizon_days, request.objective_metric, request.budget,
            )
        )
        requested_target = None
        available_budget = request.budget
        spend_capacity = (
            sum(effective_channel(selected[p.channel], p).max_daily_spend for p in profiles)
            if calendar_aware else sum(
                channel.max_daily_spend * request.horizon_days
                for channel in selected.values()
            )
        )
        saturated = request.budget > spend_capacity + max(0.01, request.budget * 1e-8)
        if saturated:
            messages.append(
                "The available budget exceeds configured inventory spend capacity; "
                "the remainder is intentionally unspent."
            )

    allocations = (
        build_calendar_allocations(outcome.cell_spend, profiles, selected, request.objective_metric)
        if calendar_aware else build_daily_allocations(
            outcome.channel_spend, request.horizon_days, selected, request.objective_metric,
        )
    )
    channel_metrics = aggregate_channel_metrics(allocations)
    summary = aggregate_plan_summary(
        allocations,
        request.objective_metric,
        available_budget=available_budget,
        requested_target=requested_target,
        inventory_saturated=saturated,
    )
    if request.task_type == TaskType.B and summary.achieved_target + max(
        1e-6, (requested_target or 0.0) * 1e-7
    ) < (requested_target or 0.0):
        raise RuntimeError("Post-solve validation failed: Type B target was not reached")

    return MediaPlanResult(
        status=ResultStatus.OPTIMAL,
        request=request,
        allocations=allocations,
        channel_metrics=channel_metrics,
        summary=summary,
        feasibility=feasibility,
        optimizer_metadata=outcome.metadata,
        messages=messages,
        temporal_profiles=profiles,
    )
