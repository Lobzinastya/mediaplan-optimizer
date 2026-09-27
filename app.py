"""Streamlit pages for media planning and campaign monitoring."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from pydantic import ValidationError

from experiments.compare_policies import run_policy_comparison
from mediaplan_optimizer.campaign import (
    POLICY_NAMES,
    CampaignState,
    future_spend_scale,
    observations_frame,
    start_campaign,
    switch_policy,
)
from mediaplan_optimizer.catalog import DEFAULT_ASSUMPTIONS_PATH, generate_catalog
from mediaplan_optimizer.control import (
    create_policy_runtime,
    ingest_and_record,
    policy_diagnostics,
    recalculate_remaining_plan,
    simulate_next_days,
)
from mediaplan_optimizer.curves import channel_response, effective_performance_rates
from mediaplan_optimizer.ingestion import FACT_TEMPLATE, parse_fact_csv
from mediaplan_optimizer.monitoring import build_monitoring
from mediaplan_optimizer.schemas import DEFAULT_CALENDAR_START, ObjectiveMetric, OptimizationRequest, PlanningMode, ResultStatus
from mediaplan_optimizer.service import optimize_media_plan
from mediaplan_optimizer.simulator import CampaignSimulator
from ui_text import APP_NAME, LANGUAGES, status_label, table_columns, translate, workflow_progress
from ui_methodology import render_methodology
from ui_components import (
    compact, csv_bytes, display_table, download_filename, metric_columns, metric_help,
    plot_chart, policy_label, research_policy_label, render_plan_result, research_help, rubles, tx, ui_frame,
)


st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")

PAGES = (
    "Overview",
    "Inventory / Generator",
    "Media Plan",
    "Fact Ingestion",
    "Campaign Monitor",
    "Adaptive Control",
    "Experiments",
    "How it works",
)
@st.cache_data(show_spinner=False)
def cached_catalog(seed: int):
    return generate_catalog(seed=seed)


@st.cache_data(show_spinner=False)
def benchmark_assumptions() -> dict:
    with Path(DEFAULT_ASSUMPTIONS_PATH).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def initialize_session() -> None:
    if "language" not in st.session_state:
        st.session_state.language = "ru"
    if "catalog_seed" not in st.session_state:
        st.session_state.catalog_seed = 42
    if "planning_catalog" not in st.session_state:
        st.session_state.planning_catalog = cached_catalog(42)
    defaults = {
        "page": "Overview",
        "nav_widget": "Overview",
        "pending_page": None,
        "draft_request": None,
        "draft_result": None,
        "draft_catalog": None,
        "campaign": None,
        "simulator": None,
        "simulator_config": None,
        "policy_runtime": None,
        "experiment_result": None,
        "experiment_config": None,
        "simulation_summary": None,
        "replan_summary": None,
        "demo_ready": False,
        "planning_mode_ui": PlanningMode.CALENDAR_AWARE.value,
        "planning_start_date": DEFAULT_CALENDAR_START,
        "inventory_reviewed": False,
        "monitor_reviewed_campaign_id": None,
        "adaptive_reviewed_campaign_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_workflow():
    campaign = st.session_state.campaign
    plan = st.session_state.draft_result
    started = campaign is not None
    return workflow_progress(
        inventory_reviewed=bool(st.session_state.inventory_reviewed),
        plan_ready=bool(plan is not None and plan.status == ResultStatus.OPTIMAL),
        campaign_started=started,
        fact_loaded=bool(started and campaign.observations),
        monitor_reviewed=bool(started and st.session_state.monitor_reviewed_campaign_id == campaign.campaign_id),
        replanned=bool(started and len(campaign.plan_versions) > 1),
        adaptive_opened=bool(started and st.session_state.adaptive_reviewed_campaign_id == campaign.campaign_id),
        experimented=st.session_state.experiment_result is not None,
    )


def page_intro(page: str) -> None:
    intro = {
        "Inventory / Generator": ("Inventory: Inspect the benchmark-derived channels and their response curves.", "After reviewing inventory, calculate the media plan."),
        "Media Plan": ("Media Plan: Enter a budget or target KPI, campaign horizon, and available channels. Calculate the allocation.", "After calculation, start the campaign."),
        "Fact Ingestion": ("Fact: Simulate delivery for a demo or upload reviewed campaign data in CSV.", "After adding fact, open campaign monitoring."),
        "Campaign Monitor": ("Monitoring: Compare the plan in force with fact and inspect the forecast.", "After monitoring, recalculate the remaining plan."),
        "Adaptive Control": ("Adaptive Control: Inspect or change the future allocation strategy.", "After adaptive control, inspect experiments."),
        "Experiments": ("Experiments: Compare strategies in synthetic scenarios; treat results as illustrative.", "Run a comparison or inspect the report."),
    }
    purpose, _ = intro[page]
    st.caption(f"**{tx('What to do here')}** · {tx(purpose)}")
    if page == "Media Plan":
        st.caption(tx("Your plan or infeasibility recommendations are a complete planning result."))
    elif page not in {"Inventory / Generator"}:
        st.caption(tx("Optional extension"))


def initialize_demo() -> None:
    """Calculate the canonical draft through the public service, without a campaign."""
    if st.session_state.campaign is not None:
        raise ValueError(tx("Finish or reset the active campaign before starting a new demo."))
    st.session_state.catalog_seed = 42
    st.session_state.planning_catalog = cached_catalog(42)
    catalog = st.session_state.planning_catalog
    request = OptimizationRequest(
        task_type="A", horizon_days=21, objective_metric="conversions",
        budget=1_200_000, included_channels=list(catalog),
        planning_mode=PlanningMode.CALENDAR_AWARE, start_date=DEFAULT_CALENDAR_START,
    )
    result = optimize_media_plan(request, catalog)
    st.session_state.draft_request = request
    st.session_state.draft_result = result
    st.session_state.draft_catalog = dict(catalog)
    st.session_state.uniform_comparison = None
    st.session_state.planning_task_ui = "A — Maximize KPI"
    st.session_state.planning_objective_ui = "conversions"
    # Let widgets supply their canonical defaults, without duplicate state defaults.
    for key in ("planning_horizon_ui", "planning_budget_ui", "planning_channels_ui"):
        st.session_state.pop(key, None)
    st.session_state.demo_ready = True
    st.session_state.planning_mode_ui = PlanningMode.CALENDAR_AWARE.value
    st.session_state.planning_start_date = DEFAULT_CALENDAR_START
    st.session_state.inventory_reviewed = False
    st.session_state.experiment_result = None
    st.session_state.experiment_config = None


def selected_parameter_table(channel) -> pd.DataFrame:
    """Show channel parameters alongside their units and configured ranges."""
    ranges = benchmark_assumptions().get("channels", {}).get(channel.name, {})
    if channel.name.startswith(("Social", "Programmatic")):
        source_category = "social and programmatic/native media"
    elif channel.name.startswith("Marketplace"):
        source_category = "marketplace sponsored advertising"
    else:
        source_category = "SMS"
    specifications = [
        ("daily_capacity", channel.daily_capacity, "contacts/day", "Daily buyable supply"),
        ("daily_reach_capacity", channel.daily_reach_capacity, "expected people/day", "Potential daily audience pool; achieved reach also depends on impressions and frequency"),
        ("base_cpm_rub", channel.base_cpm_rub, "RUB / 1,000", "Low-spend marginal inventory price"),
        ("max_daily_spend", channel.max_daily_spend, "RUB/day", "Configured daily spend cap"),
        ("average_frequency", channel.average_frequency, "impressions/person", "Expected frequency used in reach response"),
        ("base_ctr", channel.base_ctr, "fraction", "Unsaturated click-through assumption"),
        ("base_cr", channel.base_cr, "fraction", "Unsaturated post-click conversion assumption"),
        ("base_vtr", channel.base_vtr, "fraction", "Video view-through assumption, when applicable"),
        ("ctr_saturation_decay", channel.ctr_saturation_decay, "fraction", "CTR loss at full inventory saturation"),
        ("cr_saturation_decay", channel.cr_saturation_decay, "fraction", "CR loss at full inventory saturation"),
        ("response_scale_rub", channel.response_scale_rub, "RUB", "Derived buyout scale retained for compatibility"),
    ]
    rows = []
    for field, value, unit, meaning in specifications:
        range_key = "daily_reach_ratio" if field == "daily_reach_capacity" else field
        configured_range = ranges.get(range_key)
        if field == "response_scale_rub":
            configured_range = "derived from capacity and base CPM"
        elif isinstance(configured_range, list) and len(configured_range) == 2:
            configured_range = f"{configured_range[0]} to {configured_range[1]}"
        elif configured_range is None:
            configured_range = "not applicable"
        rows.append(
            {
                "parameter": field,
                "generated_value": value,
                "unit": unit,
                "meaning": meaning,
                "benchmark-derived/model range": configured_range,
                "source category": (
                    source_category if field in {"base_cpm_rub", "base_ctr", "base_cr", "base_vtr"}
                    else "synthetic model assumption / derived value"
                ),
            }
        )
    return pd.DataFrame(rows)


def go_to(page: str) -> None:
    st.session_state.pending_page = page
    st.rerun()


def campaign_or_warning() -> CampaignState | None:
    campaign = st.session_state.campaign
    if campaign is None:
        st.info(tx("Start a successful media plan before using campaign controls."))
        if st.button(tx("Create media plan"), type="primary"):
            go_to("Media Plan")
        return None
    return campaign


def render_sidebar() -> str:
    with st.sidebar:
        st.title(APP_NAME)
        st.radio(tx("Language"), tuple(LANGUAGES),
                 format_func=lambda code: LANGUAGES[code], key="language", horizontal=True)
        language = st.session_state.language
        pending_page = st.session_state.pending_page
        if pending_page is not None:
            st.session_state.nav_widget = pending_page
            st.session_state.pending_page = None
        def nav_label(item: str) -> str:
            return translate(item, language)
        st.caption(tx("Planner: Overview · Inventory · Media Plan"))
        st.caption(tx("Optional: Fact · Monitor · Adaptive Control · Experiments"))
        page = st.radio(tx("Navigate"), PAGES, format_func=nav_label, key="nav_widget")
        st.session_state.page = page
        campaign = st.session_state.campaign
        st.divider()
        if campaign is None:
            st.caption(tx("No active campaign"))
        else:
            st.caption(tx("Active: {name}", name=campaign.name))
            st.progress(campaign.current_day / campaign.horizon_days)
            st.write(
                tx("Day {day}/{horizon} · {policy}", day=campaign.current_day,
                   horizon=campaign.horizon_days, policy=policy_label(campaign.current_policy))
            )
        has_draft = st.session_state.draft_result is not None or st.session_state.draft_request is not None
        catalog_changed = (st.session_state.catalog_seed != 42
                           or st.session_state.planning_catalog != cached_catalog(42))
        with st.expander(tx("Reset controls")):
            if campaign is not None:
                confirm_campaign = st.checkbox(tx("Confirm campaign reset"))
                if st.button(tx("Reset campaign"), disabled=not confirm_campaign):
                    for key in (
                        "campaign", "simulator", "simulator_config", "policy_runtime",
                        "simulation_summary", "replan_summary",
                    ):
                        st.session_state[key] = None
                    st.rerun()
            if has_draft:
                confirm_draft = st.checkbox(tx("Confirm planning scenario reset"))
                if st.button(tx("Start new planning scenario"), disabled=not confirm_draft):
                    st.session_state.draft_request = None
                    st.session_state.draft_result = None
                    go_to("Media Plan")
            if catalog_changed:
                confirm_catalog = st.checkbox(tx("Confirm catalog reset"))
                if st.button(tx("Reset catalog to seed 42"), disabled=not confirm_catalog):
                    st.session_state.catalog_seed = 42
                    st.session_state.planning_catalog = cached_catalog(42)
                    st.session_state.draft_request = None
                    st.session_state.draft_result = None
                    st.rerun()
            if campaign is None and not has_draft and not catalog_changed:
                st.caption(tx("Nothing to reset in this session."))
        st.caption(tx("Active campaigns retain their own catalog snapshot. Catalog changes affect only new plans."))
    return page


def render_overview() -> None:
    st.title(APP_NAME)
    st.subheader(tx("Media budget optimization and planning"))
    st.write(tx("Allocate media budget across channels and campaign days, accounting for capacity, nonlinear response, and the target KPI."))
    left, right = st.columns(2)
    with left:
        st.subheader(tx("Type A — Fixed budget"))
        st.write(tx("Set a budget and horizon to maximize clicks or conversions."))
    with right:
        st.subheader(tx("Type B — Target KPI"))
        st.write(tx("Set a target and horizon to find the minimum modeled budget or an explanation of infeasibility."))
    actions = st.columns(3)
    if actions[0].button(tx("Calculate media plan"), type="primary", width="stretch"):
        go_to("Media Plan")
    if actions[1].button(tx("Inspect inventory"), width="stretch"):
        go_to("Inventory / Generator")
    if actions[2].button(tx("Run demo"), width="stretch", disabled=st.session_state.campaign is not None):
        with st.spinner(tx("Optimizing…")):
            initialize_demo()
        go_to("Media Plan")
    st.subheader(tx("Core planning flow"))
    for index, key in enumerate(("Review the catalog (optional)", "Define a Type A or Type B task",
                                 "Get a media plan or recommendations"), start=1):
        st.write(f"{index}. {tx(key)}")
    with st.expander(tx("Optional: after campaign launch")):
        st.write(tx("Load or simulate fact, monitor delivery, replan future days, and explore adaptive strategies and experiments."))
    campaign = st.session_state.campaign
    if campaign is not None:
        monitoring = build_monitoring(campaign).progress
        latest_plan = campaign.latest_plan
        last_replan = (
            tx("Day {day}/{horizon} · {policy}", day=latest_plan.created_at_day,
               horizon=campaign.horizon_days, policy=policy_label(campaign.current_policy))
            if len(campaign.plan_versions) > 1
            else tx("Not yet")
        )
        st.subheader(tx("Active campaign state"))
        cards = metric_columns(4)
        cards[0].metric(tx("Campaign day"), f"{campaign.current_day}/{campaign.horizon_days}")
        cards[1].metric(tx("Original budget"), rubles(campaign.original_budget))
        cards[2].metric(tx("Observed spend"), rubles(campaign.actual_spend))
        cards[3].metric(tx("Remaining budget"), rubles(campaign.remaining_budget))
        cards = metric_columns(4)
        cards[0].metric(tx("Original planned KPI"), compact(monitoring.planned_final_kpi))
        cards[1].metric(tx("Observed KPI"), compact(monitoring.actual_kpi_to_date))
        cards[2].metric(
            tx("Forecast KPI"),
            compact(monitoring.projected_final_kpi),
            delta=compact(monitoring.forecast_delta),
            delta_color="off",
            help=tx("Projected final KPI minus the original planned KPI."),
        )
        cards[3].metric(tx("Adaptive policy"), policy_label(campaign.current_policy))
        status = st.columns(2)
        status[0].metric(tx("Current plan version"), latest_plan.version_id)
        status[1].metric(tx("Last replan"), last_replan)
        st.caption(tx("Planning assumptions and forecasts are model outputs; observed KPI and spend come only from ingested observed or simulated fact."))
        left, middle, right = st.columns(3)
        if left.button(tx("Continue campaign"), type="primary", width="stretch"):
            go_to("Campaign Monitor")
        if middle.button(tx("Load fact"), width="stretch"):
            go_to("Fact Ingestion")
        if right.button(tx("Open adaptive control"), width="stretch"):
            go_to("Adaptive Control")


def render_inventory() -> None:
    st.title(tx("Inventory / Generator"))
    page_intro("Inventory / Generator")
    st.session_state.inventory_reviewed = True
    st.caption(tx("Public benchmark context + explicit synthetic assumptions → a reproducible planning catalog."))
    seed = st.number_input(
        tx("Catalog seed"), min_value=0, max_value=1_000_000, value=int(st.session_state.catalog_seed)
    )
    if st.button(tx("Regenerate catalog"), type="primary"):
        st.session_state.catalog_seed = int(seed)
        st.session_state.planning_catalog = cached_catalog(int(seed))
        st.session_state.draft_request = None
        st.session_state.draft_result = None
        st.success(tx("Generated planning catalog with seed {seed}.", seed=int(seed)))
    catalog = st.session_state.planning_catalog
    rows = [channel.model_dump() for channel in catalog.values()]
    with st.expander(tx("All catalog parameters")):
        display_table(ui_frame(pd.DataFrame(rows)), width="stretch", hide_index=True)

    with st.expander(tx("Public benchmark context")):
        sources = pd.DataFrame(benchmark_assumptions()["benchmark_sources"])
        visible = [
            column
            for column in ("category", "metric", "published_range", "model_range", "source_name")
            if column in sources
        ]
        display_table(ui_frame(sources[visible]), width="stretch", hide_index=True)

    selected = st.selectbox(tx("Inspect channel"), list(catalog))
    channel = catalog[selected]
    with st.expander(tx("Selected channel parameters")):
        display_table(ui_frame(selected_parameter_table(channel)), width="stretch", hide_index=True)
    spend = np.linspace(0, channel.max_daily_spend, 100)
    response = channel_response(spend, channel)
    ctr, cr = effective_performance_rates(spend, channel)
    effective_cpm = np.divide(
        spend * 1000,
        response.impressions,
        out=np.full_like(spend, channel.base_cpm_rub or np.nan),
        where=np.asarray(response.impressions) > 0,
    )
    curve_data = pd.DataFrame(
        {
            "Spend, RUB": spend,
            "Impressions / contacts": response.impressions,
            "Non-deduplicated expected reach": response.reach,
            "Clicks": response.clicks,
            "Conversions": response.conversions,
            "Effective CPM, RUB": effective_cpm,
            "Effective CTR": ctr,
            "Effective CR": cr,
        }
    )
    st.caption(tx("All reach columns mean non-deduplicated expected reach summed across channel-days, not unique campaign reach."))
    # Independent axes preserve the real scale of every response; no normalization.
    for metrics in (("Impressions / contacts", "Non-deduplicated expected reach"),
                    ("Clicks", "Conversions"), ("Effective CPM, RUB",),
                    ("Effective CTR", "Effective CR")):
        columns = st.columns(len(metrics))
        for column, metric in zip(columns, metrics):
            figure = px.line(curve_data, x="Spend, RUB", y=metric,
                             title=tx("Expected reach") if metric == "Non-deduplicated expected reach" else tx(metric),
                             labels={name: tx("Expected reach") if name == "Non-deduplicated expected reach" else tx(name)
                                     for name in curve_data.columns})
            if metric in {"Effective CTR", "Effective CR"}:
                figure.update_yaxes(tickformat=".2%")
            plot_chart(figure, target=column, width="stretch")
    st.info(tx("Effective CPM rises because progressively scarcer inventory requires more spend per delivered impression. Effective CTR and CR decline mildly as the available audience is exhausted; together these effects reduce marginal KPI per RUB at higher spend."))

    metric_help()
    if st.button(tx("Go to Media Plan →"), type="primary"):
        go_to("Media Plan")


def render_media_plan() -> None:
    st.title(tx("Media Plan"))
    page_intro("Media Plan")
    st.caption(tx("All outputs are the original plan predictions from generated planning assumptions; they are not measured campaign results."))
    if st.session_state.demo_ready and st.session_state.draft_result is not None:
        st.success(tx("Demo calculated: Calendar-aware, 21 days, RUB 1,200,000, maximum conversions, seed 42."))
    catalog = st.session_state.planning_catalog
    language = st.session_state.language
    task = st.radio(tx("Task"), ["A — Maximize KPI", "B — Minimum budget for target"],
                    format_func=lambda value: translate(value, language), horizontal=True, key="planning_task_ui")
    task_type = "A" if task.startswith("A") else "B"
    with st.form("media_plan_form"):
        mode = st.radio(
            tx("Planning mode"),
            [PlanningMode.CALENDAR_AWARE.value, PlanningMode.UNIFORM.value],
            format_func=lambda value: translate(
                "Calendar-aware" if value == PlanningMode.CALENDAR_AWARE.value else "Uniform baseline", language
            ),
            horizontal=True, key="planning_mode_ui",
        )
        if mode == PlanningMode.CALENDAR_AWARE.value:
            st.caption(tx("Calendar-aware mode optimizes each channel-day using deterministic synthetic temporal assumptions, not measured weekday forecasts."))
            start_date = st.date_input(tx("Campaign start date"), key="planning_start_date",
                                       format="DD.MM.YYYY" if language == "ru" else "YYYY-MM-DD")
        else:
            st.caption(tx("Uniform baseline treats days as exchangeable and splits each channel's optimized spend evenly."))
            start_date = None
        left, right = st.columns(2)
        horizon = left.number_input(tx("Horizon, days"), min_value=1, max_value=365, value=21, key="planning_horizon_ui")
        if task_type == "A":
            objective = right.selectbox(tx("Objective"), ["conversions", "clicks"],
                                        format_func=lambda value: translate(value, language), key="planning_objective_ui")
            budget = left.number_input(tx("Budget, RUB"), min_value=1.0, value=1_200_000.0, step=50_000.0, key="planning_budget_ui")
            target = None
        else:
            objective = right.selectbox(tx("Target metric"), ["clicks", "conversions", "reach"],
                                       format_func=lambda value: translate("Non-deduplicated expected reach" if value == "reach" else value, language))
            target = left.number_input(tx("Target value"), min_value=1.0, value=10_000.0, step=1_000.0)
            budget = None
        selected = st.multiselect(tx("Selected channels"), list(catalog), default=list(catalog), key="planning_channels_ui")
        calculate = st.form_submit_button(tx("Calculate media plan"), type="primary")
    if calculate:
        try:
            request = OptimizationRequest(
                task_type=task_type,
                horizon_days=int(horizon),
                objective_metric=objective,
                budget=budget,
                target_value=target,
                included_channels=selected,
                planning_mode=mode,
                start_date=start_date,
            )
            with st.spinner(tx("Optimizing…")):
                result = optimize_media_plan(request, catalog)
            st.session_state.draft_request = request
            st.session_state.draft_result = result
            st.session_state.draft_catalog = dict(catalog)
            st.session_state.demo_ready = False
            st.session_state.uniform_comparison = None
            st.rerun()  # Refresh state-aware sidebar controls with the completed draft.
        except (ValidationError, ValueError) as exc:
            st.error(tx("Check inputs: {details}", details=str(exc)))

    result = st.session_state.draft_result
    request = st.session_state.draft_request
    if result is not None:
        result_catalog = st.session_state.draft_catalog or catalog
        render_plan_result(result, result_catalog)
        if result.status == ResultStatus.OPTIMAL:
            st.divider()
            st.subheader(tx("Optional: continue with a campaign"))
            st.caption(tx("The media plan is complete. Optionally approve it as a campaign to load fact, monitor delivery, and replan future days."))
            st.caption(tx("The campaign is tracked inside the prototype; no external advertising platforms or APIs are called."))
            campaign_name = st.text_input(tx("Campaign name"), value=tx("Seed 42 conversion campaign"))
            replacing_campaign = st.session_state.campaign is not None
            confirm_replace = not replacing_campaign
            if replacing_campaign:
                st.warning(tx("Starting this plan will replace the active session campaign. Export any fact you need before continuing."))
                confirm_replace = st.checkbox(tx("Confirm active campaign replacement"))
            if st.button(
                tx("Start campaign workflow"),
                type="secondary",
                disabled=not confirm_replace,
            ):
                state = start_campaign(campaign_name, request, result, result_catalog)
                st.session_state.campaign = state
                st.session_state.simulator = None
                st.session_state.simulator_config = None
                st.session_state.policy_runtime = create_policy_runtime(state)
                st.session_state.simulation_summary = None
                st.session_state.replan_summary = None
                go_to("Fact Ingestion")


def ensure_simulator(state: CampaignState, seed: int, deviation: float, context: bool):
    if st.session_state.simulator is None:
        st.session_state.simulator = CampaignSimulator(
            state.planning_catalog,
            state.horizon_days,
            seed,
            parameter_deviation=deviation,
            context_strength=1.0 if context else 0.0,
        )
        st.session_state.simulator_config = {
            "seed": seed,
            "parameter_deviation": deviation,
            "contextual_variation": context,
        }
    return st.session_state.simulator


def render_fact_ingestion() -> None:
    st.title(tx("Fact Ingestion"))
    page_intro("Fact Ingestion")
    state = campaign_or_warning()
    if state is None:
        return
    current_fact = observations_frame(state)
    if not current_fact.empty:
        st.download_button(
            tx("Download campaign observed fact CSV"),
            csv_bytes(current_fact),
            file_name=download_filename("observed_fact.csv", state.campaign_id),
            mime="text/csv",
            key="download_campaign_fact",
        )
    simulation_summary = st.session_state.simulation_summary
    if simulation_summary and simulation_summary.get("campaign_id") == state.campaign_id:
        st.success(tx("Simulated fact ingested through day {day} under {policy}.",
                      day=state.current_day, policy=policy_label(state.current_policy)))
        summary_cards = metric_columns(5)
        summary_cards[0].metric(tx("Days simulated"), simulation_summary["days_advanced"])
        summary_cards[1].metric(tx("New fact rows"), simulation_summary["rows_added"])
        summary_cards[2].metric(tx("Simulated spend"), rubles(simulation_summary["spend_added"]))
        summary_cards[3].metric(tx("Simulated clicks"), compact(simulation_summary["clicks_added"]))
        summary_cards[4].metric(
            tx("Simulated conversions"), compact(simulation_summary["conversions_added"])
        )
        statuses = simulation_summary.get("status_counts", {})
        st.caption(tx("Current deterministic channel statuses: {statuses}",
                      statuses=", ".join(f"{status_label(name, st.session_state.language)}: {count}"
                                         for name, count in statuses.items()))
                   if statuses else tx("No channel status is available yet."))
        if simulation_summary["completed"]:
            st.info(tx("Simulation reached the campaign horizon; no future days remain."))
    tabs = st.tabs([tx("Simulated Fact"), tx("CSV Upload")])
    with tabs[0]:
        if state.objective == ObjectiveMetric.REACH:
            st.info(tx("The simulator does not generate observed reach. Upload observed campaign data via CSV with a reach value."))
        else:
            st.warning(tx("SIMULATED OBSERVED FACT — synthetic demo/research data, not real delivery."))
            st.metric(tx("Current completed day"), f"{state.current_day}/{state.horizon_days}")
            if st.session_state.simulator is None:
                seed = st.number_input(tx("Simulator seed"), min_value=0, value=42)
                deviation = st.slider(tx("Hidden parameter deviation"), 0.0, 0.30, 0.30, 0.05,
                                      help=tx("Hidden simulator parameters may differ by up to approximately ±30% from planning assumptions."))
                context = st.checkbox(tx("Enable weekend and fatigue variation"), value=True)
            else:
                config = st.session_state.simulator_config
                st.caption(tx("Simulator settings locked for this campaign."))
                seed = config["seed"]
                deviation = config["parameter_deviation"]
                context = config["contextual_variation"]
                st.number_input(tx("Simulator seed"), min_value=0, value=seed, disabled=True)
                st.slider(tx("Hidden parameter deviation"), 0.0, 0.30, deviation, 0.05, disabled=True)
                st.checkbox(tx("Enable weekend and fatigue variation"), value=context, disabled=True)
            quantum = st.number_input(tx("Adaptive allocation quantum, RUB"), 1_000.0, 100_000.0, 10_000.0, 1_000.0)
            buttons = st.columns(4)
            choices = [1, 3, 7, state.remaining_days]
            labels = [tx("Next day"), tx("Next 3 days"), tx("Next 7 days"), tx("Remaining campaign")]
            selected_days = None
            for column, count, label in zip(buttons, choices, labels, strict=True):
                if column.button(label, disabled=state.remaining_days == 0, width="stretch"):
                    selected_days = count
            if selected_days is not None:
                try:
                    simulator = ensure_simulator(state, int(seed), float(deviation), bool(context))
                    runtime = st.session_state.policy_runtime or create_policy_runtime(
                        state, seed=simulator.seed
                    )
                    before_day = state.current_day
                    before_rows = len(state.observations)
                    before_spend = state.actual_spend
                    before_clicks = sum(item.clicks for item in state.observations)
                    before_conversions = sum(item.conversions for item in state.observations)
                    with st.spinner(tx("Simulating observed delivery…")):
                        updated, runtime = simulate_next_days(
                            state,
                            simulator,
                            runtime,
                            days=int(selected_days),
                            quantum_rub=float(quantum),
                        )
                    st.session_state.campaign = updated
                    st.session_state.policy_runtime = runtime
                    updated_monitoring = build_monitoring(updated)
                    status_counts = updated_monitoring.channels["status"].value_counts()
                    st.session_state.simulation_summary = {
                        "campaign_id": updated.campaign_id,
                        "days_advanced": updated.current_day - before_day,
                        "rows_added": len(updated.observations) - before_rows,
                        "spend_added": updated.actual_spend - before_spend,
                        "clicks_added": (
                            sum(item.clicks for item in updated.observations) - before_clicks
                        ),
                        "conversions_added": (
                            sum(item.conversions for item in updated.observations)
                            - before_conversions
                        ),
                        "status_counts": {status: int(count) for status, count in status_counts.items()},
                        "completed": updated.remaining_days == 0,
                    }
                    st.rerun()
                except ValueError as exc:
                    st.error(tx("Check inputs: {details}", details=str(exc)))

    with tabs[1]:
        st.caption(tx("Uploaded rows become OBSERVED FACT after explicit validation and ingestion."))
        st.caption(tx("Upload complete campaign days. Missing channel/day rows count as zero; the latest uploaded day advances the campaign. Use absolute campaign day for incremental uploads. Reach-target campaigns require reach in every row; simulation does not generate reach."))
        st.download_button(
            tx("Download CSV template"),
            FACT_TEMPLATE,
            file_name=download_filename("fact_template.csv"),
            mime="text/csv",
        )
        uploaded = st.file_uploader(tx("Upload campaign fact CSV"), type=["csv"])
        validated = None
        conflicts_with_history = False
        if uploaded is not None:
            try:
                anchors = {item.date - timedelta(days=item.day - 1)
                           for item in state.observations if item.date is not None}
                if len(anchors) > 1:
                    raise ValueError("Historical date/day mapping is inconsistent; use reviewed absolute campaign days.")
                validated = parse_fact_csv(
                    uploaded.getvalue(),
                    state.planning_catalog,
                    horizon_days=state.horizon_days,
                    campaign_start_date=(state.original_request.start_date
                                         if state.original_request.planning_mode == PlanningMode.CALENDAR_AWARE
                                         else next(iter(anchors), None)),
                    require_day=(bool(state.observations) and not anchors
                                 and state.original_request.planning_mode == PlanningMode.UNIFORM),
                    daily_caps=({(p.day, p.channel): state.day_cap(p.day, p.channel)
                                 for p in state.temporal_profiles}
                                if state.original_request.planning_mode == PlanningMode.CALENDAR_AWARE
                                else None),
                )
                incoming_keys = {
                    (item.day, item.channel) for item in validated.observations
                }
                historical_keys = {
                    (item.day, item.channel) for item in state.observations
                }
                conflicts = incoming_keys.intersection(historical_keys)
                projected_spend = state.actual_spend + sum(
                    item.spend for item in validated.observations
                )
                conflicts_with_history = bool(conflicts) or (
                    projected_spend > state.original_budget + 0.01
                )
                if conflicts:
                    st.error(tx("{count} day/channel row(s) already exist in campaign history.", count=len(conflicts)))
                if projected_spend > state.original_budget + 0.01:
                    st.error(tx("Ingesting this file would exceed the original campaign budget."))
                if not conflicts_with_history:
                    st.success(tx("Schema, values, and campaign-history validation passed."))
                preview_cards = metric_columns(4)
                preview_cards[0].metric(tx("Validated rows"), len(validated.observations))
                preview_cards[1].metric(
                    tx("Day range"),
                    f"{min(item.day for item in validated.observations)}–"
                    f"{max(item.day for item in validated.observations)}",
                )
                preview_cards[2].metric(
                    tx("Channels"), len({item.channel for item in validated.observations})
                )
                preview_cards[3].metric(
                    tx("Uploaded spend"),
                    rubles(sum(item.spend for item in validated.observations)),
                )
                display_table(ui_frame(validated.preview), width="stretch", hide_index=True)
            except ValueError as exc:
                st.error(tx("Check inputs: {details}", details=str(exc)))
        if st.button(
            tx("Ingest observed fact"),
            type="primary",
            disabled=validated is None or conflicts_with_history,
        ):
            try:
                updated = ingest_and_record(
                    state,
                    validated.observations,
                    "Uploaded fact ingested",
                )
                st.session_state.campaign = updated
                seed_for_policy = (
                    st.session_state.simulator.seed
                    if st.session_state.simulator is not None
                    else 42
                )
                st.session_state.policy_runtime = create_policy_runtime(
                    updated, seed=seed_for_policy
                )
                st.success(tx("Ingested {count} rows.", count=len(validated.observations)))
                st.rerun()
            except ValueError as exc:
                st.error(tx("Check inputs: {details}", details=str(exc)))
    if state.current_day > 0 and st.button(tx("View monitoring →"), type="primary"):
        go_to("Campaign Monitor")


def render_campaign_monitor() -> None:
    st.title(tx("Campaign Monitor"))
    page_intro("Campaign Monitor")
    state = campaign_or_warning()
    if state is None:
        return
    if state.current_day > 0:
        st.session_state.monitor_reviewed_campaign_id = state.campaign_id
    monitoring = build_monitoring(state)
    p = monitoring.progress
    cards = metric_columns(5)
    cards[0].metric(tx("Original budget"), rubles(p.planned_budget))
    cards[1].metric(tx("Observed spend"), rubles(p.actual_spend))
    cards[2].metric(tx("Remaining budget"), rubles(p.remaining_budget))
    cards[3].metric(tx("Observed KPI"), compact(p.actual_kpi_to_date))
    cards[4].metric(
        tx("Forecast final KPI"),
        compact(p.projected_final_kpi),
        delta=compact(p.forecast_delta),
        delta_color="off",
        help=tx("Difference from the original planned final KPI."),
    )
    interval_lower = p.projection_interval_lower
    interval_upper = p.projection_interval_upper
    observation_days = p.projection_interval_observation_days
    if interval_lower is not None and interval_upper is not None:
        st.info(tx("Forecast range (empirical/non-calibrated): {lower}–{upper} from {days} observed day(s).",
                   lower=compact(interval_lower), upper=compact(interval_upper), days=observation_days or 0))
    else:
        st.caption(tx("Forecast range (empirical/non-calibrated): unavailable until at least 3 observed days are present (currently {days}).",
                      days=observation_days or 0))
    cards = metric_columns(4)
    cards[0].metric(tx("Budget pace"), f"{p.budget_pace:.1%}")
    cards[1].metric(tx("KPI pace"), f"{p.kpi_pace:.1%}")
    cards[2].metric(tx("Observed KPI variance vs plan in force"), f"{p.plan_vs_fact_variance:+.1%}")
    cost_name = {ObjectiveMetric.CLICKS: "CPC", ObjectiveMetric.CONVERSIONS: "CPA",
                 ObjectiveMetric.REACH: tx("Cost per non-deduplicated reach")}[state.objective]
    st.caption(tx("Reach is non-deduplicated. Status is descriptive, not a significance test; omitted fact rows count as zero through the latest observed day."))
    if future_spend_scale(state) < 1.0:
        st.warning(tx("Observed spend is ahead of the active plan. Forecast and static execution proportionally scale remaining allocations to the remaining budget; approved plan history is unchanged. Replan to approve new future allocations."))
    cards[3].metric(tx("Current {cost}", cost=cost_name), rubles(p.current_unit_cost),
                    help=tx("help.CPC") if state.objective == ObjectiveMetric.CLICKS else tx("help.CPA") if state.objective == ObjectiveMetric.CONVERSIONS else None)

    daily = monitoring.daily
    elapsed = daily[daily["day"] <= max(1, state.current_day)]
    spend_figure = go.Figure()
    if "original_planned_spend" in elapsed:
        spend_figure.add_bar(
            x=elapsed["day"],
            y=elapsed["original_planned_spend"],
            name=tx("Original plan"), marker_color="#64748B",
        )
    spend_figure.add_bar(
        x=elapsed["day"], y=elapsed["planned_spend"], name=tx("Plan in force"),
        marker_color="#0F766E",
    )
    spend_figure.add_bar(
        x=elapsed["day"], y=elapsed["actual_spend"], name=tx("Observed fact"),
        marker_color="#16A34A",
    )
    spend_figure.update_layout(
        title=tx("Daily spend: original plan / plan in force / observed fact"),
        xaxis_title=tx("Day range"), yaxis_title=tx("Spend, RUB"),
        barmode="group",
    )
    plot_chart(spend_figure, width="stretch")

    cumulative = go.Figure()
    cumulative.add_scatter(
        x=daily["day"],
        y=daily["planned_cumulative_kpi"],
        name=tx("Plan in force"),
        mode="lines",
        line={"color": "#0F766E"},
    )
    if "original_planned_cumulative_kpi" in daily:
        cumulative.add_scatter(
            x=daily["day"],
            y=daily["original_planned_cumulative_kpi"],
            name=tx("Original plan"),
            mode="lines",
            line={"dash": "dot", "color": "#64748B"},
        )
    cumulative.add_scatter(
        x=elapsed["day"],
        y=elapsed["actual_cumulative_kpi"],
        name=tx("Observed fact"),
        mode="lines+markers",
        line={"color": "#16A34A"},
    )
    if state.forecast_history:
        latest = state.forecast_history[-1]
        cumulative.add_scatter(
            x=[state.current_day, state.horizon_days],
            y=[latest.actual_kpi, latest.projected_final_kpi],
            name=tx("Forecast"),
            mode="lines",
            line={"dash": "dash", "color": "#D97706"},
        )
    cumulative.update_layout(
        title=tx("Cumulative target KPI: original plan / plan in force / observed fact / forecast"),
        xaxis_title=tx("Day range"), yaxis_title="KPI",
    )
    plot_chart(cumulative, width="stretch")
    downloads = st.columns(2)
    downloads[0].download_button(
        tx("Download monitoring daily CSV"),
        csv_bytes(daily),
        file_name=download_filename("monitoring_daily.csv", state.campaign_id),
        mime="text/csv",
        key="download_monitor_daily",
        width="stretch",
    )
    downloads[1].download_button(
        tx("Download campaign observed fact CSV"),
        csv_bytes(observations_frame(state)),
        file_name=download_filename("observed_fact.csv", state.campaign_id),
        mime="text/csv",
        key="download_monitor_fact",
        width="stretch",
    )

    st.subheader(tx("Channel status"))
    st.caption(tx("Status compares observed KPI per RUB with planned KPI per RUB using ±10% thresholds. It is descriptive and does not imply statistical significance."))
    candidate_columns = [
        "channel", "planned_spend", "actual_spend", "spend_delta", "spend_pace",
        "planned_kpi", "actual_kpi", "kpi_delta", "planned_ctr", "actual_ctr",
        "ctr_delta", "planned_cr", "actual_cr", "cr_delta", "planned_unit_cost",
        "actual_unit_cost", "unit_cost_delta", "planned_cpa", "actual_cpa",
        "cpa_delta", "status",
    ]
    display_columns = [
        column for column in candidate_columns if column in monitoring.channels.columns
    ]
    display_table(ui_frame(monitoring.channels[display_columns]), width="stretch", hide_index=True)
    st.download_button(
        tx("Download channel monitoring CSV"),
        csv_bytes(monitoring.channels),
        file_name=download_filename("monitoring_channels.csv", state.campaign_id),
        mime="text/csv",
        key="download_monitor_channels",
    )

    if state.forecast_history:
        history = pd.DataFrame([item.model_dump() for item in state.forecast_history])
        plot_chart(
            px.line(
                history,
                x="created_at_day",
                y="projected_final_kpi",
                markers=True,
                color="reason",
                title=tx("Forecast history"),
                labels=table_columns(st.session_state.language),
            ),
            width="stretch",
        )
    st.subheader(tx("Plan versions"))
    version_rows = []
    for index, version in enumerate(state.plan_versions):
        totals: dict[str, float] = {}
        for allocation in version.future_allocations:
            totals[allocation.channel] = totals.get(allocation.channel, 0.0) + allocation.spend
        main_delta = "Initial allocation"
        if index > 0:
            previous_totals: dict[str, float] = {}
            for allocation in state.plan_versions[index - 1].future_allocations:
                if allocation.day > version.created_at_day:
                    previous_totals[allocation.channel] = (
                        previous_totals.get(allocation.channel, 0.0) + allocation.spend
                    )
            deltas = {
                channel: totals.get(channel, 0.0) - previous_totals.get(channel, 0.0)
                for channel in set(totals) | set(previous_totals)
            }
            if deltas:
                channel = max(deltas, key=lambda name: abs(deltas[name]))
                main_delta = f"{channel}: {deltas[channel]:+,.0f} RUB"
        matching_forecasts = [
            forecast
            for forecast in state.forecast_history
            if forecast.created_at_day == version.created_at_day
            and forecast.reason == version.reason
        ]
        version_rows.append(
            {
                "version": version.version_id,
                "version_type": "Original plan" if index == 0 else "Revised plan",
                "created_at_day": version.created_at_day,
                "policy": policy_label(version.policy),
                "reason": version.reason,
                "remaining_budget": version.remaining_budget,
                "remaining_horizon": version.remaining_horizon,
                "forecast_at_creation": (
                    matching_forecasts[-1].projected_final_kpi
                    if matching_forecasts
                    else None
                ),
                "main_allocation_delta": main_delta,
            }
        )
    versions = pd.DataFrame(version_rows)
    display_table(ui_frame(versions), width="stretch", hide_index=True)
    metric_help()
    if state.remaining_days and state.current_day and st.button(tx("Recalculate remaining plan →"), type="primary"):
        go_to("Adaptive Control")


def version_channel_totals(state: CampaignState) -> pd.DataFrame:
    frames = []
    for version in state.plan_versions:
        frame = pd.DataFrame([row.model_dump() for row in version.future_allocations])
        if frame.empty:
            continue
        grouped = frame.groupby("channel", as_index=False)["spend"].sum()
        grouped["version"] = version.version_id
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def latest_future_spend(state: CampaignState) -> pd.DataFrame:
    frame = pd.DataFrame(
        [row.model_dump() for row in state.latest_plan.future_allocations]
    )
    if frame.empty:
        return pd.DataFrame(columns=["channel", "future_spend"])
    return (
        frame[frame["day"] > state.current_day]
        .groupby("channel", as_index=False)["spend"]
        .sum()
        .rename(columns={"spend": "future_spend"})
    )


def render_adaptive_control() -> None:
    st.title(tx("Adaptive Control"))
    page_intro("Adaptive Control")
    state = campaign_or_warning()
    if state is None:
        return
    if len(state.plan_versions) > 1:
        st.session_state.adaptive_reviewed_campaign_id = state.campaign_id
    supported_policies = (("static", "periodic_reoptimization")
                          if state.objective == ObjectiveMetric.REACH else POLICY_NAMES)
    current_index = supported_policies.index(state.current_policy)
    if state.objective == ObjectiveMetric.REACH:
        st.info(tx("Online bandit policies support clicks and conversions only, not reach."))
    progress = build_monitoring(state).progress
    st.caption(tx("Adaptive policies act only on the remaining campaign. Observed fact and all earlier plan versions remain immutable."))
    context_cards = metric_columns(4)
    context_cards[0].metric(tx("Plan version before action"), state.latest_plan.version_id)
    context_cards[1].metric(tx("Remaining budget"), rubles(state.remaining_budget))
    context_cards[2].metric(tx("Remaining days"), state.remaining_days)
    context_cards[3].metric(tx("Forecast before action"), compact(progress.projected_final_kpi))
    language = st.session_state.language
    selected = st.selectbox(
        tx("Adaptive policy for remaining campaign"),
        supported_policies,
        index=current_index,
        format_func=lambda value: policy_label(value, language),
    )
    if selected in {"thompson_sampling", "linucb"}:
        st.caption(tx("Thompson Sampling and LinUCB are research policies evaluated in a synthetic environment, not production advertising-platform controllers."))
    if selected == "linucb":
        st.caption(tx("LinUCB is an experimental contextual strategy. Results depend on exploration alpha and the synthetic scenario."))
    interval = st.selectbox(
        tx("Periodic replan interval, days"),
        [1, 3, 7],
        index=[1, 3, 7].index(state.periodic_interval_days)
        if state.periodic_interval_days in [1, 3, 7]
        else 1,
    )
    if st.button(tx("Switch policy"), disabled=selected == state.current_policy):
        try:
            updated = switch_policy(
                state, selected, periodic_interval_days=int(interval)
            )
            seed = st.session_state.simulator.seed if st.session_state.simulator else 42
            runtime = create_policy_runtime(updated, seed=seed)
            st.session_state.campaign = updated
            st.session_state.policy_runtime = runtime
            st.success(tx("Adaptive policy switched to {policy}. Observed fact, spend, and plan history were preserved.",
                          policy=policy_label(selected)))
            st.rerun()
        except ValueError as exc:
            st.error(tx("Check inputs: {details}", details=str(exc)))

    state = st.session_state.campaign
    runtime = st.session_state.policy_runtime or create_policy_runtime(state)
    if state.current_policy == "periodic_reoptimization":
        next_days = [
            day
            for day in range(state.current_day + 1, state.horizon_days + 1)
            if day != 1 and (day - 1) % state.periodic_interval_days == 0
        ]
        st.info(tx("Next scheduled replan: before day {day}", day=next_days[0])
                if next_days else tx("Next scheduled replan: none before completion"))
    if st.button(
        tx("Recalculate remaining plan"),
        type="primary",
        disabled=state.remaining_days == 0 or state.remaining_budget <= 0.01,
    ):
        try:
            before_version = state.latest_plan.version_id
            before_projection = build_monitoring(state).progress.projected_final_kpi
            before_allocations = latest_future_spend(state).rename(
                columns={"future_spend": "before_replan_spend"}
            )
            updated = recalculate_remaining_plan(state, runtime)
            after_projection = build_monitoring(updated).progress.projected_final_kpi
            after_allocations = latest_future_spend(updated).rename(
                columns={"future_spend": "after_replan_spend"}
            )
            shifts = before_allocations.merge(
                after_allocations, on="channel", how="outer"
            ).fillna(0.0)
            shifts["allocation_shift"] = (
                shifts["after_replan_spend"] - shifts["before_replan_spend"]
            )
            st.session_state.campaign = updated
            seed = st.session_state.simulator.seed if st.session_state.simulator else 42
            st.session_state.policy_runtime = create_policy_runtime(updated, seed=seed)
            st.session_state.replan_summary = {
                "campaign_id": updated.campaign_id,
                "before_version": before_version,
                "after_version": updated.latest_plan.version_id,
                "before_projection": before_projection,
                "after_projection": after_projection,
                "created_at_day": updated.current_day,
                "shifts": shifts.to_dict(orient="records"),
            }
            st.rerun()
        except ValueError as exc:
            st.error(tx("Check inputs: {details}", details=str(exc)))

    replan_summary = st.session_state.replan_summary
    if replan_summary and replan_summary.get("campaign_id") == state.campaign_id:
        st.success(tx("Created revised plan {after} from {before} at day {day}; historical observed fact was unchanged.",
                      after=replan_summary["after_version"], before=replan_summary["before_version"],
                      day=replan_summary["created_at_day"]))
        result_cards = metric_columns(3)
        result_cards[0].metric(tx("Revised plan version"), replan_summary["after_version"])
        result_cards[1].metric(
            tx("Forecast before replan"), compact(replan_summary["before_projection"])
        )
        result_cards[2].metric(
            tx("Forecast after replan"),
            compact(replan_summary["after_projection"]),
            delta=compact(
                replan_summary["after_projection"]
                - replan_summary["before_projection"]
            ),
            delta_color="off",
        )
        display_table(
            ui_frame(pd.DataFrame(replan_summary["shifts"])), width="stretch", hide_index=True
        )

    st.subheader(tx("Policy diagnostics"))
    diagnostics = policy_diagnostics(state, runtime)
    display_table(ui_frame(diagnostics), width="stretch", hide_index=True)
    if state.current_policy == "thompson_sampling":
        st.caption(tx("Posterior mean and uncertainty summarize observed KPI events per impression; selection also includes current marginal KPI per RUB."))
    elif state.current_policy == "linucb":
        st.warning(tx("LinUCB combines estimated context contribution with an uncertainty bonus. Offline results show material sensitivity to alpha; no absolute superiority is claimed."))
    elif state.current_policy == "periodic_reoptimization":
        beliefs = build_monitoring(state).beliefs
        belief_rows = [
            {
                "channel": name,
                "updated_ctr": channel.base_ctr,
                "updated_cr": channel.base_cr,
                "original_ctr": state.planning_catalog[name].base_ctr,
                "original_cr": state.planning_catalog[name].base_cr,
            }
            for name, channel in beliefs.items()
        ]
        display_table(ui_frame(pd.DataFrame(belief_rows)), width="stretch", hide_index=True)

    totals = version_channel_totals(state)
    if not totals.empty:
        plot_chart(
            px.bar(
                totals,
                x="channel",
                y="spend",
                color="version",
                barmode="group",
                title=tx("Future allocation by plan version"),
                labels=table_columns(st.session_state.language),
            ),
            width="stretch",
        )
    research_help()
    if st.button(tx("Open experiments →")):
        go_to("Experiments")


def render_simulator_debug() -> None:
    with st.expander(tx("Simulator debug / evaluation-only")):
        debug_seed = st.number_input(tx("Simulator truth seed"), min_value=0, value=42)
        truth = CampaignSimulator(st.session_state.planning_catalog, 21, int(debug_seed)).truth_snapshot()
        st.warning(tx("SIMULATED HIDDEN TRUTH — evaluation/debug only; adaptive policies cannot access it."))
        display_table(
            ui_frame(pd.DataFrame(
                [{"channel": name, **values.__dict__} for name, values in truth.items()]
            )),
            width="stretch",
            hide_index=True,
        )


def render_experiments() -> None:
    st.title(tx("Policy Experiments"))
    page_intro("Experiments")
    st.caption(tx("The research comparison uses a common Uniform V1 baseline and a synthetic environment. It is separate from the main Calendar-aware media planner."))
    st.caption(tx("Compare adaptive policies on paired synthetic scenarios. The Oracle is evaluation-only and is never available to a deployable policy."))
    with st.form("experiment_form"):
        columns = st.columns(3)
        seeds = columns[0].number_input(tx("Paired seeds"), 1, 20, 5)
        horizon = columns[1].number_input(tx("Horizon"), 3, 60, 21)
        budget = columns[2].number_input(tx("Budget, RUB"), 50_000.0, 10_000_000.0, 1_200_000.0, 50_000.0)
        language = st.session_state.language
        objective = columns[0].selectbox(tx("Objective"), ["conversions", "clicks"],
                                         format_func=lambda value: translate(value, language))
        quantum = columns[1].number_input(tx("Quantum, RUB"), 1_000.0, 100_000.0, 10_000.0, 1_000.0)
        alpha = columns[2].slider(tx("LinUCB alpha"), 0.0, 2.0, 0.7, 0.1, help=tx("help.linucb"))
        run = st.form_submit_button(tx("Run comparison"), type="primary")
    if run:
        configuration = {
            "catalog_seed": int(st.session_state.catalog_seed),
            "planning_catalog": {name: channel.model_dump(mode="json")
                                 for name, channel in st.session_state.planning_catalog.items()},
            "seeds": list(range(int(seeds))),
            "budget": float(budget),
            "horizon_days": int(horizon),
            "objective": objective,
            "quantum_rub": float(quantum),
            "reoptimization_interval_days": 3,
            "thompson_prior_strength": 5_000.0,
            "linucb_alpha": float(alpha),
            "parameter_deviation": 0.20,
            "context_strength": 1.0,
        }
        with st.spinner(tx("Running paired policy simulations…")):
            st.session_state.experiment_result = run_policy_comparison(
                st.session_state.planning_catalog,
                configuration["seeds"],
                budget=float(budget),
                horizon_days=int(horizon),
                objective=ObjectiveMetric(objective),
                quantum_rub=float(quantum),
                linucb_alpha=float(alpha),
            )
        st.session_state.experiment_config = configuration
    if st.session_state.experiment_result is None:
        st.info(tx("Configure a bounded comparison and run it explicitly."))
        research_help()
        render_simulator_debug()
        return
    summary, runs, daily, _channels = st.session_state.experiment_result
    configuration = st.session_state.experiment_config or {"objective": objective}
    wins = (
        runs.assign(win=runs["kpi_delta_vs_static"] > 0)
        .groupby("policy", as_index=False)["win"]
        .sum()
        .rename(columns={"win": "wins_vs_static"})
    )
    table = summary.merge(wins, on="policy", how="left")
    table.insert(1, "policy_label", table["policy"].map(policy_label))
    display_table(ui_frame(table.assign(policy_label=table["policy"].map(research_policy_label))),
                  width="stretch", hide_index=True)
    download_columns = st.columns(3)
    download_columns[0].download_button(
        tx("Download experiment summary CSV"),
        csv_bytes(table),
        file_name=download_filename("policy_experiment_summary.csv"),
        mime="text/csv",
        key="download_experiment_summary",
        width="stretch",
    )
    labeled_runs = runs.copy()
    labeled_runs.insert(1, "policy_label", labeled_runs["policy"].map(policy_label))
    download_columns[1].download_button(
        tx("Download experiment runs CSV"),
        csv_bytes(labeled_runs),
        file_name=download_filename("policy_experiment_runs.csv"),
        mime="text/csv",
        key="download_experiment_runs",
        width="stretch",
    )
    download_columns[2].download_button(
        tx("Download experiment config JSON"),
        json.dumps(configuration, indent=2).encode("utf-8"),
        file_name=download_filename("policy_experiment_config.json"),
        mime="application/json",
        key="download_experiment_config",
        width="stretch",
    )
    chart_runs = runs.assign(policy_label=runs["policy"].map(research_policy_label))
    chart_summary = summary.assign(policy_label=summary["policy"].map(research_policy_label))
    plot_chart(
        px.box(
            chart_runs,
            x="policy_label",
            y="total_kpi",
            points="all",
            title=tx("KPI distribution"),
            labels={"policy_label": tx("Adaptive policy"), "total_kpi": "KPI"},
        ),
        width="stretch",
    )
    plot_chart(
        px.bar(
            chart_summary,
            x="policy_label",
            y="mean_regret_to_oracle",
            title=tx("Mean regret to Oracle (evaluation-only)"),
            labels={"policy_label": tx("Adaptive policy"),
                    "mean_regret_to_oracle": table_columns(st.session_state.language).get("mean_regret_to_oracle", "Mean regret to Oracle")},
        ),
        width="stretch",
    )
    metric = (
        "cumulative_conversions"
        if configuration.get("objective") == "conversions"
        else "cumulative_clicks"
    )
    curves = daily.groupby(["policy", "day"], as_index=False)[metric].mean()
    curves["policy_label"] = curves["policy"].map(research_policy_label)
    plot_chart(
        px.line(
            curves,
            x="day",
            y=metric,
            color="policy_label",
            title=tx("Mean cumulative KPI"),
            labels={"policy_label": tx("Adaptive policy"), metric: "KPI"},
        ),
        width="stretch",
    )
    plot_chart(
        px.box(
            chart_runs[chart_runs["policy"] != "static"],
            x="policy_label",
            y="kpi_delta_vs_static",
            points="all",
            title=tx("Paired KPI difference versus static"),
            labels={"policy_label": tx("Adaptive policy"), "kpi_delta_vs_static": "KPI"},
        ),
        width="stretch",
    )
    st.warning(tx("Oracle (evaluation-only) uses simulator truth and is not a deployable policy. Simulator truth is synthetic. Results depend on planner misspecification, horizon, noise, context strength, and hyperparameters. Periodic replanning can be competitive; exploration can hurt with accurate priors or short campaigns; LinUCB alpha sensitivity is nontrivial."))
    research_help()
    render_simulator_debug()


initialize_session()
page = render_sidebar()
if page == "Overview":
    render_overview()
elif page == "Inventory / Generator":
    render_inventory()
elif page == "Media Plan":
    render_media_plan()
elif page == "Campaign Monitor":
    render_campaign_monitor()
elif page == "Fact Ingestion":
    render_fact_ingestion()
elif page == "Adaptive Control":
    render_adaptive_control()
elif page == "Experiments":
    render_experiments()
elif page == "How it works":
    render_methodology()
