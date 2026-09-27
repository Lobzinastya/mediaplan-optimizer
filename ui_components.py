"""Streamlit formatting and plan-result views."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mediaplan_optimizer.calendar import effective_channel
from mediaplan_optimizer.control import allocation_explanations
from mediaplan_optimizer.curves import marginal_metric_response
from mediaplan_optimizer.feasibility import analyze_feasibility
from mediaplan_optimizer.metrics import aggregate_weekly_metrics
from mediaplan_optimizer.schemas import PlanningMode, ResultStatus
from mediaplan_optimizer.service import optimize_media_plan
from ui_text import APP_SLUG, feasibility_text, localize_frame, table_columns, translate


POLICY_LABELS = {
    "static": "Static policy",
    "periodic_reoptimization": "Periodic reoptimization",
    "thompson_sampling": "Thompson Sampling",
    "linucb": "LinUCB",
}
EXPERIMENT_POLICY_LABELS = {
    **POLICY_LABELS,
    "oracle": "Oracle (evaluation-only)",
}


TEAL = "#0F766E"
TEAL_SCALE = ["#F0FDFA", "#CCFBF1", "#14B8A6", "#0F766E", "#115E59"]


def download_filename(artifact: str, campaign_id: str | None = None) -> str:
    """ASCII-only display filename; payload and analytical schema are unchanged."""
    parts = [APP_SLUG]
    if campaign_id is not None:
        parts.append(re.sub(r"[^A-Za-z0-9_-]+", "_", campaign_id).strip("_") or "campaign")
    parts.append(re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact).strip("._") or "export")
    return "_".join(parts)


def metric_columns(count: int):
    """Keep summary labels readable with at most three cards per row."""
    return [column for start in range(0, count, 3)
            for column in st.columns(min(3, count - start))]


def style_chart(figure):
    """Presentation only: never transform trace data."""
    defaults = list(figure.layout.template.layout.colorway or ()) + px.colors.qualitative.Plotly
    figure.update_layout(
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        colorway=[TEAL, "#14B8A6", "#64748B", "#115E59", "#94A3B8"],
        font=dict(family="Arial, sans-serif", size=14, color="#0F172A"),
        title=dict(font=dict(size=18)), margin=dict(l=24, r=24, t=64, b=64),
        legend=dict(orientation="h", y=-0.25, x=0, title_text=""),
        hovermode="closest",
    )
    figure.update_xaxes(gridcolor="#E2E8F0", zeroline=False, automargin=True)
    figure.update_yaxes(gridcolor="#E2E8F0", zeroline=False, automargin=True)
    # Replace both Plotly defaults and Streamlit's theme placeholder colors.
    palette = [TEAL, "#14B8A6", "#64748B", "#115E59", "#94A3B8"]
    for index, trace in enumerate(figure.data):
        color = palette[index % len(palette)]
        if trace.type in {"scatter", "box", "bar"}:
            if getattr(trace.marker, "color", None) is None or (
                isinstance(trace.marker.color, str) and trace.marker.color in defaults
            ):
                trace.marker.color = color
        if trace.type == "scatter" and (trace.line.color is None or trace.line.color in defaults):
            trace.line.color = color
    return figure


def plot_chart(figure, *, target=None, **kwargs):
    return (target or st).plotly_chart(
        style_chart(figure), theme=None,
        config={"displayModeBar": False, "displaylogo": False}, **kwargs,
    )


def display_table(frame: pd.DataFrame, **kwargs):
    """Style a translated copy without rounding stored or exported values."""
    language = st.session_state.get("language", "ru")
    source_names = {label: source for source, label in table_columns(language).items()}
    formats = {}
    for column in frame.select_dtypes(include="number").columns:
        source = source_names.get(column, column)
        if source in {"ctr", "cr", "vtr", "inventory_saturation"} or source.endswith(("_ctr", "_cr", "_pace")):
            formats[column] = "{:.2%}"
        elif source == "marginal_kpi_per_rub":
            formats[column] = "{:,.6f}"
        elif "factor" in source or source in {"relative_marginal_efficiency", "generated_value", "base_ctr", "base_cr", "base_vtr"}:
            formats[column] = "{:,.4f}"
        elif pd.api.types.is_integer_dtype(frame[column]):
            formats[column] = "{:,.0f}"
        else:
            formats[column] = "{:,.2f}"
    return st.dataframe(frame.style.format(formats, na_rep="—"), **kwargs)


def rubles(value: float | None) -> str:
    return "—" if value is None else f"₽{value:,.0f}"


def compact(value: float | None) -> str:
    if value is None:
        return "—"
    absolute = abs(value)
    if absolute >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:,.2f}T"
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:,.2f}K"
    return f"{value:,.2f}"


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a displayed table consistently for download."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def policy_label(value: str, language: str | None = None) -> str:
    label = EXPERIMENT_POLICY_LABELS.get(value, value.replace("_", " ").title())
    language = language or st.session_state.get("language", "ru")
    return translate(label, language) if label in {
        "Static policy", "Periodic reoptimization", "Oracle (evaluation-only)"
    } else label


def research_policy_label(value: str) -> str:
    """Standalone Uniform research labels; campaign labels and CSVs stay unchanged."""
    return tx("Static baseline (Uniform V1)") if value == "static" else policy_label(value)


def tx(key: str, **values: object) -> str:
    return translate(key, st.session_state.get("language", "ru"), **values)


def ui_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return localize_frame(frame, st.session_state.get("language", "ru"))


def metric_help() -> None:
    with st.expander(tx("Planner metrics")):
        for name in ("CPM", "CPC", "CPA", "CTR", "CR", "VTR", "saturation"):
            display_name = tx("Saturation") if name == "saturation" else name
            st.write(f"{display_name}: {tx('help.' + name)}")


def research_help() -> None:
    with st.expander(tx("Research methods")):
        st.caption(tx("Thompson Sampling and LinUCB are research policies evaluated in a synthetic environment, not production advertising-platform controllers."))
        st.caption(tx("LinUCB is an experimental contextual strategy. Results depend on exploration alpha and the synthetic scenario."))
        for name, key in (("Oracle", "oracle"), ("Thompson Sampling", "thompson"), ("LinUCB", "linucb")):
            st.write(f"{name}: {tx('help.' + key)}")


def render_infeasible(result, catalog) -> None:
    cached = st.session_state.get("feasibility_budget_result")
    feasibility = cached[1] if cached is not None and cached[0] is result else result.feasibility
    st.error(tx("The requested target is infeasible for the selected assumptions."))
    if feasibility is None:
        return
    cards = st.columns(3)
    cards[0].metric(tx("Requested"), compact(feasibility.requested_value))
    cards[1].metric(tx("Maximum"), compact(feasibility.maximum_achievable))
    cards[2].metric(tx("Gap"), compact(feasibility.target_gap))
    st.subheader(tx("What can you change?"))
    explanation, recommendations = feasibility_text(
        feasibility, result.request.objective_metric.value, st.session_state.language,
        result.request.horizon_days,
    )
    st.write(explanation)
    for recommendation in recommendations:
        st.write(f"• {recommendation}")
    if feasibility.estimated_budget_for_recommended_target is None and feasibility.recommended_target is not None:
        st.caption(tx("Budget estimates may take several seconds; they use the same Type B optimizer and are model estimates, not guarantees."))
        if st.button(tx("Estimate budgets for feasible options")):
            included = set(result.request.included_channels or catalog)
            excluded = set(result.request.excluded_channels or ())
            selected = {name: channel for name, channel in catalog.items()
                        if name in included and name not in excluded}
            with st.spinner(tx("Estimating counterfactual budgets…")):
                detailed = analyze_feasibility(
                    selected, catalog, result.request.horizon_days,
                    result.request.objective_metric, result.request.target_value,
                    estimate_budgets=True,
                    planning_mode=result.request.planning_mode,
                    start_date=result.request.start_date,
                    calendar_profile_seed=result.request.calendar_profile_seed,
                )
            st.session_state.feasibility_budget_result = (result, detailed)
            st.rerun()


def render_plan_result(result, catalog) -> None:
    cached = st.session_state.get("uniform_comparison")
    if cached is not None and cached[0] is not result:
        st.session_state.uniform_comparison = None
    st.caption(tx("All reach columns mean non-deduplicated expected reach summed across channel-days, not unique campaign reach."))
    if result.status == ResultStatus.INFEASIBLE:
        render_infeasible(result, catalog)
        return
    summary = result.summary
    if summary is None:
        st.error(tx("No plan summary returned."))
        return
    calendar_aware = result.request.planning_mode == PlanningMode.CALENDAR_AWARE
    if calendar_aware:
        st.caption(tx("DETERMINISTIC SYNTHETIC TEMPORAL ASSUMPTIONS — not measured seasonality or platform forecasts."))
    objective = result.request.objective_metric.value
    complement = "clicks" if objective == "conversions" else "conversions"
    cards = metric_columns(5)
    if result.request.task_type.value == "A":
        cards[0].metric(tx("Planned spend"), rubles(summary.spend))
        cards[1].metric(tx("Forecast clicks") if objective == "clicks" else tx("Forecast conversions"),
                        compact(summary.achieved_target))
        cards[2].metric(tx(complement), compact(getattr(summary, complement)))
        cards[3].metric(tx("Effective CPM"), rubles(summary.cpm), help=tx("help.CPM"))
        cards[4].metric(tx("Expected unused"), rubles(summary.unspent_budget))
    else:
        cards[0].metric(tx("Minimum modeled budget"), rubles(summary.spend))
        cards[1].metric(tx("Target value"), compact(result.request.target_value))
        cards[2].metric(tx("Achieved"), compact(summary.achieved_target))
        cards[3].metric(tx("Effective CPM"), rubles(summary.cpm), help=tx("help.CPM"))
        cards[4].metric(tx(complement), compact(getattr(summary, complement)))
    if summary.unspent_budget is not None and summary.unspent_budget > 0.01:
        st.info(tx("Budget remains unused because every selected channel reached its configured daily spend/inventory limit; forcing more spend would violate the planning assumptions."))

    st.subheader(tx("Whole-plan metrics"))
    whole_plan = metric_columns(9)
    for card, label, value in zip(whole_plan[:3], (
        tx("Non-deduplicated expected reach"), tx("clicks"), tx("conversions"),
    ), (summary.reach, summary.clicks, summary.conversions), strict=True):
        card.metric(label, compact(value))
    for card, label, value in zip(whole_plan[3:6], ("CTR", "VTR", "CR"),
                                  (summary.ctr, summary.vtr, summary.cr), strict=True):
        card.metric(label, "—" if value is None else f"{value:.3%}")
    for card, label, value in zip(whole_plan[6:], (tx("Effective CPM"), "CPC", "CPA"),
                                  (summary.cpm, summary.cpc, summary.cpa), strict=True):
        card.metric(label, rubles(value))

    channel_data = pd.DataFrame([item.model_dump() for item in result.channel_metrics])
    explanations = allocation_explanations(result, catalog)
    channel_data = channel_data.merge(
        explanations[["channel", "inventory_saturation", "marginal_kpi_per_rub"]],
        on="channel",
        how="left",
    )
    plot_chart(
        px.bar(channel_data[channel_data["spend"] > 0].sort_values("spend", ascending=False), x="channel", y="spend", title=tx("Budget allocation"),
               labels=table_columns(st.session_state.language)),
        width="stretch",
    )
    with st.expander(tx("Detailed channel results")):
        display_table(ui_frame(channel_data), width="stretch", hide_index=True)
    st.download_button(
        tx("Download channel summary CSV"),
        csv_bytes(channel_data),
        file_name=download_filename("channel_summary.csv"),
        mime="text/csv",
        key="download_plan_channels",
    )

    allocations = pd.DataFrame([row.model_dump() for row in result.allocations])
    objective = result.request.objective_metric.value
    daily = allocations.groupby("day", as_index=False).agg(
        spend=("spend", "sum"), kpi=(objective, "sum")
    )
    daily["cumulative_kpi"] = daily["kpi"].cumsum()
    if calendar_aware:
        first_by_day = allocations.drop_duplicates("day").set_index("day")
        daily["date"] = daily["day"].map(first_by_day["date"])
        daily["weekday"] = daily["day"].map(first_by_day["weekday"])
        plot_chart(
            px.bar(daily, x="date", y="spend", title=tx("Daily total spend"),
                   labels=table_columns(st.session_state.language)),
            width="stretch",
        )
        heat = allocations.pivot(index="channel", columns="day", values="spend")
        plot_chart(
            go.Figure(data=go.Heatmap(
                z=heat.to_numpy(), x=heat.columns.tolist(), y=heat.index.tolist(),
                colorscale=TEAL_SCALE, colorbar={"title": "₽"},
            )).update_layout(title=tx("Channel × day spend"), xaxis_title=tx("Campaign day")),
            width="stretch",
        )
    plot_chart(
        px.line(daily, x="date" if calendar_aware else "day", y="cumulative_kpi", title=f"{tx('Daily cumulative prediction')} · {tx(objective)}",
                labels=table_columns(st.session_state.language)),
        width="stretch",
    )
    if calendar_aware:
        display_table(ui_frame(daily), width="stretch", hide_index=True)
        with st.expander(tx("Calendar day profiles")):
            display_table(ui_frame(pd.DataFrame([p.model_dump() for p in result.temporal_profiles])),
                         width="stretch", hide_index=True)
        profiles = {(p.day, p.channel): p for p in result.temporal_profiles}
        economics = []
        for row in result.allocations:
            profile = profiles[(row.day, row.channel)]
            channel = effective_channel(catalog[row.channel], profile)
            economics.append({
                "day": row.day, "date": row.date, "channel": row.channel,
                "spend": row.spend, "daily_cap": channel.max_daily_spend,
                "marginal_kpi_per_rub": marginal_metric_response(
                    row.spend, channel, result.request.objective_metric,
                ),
                "ctr_factor": profile.ctr_factor, "cr_factor": profile.cr_factor,
                "cpm_factor": profile.cpm_factor,
            })
        with st.expander(tx("Temporal allocation economics")):
            display_table(ui_frame(pd.DataFrame(economics)), width="stretch", hide_index=True)
    st.download_button(
        tx("Download daily plan CSV"),
        csv_bytes(allocations),
        file_name=download_filename("daily_plan.csv"),
        mime="text/csv",
        key="download_daily_plan",
    )
    st.subheader(tx("Weekly predicted delivery"))
    display_table(
        ui_frame(pd.DataFrame(aggregate_weekly_metrics(result.allocations))),
        width="stretch",
        hide_index=True,
    )
    st.subheader(tx("Why this allocation?"))
    display_table(ui_frame(explanations[[
        "channel", "planned_spend", "marginal_kpi_per_rub", "inventory_saturation", "efficiency_rank",
    ]]), width="stretch", hide_index=True)
    with st.expander(tx("Detailed allocation explanation")):
        display_table(ui_frame(explanations), width="stretch", hide_index=True)
    st.caption(tx("Explanations use deterministic marginal KPI efficiency, saturation, and spend caps."))
    if calendar_aware:
        with st.expander(tx("Uniform baseline comparison")):
            type_a = result.request.task_type.value == "A"
            st.caption(tx("Compare achieved KPI for the same budget, channels, horizon, and catalog.")
                       if type_a else tx("Uniform baseline is solved independently for the same target, channels, horizon, and catalog. Compare required model budgets."))
            if st.button(tx("Calculate baseline")):
                baseline_request = result.request.model_copy(update={
                    "planning_mode": PlanningMode.UNIFORM,
                })
                with st.spinner(tx("Calculating Uniform baseline…")):
                    baseline = optimize_media_plan(baseline_request, catalog)
                st.session_state.uniform_comparison = (result, baseline)
            comparison = st.session_state.get("uniform_comparison")
            if comparison is not None and comparison[0] is result:
                baseline = comparison[1]
                st.subheader(tx("Uniform vs Calendar-aware"))
                if baseline.summary is None:
                    st.warning(tx("The Uniform baseline cannot meet this target under the same constraints. No budget difference is available."))
                    if baseline.feasibility is not None:
                        st.write(feasibility_text(
                            baseline.feasibility, objective, st.session_state.language,
                            result.request.horizon_days,
                        )[0])
                else:
                    calendar_value = summary.achieved_target if type_a else summary.spend
                    uniform_value = baseline.summary.achieved_target if type_a else baseline.summary.spend
                    difference = calendar_value - uniform_value
                    percent = f"{difference / uniform_value:+.2%}" if uniform_value else "—"
                    cards = metric_columns(3)
                    formatter = compact if type_a else rubles
                    cards[0].metric(tx("Calendar KPI") if type_a else tx("Calendar required budget"),
                                    formatter(calendar_value))
                    cards[1].metric(tx("Uniform KPI") if type_a else tx("Uniform required budget"),
                                    formatter(uniform_value))
                    cards[2].metric(tx("KPI difference") if type_a else tx("Budget difference"),
                                    f"{difference:+,.2f}", delta=percent, delta_color="off",
                                    help=tx("Calendar minus Uniform; percentage relative to Uniform."))
                    rows = []
                    for label, plan in (("Calendar-aware", result), ("Uniform baseline", baseline)):
                        day_spend = pd.DataFrame([a.model_dump() for a in plan.allocations]).groupby("day")["spend"].sum()
                        rows.append({
                            "mode": tx(label), "spend": plan.summary.spend,
                            **({"requested_target": result.request.target_value} if not type_a else {}),
                            "total_kpi": plan.summary.achieved_target,
                            "unit_cost": (plan.summary.spend / plan.summary.achieved_target
                                          if plan.summary.achieved_target else None),
                            "daily_spend_sd": float(day_spend.std(ddof=0)),
                        })
                    display_table(ui_frame(pd.DataFrame(rows)), width="stretch", hide_index=True)
                    st.caption(tx("Unit cost is spend per unit of the selected KPI (CPC for clicks, CPA for conversions)."))
                st.caption(tx("Temporal variation is modeled, not measured; differences are scenario-specific."))
    metric_help()
