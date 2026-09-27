"""UI-only regression checks: state, display precision, branding, and charts."""

import ast
import json
from pathlib import Path
import re
import tomllib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.campaign import start_campaign
from mediaplan_optimizer.schemas import DEFAULT_CALENDAR_START, OptimizationRequest, PlanningMode
from mediaplan_optimizer.service import optimize_media_plan
from ui_components import compact, csv_bytes, display_table, download_filename, metric_columns, rubles, style_chart
from ui_text import APP_NAME, APP_SLUG, localize_frame, translate


ROOT = Path(__file__).resolve().parents[1]


def button(app, key, language):
    return next(x for x in app.button if x.label == translate(key, language))


def navigate(app, key, language):
    option = next(x for x in app.sidebar.radio[1].options if x.endswith(translate(key, language)))
    app.sidebar.radio[1].set_value(option).run(timeout=30)
    assert not app.exception


@pytest.mark.parametrize("language", ["ru", "en"])
def test_planner_identity_inventory_and_research_separation(language):
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == APP_NAME
    assert app.sidebar.title[0].value == APP_NAME
    navigation_labels = list(app.sidebar.radio[1].options)
    assert translate("Start here", language) not in [x.value for x in app.subheader]
    for key in ("Type A — Fixed budget", "Type B — Target KPI", "Core planning flow"):
        assert translate(key, language) in [x.value for x in app.subheader]
    labels = [x.label for x in app.button]
    for key in ("Reset campaign", "Reset catalog to seed 42", "Start new planning scenario"):
        assert translate(key, language) not in labels
    button(app, "Inspect inventory", language).click().run(timeout=30)
    assert app.session_state["page"] == "Inventory / Generator"
    assert not app.exception
    app.run(timeout=30)
    assert app.session_state["page"] == "Inventory / Generator"
    assert app.sidebar.radio[1].options == navigation_labels
    expanders = [x.label for x in app.expander]
    for key in ("All catalog parameters", "Public benchmark context",
                "Selected channel parameters", "Planner metrics"):
        assert translate(key, language) in expanders
    assert translate("Research methods", language) not in expanders
    assert translate("Simulator debug / evaluation-only", language) not in expanders
    figures = [json.loads(x.proto.spec) for x in app.get("plotly_chart")]
    assert len(figures) == 7
    assert all(len(fig["data"]) == 1 for fig in figures)
    assert all(fig["data"][0]["line"]["color"] == "#0F766E" for fig in figures)
    assert list(app.session_state["planning_catalog"]) == list(generate_catalog(seed=42))
    navigate(app, "Experiments", language)
    assert translate("Research methods", language) in [x.label for x in app.expander]
    assert translate("Simulator debug / evaluation-only", language) in [x.label for x in app.expander]


@pytest.mark.parametrize("language", ["ru", "en"])
def test_task_aware_summary_and_state_aware_resets(language):
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    button(app, "Calculate media plan", language).click().run(timeout=30)
    assert app.session_state["page"] == "Media Plan"
    button(app, "Calculate media plan", language).click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 14  # Five original cards plus nine whole-plan metrics.
    assert [x.label for x in app.metric[:5]] == [translate(key, language) for key in (
        "Planned spend", "Forecast conversions", "clicks", "Effective CPM", "Expected unused",
    )]
    assert translate("Forecast conversions", language) in [x.label for x in app.metric]
    assert translate("Original planned KPI", language) not in [x.label for x in app.metric]
    assert translate("Optional: continue with a campaign", language) in [x.value for x in app.subheader]
    assert button(app, "Start new planning scenario", language).disabled
    button(app, "Start campaign workflow", language).click().run(timeout=30)
    assert app.session_state["campaign"] is not None
    assert button(app, "Reset campaign", language).disabled
    campaign = app.session_state["campaign"]
    next(x for x in app.checkbox if x.label == translate("Confirm campaign reset", language)).check().run(timeout=30)
    button(app, "Reset campaign", language).click().run(timeout=30)
    assert not app.exception
    assert app.session_state["campaign"] is None
    assert campaign.current_day == 0
    assert translate("Reset campaign", language) not in [x.label for x in app.button]
    app.session_state["catalog_seed"] = 43
    app.session_state["planning_catalog"] = generate_catalog(seed=43)
    app.run(timeout=30)
    assert button(app, "Reset catalog to seed 42", language).disabled
    next(x for x in app.checkbox if x.label == translate("Confirm catalog reset", language)).check().run(timeout=30)
    button(app, "Reset catalog to seed 42", language).click().run(timeout=30)
    assert app.session_state["planning_catalog"] == generate_catalog(seed=42)
    assert translate("Reset catalog to seed 42", language) not in [x.label for x in app.button]
    navigate(app, "Media Plan", language)
    next(x for x in app.radio if x.label == translate("Task", language)).set_value(
        translate("B — Minimum budget for target", language)
    ).run(timeout=30)
    button(app, "Calculate media plan", language).click().run(timeout=30)
    assert not app.exception
    assert app.session_state["draft_result"].status.value == "optimal"
    assert len(app.metric) == 14
    assert [x.label for x in app.metric[:5]] == [translate(key, language) for key in (
        "Minimum modeled budget", "Target value", "Achieved", "Effective CPM", "conversions",
    )]
    assert translate("Minimum modeled budget", language) in [x.label for x in app.metric]
    assert translate("Expected unused", language) not in [x.label for x in app.metric]


def test_display_formatting_preserves_analytical_values_and_csv(monkeypatch):
    frame = pd.DataFrame({"channel": ["SMS"], "spend": [617514.206812345],
                          "ctr": [0.0123456789], "cpm_factor": [1.123456789],
                          "marginal_kpi_per_rub": [0.0023456789]})
    snapshot = frame.copy(deep=True)
    payload = csv_bytes(frame)
    captured = []
    monkeypatch.setattr("ui_components.st.dataframe", lambda styled, **kwargs: captured.append(styled))
    for language in ("ru", "en"):
        monkeypatch.setattr("ui_components.st.session_state", {"language": language})
        displayed = localize_frame(frame, language)
        display_table(displayed)
        pd.testing.assert_frame_equal(captured[-1].data, displayed)
        html = captured[-1].to_html()
        assert "617,514.21" in html
        assert "1.23%" in html
        assert "1.1235" in html
        assert "0.002346" in html
    assert csv_bytes(frame) == payload
    assert b"617514.206812345" in payload
    pd.testing.assert_frame_equal(frame, snapshot)


def test_chart_style_preserves_data_and_semantic_colors():
    figure = px.line(x=[0, 1, 2], y=[0.0, 0.123456789, 5.0])
    values = list(figure.data[0].y)
    figure = style_chart(figure)
    assert list(figure.data[0].y) == values
    assert figure.data[0].line.color == "#0F766E"
    assert figure.layout.paper_bgcolor == "#FFFFFF"
    for color in ("#0F766E", "#64748B", "#16A34A", "#D97706"):
        semantic = go.Figure(go.Scatter(x=[1], y=[2], line_color=color))
        assert style_chart(semantic).data[0].line.color == color


def test_summary_rows_have_at_most_three_cards(monkeypatch):
    assert compact(1e12) == "1.00T"
    assert compact(1e9) == "1.00B"
    widths = []
    monkeypatch.setattr("ui_components.st.columns", lambda count: widths.append(count) or [None] * count)
    assert len(metric_columns(5)) == 5
    assert widths == [3, 2]
    widths.clear()
    assert len(metric_columns(9)) == 9
    assert widths == [3, 3, 3]


def test_theme_and_download_filenames():
    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
    assert config["theme"]["base"] == "light"
    assert config["theme"]["primaryColor"] == "#0F766E"
    assert download_filename("daily_plan.csv") == APP_SLUG + "_daily_plan.csv"
    assert download_filename("observed_fact.csv", "campaign-42") == APP_SLUG + "_campaign-42_observed_fact.csv"
    for campaign_id in ("Кампания 42", "../../private", "", "a/b:c"):
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", download_filename("observed_fact.csv", campaign_id))
    for path in (ROOT / "app.py", ROOT / "ui_components.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kwarg in node.keywords:
                    if kwarg.arg == "file_name":
                        assert isinstance(kwarg.value, ast.Call)
                        assert kwarg.value.func.id == "download_filename"


@pytest.mark.parametrize("language", ["ru", "en"])
def test_one_click_demo_calculates_canonical_draft_and_protects_active_campaign(language, monkeypatch, caplog):
    monkeypatch.setattr("streamlit.elements.lib.policies._shown_default_value_warning", False)
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    app.session_state["planning_catalog"] = generate_catalog(seed=43)
    app.session_state["catalog_seed"] = 43
    app.session_state["planning_horizon_ui"] = 3
    app.session_state["planning_budget_ui"] = 50_000.0
    app.session_state["planning_objective_ui"] = "clicks"
    app.session_state["planning_task_ui"] = "B — Minimum budget for target"
    button(app, "Run demo", language).click().run(timeout=30)
    assert not app.exception
    assert app.session_state["page"] == "Media Plan"
    assert app.session_state["campaign"] is None
    request = app.session_state["draft_request"]
    catalog = generate_catalog(seed=42)
    expected = OptimizationRequest(
        task_type="A", horizon_days=21, objective_metric="conversions",
        budget=1_200_000, included_channels=list(catalog),
        planning_mode="calendar_aware", start_date=DEFAULT_CALENDAR_START,
    )
    assert request == expected
    assert app.session_state["draft_catalog"] == catalog
    result = app.session_state["draft_result"]
    assert result.model_dump() == optimize_media_plan(expected, catalog).model_dump()
    assert result.status.value == "optimal"
    assert any("seed 42" in x.value for x in app.success)
    assert next(x for x in app.number_input if x.label == translate("Horizon, days", language)).value == 21
    button(app, "Calculate media plan", language).click().run(timeout=30)
    assert app.session_state["draft_result"].model_dump() == result.model_dump()
    assert "was created with a default value" not in caplog.text
    button(app, "Start campaign workflow", language).click().run(timeout=30)
    campaign = app.session_state["campaign"]
    navigate(app, "Overview", language)
    assert button(app, "Run demo", language).disabled
    assert app.session_state["campaign"] is campaign


@pytest.mark.parametrize("language", ["ru", "en"])
@pytest.mark.parametrize("task", ["A", "B"])
def test_comparison_semantics_snapshot_and_invalidation(language, task, monkeypatch):
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    catalog = generate_catalog(seed=43)
    request = OptimizationRequest(
        task_type=task, horizon_days=7, objective_metric="clicks",
        included_channels=["SMS", "Social Network 1"], excluded_channels=["Programmatic"],
        planning_mode="calendar_aware", start_date=DEFAULT_CALENDAR_START,
        calendar_profile_seed=11,
        **({"budget": 100_000} if task == "A" else {"target_value": 1_000}),
    )
    result = optimize_media_plan(request, catalog)
    campaign = start_campaign("Comparison", request, result, catalog)
    snapshot = result.model_dump()
    campaign_snapshot = campaign.model_dump()
    app.session_state["draft_result"] = result
    app.session_state["draft_request"] = request
    app.session_state["draft_catalog"] = catalog
    app.session_state["planning_catalog"] = generate_catalog(seed=44)
    app.session_state["campaign"] = campaign
    calls = []
    def baseline_service(baseline_request, baseline_catalog):
        calls.append((baseline_request, baseline_catalog))
        return optimize_media_plan(baseline_request, baseline_catalog)
    monkeypatch.setattr("ui_components.optimize_media_plan", baseline_service)
    navigate(app, "Media Plan", language)
    button(app, "Calculate baseline", language).click().run(timeout=30)
    assert not app.exception
    compared, uniform = app.session_state["uniform_comparison"]
    assert compared is result
    assert calls[0][0].model_dump() == {**request.model_dump(), "planning_mode": PlanningMode.UNIFORM}
    assert calls[0][1] == catalog
    assert result.model_dump() == snapshot
    assert app.session_state["campaign"].model_dump() == campaign_snapshot
    primary = "KPI difference" if task == "A" else "Budget difference"
    other = "Budget difference" if task == "A" else "KPI difference"
    labels = [x.label for x in app.metric]
    assert translate(primary, language) in labels
    assert translate(other, language) not in labels
    difference = ((result.summary.achieved_target - uniform.summary.achieved_target)
                  if task == "A" else result.summary.spend - uniform.summary.spend)
    denominator = uniform.summary.achieved_target if task == "A" else uniform.summary.spend
    metric = next(x for x in app.metric if x.label == translate(primary, language))
    assert metric.value == f"{difference:+,.2f}"
    assert metric.delta == f"{difference / denominator:+.2%}"
    updated = request.model_copy(update={"budget": 120_000} if task == "A" else {"target_value": 1_200})
    app.session_state["draft_request"] = updated
    app.session_state["draft_result"] = optimize_media_plan(updated, catalog)
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["uniform_comparison"] is None
    assert translate(primary, language) not in [x.label for x in app.metric]


@pytest.mark.parametrize("language", ["ru", "en"])
@pytest.mark.parametrize("mode", ["calendar_aware", "uniform"])
def test_reach_campaign_disables_bandits_and_simulation_but_keeps_csv(language, mode):
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    navigate(app, "Media Plan", language)
    objective = next(x for x in app.selectbox if x.label == translate("Objective", language))
    assert set(objective.options) == {translate(x, language) for x in ("clicks", "conversions")}
    task = next(x for x in app.radio if x.label == translate("Task", language))
    task.set_value("B — Minimum budget for target").run(timeout=30)
    target_metric = next(x for x in app.selectbox if x.label == translate("Target metric", language))
    assert set(target_metric.options) == {
        translate(x, language) for x in ("clicks", "conversions", "Non-deduplicated expected reach")
    }
    target_metric.set_value("reach").run(timeout=30)
    # Switching back from reach must not make Type A inherit an invalid objective.
    next(x for x in app.radio if x.label == translate("Task", language)).set_value(
        "A — Maximize KPI"
    ).run(timeout=30)
    objective = next(x for x in app.selectbox if x.label == translate("Objective", language))
    assert objective.value in ("clicks", "conversions")
    assert set(objective.options) == {translate(x, language) for x in ("clicks", "conversions")}
    next(x for x in app.radio if x.label == translate("Task", language)).set_value(
        "B — Minimum budget for target"
    ).run(timeout=30)
    next(x for x in app.radio if x.label == translate("Planning mode", language)).set_value(mode).run(timeout=30)
    next(x for x in app.selectbox if x.label == translate("Target metric", language)).set_value("reach")
    next(x for x in app.number_input if x.label == translate("Horizon, days", language)).set_value(3)
    next(x for x in app.number_input if x.label == translate("Target value", language)).set_value(10_000)
    button(app, "Calculate media plan", language).click().run(timeout=30)
    assert not app.exception
    result = app.session_state["draft_result"]
    assert result.request.task_type.value == "B"
    assert result.request.objective_metric.value == "reach"
    assert result.request.planning_mode.value == mode
    assert result.status.value == "optimal"
    assert result.summary.reach >= 10_000 * (1 - 1e-7)
    assert result.summary.achieved_target == result.summary.reach
    assert translate("Non-deduplicated expected reach", language) in [x.label for x in app.metric]
    button(app, "Start campaign workflow", language).click().run(timeout=30)
    assert not app.exception
    assert app.session_state["page"] == "Fact Ingestion"
    campaign = app.session_state["campaign"]
    assert campaign.objective.value == "reach"
    assert campaign.original_result == result
    assert campaign.current_day == 0
    assert campaign.latest_plan.version_id == "v1"
    assert app.session_state["simulator"] is None
    for key in ("Next day", "Next 3 days", "Next 7 days", "Remaining campaign"):
        assert translate(key, language) not in [x.label for x in app.button]
    assert not app.slider
    assert translate("CSV Upload", language) in [x.label for x in app.tabs]
    assert len(app.get("file_uploader")) == 1
    assert any("reach" in x.value for x in app.info)
    navigate(app, "Adaptive Control", language)
    policies = next(x for x in app.selectbox if x.label == translate("Adaptive policy for remaining campaign", language))
    assert policies.options == [translate("Static policy", language), translate("Periodic reoptimization", language)]
    assert "Thompson Sampling" not in policies.options and "LinUCB" not in policies.options
    assert translate("Online bandit policies support clicks and conversions only, not reach.", language) in [x.value for x in app.info]
    assert app.session_state["campaign"] is campaign


@pytest.mark.parametrize("language", ["ru", "en"])
def test_synthetic_fact_shows_readable_locked_settings_without_raw_dict(language):
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    button(app, "Run demo", language).click().run(timeout=30)
    button(app, "Start campaign workflow", language).click().run(timeout=30)
    assert not app.exception
    next(x for x in app.number_input if x.label == translate("Simulator seed", language)).set_value(17)
    next(x for x in app.slider if x.label == translate("Hidden parameter deviation", language)).set_value(0.15)
    next(x for x in app.checkbox if x.label == translate("Enable weekend and fatigue variation", language)).uncheck()
    button(app, "Next day", language).click().run(timeout=30)
    assert not app.exception
    assert app.session_state["campaign"].current_day == 1
    assert app.session_state["simulator_config"] == {
        "seed": 17, "parameter_deviation": 0.15, "contextual_variation": False,
    }
    visible = "\n".join(str(x.value) for kind in ("caption", "markdown", "text", "code", "json") for x in app.get(kind))
    assert str(app.session_state["simulator_config"]) not in visible
    assert "parameter_deviation" not in visible and "contextual_variation" not in visible
    seed = next(x for x in app.number_input if x.label == translate("Simulator seed", language))
    deviation = next(x for x in app.slider if x.label == translate("Hidden parameter deviation", language))
    context = next(x for x in app.checkbox if x.label == translate("Enable weekend and fatigue variation", language))
    assert seed.value == 17 and seed.disabled
    assert deviation.value == 0.15 and deviation.disabled
    assert context.value is False and context.disabled


def test_uniform_comparison_handles_infeasible_baseline():
    catalog = generate_catalog(seed=42)
    request = OptimizationRequest(task_type="B", objective_metric="conversions",
                                  horizon_days=1, target_value=3954, planning_mode="calendar_aware")
    result = optimize_media_plan(request, catalog)
    assert result.status.value == "optimal"
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    app.session_state["draft_request"] = request
    app.session_state["draft_result"] = result
    app.session_state["draft_catalog"] = catalog
    navigate(app, "Media Plan", "ru")
    button(app, "Calculate baseline", "ru").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["uniform_comparison"][1].status.value == "infeasible"
    assert any("Uniform baseline" in x.value for x in app.warning)
    assert translate("Budget difference") not in [x.label for x in app.metric]


@pytest.mark.parametrize("language", ["ru", "en"])
@pytest.mark.parametrize("scenario", ["type_a", "type_b", "summary_contract", "undefined"])
def test_whole_plan_metrics_render_summary_without_recalculation(language, scenario):
    catalog = generate_catalog(seed=42)
    request = OptimizationRequest(
        task_type="B" if scenario == "type_b" else "A",
        horizon_days=7, planning_mode="calendar_aware",
        objective_metric="clicks" if scenario == "type_b" else "conversions",
        **({"target_value": 1_000} if scenario == "type_b" else {"budget": 100_000}),
    )
    result = optimize_media_plan(request, catalog)
    if scenario == "summary_contract":
        # Deliberately distinguish contract values from channel/day aggregates:
        # presentation must read PlanSummary, not rebuild these metrics.
        result.summary = result.summary.model_copy(update={
            "reach": 123_456.789, "clicks": 2_345.678, "conversions": 67.89,
            "ctr": 0.012345, "vtr": 0.234567, "cr": 0.067891,
            "cpm": 123.4, "cpc": 12.4, "cpa": 567.4,
        })
    elif scenario == "undefined":
        # Exercise the nullable summary fields independently of channel rows.
        result.summary = result.summary.model_copy(update={
            "vtr": None, "cpm": None, "cpc": None, "cpa": None,
        })
    snapshot = result.model_dump()
    daily = pd.DataFrame([row.model_dump() for row in result.allocations])
    channels = pd.DataFrame([row.model_dump() for row in result.channel_metrics])
    exports = (csv_bytes(daily), csv_bytes(channels))
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    app.session_state["draft_request"] = request
    app.session_state["draft_result"] = result
    app.session_state["draft_catalog"] = catalog
    navigate(app, "Media Plan", language)

    assert not app.exception
    assert translate("Whole-plan metrics", language) in [x.value for x in app.subheader]
    assert len(app.metric) == 14
    metrics = list(app.metric)[5:14]
    assert [item.label for item in metrics] == [
        translate("Non-deduplicated expected reach", language),
        translate("clicks", language), translate("conversions", language),
        "CTR", "VTR", "CR", translate("Effective CPM", language), "CPC", "CPA",
    ]
    summary = result.summary
    assert [item.value for item in metrics] == [
        compact(summary.reach), compact(summary.clicks), compact(summary.conversions),
        f"{summary.ctr:.3%}", "—" if summary.vtr is None else f"{summary.vtr:.3%}",
        f"{summary.cr:.3%}", rubles(summary.cpm), rubles(summary.cpc), rubles(summary.cpa),
    ]
    if scenario == "summary_contract":
        assert [item.value for item in metrics] == [
            "123.46K", "2.35K", "67.89", "1.234%", "23.457%", "6.789%",
            "₽123", "₽12", "₽567",
        ]
    if scenario == "undefined":
        assert [metrics[index].value for index in (4, 6, 7, 8)] == ["—"] * 4
    assert translate(
        "All reach columns mean non-deduplicated expected reach summed across channel-days, not unique campaign reach.",
        language,
    ) in [item.value for item in app.caption]
    assert result.model_dump() == snapshot
    assert exports == (
        csv_bytes(pd.DataFrame([row.model_dump() for row in result.allocations])),
        csv_bytes(pd.DataFrame([row.model_dump() for row in result.channel_metrics])),
    )
