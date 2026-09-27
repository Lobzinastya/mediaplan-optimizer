"""Presentation invariants: translation coverage and state-derived guidance."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.schemas import ObjectiveMetric, OptimizationRequest, TaskType
from mediaplan_optimizer.service import optimize_media_plan
from ui_text import RU, TABLE_RU, WORKFLOW_STEPS, feasibility_text, localize_frame, localized_columns, translate, workflow_progress
from ui_components import csv_bytes, policy_label


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_literal_ui_keys_and_workflow_steps_have_russian_translation():
    keys = {
        node.args[0].value
        for path in (APP_PATH, APP_PATH.with_name("ui_components.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tx"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert not (keys | set(WORKFLOW_STEPS)) - RU.keys()
    assert translate("Overview", "ru") != translate("Overview", "en")


def test_workflow_progress_is_eight_state_derived_steps():
    empty = dict(inventory_reviewed=False, plan_ready=False, campaign_started=False,
                 fact_loaded=False, monitor_reviewed=False, replanned=False,
                 adaptive_opened=False, experimented=False)
    initial = workflow_progress(**empty)
    assert len(initial.completed) == 8
    assert initial.next_step == "Inventory"
    assert not any(initial.completed)
    ready = {**empty, "inventory_reviewed": True, "plan_ready": True,
             "campaign_started": True, "fact_loaded": True}
    mid = workflow_progress(**ready)
    assert mid.completed[:4] == (True,) * 4
    assert mid.next_step == "Monitoring"
    assert workflow_progress(**dict.fromkeys(empty, True)).next_step is None


def test_infeasible_guidance_is_localized_without_changing_service_result():
    catalog = generate_catalog(seed=42)
    result = optimize_media_plan(
        OptimizationRequest(task_type=TaskType.B, objective_metric=ObjectiveMetric.CONVERSIONS,
                            horizon_days=1, target_value=1e12), catalog
    )
    assert result.feasibility is not None
    english = feasibility_text(result.feasibility, "conversions", "en")
    russian = feasibility_text(result.feasibility, "conversions", "ru")
    assert english[0] == result.feasibility.explanation
    assert russian[0] != english[0]
    assert len(russian[1]) >= 1


@pytest.mark.parametrize("language", ["ru", "en"])
def test_infeasible_ui_renders_structured_counterfactuals(language: str):
    catalog = generate_catalog(seed=42)
    request = OptimizationRequest(
        task_type=TaskType.B, objective_metric=ObjectiveMetric.CLICKS,
        horizon_days=1, target_value=20_000, included_channels=["SMS"],
    )
    result = optimize_media_plan(request, catalog)
    assert result.feasibility is not None
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    app.session_state["draft_request"] = request
    app.session_state["draft_result"] = result
    app.session_state["draft_catalog"] = catalog
    option = next(item for item in app.sidebar.radio[1].options
                  if item.endswith(translate("Media Plan", language)))
    app.sidebar.radio[1].set_value(option).run(timeout=30)
    assert not app.exception
    assert any(item.value == translate("What can you change?", language)
               for item in app.subheader)
    recommendation_text = "\n".join(item.value for item in app.markdown)
    assert f"{result.feasibility.recommended_target:,.2f}" in recommendation_text
    assert str(result.feasibility.minimum_feasible_horizon) in recommendation_text
    assert any(item.label == translate("Estimate budgets for feasible options", language)
               for item in app.button)


def test_language_persists_across_pages_and_demo_initialization_is_safe():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    assert app.session_state["language"] == "ru"
    assert any(item.label == translate("Run demo") for item in app.button)
    next(item for item in app.button if item.label == translate("Run demo")).click().run(timeout=30)
    assert not app.exception
    assert app.session_state["catalog_seed"] == 42
    assert app.session_state["demo_ready"] is True
    assert app.session_state["campaign"] is None
    assert app.session_state["draft_result"].status.value == "optimal"
    assert app.session_state["page"] == "Media Plan"
    assert app.session_state["draft_result"].summary.conversions == pytest.approx(3077.0820480462485)
    app.sidebar.radio[0].set_value("English").run(timeout=30)
    assert not app.exception
    assert app.session_state["language"] == "en"
    assert app.sidebar.radio[1].options[0] == "Overview"
    option = next(item for item in app.sidebar.radio[1].options if item.endswith("Media Plan"))
    app.sidebar.radio[1].set_value(option).run(timeout=30)
    assert not app.exception
    assert app.session_state["language"] == "en"
    assert any(item.label == "Calculate media plan" for item in app.button)
    assert next(item for item in app.number_input if item.label == "Budget, RUB").value == 1_200_000.0
    assert next(item for item in app.number_input if item.label == "Horizon, days").value == 21
    next(item for item in app.button if item.label == "Calculate media plan").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["draft_result"].request.planning_mode.value == "calendar_aware"
    assert abs(app.session_state["draft_result"].summary.conversions - 3077.0820480462485) < 1e-6


def test_all_pages_render_in_english():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    app.sidebar.radio[0].set_value("English").run(timeout=30)
    pages = ("Overview", "Inventory / Generator", "Media Plan", "Fact Ingestion",
             "Campaign Monitor", "Adaptive Control", "Experiments", "How it works")
    assert [option.lstrip("✓●○ ") for option in app.sidebar.radio[1].options] == list(pages)
    for page in pages[1:]:
        option = next(item for item in app.sidebar.radio[1].options if item.endswith(page))
        app.sidebar.radio[1].set_value(option).run(timeout=30)
        assert not app.exception, page


def test_all_table_column_translations_are_unique_and_preserve_source_schema():
    source_columns = list(TABLE_RU)
    frame = pd.DataFrame(columns=source_columns)
    for language in ("ru", "en"):
        displayed = localize_frame(frame, language)
        assert displayed.columns.is_unique
        assert list(frame.columns) == source_columns
    assert localized_columns(["policy", "policy_label"], "ru") == ["Код стратегии", "Стратегия"]
    with pytest.raises(ValueError, match="Duplicate localized dataframe columns.*policy"):
        localized_columns(["policy", "policy"], "en")
    facts = pd.DataFrame({"weekday": [0], "status": ["On track"], "spend": [123.45]})
    snapshot = facts.copy(deep=True)
    assert localize_frame(facts, "ru").iloc[0].tolist() == ["Понедельник", RU["On track"], 123.45]
    pd.testing.assert_frame_equal(localize_frame(facts, "en"), facts.assign(weekday="Monday"))
    pd.testing.assert_frame_equal(facts, snapshot)


@pytest.mark.parametrize("language", ["ru", "en"])
def test_experiment_results_table_renders_with_unique_columns(language: str, monkeypatch):
    exported = []
    def capture_csv(frame):
        exported.append(frame.copy(deep=True))
        return csv_bytes(frame)
    monkeypatch.setattr("ui_components.csv_bytes", capture_csv)
    results = APP_PATH.parent / "experiments" / "results"
    summary = pd.read_csv(results / "summary.csv")
    runs = pd.read_csv(results / "runs.csv")
    daily = pd.read_csv(results / "daily_curves.csv")
    original_summary_columns = list(summary.columns)
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    app.session_state["experiment_result"] = (summary, runs, daily, pd.DataFrame())
    app.session_state["experiment_config"] = {"objective": "conversions"}
    option = next(item for item in app.sidebar.radio[1].options
                  if item.endswith(translate("Experiments", language)))
    app.sidebar.radio[1].set_value(option).run(timeout=30)
    assert not app.exception
    rendered = app.dataframe[0].value
    assert rendered.columns.is_unique
    assert list(summary.columns) == original_summary_columns
    label_column = TABLE_RU["policy_label"] if language == "ru" else "policy_label"
    id_column = TABLE_RU["policy"] if language == "ru" else "policy"
    assert rendered.loc[rendered[id_column] == "static", label_column].iloc[0] == "Static baseline (Uniform V1)"
    assert policy_label("static", language) != "Static baseline (Uniform V1)"
    for frame in exported:
        assert frame.loc[frame["policy"] == "static", "policy_label"].iloc[0] == policy_label("static", language)
    assert set(summary["policy"]) == set(rendered[id_column])
    if language == "ru":
        assert "Код стратегии" in rendered.columns
        assert "Стратегия" in rendered.columns
        assert "Средняя разница с базовой стратегией" in rendered.columns
    else:
        assert "policy" in rendered.columns
        assert "policy_label" in rendered.columns
