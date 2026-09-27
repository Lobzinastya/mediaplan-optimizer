from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest
from mediaplan_optimizer.catalog import generate_catalog
from ui_text import translate


def navigate(app: AppTest, page: str) -> AppTest:
    navigation = app.sidebar.radio[1]
    option = next(option for option in navigation.options if option.endswith(translate(page)))
    return navigation.set_value(option).run(timeout=30)


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == translate(label))


def test_all_pages_render_without_running_experiments():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=30)
    assert not app.exception
    assert app.session_state["language"] == "ru"
    pages = (
        "Overview", "Inventory / Generator", "Media Plan", "Fact Ingestion",
        "Campaign Monitor", "Adaptive Control", "Experiments", "How it works",
    )
    assert [option.lstrip("✓●○ ") for option in app.sidebar.radio[1].options] == [translate(page) for page in pages]
    for page in pages[1:]:
        app = navigate(app, page)
        assert not app.exception, f"{page} raised: {list(app.exception)}"


def test_canonical_ui_flow_plan_start_simulate_monitor_and_replan():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=30)

    app = navigate(app, "Media Plan")
    mode = next(item for item in app.radio if item.label == translate("Planning mode"))
    mode.set_value(translate("Uniform baseline")).run(timeout=30)
    button(app, "Calculate media plan").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["draft_result"].status.value == "optimal"
    assert abs(app.session_state["draft_result"].summary.conversions - 3069.9809275260704) < 1e-6

    button(app, "Start campaign workflow").click().run(
        timeout=30
    )
    assert not app.exception
    assert app.session_state["campaign"].current_day == 0
    assert app.session_state["campaign"].latest_plan.version_id == "v1"
    deviation = next(x for x in app.slider if x.label == translate("Hidden parameter deviation"))
    assert deviation.value == 0.30

    button(app, "Next day").click().run(
        timeout=30
    )
    assert not app.exception
    assert app.session_state["campaign"].current_day == 1
    assert app.session_state["simulation_summary"]["rows_added"] > 0
    assert app.session_state["simulator_config"]["parameter_deviation"] == 0.30

    app = navigate(app, "Campaign Monitor")
    assert not app.exception

    app = navigate(app, "Adaptive Control")
    policies = next(x for x in app.selectbox if x.label == translate("Adaptive policy for remaining campaign"))
    assert len(policies.options) == 4
    assert "Thompson Sampling" in policies.options and "LinUCB" in policies.options
    button(app, "Recalculate remaining plan").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["campaign"].latest_plan.version_id == "v2"
    assert app.session_state["replan_summary"]["before_version"] == "v1"
    assert app.session_state["replan_summary"]["after_version"] == "v2"


def test_task_switch_renders_target_before_submit_and_keeps_campaign_catalog():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=30)
    app = navigate(app, "Media Plan")
    task = next(item for item in app.radio if item.label == translate("Task"))
    task.set_value(translate("B — Minimum budget for target")).run(timeout=30)
    assert any(item.label == translate("Target value") for item in app.number_input)
    assert not any(item.label == translate("Budget, RUB") for item in app.number_input)
    next(item for item in app.radio if item.label == translate("Task")).set_value(translate("A — Maximize KPI")).run(timeout=30)
    button(app, "Calculate media plan").click().run(timeout=30)
    snapshot = app.session_state["draft_catalog"]
    app.session_state["planning_catalog"] = generate_catalog(seed=43)
    app.run(timeout=30)
    button(app, "Start campaign workflow").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["campaign"].planning_catalog == snapshot
