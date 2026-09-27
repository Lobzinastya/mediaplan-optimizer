"""Bilingual planning-mode controls and dated temporal views."""

from pathlib import Path
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from ui_text import translate


APP = Path(__file__).resolve().parents[1] / "app.py"


def _plan_page(language: str):
    app = AppTest.from_file(APP).run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    option = next(x for x in app.sidebar.radio[1].options
                  if x.endswith(translate("Media Plan", language)))
    app.sidebar.radio[1].set_value(option).run(timeout=30)
    assert not app.exception
    return app


@pytest.mark.parametrize("language", ["ru", "en"])
def test_calendar_mode_start_date_and_temporal_views_render(language: str):
    app = _plan_page(language)
    mode = next(x for x in app.radio if x.label == translate("Planning mode", language))
    assert mode.value == "calendar_aware"
    date_control = next(x for x in app.date_input
                        if x.label == translate("Campaign start date", language))
    assert date_control.value.isoformat() == "2026-09-21"
    assert date_control.proto.format == ("DD.MM.YYYY" if language == "ru" else "YYYY-MM-DD")
    if language == "en":
        date_control.set_value(date(2026, 9, 24)).run(timeout=30)
    next(x for x in app.button if x.label == translate("Calculate media plan", language)).click().run(timeout=30)
    assert not app.exception
    result = app.session_state["draft_result"]
    assert result.request.planning_mode.value == "calendar_aware"
    assert result.allocations[0].date.isoformat() == (
        "2026-09-24" if language == "en" else "2026-09-21"
    )
    assert len(result.temporal_profiles) == len(result.allocations)
    assert len(app.get("plotly_chart")) >= 4
    assert any(x.label == translate("Calculate baseline", language) for x in app.button)


def test_calendar_ui_uniform_ablation_renders_without_changing_calendar_plan():
    app = _plan_page("ru")
    next(x for x in app.button if x.label == translate("Calculate media plan")).click().run(timeout=30)
    calendar_result = app.session_state["draft_result"]
    next(x for x in app.button if x.label == translate("Calculate baseline")).click().run(timeout=30)
    assert not app.exception
    compared, uniform = app.session_state["uniform_comparison"]
    assert compared is calendar_result
    assert uniform.summary.conversions == pytest.approx(3069.9809275260704)
    assert calendar_result.summary.conversions == pytest.approx(3077.0820480462485)


@pytest.mark.parametrize("language", ["ru", "en"])
def test_ui_uniform_mode_keeps_historical_canonical_result(language: str):
    app = _plan_page(language)
    mode = next(x for x in app.radio if x.label == translate("Planning mode", language))
    mode.set_value(translate("Uniform baseline", language)).run(timeout=30)
    assert not app.exception
    assert not any(x.label == translate("Campaign start date", language) for x in app.date_input)
    next(x for x in app.button if x.label == translate("Calculate media plan", language)).click().run(timeout=30)
    assert not app.exception
    result = app.session_state["draft_result"]
    assert result.request.planning_mode.value == "uniform"
    assert result.summary.conversions == pytest.approx(3069.9809275260704)
