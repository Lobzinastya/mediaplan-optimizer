"""The methodology stays file-backed, readable and independent of campaign state."""

import ast
from copy import deepcopy
from pathlib import Path
import re

import pytest
from streamlit.testing.v1 import AppTest

from mediaplan_optimizer.campaign import start_campaign
from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.schemas import OptimizationRequest
from mediaplan_optimizer.service import optimize_media_plan
from ui_methodology import DOCUMENT_PATH, load_methodology, streamlit_markdown
from ui_text import translate


ROOT = Path(__file__).resolve().parents[1]


def test_document_loads_core_and_adaptive_sections():
    assert DOCUMENT_PATH.is_file()
    title, introduction, sections = load_methodology()
    assert title and introduction
    assert len(sections) == 17
    headings = [heading for heading, _ in sections]
    for heading in ("1. Что задаёт пользователь", "5. Uniform V1",
                    "6. Type B и проверка достижимости", "7. Calendar-aware V2",
                    "11. Что меняется, когда появляется факт",
                    "12. Thompson Sampling", "13. LinUCB",
                    "14. Что показывает адаптивная часть"):
        assert heading in headings
    text = DOCUMENT_PATH.read_text(encoding="utf-8")
    assert "—" not in text and "–" not in text
    assert "(docs/MATHEMATICAL_MODEL.md)" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_loader_reads_current_file_and_preserves_tables(tmp_path):
    path = tmp_path / "model.md"
    table = "| KPI | Value |\n| --- | --- |\n| clicks | 10 |"
    path.write_text("# Title\n\nIntroduction\n\n## First\n\n" + table, encoding="utf-8")
    assert load_methodology(path) == ("Title", "Introduction", [("First", table)])
    assert streamlit_markdown(table) == table
    path.write_text("# Changed\n\nNew introduction\n\n## Second\n\nCurrent text", encoding="utf-8")
    assert load_methodology(path)[0] == "Changed"


def test_math_delimiters_change_but_equations_do_not():
    source = r"Value \(x=-1\)." + "\n" + r"\[y = \frac{x}{2} - 1\]"
    assert streamlit_markdown(source) == "Value $x=-1$.\n\n$$\ny = \\frac{x}{2} - 1\n$$\n"
    source = DOCUMENT_PATH.read_text(encoding="utf-8")
    rendered = streamlit_markdown(source)
    assert rendered.count("$$") == 2 * source.count(r"\[") == 56
    assert r"\[" not in rendered and r"\(" not in rendered


def test_ui_does_not_embed_the_document_or_write_session_state():
    _, introduction, sections = load_methodology()
    paragraphs = [introduction, *(body for _, body in sections)]
    for name in ("app.py", "ui_text.py", "ui_components.py", "ui_methodology.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        assert not any(paragraph in literal for paragraph in paragraphs for literal in literals)
    source = (ROOT / "ui_methodology.py").read_text(encoding="utf-8")
    assert "session_state" not in source


@pytest.fixture(scope="module")
def approved_plan():
    catalog = generate_catalog(seed=42)
    request = OptimizationRequest(task_type="A", horizon_days=21,
                                  objective_metric="conversions", budget=1_200_000,
                                  planning_mode="calendar_aware")
    plan = optimize_media_plan(request, catalog)
    return catalog, request, plan


@pytest.mark.parametrize("language", ["ru", "en"])
@pytest.mark.parametrize("with_campaign", [False, True])
def test_methodology_page_is_read_only_and_bilingual(language, with_campaign, approved_plan):
    catalog, request, plan = approved_plan
    app = AppTest.from_file(ROOT / "app.py").run(timeout=30)
    if language == "en":
        app.sidebar.radio[0].set_value("English").run(timeout=30)
    if with_campaign:
        app.session_state["draft_request"] = request.model_copy(deep=True)
        app.session_state["draft_result"] = plan.model_copy(deep=True)
        app.session_state["draft_catalog"] = dict(catalog)
        app.session_state["campaign"] = start_campaign("Read-only check", request, plan, catalog)
        app.session_state["demo_ready"] = True
    keys = ("draft_request", "draft_result", "draft_catalog", "campaign", "demo_ready",
            "planning_catalog", "catalog_seed", "planning_mode_ui", "planning_start_date",
            "experiment_result", "experiment_config", "inventory_reviewed",
            "monitor_reviewed_campaign_id", "adaptive_reviewed_campaign_id")
    before = deepcopy({key: app.session_state[key] for key in keys})
    label = translate("How it works", language)
    assert label in app.sidebar.radio[1].options
    app.sidebar.radio[1].set_value(label).run(timeout=30)
    assert not app.exception
    title, introduction, sections = load_methodology()
    assert app.title[0].value == title
    assert introduction in [item.value for item in app.markdown]
    expanders = {item.label: item for item in app.expander}
    for heading, body in sections:
        assert heading in expanders
        assert not expanders[heading].proto.expanded
        displayed = "\n".join(item.value for item in expanders[heading].markdown)
        assert streamlit_markdown(re.sub(r"\[([^\]]+)\]\((\.\./[^)]+)\)", r"\1", body)) == displayed
    displayed = "\n".join(item.value for item in app.markdown)
    assert "Thompson Sampling" in displayed and "LinUCB" in displayed and "Oracle" in displayed
    assert not app.get("plotly_chart")
    assert {key: app.session_state[key] for key in keys} == before
    app.run(timeout=30)
    assert not app.exception
    assert {key: app.session_state[key] for key in keys} == before
