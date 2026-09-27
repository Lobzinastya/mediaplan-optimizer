"""Read-only canonical/report/export verification using production code.

Does not regenerate checked-in experiment tables or rewrite report snapshots.
"""
from __future__ import annotations

import json
import hashlib
from datetime import date
import sys
from io import BytesIO
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory

import nbformat
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mediaplan_optimizer.campaign import add_observations, observations_frame, start_campaign
from mediaplan_optimizer.catalog import generate_catalog, load_catalog
from mediaplan_optimizer.control import recalculate_remaining_plan
from mediaplan_optimizer.ingestion import parse_fact_csv
from mediaplan_optimizer.monitoring import build_monitoring
from mediaplan_optimizer.schemas import OptimizationRequest
from mediaplan_optimizer.service import optimize_media_plan
from scripts.generate_report import MARKDOWN_PATH, HTML_PATH, generate_report, render_html
from ui_components import csv_bytes


def artifact_hashes() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (REPO_ROOT / "experiments" / "results").iterdir() if path.is_file()}


def canonical_data() -> dict:
    """Recheck planner outputs; read research evidence without rerunning it."""
    catalog = load_catalog()
    requests = {
        "type_a": dict(task_type="A", horizon_days=21, objective_metric="conversions", budget=1_200_000),
        "type_b": dict(task_type="B", horizon_days=14, objective_metric="clicks", target_value=10_000),
        "calendar_type_a": dict(task_type="A", horizon_days=21, objective_metric="conversions",
                                budget=1_200_000, planning_mode="calendar_aware",
                                start_date=date(2026, 9, 21), calendar_profile_seed=42),
        "infeasible": dict(task_type="B", horizon_days=1, objective_metric="conversions", target_value=1e12),
    }
    data = {name: optimize_media_plan(OptimizationRequest(**values), catalog)
            for name, values in requests.items()}
    data["catalog"] = catalog
    for key, filename in (("adaptive_summary", "summary.csv"), ("adaptive_runs", "runs.csv"),
                          ("adaptive_daily", "daily_curves.csv"), ("adaptive_channels", "channel_totals.csv")):
        data[key] = pd.read_csv(REPO_ROOT / "experiments" / "results" / filename)
    return data


def verify_document_values(markdown: str, data: dict) -> None:
    """Check the actual scenario/table rows, not just numbers anywhere in the text."""
    def row(label: str) -> str:
        matches = [line for line in markdown.splitlines() if line.startswith(f"| {label} |")]
        assert len(matches) == 1, f"Expected one report row for {label}"
        return matches[0]

    def number(value: float) -> str:
        return f"{value:,.2f}".replace(",", " ").replace(".", ",")

    for label, key in (("Uniform A", "type_a"), ("Calendar A", "calendar_type_a")):
        summary = data[key].summary
        assert number(summary.conversions) in row(label)
        assert number(summary.cpa) in row(label)
    assert number(data["type_b"].summary.spend) in row("Uniform B")
    feasibility = data["infeasible"].feasibility
    assert number(feasibility.maximum_achievable) in row("Недостижимый B")
    assert feasibility.reason.value in row("Недостижимый B")
    names = {"static": "Static", "periodic_reoptimization": "Periodic",
             "thompson_sampling": "Thompson Sampling", "linucb": "LinUCB", "oracle": "Oracle"}
    for item in data["adaptive_summary"].itertuples(index=False):
        cells = [cell.strip() for cell in row(names[item.policy]).strip("|").split("|")]
        assert cells[1:] == [number(item.mean_kpi), number(item.median_kpi)]


def main() -> None:
    started = perf_counter()
    original_hashes = artifact_hashes()
    data = canonical_data()
    catalog = data["catalog"]
    assert catalog == generate_catalog(seed=42)
    plan = data["type_a"]
    assert np.isclose(plan.summary.conversions, 3069.9809275260704, rtol=0, atol=1e-7)
    assert np.isclose(data["type_b"].summary.spend, 104138.52082512155, rtol=0, atol=0.01)
    assert data["infeasible"].feasibility.reason.value == "MARKET_CAPACITY"
    assert np.isclose(data["calendar_type_a"].summary.conversions, 3077.0820480462485, rtol=0, atol=1e-7)
    assert np.isclose(data["infeasible"].feasibility.maximum_achievable, 3952.4774377056415, rtol=0, atol=1e-7)
    assert data["type_b"].summary.achieved_target + 0.001 >= 10_000

    state = start_campaign("Release artifact check", plan.request, plan, catalog)
    uploaded = parse_fact_csv(str(REPO_ROOT / "examples" / "fact_first_3_days.csv"), catalog, horizon_days=21)
    observed = add_observations(state, uploaded.observations)
    revised = recalculate_remaining_plan(observed)
    assert revised.observations == observed.observations
    monitor = build_monitoring(revised)
    tables = {
        "daily_plan": pd.DataFrame([row.model_dump() for row in plan.allocations]),
        "channel_summary": pd.DataFrame([row.model_dump() for row in plan.channel_metrics]),
        "campaign_fact": observations_frame(revised),
        "monitor_daily": monitor.daily,
        "monitor_channels": monitor.channels,
        "experiment_summary": data["adaptive_summary"],
        "experiment_runs": data["adaptive_runs"],
    }
    sizes = {}
    for name, frame in tables.items():
        payload = csv_bytes(frame)
        restored = pd.read_csv(BytesIO(payload), encoding="utf-8-sig")
        assert payload.startswith(b"\xef\xbb\xbf") and not restored.empty
        assert restored.columns.tolist() == frame.columns.tolist()
        assert len(restored) == len(frame)
        assert not np.isinf(restored.select_dtypes(include="number").to_numpy()).any()
        sizes[name] = len(payload)
    roundtrip = parse_fact_csv(csv_bytes(tables["campaign_fact"]), catalog, horizon_days=21)
    assert roundtrip.observations == revised.observations

    artifact_dir = REPO_ROOT / "experiments" / "results"
    configuration = json.loads((artifact_dir / "configuration.json").read_text(encoding="utf-8"))
    assert configuration["planning_catalog"] == {name: c.model_dump(mode="json") for name, c in catalog.items()}
    assert json.loads(json.dumps(configuration)) == configuration
    runs = data["adaptive_runs"]
    assert set(configuration["seeds"]) == set(runs.seed.unique()) == set(range(20))
    assert not runs.duplicated(["policy", "seed"]).any()
    summary = data["adaptive_summary"].set_index("policy")
    assert set(summary.index) == set(runs.policy.unique())
    for policy, group in runs.groupby("policy"):
        assert set(group.seed) == set(configuration["seeds"])
        assert summary.loc[policy, "runs"] == len(group)
        for field, expected in (("mean_kpi", group.total_kpi.mean()),
                                ("median_kpi", group.total_kpi.median()),
                                ("std_kpi", group.total_kpi.std()),
                                ("mean_regret_to_oracle", group.regret_to_oracle.mean())):
            assert np.isclose(summary.loc[policy, field], expected, rtol=1e-10, atol=1e-8)
    for key in ("adaptive_summary", "adaptive_runs", "adaptive_daily", "adaptive_channels"):
        assert np.isfinite(data[key].select_dtypes(include="number").to_numpy()).all()
    assert (runs.total_spend <= 1_200_000 + 1e-6).all()

    assert MARKDOWN_PATH.is_file() and HTML_PATH.is_file()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    html = HTML_PATH.read_text(encoding="utf-8")
    verify_document_values(markdown, data)
    assert html == render_html(markdown), "HTML is stale: run scripts/generate_report.py"
    with TemporaryDirectory(prefix="mediaplan-report-") as temporary:
        generated = Path(temporary) / "report.html"
        generate_report(generated)
        assert generated.read_text(encoding="utf-8") == html
    assert artifact_hashes() == original_hashes, "Report verification/generation changed research artifacts"
    assert "<!doctype html>" in html.lower()
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "demo.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert all(output.output_type != "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))
    print(json.dumps({
        "status": "passed", "export_bytes": sizes,
        "report_characters": {"markdown": len(markdown), "html": len(html)},
        "adaptive_means": dict(zip(data["adaptive_summary"].policy, data["adaptive_summary"].mean_kpi)),
        "saved_research_summary_match": True, "catalog_snapshot_match": True,
        "html_matches_methodology": True, "experiment_artifacts_unchanged": True,
        "canonical_type_a": plan.summary.conversions,
        "canonical_type_b_spend": data["type_b"].summary.spend,
        "elapsed_seconds": perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
