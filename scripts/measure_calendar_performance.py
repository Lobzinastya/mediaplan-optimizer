"""Single-run V1/V2 planning timings; no artifacts are rewritten."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from time import perf_counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mediaplan_optimizer.calendar import generate_day_profiles
from mediaplan_optimizer.calendar_optimizer import optimize_type_a_calendar, optimize_type_b_calendar
from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.optimizer import optimize_type_a, optimize_type_b
from mediaplan_optimizer.schemas import ObjectiveMetric


def timed(function):
    started = perf_counter()
    outcome = function()
    return round(perf_counter() - started, 3), outcome


def main() -> None:
    catalog = generate_catalog(42)
    start = date(2026, 9, 21)
    profiles_21 = generate_day_profiles(catalog, start, 21, 42)
    profiles_14 = generate_day_profiles(catalog, start, 14, 42)
    profiles_365 = generate_day_profiles(catalog, start, 365, 42)
    operations = {
        "uniform_type_a_21": lambda: optimize_type_a(catalog, 21, ObjectiveMetric.CONVERSIONS, 1_200_000),
        "calendar_type_a_21": lambda: optimize_type_a_calendar(catalog, profiles_21, ObjectiveMetric.CONVERSIONS, 1_200_000),
        "calendar_type_a_365": lambda: optimize_type_a_calendar(catalog, profiles_365, ObjectiveMetric.CONVERSIONS, 1_200_000),
        "uniform_type_b_14": lambda: optimize_type_b(catalog, 14, ObjectiveMetric.CLICKS, 10_000),
        "calendar_type_b_14": lambda: optimize_type_b_calendar(catalog, profiles_14, ObjectiveMetric.CLICKS, 10_000),
        "calendar_type_b_21": lambda: optimize_type_b_calendar(catalog, profiles_21, ObjectiveMetric.CLICKS, 10_000),
    }
    output = {}
    for name, function in operations.items():
        seconds, result = timed(function)
        output[name] = {"seconds": seconds, "achieved_kpi": result.achieved_value}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
