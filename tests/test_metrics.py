import pytest

from mediaplan_optimizer.curves import ChannelResponse
from mediaplan_optimizer.metrics import (
    aggregate_channel_metrics,
    aggregate_plan_summary,
    aggregate_weekly_metrics,
    build_daily_allocations,
)
from mediaplan_optimizer.schemas import ChannelConfig, DailyAllocation, ObjectiveMetric


def allocation(
    *,
    channel: str,
    day: int,
    spend: float,
    impressions: float,
    clicks: float,
    conversions: float,
    video_views: float | None = None,
) -> DailyAllocation:
    return DailyAllocation(
        day=day,
        channel=channel,
        spend=spend,
        impressions=impressions,
        reach=impressions / 2,
        clicks=clicks,
        conversions=conversions,
        video_views=video_views,
        cumulative_target_kpi=0,
    )


def test_plan_rates_and_costs_are_ratios_of_totals() -> None:
    records = [
        allocation(
            channel="large",
            day=1,
            spend=100,
            impressions=1000,
            clicks=100,
            conversions=10,
            video_views=500,
        ),
        allocation(
            channel="small",
            day=1,
            spend=50,
            impressions=100,
            clicks=50,
            conversions=20,
        ),
    ]

    summary = aggregate_plan_summary(records, ObjectiveMetric.CONVERSIONS)

    assert summary.ctr == pytest.approx(150 / 1100)
    assert summary.cr == pytest.approx(30 / 150)
    assert summary.vtr == pytest.approx(500 / 1000)
    assert summary.cpm == pytest.approx(150 / 1100 * 1000)
    assert summary.cpc == pytest.approx(150 / 150)
    assert summary.cpa == pytest.approx(150 / 30)
    assert summary.achieved_target == 30


def test_channel_metrics_are_weighted_and_preserve_video_applicability() -> None:
    records = [
        allocation(
            channel="video",
            day=1,
            spend=10,
            impressions=100,
            clicks=10,
            conversions=1,
            video_views=40,
        ),
        allocation(
            channel="video",
            day=2,
            spend=30,
            impressions=300,
            clicks=15,
            conversions=3,
            video_views=180,
        ),
        allocation(
            channel="sms", day=1, spend=5, impressions=20, clicks=0, conversions=0
        ),
    ]

    video, sms = aggregate_channel_metrics(records)
    assert video.ctr == pytest.approx(25 / 400)
    assert video.cr == pytest.approx(4 / 25)
    assert video.vtr == pytest.approx(220 / 400)
    assert video.cpm == pytest.approx(100)
    assert video.cpc == pytest.approx(1.6)
    assert video.cpa == pytest.approx(10)
    assert sms.video_views is None
    assert sms.vtr is None


def test_zero_denominators_have_zero_rates_and_none_costs() -> None:
    records = [
        allocation(
            channel="empty",
            day=1,
            spend=0,
            impressions=0,
            clicks=0,
            conversions=0,
            video_views=0,
        )
    ]

    metric = aggregate_channel_metrics(records)[0]
    summary = aggregate_plan_summary(records, "clicks", available_budget=10)

    assert (metric.ctr, metric.cr, metric.vtr) == (0.0, 0.0, 0.0)
    assert (metric.cpm, metric.cpc, metric.cpa) == (None, None, None)
    assert (summary.ctr, summary.cr, summary.vtr) == (0.0, 0.0, 0.0)
    assert summary.unspent_budget == 10


def test_daily_reconstruction_is_even_and_cumulative_is_global(monkeypatch) -> None:
    def fake_response(spend: float, channel: ChannelConfig) -> ChannelResponse:
        multiplier = 1 if channel.name == "a" else 10
        return ChannelResponse(
            impressions=spend * 100,
            reach=spend * 50,
            clicks=spend * multiplier,
            conversions=spend * multiplier / 2,
            video_views=None,
        )

    monkeypatch.setattr("mediaplan_optimizer.metrics.channel_response", fake_response)
    base = dict(
        channel_type="media",
        daily_capacity=1000,
        daily_reach_capacity=500,
        response_scale_rub=100,
        max_daily_spend=100,
        average_frequency=2,
        base_ctr=0.1,
        base_cr=0.1,
    )
    catalog = {
        "a": ChannelConfig(name="a", **base),
        "b": ChannelConfig(name="b", **base),
    }

    records = build_daily_allocations(
        {"a": 30, "b": 60}, 3, catalog, ObjectiveMetric.CLICKS
    )

    assert [record.day for record in records] == [1, 1, 2, 2, 3, 3]
    assert [record.spend for record in records] == [10, 20, 10, 20, 10, 20]
    assert [record.cumulative_target_kpi for record in records] == [
        210,
        210,
        420,
        420,
        630,
        630,
    ]


def test_reconstruction_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        build_daily_allocations({}, 0, {}, "clicks")
    with pytest.raises(KeyError, match="missing"):
        build_daily_allocations({"missing": 1}, 1, {}, "clicks")


def test_negligible_floating_point_budget_residual_is_reported_as_zero() -> None:
    records = [
        allocation(
            channel="channel",
            day=1,
            spend=0.1,
            impressions=1,
            clicks=0.1,
            conversions=0.01,
        )
    ]
    summary = aggregate_plan_summary(
        records, "clicks", available_budget=0.1000000000001
    )
    assert summary.unspent_budget == 0.0


def test_weekly_aggregation_keeps_partial_week_and_cumulative_kpi() -> None:
    records = [
        DailyAllocation(
            day=day,
            channel="channel",
            spend=10,
            impressions=100,
            reach=40,
            clicks=5,
            conversions=1,
            cumulative_target_kpi=day * 5,
        )
        for day in range(1, 10)
    ]

    weeks = aggregate_weekly_metrics(records)

    assert [(week["start_day"], week["end_day"]) for week in weeks] == [
        (1, 7),
        (8, 9),
    ]
    assert [week["spend"] for week in weeks] == [70, 20]
    assert [week["non_deduplicated_reach"] for week in weeks] == [280, 80]
    assert [week["cumulative_target_kpi"] for week in weeks] == [35, 45]
