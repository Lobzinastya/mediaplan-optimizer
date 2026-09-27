from __future__ import annotations

from datetime import date
from io import BytesIO, StringIO

import pandas as pd
import pytest

from mediaplan_optimizer.campaign import CampaignObservation, add_observations, observations_frame
from mediaplan_optimizer.ingestion import parse_fact_csv, validate_fact_dataframe


VALID = """day,channel,spend,impressions,clicks,conversions
1,Programmatic,50000,300000,1200,35
1,Marketplace 1,30000,80000,1700,90
"""


def test_valid_csv_normalizes_to_shared_observation_contract(product_catalog):
    result = parse_fact_csv(VALID, product_catalog, horizon_days=21)
    assert len(result.observations) == 2
    assert all(item.source == "upload" for item in result.observations)
    assert list(result.preview.columns) == [
        "day",
        "date",
        "channel",
        "spend",
        "impressions",
        "clicks",
        "conversions",
        "reach",
        "video_views",
        "source",
    ]


def test_date_only_csv_maps_calendar_days(product_catalog):
    csv = """date,channel,spend,impressions,clicks,conversions
2026-01-05,Programmatic,100,1000,10,1
2026-01-07,Programmatic,100,1000,10,1
"""
    result = parse_fact_csv(csv, product_catalog, horizon_days=21)
    assert [item.day for item in result.observations] == [1, 3]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"day": [1]}), "missing required columns"),
        (
            pd.DataFrame(
                [[1, "Unknown", 1, 10, 1, 0]],
                columns=["day", "channel", "spend", "impressions", "clicks", "conversions"],
            ),
            "unknown channels",
        ),
        (
            pd.DataFrame(
                [[1, "Programmatic", -1, 10, 1, 0]],
                columns=["day", "channel", "spend", "impressions", "clicks", "conversions"],
            ),
            "spend must be nonnegative",
        ),
        (
            pd.DataFrame(
                [[1, "Programmatic", 1, 10, 11, 0]],
                columns=["day", "channel", "spend", "impressions", "clicks", "conversions"],
            ),
            "clicks cannot exceed",
        ),
        (
            pd.DataFrame(
                [[1, "Programmatic", 1, 10, 2, 3]],
                columns=["day", "channel", "spend", "impressions", "clicks", "conversions"],
            ),
            "conversions cannot exceed",
        ),
    ],
)
def test_invalid_uploads_are_rejected(product_catalog, frame, message):
    with pytest.raises(ValueError, match=message):
        validate_fact_dataframe(frame, product_catalog, horizon_days=21)


def test_duplicate_rows_are_rejected(product_catalog):
    frame = pd.read_csv(StringIO(VALID))
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_fact_dataframe(frame, product_catalog, horizon_days=21)


def test_exported_fact_with_null_optional_values_roundtrips(product_state):
    state = add_observations(product_state, [CampaignObservation(day=1, channel="Programmatic", spend=100,
                                       impressions=1000, clicks=10, conversions=1)])
    payload = observations_frame(state).to_csv(index=False).encode("utf-8-sig")
    parsed = parse_fact_csv(payload, state.planning_catalog, horizon_days=21)
    assert parsed.observations == state.observations
    assert pd.read_csv(BytesIO(payload)).columns.tolist() == parsed.preview.columns.tolist()


def test_simulated_export_preserves_provenance(product_state):
    observation = CampaignObservation(day=1, channel="Programmatic", spend=100,
                                       impressions=1000, clicks=10, conversions=1).model_copy(update={"source": "simulator"})
    state = add_observations(product_state, [observation])
    parsed = parse_fact_csv(observations_frame(state).to_csv(index=False).encode("utf-8-sig"),
                            state.planning_catalog, horizon_days=21)
    assert parsed.observations[0].source == "simulator"


@pytest.mark.parametrize("value", ["garbage", "inf", "-1"])
def test_optional_fact_values_reject_invalid_nonempty_data(product_catalog, value):
    payload = "day,channel,spend,impressions,clicks,conversions,reach\n1,Programmatic,100,1000,10,1," + value
    with pytest.raises(ValueError):
        parse_fact_csv(payload, product_catalog, horizon_days=21)


def test_incremental_date_upload_uses_campaign_anchor(product_catalog):
    payload = "date,channel,spend,impressions,clicks,conversions\n2026-01-07,Programmatic,100,1000,10,1"
    parsed = parse_fact_csv(payload, product_catalog, horizon_days=21, campaign_start_date=date(2026, 1, 5))
    assert parsed.observations[0].day == 3
    with pytest.raises(ValueError, match="absolute campaign day"):
        parse_fact_csv(payload, product_catalog, horizon_days=21, require_day=True)


def test_conflicting_calendar_mapping_is_rejected(product_catalog):
    payload = "day,date,channel,spend,impressions,clicks,conversions\n2,2026-01-07,Programmatic,100,1000,10,1"
    with pytest.raises(ValueError, match="consistent campaign start"):
        parse_fact_csv(payload, product_catalog, horizon_days=21, campaign_start_date=date(2026, 1, 5))


def test_out_of_order_csv_is_normalized_chronologically(product_catalog):
    csv = """day,channel,spend,impressions,clicks,conversions
3,Programmatic,300,3000,30,3
1,Marketplace 1,100,1000,10,1
2,Programmatic,200,2000,20,2
"""

    result = parse_fact_csv(csv, product_catalog, horizon_days=21)

    assert [(item.day, item.channel) for item in result.observations] == [
        (1, "Marketplace 1"),
        (2, "Programmatic"),
        (3, "Programmatic"),
    ]
    assert result.preview["day"].tolist() == [1, 2, 3]
