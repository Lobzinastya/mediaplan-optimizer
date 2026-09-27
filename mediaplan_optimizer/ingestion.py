"""CSV fact validation and normalization into CampaignObservation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, timedelta
from io import BytesIO, StringIO
from typing import IO, Mapping

import numpy as np
import pandas as pd

from .campaign import CampaignObservation
from .schemas import ChannelConfig


FACT_VALUE_COLUMNS = ["spend", "impressions", "clicks", "conversions"]
OPTIONAL_VALUE_COLUMNS = ["reach", "video_views"]
FACT_TEMPLATE = """day,channel,spend,impressions,clicks,conversions
1,Programmatic,50000,300000,1200,35
1,Marketplace 1,30000,80000,1700,90
"""


@dataclass(frozen=True)
class FactValidationResult:
    preview: pd.DataFrame
    observations: tuple[CampaignObservation, ...]


def _normalize_dates(
    frame: pd.DataFrame, campaign_start_date: Date | None = None
) -> pd.DataFrame:
    result = frame.copy()
    if "date" in result.columns:
        parsed = pd.to_datetime(result["date"], errors="coerce")
        if (parsed.isna() & result["date"].notna()).any():
            raise ValueError("date contains invalid values")
        result["date"] = parsed.dt.date.where(parsed.notna(), None)
    if "day" not in result.columns:
        if "date" not in result.columns:
            raise ValueError("fact requires either a day or date column")
        if result["date"].isna().any():
            raise ValueError("date is required when day is absent")
        first = campaign_start_date or min(result["date"])
        result["day"] = result["date"].map(lambda value: (value - first).days + 1)
    return result


def validate_fact_dataframe(
    frame: pd.DataFrame,
    catalog: Mapping[str, ChannelConfig],
    *,
    horizon_days: int,
    campaign_start_date: Date | None = None,
    require_day: bool = False,
    daily_caps: Mapping[tuple[int, str], float] | None = None,
) -> FactValidationResult:
    """Validate uploaded rows without coercing or repairing invalid facts."""
    if frame.empty:
        raise ValueError("fact CSV is empty")
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    if normalized.columns.duplicated().any():
        raise ValueError("duplicate normalized column names are not allowed")
    if require_day and "day" not in normalized:
        raise ValueError("include absolute campaign day for subsequent uploads without a calendar anchor")
    required = {"channel", *FACT_VALUE_COLUMNS}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    normalized = _normalize_dates(normalized, campaign_start_date)

    numeric_columns = ["day", *FACT_VALUE_COLUMNS]
    numeric_columns.extend(
        column for column in OPTIONAL_VALUE_COLUMNS if column in normalized.columns
    )
    for column in numeric_columns:
        converted = pd.to_numeric(normalized[column], errors="coerce")
        optional = column in OPTIONAL_VALUE_COLUMNS
        invalid_missing = converted.isna() & (normalized[column].notna() if optional else True)
        if invalid_missing.any() or not np.isfinite(converted.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite numeric values")
        normalized[column] = converted

    if (normalized["day"] % 1 != 0).any():
        raise ValueError("day must contain integers")
    normalized["day"] = normalized["day"].astype(int)
    if (normalized["day"] <= 0).any() or (normalized["day"] > horizon_days).any():
        raise ValueError(f"day must be between 1 and {horizon_days}")
    if "date" not in normalized and campaign_start_date is not None:
        normalized["date"] = normalized["day"].map(
            lambda day: campaign_start_date + timedelta(days=int(day) - 1)
        )
    if "date" in normalized:
        anchors = {row.date - timedelta(days=int(row.day) - 1)
                   for row in normalized.itertuples() if pd.notna(row.date)}
        if len(anchors) > 1 or (campaign_start_date and anchors and anchors != {campaign_start_date}):
            raise ValueError("date and day must use one consistent campaign start date")
    normalized["channel"] = normalized["channel"].astype(str).str.strip()
    unknown = sorted(set(normalized["channel"]) - set(catalog))
    if unknown:
        raise ValueError(f"unknown channels: {', '.join(unknown)}")
    if normalized.duplicated(subset=["day", "channel"]).any():
        raise ValueError("duplicate day/channel rows are not allowed")
    for column in numeric_columns[1:]:
        if (normalized[column] < 0).any():
            raise ValueError(f"{column} must be nonnegative")
    if (normalized["clicks"] > normalized["impressions"]).any():
        raise ValueError("clicks cannot exceed impressions")
    if (normalized["conversions"] > normalized["clicks"]).any():
        raise ValueError("conversions cannot exceed clicks")
    for column in OPTIONAL_VALUE_COLUMNS:
        if column in normalized.columns and (
            normalized[column] > normalized["impressions"]
        ).any():
            raise ValueError(f"{column} cannot exceed impressions")
    for row in normalized.itertuples(index=False):
        cap = (daily_caps.get((row.day, row.channel), catalog[row.channel].max_daily_spend)
               if daily_caps is not None else catalog[row.channel].max_daily_spend)
        if row.spend > cap + 1e-7:
            raise ValueError(f"daily spend cap exceeded for {row.channel} on day {row.day}")

    observations = tuple(
        CampaignObservation(
            day=int(row["day"]),
            date=None if pd.isna(row.get("date")) else row["date"],
            channel=str(row["channel"]),
            spend=float(row["spend"]),
            impressions=float(row["impressions"]),
            clicks=float(row["clicks"]),
            conversions=float(row["conversions"]),
            reach=(None if pd.isna(row.get("reach")) else float(row["reach"])),
            video_views=(
                None if pd.isna(row.get("video_views")) else float(row["video_views"])
            ),
            source=row.get("source", "upload"),
        )
        for row in normalized.to_dict(orient="records")
    )
    ordered = tuple(sorted(observations, key=lambda item: (item.day, item.channel)))
    preview = pd.DataFrame([item.model_dump(mode="json") for item in ordered])
    return FactValidationResult(preview=preview, observations=ordered)


def parse_fact_csv(
    source: str | bytes | IO[str] | IO[bytes],
    catalog: Mapping[str, ChannelConfig],
    *,
    horizon_days: int,
    campaign_start_date: Date | None = None,
    require_day: bool = False,
    daily_caps: Mapping[tuple[int, str], float] | None = None,
) -> FactValidationResult:
    if isinstance(source, bytes):
        readable: str | IO[str] | IO[bytes] = BytesIO(source)
    elif isinstance(source, str) and "\n" in source:
        readable = StringIO(source)
    else:
        readable = source
    try:
        frame = pd.read_csv(readable)
    except Exception as exc:
        raise ValueError(f"unable to parse CSV: {exc}") from exc
    return validate_fact_dataframe(
        frame, catalog, horizon_days=horizon_days,
        campaign_start_date=campaign_start_date, require_day=require_day,
        daily_caps=daily_caps,
    )
