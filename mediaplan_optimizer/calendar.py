"""Deterministic synthetic weekday profiles, separate from benchmark catalog data."""

from __future__ import annotations

import hashlib
from datetime import date as Date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from .schemas import ChannelConfig, TemporalProfile


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "calendar_profiles.yaml"
FACTOR_NAMES = ("supply", "cpm", "ctr", "cr", "max_spend")


@lru_cache(maxsize=1)
def load_calendar_config() -> dict:
    with DEFAULT_PROFILE_PATH.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data.get("provenance") != "pure_synthetic_temporal_scenario":
        raise ValueError("calendar config must identify synthetic provenance")
    for category, factors in data["profiles"].items():
        for name in FACTOR_NAMES:
            values = factors[name]
            if len(values) != 7 or any(not 0.8 <= float(x) <= 1.2 for x in values):
                raise ValueError(f"invalid {category}/{name} weekday factors")
    return data


def _category(channel_name: str) -> str:
    if channel_name.startswith("Social"):
        return "social"
    if channel_name.startswith("Marketplace"):
        return "marketplace"
    if channel_name == "SMS":
        return "sms"
    return "programmatic"


def _weekly_factors(seed: int, channel: str, name: str, week_index: int) -> np.ndarray:
    base = np.asarray(load_calendar_config()["profiles"][_category(channel)][name], dtype=float)
    jitter = np.empty(7, dtype=float)
    for weekday in range(7):
        token = f"{seed}:{channel}:{name}:{week_index}:{weekday}".encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big")
        jitter[weekday] = (value / (2**64 - 1) - 0.5) * 0.008
    factors = base * (1.0 + jitter)
    factors /= factors.mean()
    if np.any((factors < 0.8) | (factors > 1.2)):
        raise ValueError("normalized temporal factors exceed supported bounds")
    return factors


def generate_day_profiles(
    catalog: Mapping[str, ChannelConfig],
    start_date: Date,
    horizon_days: int,
    seed: int = 42,
    *,
    day_offset: int = 0,
) -> tuple[TemporalProfile, ...]:
    """Generate reproducible absolute campaign days; no implicit wall-clock date."""
    if horizon_days <= 0 or day_offset < 0 or seed < 0:
        raise ValueError("horizon, offset, and seed must be valid")
    weekly: dict[tuple[str, str, int], np.ndarray] = {}
    result: list[TemporalProfile] = []
    for day in range(day_offset + 1, day_offset + horizon_days + 1):
        current_date = start_date + timedelta(days=day - 1)
        weekday = current_date.weekday()
        week_index = (day - 1) // 7
        for channel in catalog:
            factors = {}
            for name in FACTOR_NAMES:
                key = (channel, name, week_index)
                if key not in weekly:
                    weekly[key] = _weekly_factors(seed, channel, name, week_index)
                factors[name] = weekly[key]
            result.append(
                TemporalProfile(
                    channel=channel, day=day, date=current_date, weekday=weekday,
                    supply_factor=float(factors["supply"][weekday]),
                    cpm_factor=float(factors["cpm"][weekday]),
                    ctr_factor=float(factors["ctr"][weekday]),
                    cr_factor=float(factors["cr"][weekday]),
                    max_spend_factor=float(factors["max_spend"][weekday]),
                )
            )
    return tuple(result)


def effective_channel(base: ChannelConfig, profile: TemporalProfile) -> ChannelConfig:
    """Derive one day-specific channel without mutating the approved base catalog."""
    supply = profile.supply_factor
    cpm = profile.cpm_factor
    return base.model_copy(update={
        "daily_capacity": base.daily_capacity * supply,
        "daily_reach_capacity": base.daily_reach_capacity * supply,
        "base_cpm_rub": None if base.base_cpm_rub is None else base.base_cpm_rub * cpm,
        "response_scale_rub": base.response_scale_rub * supply * cpm,
        "base_ctr": min(1.0, base.base_ctr * profile.ctr_factor),
        "base_cr": min(1.0, base.base_cr * profile.cr_factor),
        "max_daily_spend": base.max_daily_spend * profile.max_spend_factor,
    })
