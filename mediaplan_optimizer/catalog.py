"""Deterministic synthetic channel catalog generation and persistence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .schemas import ChannelConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSUMPTIONS_PATH = PROJECT_ROOT / "config" / "benchmark_sources.yaml"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "config" / "channels.yaml"

REQUIRED_CHANNELS = (
    "Social Network 1",
    "Social Network 2",
    "Social Network 3",
    "Programmatic",
    "Marketplace 1",
    "Marketplace 2",
    "Marketplace 3",
    "SMS",
)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _check_inventory(names: set[str]) -> None:
    required = set(REQUIRED_CHANNELS)
    if names != required:
        missing = sorted(required - names)
        extra = sorted(names - required)
        raise ValueError(f"Catalog inventory mismatch; missing={missing}, extra={extra}")


def _unit_sample(seed: int, channel: str, field: str) -> float:
    """Return a stable pseudo-random value independent of Python RNG versions."""

    payload = f"{seed}:{channel}:{field}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / ((1 << 64) - 1)


def _sample_range(value: Any, seed: int, channel: str, field: str) -> float:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{channel}.{field} must be a two-item [minimum, maximum] range")
    low, high = (float(item) for item in value)
    if not 0 <= low <= high:
        raise ValueError(f"Invalid range for {channel}.{field}: {value}")
    return low + (high - low) * _unit_sample(seed, channel, field)


def generate_catalog(
    seed: int = 42,
    assumptions_path: str | Path | None = None,
) -> dict[str, ChannelConfig]:
    """Generate the eight-channel synthetic catalog from documented ranges."""

    path = Path(assumptions_path) if assumptions_path is not None else DEFAULT_ASSUMPTIONS_PATH
    document = _read_yaml(path)
    profiles = document.get("channels")
    if not isinstance(profiles, Mapping):
        raise ValueError("Assumptions YAML must contain a 'channels' mapping")
    _check_inventory(set(profiles))

    catalog: dict[str, ChannelConfig] = {}
    for name in REQUIRED_CHANNELS:
        profile = profiles[name]
        if not isinstance(profile, Mapping):
            raise ValueError(f"Assumptions for {name} must be a mapping")

        capacity = round(_sample_range(profile["daily_capacity"], seed, name, "daily_capacity"))
        reach_ratio = _sample_range(
            profile["daily_reach_ratio"], seed, name, "daily_reach_ratio"
        )
        vtr_range = profile.get("base_vtr")
        vtr = (
            None
            if vtr_range is None
            else round(_sample_range(vtr_range, seed, name, "base_vtr"), 6)
        )
        base_cpm = round(
            _sample_range(profile["base_cpm_rub"], seed, name, "base_cpm_rub"), 4
        )
        response_scale = round(capacity * base_cpm / 1000.0)
        catalog[name] = ChannelConfig(
            name=name,
            channel_type=str(profile.get("channel_type", "media")),
            daily_capacity=float(capacity),
            daily_reach_capacity=float(round(capacity * reach_ratio)),
            base_cpm_rub=base_cpm,
            response_scale_rub=float(response_scale),
            max_daily_spend=float(
                round(_sample_range(profile["max_daily_spend"], seed, name, "max_daily_spend"))
            ),
            average_frequency=round(
                _sample_range(profile["average_frequency"], seed, name, "average_frequency"), 4
            ),
            base_ctr=round(_sample_range(profile["base_ctr"], seed, name, "base_ctr"), 6),
            base_cr=round(_sample_range(profile["base_cr"], seed, name, "base_cr"), 6),
            base_vtr=vtr,
            ctr_saturation_decay=round(
                _sample_range(
                    profile["ctr_saturation_decay"],
                    seed,
                    name,
                    "ctr_saturation_decay",
                ),
                6,
            ),
            cr_saturation_decay=round(
                _sample_range(
                    profile["cr_saturation_decay"],
                    seed,
                    name,
                    "cr_saturation_decay",
                ),
                6,
            ),
        )
    return catalog


def load_catalog(path: str | Path | None = None) -> dict[str, ChannelConfig]:
    """Load and validate a saved catalog (the bundled seed-42 catalog by default)."""

    catalog_path = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    document = _read_yaml(catalog_path)
    records: Any = document.get("channels", document)
    if isinstance(records, list):
        records = {record["name"]: record for record in records}
    if not isinstance(records, Mapping):
        raise ValueError("Catalog YAML must contain a channel mapping or list")
    _check_inventory(set(records))

    result: dict[str, ChannelConfig] = {}
    for name in REQUIRED_CHANNELS:
        values = records[name]
        if not isinstance(values, Mapping):
            raise ValueError(f"Catalog entry for {name} must be a mapping")
        payload = dict(values)
        payload.setdefault("name", name)
        if payload["name"] != name:
            raise ValueError(f"Catalog key and embedded name disagree for {name}")
        result[name] = ChannelConfig.model_validate(payload)
    return result


def save_catalog(
    catalog: Mapping[str, ChannelConfig],
    path: str | Path | None = None,
    seed: int = 42,
) -> Path:
    """Save a catalog in stable inventory order and return its destination."""

    _check_inventory(set(catalog))
    destination = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "provenance": "benchmark_derived_synthetic_assumptions",
            "seed": seed,
            "notice": (
                "Illustrative values generated from broad public benchmark-derived "
                "ranges; not forecasts or vendor commitments."
            ),
        },
        "channels": {
            name: catalog[name].model_dump(mode="json") for name in REQUIRED_CHANNELS
        },
    }
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    destination.write_text(
        "# GENERATED FILE: deterministic benchmark-derived synthetic catalog (seed "
        f"{seed}).\n"
        "# Values use broad public benchmark context; they are not forecasts.\n"
        + rendered,
        encoding="utf-8",
        newline="\n",
    )
    return destination
