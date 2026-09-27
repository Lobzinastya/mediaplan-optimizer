"""Seeded campaign environment with hidden truth and noisy feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .curves import buyout_scale_rub
from .schemas import ChannelConfig


@dataclass(frozen=True)
class HiddenChannelParameters:
    ctr_multiplier: float
    cr_multiplier: float
    scale_multiplier: float
    weekend_lift: float
    fatigue_strength: float


@dataclass(frozen=True)
class ChunkObservation:
    day: int
    channel: str
    spend: float
    impressions: int
    clicks: int
    conversions: int


@dataclass(frozen=True)
class EnvironmentDiagnostics:
    expected_impressions: float
    effective_ctr: float
    effective_cr: float


def generate_hidden_parameters(
    catalog: Mapping[str, ChannelConfig],
    seed: int,
    parameter_deviation: float = 0.20,
    context_strength: float = 1.0,
) -> dict[str, HiddenChannelParameters]:
    """Create moderate, reproducible deviations from planner beliefs."""
    if not 0 <= parameter_deviation <= 0.30:
        raise ValueError("parameter_deviation must be between 0 and 0.30")
    if not 0 <= context_strength <= 2.0:
        raise ValueError("context_strength must be between 0 and 2")
    rng = np.random.default_rng(seed)
    return {
        name: HiddenChannelParameters(
            ctr_multiplier=float(
                rng.uniform(1.0 - parameter_deviation, 1.0 + parameter_deviation)
            ),
            cr_multiplier=float(
                rng.uniform(1.0 - parameter_deviation, 1.0 + parameter_deviation)
            ),
            scale_multiplier=float(
                rng.uniform(1.0 - parameter_deviation, 1.0 + parameter_deviation)
            ),
            weekend_lift=float(rng.uniform(-0.15, 0.15) * context_strength),
            fatigue_strength=float(rng.uniform(0.05, 0.20) * context_strength),
        )
        for name in catalog
    }


class CampaignSimulator:
    """Hidden environment. Deployable policies receive only observations."""

    def __init__(
        self,
        catalog: Mapping[str, ChannelConfig],
        horizon_days: int,
        seed: int,
        parameter_deviation: float = 0.20,
        context_strength: float = 1.0,
    ) -> None:
        self.catalog = dict(catalog)
        self.horizon_days = horizon_days
        self.seed = seed
        self.parameter_deviation = parameter_deviation
        self.context_strength = context_strength
        self._truth = generate_hidden_parameters(
            catalog, seed, parameter_deviation, context_strength
        )
        self._rng = np.random.default_rng(seed + 10_000_019)
        self._observed_daily_impressions: dict[tuple[int, str], int] = {}

    def truth_snapshot(self) -> dict[str, HiddenChannelParameters]:
        """Evaluation-only copy used by tests and the explicitly labeled oracle."""
        return dict(self._truth)

    def oracle_catalog(self) -> dict[str, ChannelConfig]:
        """Evaluation-only catalog containing hidden base parameters."""
        result: dict[str, ChannelConfig] = {}
        for name, channel in self.catalog.items():
            truth = self._truth[name]
            true_scale = buyout_scale_rub(channel) * truth.scale_multiplier
            true_base_cpm = (
                None
                if channel.base_cpm_rub is None
                else channel.base_cpm_rub * truth.scale_multiplier
            )
            result[name] = channel.model_copy(
                update={
                    "base_ctr": min(1.0, channel.base_ctr * truth.ctr_multiplier),
                    "base_cr": min(1.0, channel.base_cr * truth.cr_multiplier),
                    "base_cpm_rub": true_base_cpm,
                    "response_scale_rub": true_scale,
                }
            )
        return result

    def diagnostics(
        self,
        channel_name: str,
        day: int,
        spend: float,
        daily_spend_before: float,
        cumulative_spend_before: float,
    ) -> EnvironmentDiagnostics:
        channel = self.catalog[channel_name]
        truth = self._truth[channel_name]
        true_scale = buyout_scale_rub(channel) * truth.scale_multiplier
        before_fraction = -np.expm1(-daily_spend_before / true_scale)
        after_fraction = -np.expm1(-(daily_spend_before + spend) / true_scale)
        expected_impressions = channel.daily_capacity * (
            after_fraction - before_fraction
        )

        weekend = float((day - 1) % 7 in (5, 6))
        cumulative_fraction = min(
            1.0,
            cumulative_spend_before
            / max(buyout_scale_rub(channel) * self.horizon_days, 1.0),
        )
        ctr = (
            channel.base_ctr
            * truth.ctr_multiplier
            * (1.0 - channel.ctr_saturation_decay * after_fraction)
            * (1.0 + truth.weekend_lift * weekend)
            * (1.0 - truth.fatigue_strength * cumulative_fraction)
        )
        cr = (
            channel.base_cr
            * truth.cr_multiplier
            * (1.0 - channel.cr_saturation_decay * after_fraction)
        )
        return EnvironmentDiagnostics(
            expected_impressions=max(0.0, float(expected_impressions)),
            effective_ctr=float(np.clip(ctr, 0.0, 1.0)),
            effective_cr=float(np.clip(cr, 0.0, 1.0)),
        )

    def simulate_chunk(
        self,
        channel_name: str,
        day: int,
        spend: float,
        daily_spend_before: float,
        cumulative_spend_before: float,
        *,
        daily_cap_override: float | None = None,
    ) -> ChunkObservation:
        channel = self.catalog[channel_name]
        daily_cap = channel.max_daily_spend if daily_cap_override is None else daily_cap_override
        tolerance = max(1e-7, daily_cap * 1e-10)
        if spend <= 0 or not np.isfinite(spend):
            raise ValueError("chunk spend must be finite and positive")
        if daily_spend_before + spend > daily_cap + tolerance:
            raise ValueError(f"daily spend cap exceeded for {channel_name}")

        diagnostics = self.diagnostics(
            channel_name,
            day,
            spend,
            daily_spend_before,
            cumulative_spend_before,
        )
        key = (day, channel_name)
        already_observed = self._observed_daily_impressions.get(key, 0)
        remaining_inventory = max(0, int(channel.daily_capacity) - already_observed)
        if remaining_inventory:
            probability = min(
                1.0, diagnostics.expected_impressions / remaining_inventory
            )
            impressions = int(self._rng.binomial(remaining_inventory, probability))
        else:
            impressions = 0
        self._observed_daily_impressions[key] = already_observed + impressions
        clicks = int(self._rng.binomial(impressions, diagnostics.effective_ctr))
        conversions = int(self._rng.binomial(clicks, diagnostics.effective_cr))
        return ChunkObservation(
            day=day,
            channel=channel_name,
            spend=float(spend),
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
        )
