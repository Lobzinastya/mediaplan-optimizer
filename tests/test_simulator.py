from __future__ import annotations

import pytest

from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.simulator import CampaignSimulator


def test_fixed_seed_replays_identical_action_sequence() -> None:
    catalog = generate_catalog(seed=42)
    actions = [
        ("Social Network 1", 1, catalog["Social Network 1"].max_daily_spend * 0.2, 0.0, 0.0),
        ("Social Network 1", 1, catalog["Social Network 1"].max_daily_spend * 0.3, catalog["Social Network 1"].max_daily_spend * 0.2, 1.0),
        ("Programmatic", 2, catalog["Programmatic"].max_daily_spend * 0.4, 0.0, 0.0),
    ]

    def replay() -> list[object]:
        simulator = CampaignSimulator(catalog, horizon_days=7, seed=123)
        return [simulator.simulate_chunk(*action) for action in actions]

    assert replay() == replay()


def test_hidden_truth_is_moderate_and_oracle_adjusts_planner_beliefs() -> None:
    catalog = generate_catalog(seed=42)
    simulator = CampaignSimulator(catalog, horizon_days=14, seed=123)
    truth = simulator.truth_snapshot()
    oracle = simulator.oracle_catalog()

    for name, parameters in truth.items():
        assert 0.8 <= parameters.ctr_multiplier <= 1.2
        assert 0.8 <= parameters.cr_multiplier <= 1.2
        assert 0.8 <= parameters.scale_multiplier <= 1.2
        assert -0.15 <= parameters.weekend_lift <= 0.15
        assert 0.05 <= parameters.fatigue_strength <= 0.20
        assert (
            oracle[name].base_ctr != catalog[name].base_ctr
            or oracle[name].base_cr != catalog[name].base_cr
            or oracle[name].response_scale_rub != catalog[name].response_scale_rub
        )


def test_diagnostics_are_bounded_and_expected_volume_is_nonnegative() -> None:
    catalog = generate_catalog(seed=42)
    simulator = CampaignSimulator(catalog, horizon_days=14, seed=123)

    for name, channel in catalog.items():
        diagnostics = simulator.diagnostics(
            name,
            day=6,
            spend=channel.max_daily_spend * 0.25,
            daily_spend_before=0.0,
            cumulative_spend_before=0.0,
        )
        assert diagnostics.expected_impressions >= 0
        assert 0 <= diagnostics.effective_ctr <= 1
        assert 0 <= diagnostics.effective_cr <= 1


def test_observations_are_integer_nonnegative_and_ordered() -> None:
    catalog = generate_catalog(seed=42)
    simulator = CampaignSimulator(catalog, horizon_days=7, seed=123)
    channel = catalog["Marketplace 1"]
    observation = simulator.simulate_chunk(
        "Marketplace 1", 1, channel.max_daily_spend * 0.5, 0.0, 0.0
    )

    assert isinstance(observation.impressions, int)
    assert isinstance(observation.clicks, int)
    assert isinstance(observation.conversions, int)
    assert observation.impressions >= 0
    assert 0 <= observation.clicks <= observation.impressions
    assert 0 <= observation.conversions <= observation.clicks


def test_daily_spend_cap_is_enforced() -> None:
    catalog = generate_catalog(seed=42)
    simulator = CampaignSimulator(catalog, horizon_days=7, seed=123)
    channel = catalog["SMS"]

    with pytest.raises(ValueError, match="daily spend cap exceeded"):
        simulator.simulate_chunk(
            "SMS", 1, channel.max_daily_spend * 1.01, 0.0, 0.0
        )

    simulator.simulate_chunk("SMS", 1, channel.max_daily_spend * 0.6, 0.0, 0.0)
    with pytest.raises(ValueError, match="daily spend cap exceeded"):
        simulator.simulate_chunk(
            "SMS", 1, channel.max_daily_spend * 0.5, channel.max_daily_spend * 0.6, 0.0
        )


def test_observed_daily_impressions_never_exceed_capacity() -> None:
    catalog = generate_catalog(seed=42)
    simulator = CampaignSimulator(catalog, horizon_days=7, seed=123)
    channel = catalog["Social Network 2"]
    chunk_spend = channel.max_daily_spend * 0.09
    total_impressions = 0

    for index in range(10):
        observation = simulator.simulate_chunk(
            "Social Network 2",
            1,
            chunk_spend,
            chunk_spend * index,
            chunk_spend * index,
        )
        total_impressions += observation.impressions
        assert total_impressions <= channel.daily_capacity

