from __future__ import annotations

from mediaplan_optimizer.catalog import REQUIRED_CHANNELS, generate_catalog, load_catalog
from mediaplan_optimizer.schemas import ChannelConfig
from mediaplan_optimizer.curves import channel_response


def test_catalog_has_exact_inventory() -> None:
    catalog = generate_catalog()
    assert tuple(catalog) == REQUIRED_CHANNELS
    assert all(isinstance(channel, ChannelConfig) for channel in catalog.values())


def test_generation_is_deterministic_and_seeded() -> None:
    first = generate_catalog(seed=42)
    second = generate_catalog(seed=42)
    other = generate_catalog(seed=43)
    assert first == second
    assert first != other


def test_bundled_catalog_is_seed_42() -> None:
    assert load_catalog() == generate_catalog(seed=42)


def test_catalog_parameters_are_coherent() -> None:
    for name, channel in generate_catalog().items():
        assert channel.name == name
        assert 0 < channel.daily_reach_capacity <= channel.daily_capacity
        assert channel.response_scale_rub > 0
        assert channel.base_cpm_rub is not None and channel.base_cpm_rub > 0
        assert channel.response_scale_rub == round(
            channel.daily_capacity * channel.base_cpm_rub / 1000
        )
        assert channel.max_daily_spend > 0
        assert channel.average_frequency >= 1
        assert 0 <= channel.base_ctr <= 1
        assert 0 <= channel.base_cr <= 1
        assert 0 <= channel.ctr_saturation_decay <= 0.5
        assert 0 <= channel.cr_saturation_decay <= 0.5
        if channel.base_vtr is not None:
            assert 0 <= channel.base_vtr <= 1
    assert generate_catalog()["SMS"].channel_type == "sms"
    assert generate_catalog()["SMS"].base_vtr is None


def test_generated_cost_and_saturation_scales_are_internally_plausible() -> None:
    """Catch unit mismatches between spend, scale, and daily inventory."""
    for channel in generate_catalog(seed=42).values():
        response = channel_response(channel.max_daily_spend, channel)
        low_spend_cpm = channel.base_cpm_rub
        cap_cpm = 1000 * channel.max_daily_spend / response.impressions
        delivered_fraction = response.impressions / channel.daily_capacity
        reach_fraction = response.reach / channel.daily_reach_capacity

        assert low_spend_cpm is not None
        assert 100 <= low_spend_cpm <= 600
        assert low_spend_cpm < cap_cpm <= 1_600
        assert 0.85 <= delivered_fraction < 1
        assert 0.45 <= reach_fraction < 0.70
