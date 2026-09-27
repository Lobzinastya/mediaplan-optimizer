from __future__ import annotations

import numpy as np
import pytest

from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.curves import (
    buyout_scale_rub,
    channel_response,
    effective_performance_rates,
    marginal_metric_response,
    metric_response,
)
from mediaplan_optimizer.schemas import ChannelConfig, ObjectiveMetric


@pytest.mark.parametrize("channel", generate_catalog().values(), ids=lambda c: c.name)
def test_curve_is_finite_nonnegative_monotone_bounded_and_concave(channel) -> None:
    spend = np.linspace(0, channel.max_daily_spend * 20, 501)
    response = channel_response(spend, channel)
    for metric in (response.impressions, response.reach, response.clicks, response.conversions):
        assert np.all(np.isfinite(metric))
        assert np.all(metric >= 0)
        assert np.all(np.diff(metric) >= -1e-9)
    within_cap = np.linspace(0, channel.max_daily_spend, 501)
    for metric in ("clicks", "conversions"):
        assert np.all(np.diff(metric_response(within_cap, channel, metric), n=2) <= 1e-7)
    assert np.all(response.impressions <= channel.daily_capacity + 1e-8)
    assert np.all(response.reach <= channel.daily_reach_capacity + 1e-8)
    if response.video_views is not None:
        assert np.all(np.diff(response.video_views) >= -1e-9)
        assert np.all(response.video_views <= channel.daily_capacity * channel.base_vtr + 1e-8)


def test_zero_spend_and_rates() -> None:
    for channel in generate_catalog().values():
        response = channel_response(0, channel)
        assert response.impressions == response.reach == response.clicks == 0
        assert response.conversions == 0
        assert response.video_views in (None, 0)

        zero_ctr, zero_cr = effective_performance_rates(0, channel)
        assert zero_ctr == pytest.approx(channel.base_ctr)
        assert zero_cr == pytest.approx(channel.base_cr)

        positive = channel_response(channel.response_scale_rub, channel)
        effective_ctr, effective_cr = effective_performance_rates(
            channel.response_scale_rub, channel
        )
        assert positive.clicks / positive.impressions == pytest.approx(effective_ctr)
        assert positive.conversions / positive.clicks == pytest.approx(effective_cr)
        assert effective_ctr <= channel.base_ctr
        assert effective_cr <= channel.base_cr
        if positive.video_views is not None:
            assert positive.video_views / positive.impressions == pytest.approx(channel.base_vtr)


def test_metric_response_supports_scalar_and_arrays() -> None:
    channel = generate_catalog()["Social Network 1"]
    assert metric_response(1000, channel, "clicks") > 0
    values = metric_response(np.array([0, 1000, 2000]), channel, "conversions")
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)


def test_invalid_spend_and_metric_are_rejected() -> None:
    channel = generate_catalog()["Programmatic"]
    with pytest.raises(ValueError):
        channel_response(-1, channel)
    with pytest.raises(ValueError):
        channel_response(float("inf"), channel)
    with pytest.raises(ValueError):
        metric_response(1, channel, "revenue")
    with pytest.raises(ValueError, match="Unsupported objective metric.*revenue"):
        marginal_metric_response(1, channel, "revenue")


@pytest.mark.parametrize("channel", generate_catalog().values(), ids=lambda c: c.name)
def test_unit_costs_worsen_smoothly_as_inventory_saturates(channel) -> None:
    spends = channel.response_scale_rub * np.array([1e-6, 0.1, 1.0, 2.0])
    response = channel_response(spends, channel)
    cpm = spends / response.impressions * 1000
    cpc = spends / response.clicks
    cpa = spends / response.conversions

    for unit_cost in (cpm, cpc, cpa):
        assert np.all(np.isfinite(unit_cost))
        assert np.all(np.diff(unit_cost) > 0)


@pytest.mark.parametrize("channel", generate_catalog().values(), ids=lambda c: c.name)
def test_effective_rates_are_bounded_and_do_not_improve_with_saturation(channel) -> None:
    spends = channel.response_scale_rub * np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    ctr, cr = effective_performance_rates(spends, channel)

    for rates in (ctr, cr):
        assert np.all(np.isfinite(rates))
        assert np.all((0 <= rates) & (rates <= 1))
        assert np.all(np.diff(rates) <= 1e-12)


def test_explicit_base_cpm_controls_inventory_buyout() -> None:
    channel = generate_catalog()["Social Network 1"]
    spend = 100_000
    cheaper = channel.model_copy(update={"base_cpm_rub": channel.base_cpm_rub / 2})
    costlier = channel.model_copy(update={"base_cpm_rub": channel.base_cpm_rub * 2})

    assert channel_response(spend, cheaper).impressions > channel_response(
        spend, costlier
    ).impressions


@pytest.mark.parametrize("decays", [(0.5, 0), (0, 0.5), (1 / 3, 1 / 3), (0.2, 0.4)])
def test_valid_decay_boundary_is_monotone_and_concave(product_catalog, decays):
    data = next(iter(product_catalog.values())).model_dump()
    data.update(ctr_saturation_decay=decays[0], cr_saturation_decay=decays[1])
    channel = ChannelConfig(**data)
    spends = np.linspace(0, 30 * buyout_scale_rub(channel), 501)
    for metric in ObjectiveMetric:
        values = metric_response(spends, channel, metric)
        marginal = np.array([marginal_metric_response(float(s), channel, metric) for s in spends])
        assert np.isfinite(values).all()
        assert np.diff(values).min() >= -1e-8
        assert marginal.min() >= -1e-12
        assert np.diff(marginal).max() <= 1e-12
