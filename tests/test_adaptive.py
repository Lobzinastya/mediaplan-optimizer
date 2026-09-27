"""Integration coverage for deployable adaptive campaign policies."""

import numpy as np
import pandas as pd
import pytest

from mediaplan_optimizer.adaptive import (
    ACTION_COLUMNS,
    CHANNEL_COLUMNS,
    DAILY_COLUMNS,
    CampaignRunConfig,
    run_adaptive_campaign,
)
from mediaplan_optimizer.bandits import (
    LinUCBPolicy,
    PeriodicReoptimizationPolicy,
    StaticPolicy,
    ThompsonSamplingPolicy,
)
from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.schemas import ObjectiveMetric


@pytest.fixture(scope="module")
def catalog():
    return generate_catalog(seed=42)


@pytest.fixture(scope="module")
def config():
    # A deliberately small, but fully spendable, campaign keeps this test fast.
    return CampaignRunConfig(
        budget=90_000.0, horizon_days=3, quantum_rub=30_000.0, seed=17,
    )


def _policies():
    return [
        StaticPolicy(),
        PeriodicReoptimizationPolicy(interval_days=2),
        ThompsonSamplingPolicy(prior_strength=100.0),
        LinUCBPolicy(alpha=0.5),
    ]


def _finite_frame(frame: pd.DataFrame) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_all_policies_complete_campaign_with_stable_outputs(catalog, config):
    for policy in _policies():
        result = run_adaptive_campaign(policy, catalog, config)

        assert len(result.daily) == config.horizon_days
        assert not result.actions.empty
        assert list(result.daily.columns) == DAILY_COLUMNS
        assert list(result.channels.columns) == CHANNEL_COLUMNS
        assert list(result.actions.columns) == ACTION_COLUMNS
        assert result.total_spend <= config.budget + 1e-6
        assert result.total_spend == pytest.approx(config.budget, abs=1e-6)
        _finite_frame(result.daily)
        _finite_frame(result.channels)
        _finite_frame(result.actions)

        # A policy may split a day's allocation into several chunks, but never
        # exceed a channel's configured daily cap.
        by_day_channel = result.actions.groupby(["day", "channel"])["spend"].sum()
        for (day, channel), spend in by_day_channel.items():
            assert spend <= catalog[channel].max_daily_spend + 1e-6


def test_fixed_seed_and_policy_are_reproducible(catalog, config):
    for policy_factory in (StaticPolicy, PeriodicReoptimizationPolicy,
                           ThompsonSamplingPolicy, LinUCBPolicy):
        first = run_adaptive_campaign(policy_factory(), catalog, config)
        second = run_adaptive_campaign(policy_factory(), catalog, config)
        pd.testing.assert_frame_equal(first.daily, second.daily)
        pd.testing.assert_frame_equal(first.channels, second.channels)
        pd.testing.assert_frame_equal(first.actions, second.actions)


def test_adaptive_objectives_accept_clicks_and_reject_reach():
    assert CampaignRunConfig(objective=ObjectiveMetric.CLICKS).objective == ObjectiveMetric.CLICKS
    with pytest.raises(ValueError, match="clicks or conversions"):
        CampaignRunConfig(objective=ObjectiveMetric.REACH)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_adaptive_budget_is_rejected(value):
    with pytest.raises(ValueError):
        CampaignRunConfig(budget=value)
