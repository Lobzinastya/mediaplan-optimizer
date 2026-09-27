
import pytest

from types import SimpleNamespace

import numpy as np

from mediaplan_optimizer import bandits
from mediaplan_optimizer.bandits import (
    DecisionContext,
    LinUCBPolicy,
    PeriodicReoptimizationPolicy,
    StaticPolicy,
    ThompsonSamplingPolicy,
)
from mediaplan_optimizer.schemas import ChannelConfig, ObjectiveMetric
from mediaplan_optimizer.simulator import ChunkObservation


def _channel(name: str, *, ctr: float = 0.1, cr: float = 0.2) -> ChannelConfig:
    return ChannelConfig(
        name=name,
        daily_capacity=100_000,
        daily_reach_capacity=80_000,
        response_scale_rub=100_000,
        max_daily_spend=100_000,
        average_frequency=1.5,
        base_ctr=ctr,
        base_cr=cr,
    )


def _context(channel: str, *, vector=None, expected=10.0, marginal=1.0) -> DecisionContext:
    return DecisionContext(
        channel=channel,
        day=1,
        vector=np.asarray(vector if vector is not None else [1.0] * 7),
        proposed_spend=100.0,
        expected_kpi=expected,
        marginal_expected_kpi_per_rub=marginal,
        daily_spend=0.0,
        cumulative_spend=0.0,
    )


def _observation(*, clicks=0, conversions=0, impressions=10, channel="a") -> ChunkObservation:
    return ChunkObservation(1, channel, 100.0, impressions, clicks, conversions)


def test_thompson_updates_beta_posterior_arithmetic_exactly():
    policy = ThompsonSamplingPolicy(prior_strength=5.0)
    policy.reset({"a": _channel("a", ctr=0.1)}, 100, 1, ObjectiveMetric.CLICKS, 7)
    assert policy.alpha["a"] == 0.5
    assert policy.beta["a"] == 4.5

    policy.update(_context("a"), _observation(clicks=3, impressions=10))
    assert policy.alpha["a"] == 3.5
    assert policy.beta["a"] == 11.5


def test_thompson_fixed_seed_is_deterministic_and_explores_equal_channels():
    catalog = {name: _channel(name) for name in ("a", "b")}
    contexts = [_context("a"), _context("b")]
    first = ThompsonSamplingPolicy(prior_strength=1.0)
    second = ThompsonSamplingPolicy(prior_strength=1.0)
    for policy in (first, second):
        policy.reset(catalog, 100, 1, ObjectiveMetric.CLICKS, 123)
    choices_first = [first.select_channel(contexts) for _ in range(50)]
    choices_second = [second.select_channel(contexts) for _ in range(50)]
    assert choices_first == choices_second
    assert set(choices_first) == {"a", "b"}


def test_thompson_strong_evidence_shifts_preference():
    catalog = {"a": _channel("a"), "b": _channel("b")}
    policy = ThompsonSamplingPolicy(prior_strength=2.0)
    policy.reset(catalog, 100, 1, ObjectiveMetric.CLICKS, 9)
    for _ in range(5):
        policy.update(_context("a"), _observation(clicks=80, impressions=100, channel="a"))
        policy.update(_context("b"), _observation(clicks=5, impressions=100, channel="b"))
    assert policy.alpha["a"] > policy.alpha["b"]
    assert policy.beta["a"] < policy.beta["b"]
    choices = [policy.select_channel([_context("a"), _context("b")]) for _ in range(100)]
    assert choices.count("a") > 90


def test_linucb_update_matches_ridge_statistics_and_scores_are_finite():
    policy = LinUCBPolicy(alpha=0.4, regularization=2.0)
    policy.reset({"a": _channel("a")}, 100, 1, ObjectiveMetric.CLICKS, 1)
    vector = np.array([1.0, 0.2, 0.0, 0.4, 0.1, 0.8, 1.0])
    context = _context("a", vector=vector, expected=10.0)
    policy.update(context, _observation(clicks=4, impressions=10))
    expected_a = 2.0 * np.eye(7) + np.outer(vector, vector)
    expected_b = 0.4 * vector
    np.testing.assert_allclose(policy.A["a"], expected_a)
    np.testing.assert_allclose(policy.b["a"], expected_b)
    assert np.isfinite(policy.A["a"]).all()
    assert np.isfinite(policy.b["a"]).all()
    assert policy.select_channel([context]) == "a"


def test_linucb_contextual_reward_prefers_rewarded_channel():
    policy = LinUCBPolicy(alpha=0.0)
    policy.reset({"a": _channel("a"), "b": _channel("b")}, 100, 1, ObjectiveMetric.CLICKS, 1)
    va = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vb = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    policy.update(_context("a", vector=va), _observation(clicks=10, impressions=10, channel="a"))
    policy.update(_context("b", vector=vb), _observation(clicks=0, impressions=10, channel="b"))
    assert policy.select_channel([_context("a", vector=va), _context("b", vector=vb)]) == "a"


def test_periodic_reoptimization_waits_then_updates_beliefs_and_preserves_plan_inputs(monkeypatch):
    calls = []

    def fake_optimize(catalog, horizon, objective, budget):
        calls.append((catalog, horizon, objective, budget))
        return SimpleNamespace(channel_spend={"a": min(100.0, budget), "b": max(0.0, budget - 100.0)})

    monkeypatch.setattr(bandits, "optimize_type_a", fake_optimize)
    catalog = {"a": _channel("a", ctr=0.1, cr=0.2), "b": _channel("b", ctr=0.2, cr=0.3)}
    policy = PeriodicReoptimizationPolicy(interval_days=3, prior_impressions=10, prior_clicks=10)
    policy.reset(catalog, 1_000, 7, ObjectiveMetric.CLICKS, 1)
    policy.on_day_start(1, 1_000, 7)
    assert len(calls) == 1
    before = policy.channel_belief("a")
    policy.update(_context("a"), _observation(clicks=8, impressions=10, channel="a"))
    policy.on_day_start(2, 900, 6)
    policy.on_day_start(3, 800, 5)
    assert policy.channel_belief("a") == before
    policy.on_day_start(4, 700, 4)
    assert len(calls) == 2
    assert calls[-1][1:] == (4, ObjectiveMetric.CLICKS, 700)
    assert policy.channel_belief("a").base_ctr == (0.1 * 10 + 8) / 20


def test_static_policy_uses_planner_quota_and_never_learns(monkeypatch):
    monkeypatch.setattr(
        bandits,
        "optimize_type_a",
        lambda *args: SimpleNamespace(channel_spend={"a": 60.0, "b": 40.0}),
    )
    catalog = {"a": _channel("a"), "b": _channel("b")}
    policy = StaticPolicy()
    policy.reset(catalog, 100, 2, ObjectiveMetric.CLICKS, 1)
    context = _context("a")
    assert policy.allocation_limit(context) == 30.0
    assert policy.select_channel([context, _context("b")]) == "a"
    belief = policy.channel_belief("a")
    policy.update(context, _observation(clicks=10))
    assert policy.channel_belief("a") == belief


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_thompson_boundary_rates_have_proper_beta_prior(product_catalog, rate):
    base = product_catalog["Programmatic"]
    catalog = {base.name: base.model_copy(update={"base_ctr": rate, "base_cr": rate})}
    policy = ThompsonSamplingPolicy()
    policy.reset(catalog, 1000, 1, ObjectiveMetric.CONVERSIONS, 42)
    assert policy.alpha[base.name] > 0
    assert policy.beta[base.name] > 0
    assert np.isfinite(policy.rng.beta(policy.alpha[base.name], policy.beta[base.name]))
