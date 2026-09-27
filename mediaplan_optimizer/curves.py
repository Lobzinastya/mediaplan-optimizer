"""Pure response curves for daily channel spend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .schemas import ChannelConfig, ObjectiveMetric


ResponseValue: TypeAlias = float | np.ndarray


@dataclass(frozen=True)
class ChannelResponse:
    impressions: ResponseValue
    reach: ResponseValue
    clicks: ResponseValue
    conversions: ResponseValue
    video_views: ResponseValue | None


def buyout_scale_rub(channel: ChannelConfig) -> float:
    """Return the spend scale implied by low-spend marginal CPM.

    `response_scale_rub` remains as a backward-compatible fallback for callers
    that construct a ChannelConfig without the additive CPM field.
    """
    if channel.base_cpm_rub is None:
        return channel.response_scale_rub
    return channel.daily_capacity * channel.base_cpm_rub / 1000.0


def _validated_spend(spend: float | np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(spend, dtype=float)
    scalar = values.ndim == 0
    if not np.all(np.isfinite(values)):
        raise ValueError("spend must contain only finite values")
    if np.any(values < 0):
        raise ValueError("spend must be nonnegative")
    return values, scalar


def _native(value: np.ndarray, scalar: bool) -> ResponseValue:
    return float(value) if scalar else value


def channel_response(
    spend: float | np.ndarray, channel: ChannelConfig
) -> ChannelResponse:
    """Calculate a scalar or vectorized response for nonnegative daily spend."""

    values, scalar = _validated_spend(spend)
    # -expm1(-x) is accurate close to zero and approaches one without overflow.
    scale = buyout_scale_rub(channel)
    saturation = -np.expm1(-values / scale)
    impressions = channel.daily_capacity * saturation
    reach_denominator = channel.daily_reach_capacity * channel.average_frequency
    reach = channel.daily_reach_capacity * -np.expm1(-impressions / reach_denominator)
    effective_ctr = channel.base_ctr * (
        1.0 - channel.ctr_saturation_decay * saturation
    )
    effective_cr = channel.base_cr * (
        1.0 - channel.cr_saturation_decay * saturation
    )
    clicks = impressions * effective_ctr
    conversions = clicks * effective_cr
    video_views = None if channel.base_vtr is None else impressions * channel.base_vtr
    return ChannelResponse(
        impressions=_native(impressions, scalar),
        reach=_native(reach, scalar),
        clicks=_native(clicks, scalar),
        conversions=_native(conversions, scalar),
        video_views=None if video_views is None else _native(video_views, scalar),
    )


def effective_performance_rates(
    spend: float | np.ndarray, channel: ChannelConfig
) -> tuple[ResponseValue, ResponseValue]:
    """Return bounded CTR and post-click CR after audience-exhaustion decay."""
    values, scalar = _validated_spend(spend)
    saturation = -np.expm1(-values / buyout_scale_rub(channel))
    ctr = channel.base_ctr * (1.0 - channel.ctr_saturation_decay * saturation)
    cr = channel.base_cr * (1.0 - channel.cr_saturation_decay * saturation)
    return _native(ctr, scalar), _native(cr, scalar)


def marginal_metric_response(
    spend: float, channel: ChannelConfig, metric: ObjectiveMetric
) -> float:
    """Derivative of one-day response, used by the concave optimizer."""
    values, _ = _validated_spend(spend)
    scale = buyout_scale_rub(channel)
    exp_term = float(np.exp(-values / scale))
    saturation = 1.0 - exp_term
    d_saturation = exp_term / scale
    if metric == ObjectiveMetric.CLICKS:
        derivative = (
            channel.daily_capacity
            * channel.base_ctr
            * d_saturation
            * (1.0 - 2.0 * channel.ctr_saturation_decay * saturation)
        )
        return float(derivative)
    if metric == ObjectiveMetric.CONVERSIONS:
        ctr_decay = channel.ctr_saturation_decay
        cr_decay = channel.cr_saturation_decay
        polynomial_derivative = (
            1.0
            - 2.0 * (ctr_decay + cr_decay) * saturation
            + 3.0 * ctr_decay * cr_decay * saturation**2
        )
        return float(
            channel.daily_capacity
            * channel.base_ctr
            * channel.base_cr
            * d_saturation
            * polynomial_derivative
        )
    if metric == ObjectiveMetric.REACH:
        impressions = channel.daily_capacity * saturation
        d_impressions = channel.daily_capacity * d_saturation
        reach_pressure = impressions / (
            channel.daily_reach_capacity * channel.average_frequency
        )
        return float(
            np.exp(-reach_pressure) * d_impressions / channel.average_frequency
        )
    raise ValueError(f"Unsupported objective metric: {metric!r}")


def metric_response(
    spend: float | np.ndarray,
    channel: ChannelConfig,
    metric: str | ObjectiveMetric,
) -> ResponseValue:
    """Return one response metric by its public name."""

    metric_name = metric.value if isinstance(metric, ObjectiveMetric) else str(metric)
    if metric_name not in ChannelResponse.__dataclass_fields__:
        valid = ", ".join(ChannelResponse.__dataclass_fields__)
        raise ValueError(f"Unknown metric {metric_name!r}; expected one of: {valid}")
    value = getattr(channel_response(spend, channel), metric_name)
    if value is None:
        # A non-video channel has no meaningful video-view response.
        return 0.0 if np.asarray(spend).ndim == 0 else np.zeros_like(spend, dtype=float)
    return value
