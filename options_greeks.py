# options_greeks.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, List
from scipy.stats import norm


@dataclass
class LegGreeks:
    delta: float
    gamma: float
    theta: float   # 每日 Theta（美元 / 張）
    vega: float    # 每 1% IV 變化（美元 / 張）


@dataclass
class SpreadGreeks:
    short_delta: float
    long_delta: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    iv_short: Optional[float] = None
    iv_long: Optional[float] = None


def _bs_d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def calc_leg_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    multiplier: int = 100
) -> LegGreeks:
    """計算單腿 Greeks（回傳每張合約數值）"""
    if T <= 0 or sigma is None or sigma <= 0:
        return LegGreeks(0.0, 0.0, 0.0, 0.0)

    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    if option_type.lower() in ("call", "c"):
        delta = norm.cdf(d1)
        theta = (
            -S * pdf_d1 * sigma / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -S * pdf_d1 * sigma / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    gamma = pdf_d1 / (S * sigma * math.sqrt(T)) if S > 0 else 0.0
    vega = S * pdf_d1 * math.sqrt(T) / 100

    return LegGreeks(
        delta=delta,
        gamma=gamma,
        theta=theta * multiplier,
        vega=vega * multiplier
    )


def calc_spread_greeks(
    legs: List,
    spot: float,
    dte: int,
    r: float = 0.04
) -> SpreadGreeks:
    """
    legs[0] = 短腿（賣出）
    legs[1] = 長腿（買入）
    """
    T = max(dte, 0.5) / 365.0

    short_leg = legs[0]
    long_leg = legs[1]

    def _get_type(leg) -> str:
        side = str(getattr(leg, "side", "")).lower()
        right = str(getattr(leg, "right", "")).lower()
        if "p" in side or "put" in side or right == "put":
            return "put"
        return "call"

    short_type = _get_type(short_leg)
    long_type = _get_type(long_leg)

    sigma_short = getattr(short_leg, "iv", None)
    sigma_long = getattr(long_leg, "iv", None)

    if not sigma_short or sigma_short < 0.05:
        sigma_short = 0.18
    if not sigma_long or sigma_long < 0.05:
        sigma_long = 0.18

    g_short = calc_leg_greeks(spot, short_leg.strike, T, r, sigma_short, short_type)
    g_long = calc_leg_greeks(spot, long_leg.strike, T, r, sigma_long, long_type)

    # 賣出為負，買入為正
    net_delta = -g_short.delta + g_long.delta
    net_gamma = -g_short.gamma + g_long.gamma
    net_theta = -g_short.theta + g_long.theta
    net_vega = -g_short.vega + g_long.vega

    return SpreadGreeks(
        short_delta=g_short.delta,
        long_delta=g_long.delta,
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_theta=net_theta,
        net_vega=net_vega,
        iv_short=sigma_short,
        iv_long=sigma_long
    )