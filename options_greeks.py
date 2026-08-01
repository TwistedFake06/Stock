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


def _option_right(leg) -> str:
    """Resolve put/call from leg.right (never from buy/sell side)."""
    right = str(getattr(leg, "right", "")).lower()
    if right in ("put", "p") or right.startswith("put"):
        return "put"
    return "call"


def _leg_side(leg) -> str:
    side = str(getattr(leg, "side", "")).lower()
    if side in ("sell", "short", "s"):
        return "sell"
    if side in ("buy", "long", "b"):
        return "buy"
    return side


def _sigma_or_default(leg, default: float = 0.18) -> float:
    sigma = getattr(leg, "iv", None)
    try:
        if sigma is not None and float(sigma) >= 0.05:
            return float(sigma)
    except (TypeError, ValueError):
        pass
    return default


def calc_spread_greeks(
    legs: List,
    spot: float,
    dte: int,
    r: float = 0.04
) -> SpreadGreeks:
    """
    Net Greeks for a vertical (or multi-leg) spread.

    Resolves short/long by each leg's ``side`` (sell/buy), not list index —
    debit builders store [long, short] while credit builders store [short, long].
    """
    if not legs:
        return SpreadGreeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    T = max(dte, 0.5) / 365.0

    short_legs = [lg for lg in legs if _leg_side(lg) == "sell"]
    long_legs = [lg for lg in legs if _leg_side(lg) == "buy"]

    # Fallback only if side metadata is missing (legacy callers)
    if not short_legs and len(legs) >= 1:
        short_legs = [legs[0]]
    if not long_legs and len(legs) >= 2:
        long_legs = [legs[1]]
    elif not long_legs and legs:
        long_legs = [legs[-1]]

    short_leg = short_legs[0]
    long_leg = long_legs[0]

    sigma_short = _sigma_or_default(short_leg)
    sigma_long = _sigma_or_default(long_leg)

    g_short = calc_leg_greeks(
        spot, float(short_leg.strike), T, r, sigma_short, _option_right(short_leg)
    )
    g_long = calc_leg_greeks(
        spot, float(long_leg.strike), T, r, sigma_long, _option_right(long_leg)
    )

    # Aggregate all legs by side (robust if >2 legs later)
    net_delta = 0.0
    net_gamma = 0.0
    net_theta = 0.0
    net_vega = 0.0
    for leg in legs:
        side = _leg_side(leg)
        sign = 1.0 if side == "buy" else -1.0
        g = calc_leg_greeks(
            spot,
            float(leg.strike),
            T,
            r,
            _sigma_or_default(leg),
            _option_right(leg),
        )
        net_delta += sign * g.delta
        net_gamma += sign * g.gamma
        net_theta += sign * g.theta
        net_vega += sign * g.vega

    return SpreadGreeks(
        short_delta=g_short.delta,
        long_delta=g_long.delta,
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_theta=net_theta,
        net_vega=net_vega,
        iv_short=sigma_short,
        iv_long=sigma_long,
    )