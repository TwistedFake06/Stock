"""Black-Scholes estimates for vertical credit spreads (backtest engine)."""

from __future__ import annotations

import math

from scipy.stats import norm


def bs_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str = "put",
) -> float:
    """Return the Black-Scholes value for one option share."""
    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0) if option_type == "call" else max(strike - spot, 0)

    sqrt_time = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_time)
    d2 = d1 - volatility * sqrt_time

    if option_type == "call":
        return float(
            spot * norm.cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2)
        )
    return float(
        strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    )


def mark_credit_vertical(
    spot: float,
    short_strike: float,
    long_strike: float,
    dte: int,
    volatility: float,
    option_type: str,
    rate: float = 0.04,
) -> float:
    """Mark-to-market credit (short premium - long premium) per share."""
    t = max(dte, 0) / 365.0
    short_px = bs_price(spot, short_strike, t, rate, volatility, option_type)
    long_px = bs_price(spot, long_strike, t, rate, volatility, option_type)
    return max(short_px - long_px, 0.0)


def estimate_bull_put(
    spot: float,
    short_strike: float,
    long_strike: float,
    dte: int,
    volatility: float,
    rate: float = 0.04,
) -> dict[str, float]:
    """Estimate a one-contract Bull Put Credit Spread from Black-Scholes prices."""
    credit = mark_credit_vertical(
        spot, short_strike, long_strike, max(dte, 1), volatility, "put", rate
    )
    credit = max(credit, 0.01)
    width = short_strike - long_strike
    max_profit = credit * 100
    max_loss = max(width - credit, 0) * 100
    return {
        "strategy": "bull_put",
        "option_type": "put",
        "credit": round(credit, 2),
        "width": width,
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "breakeven": round(short_strike - credit, 2),
        "credit_width_ratio": round(credit / width, 3) if width > 0 else 0.0,
        "short_strike": float(short_strike),
        "long_strike": float(long_strike),
        "dte": int(dte),
    }


def estimate_bear_call(
    spot: float,
    short_strike: float,
    long_strike: float,
    dte: int,
    volatility: float,
    rate: float = 0.04,
) -> dict[str, float]:
    """Estimate a one-contract Bear Call Credit Spread (short lower call / long higher)."""
    credit = mark_credit_vertical(
        spot, short_strike, long_strike, max(dte, 1), volatility, "call", rate
    )
    credit = max(credit, 0.01)
    width = long_strike - short_strike
    max_profit = credit * 100
    max_loss = max(width - credit, 0) * 100
    return {
        "strategy": "bear_call",
        "option_type": "call",
        "credit": round(credit, 2),
        "width": width,
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "breakeven": round(short_strike + credit, 2),
        "credit_width_ratio": round(credit / width, 3) if width > 0 else 0.0,
        "short_strike": float(short_strike),
        "long_strike": float(long_strike),
        "dte": int(dte),
    }
