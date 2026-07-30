"""Black-Scholes estimates for vertical spreads without historical option chains."""

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
        return float(spot * norm.cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2))
    return float(strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def estimate_bull_put(
    spot: float,
    short_strike: float,
    long_strike: float,
    dte: int,
    volatility: float,
    rate: float = 0.04,
) -> dict[str, float]:
    """Estimate a one-contract Bull Put Credit Spread from Black-Scholes prices."""
    time_to_expiry = max(dte, 1) / 365.0
    short_price = bs_price(spot, short_strike, time_to_expiry, rate, volatility, "put")
    long_price = bs_price(spot, long_strike, time_to_expiry, rate, volatility, "put")
    credit = max(short_price - long_price, 0.01)
    width = short_strike - long_strike
    max_profit = credit * 100
    max_loss = max(width - credit, 0) * 100

    return {
        "credit": round(credit, 2),
        "width": width,
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "breakeven": round(short_strike - credit, 2),
        "credit_width_ratio": round(credit / width, 3) if width > 0 else 0.0,
    }