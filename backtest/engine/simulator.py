"""Simplified exit rules for daily-price vertical-spread backtests."""

from __future__ import annotations

from collections.abc import Sequence


def simulate_trade(
    future_closes: Sequence[float],
    spread: dict[str, float],
    hold_days: int = 14,
    take_profit_pct: float = 0.5,
) -> tuple[int, float, str]:
    """Simulate a simplified exit and return held days, P&L, and exit reason."""
    if not future_closes or hold_days <= 0:
        return 0, 0.0, "insufficient_data"

    max_profit = spread["max_profit"]
    max_loss = spread["max_loss"]
    breakeven = spread["breakeven"]
    window = future_closes[:hold_days]

    for index, price in enumerate(window):
        if price >= breakeven * 1.005:
            return index + 1, round(max_profit * take_profit_pct, 2), "50%_profit"
        if price <= breakeven * 0.97:
            return index + 1, round(-max_loss * 0.75, 2), "stop_loss"

    final_price = window[-1]
    if final_price >= breakeven:
        return len(window), round(max_profit * 0.65, 2), "time_exit_win"
    return len(window), round(-max_loss * 0.5, 2), "time_exit_loss"