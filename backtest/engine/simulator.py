"""Mark-to-market exits for vertical credit-spread backtests."""

from __future__ import annotations

from collections.abc import Sequence

from backtest.engine.spread_sim import mark_credit_vertical


def simulate_trade(
    future_closes: Sequence[float],
    spread: dict[str, float | str],
    hold_days: int = 14,
    take_profit_pct: float = 0.5,
    stop_loss_pct: float = 2.0,
    entry_vol: float | None = None,
    use_mtm: bool = True,
    slip_per_share: float = 0.03,
    commission_round_trip: float = 2.6,
) -> tuple[int, float, str]:
    """
    Simulate a credit vertical and return (days_held, pnl_$, exit_reason).

    Costs:
    - slip_per_share: bid/ask friction on close ($/share)
    - commission_round_trip: total $ for open+close of 1 contract (default ~$0.65×4 legs
      if each leg open/close, simplified to $2.6/contract RT)
    """
    if not future_closes or hold_days <= 0:
        return 0, 0.0, "insufficient_data"

    max_profit = float(spread["max_profit"])
    max_loss = float(spread["max_loss"])
    window = list(future_closes[:hold_days])
    if not window:
        return 0, 0.0, "insufficient_data"

    if not use_mtm:
        return _simulate_legacy(window, spread, take_profit_pct)

    entry_credit = float(spread["credit"])
    short_k = float(spread["short_strike"])
    long_k = float(spread["long_strike"])
    opt_type = str(spread.get("option_type") or "put")
    vol = float(entry_vol if entry_vol is not None else spread.get("hist_vol") or 0.20)
    vol = max(0.08, min(vol, 0.90))
    dte0 = int(spread.get("dte") or hold_days)

    tp = take_profit_pct * max_profit
    sl = stop_loss_pct * max_loss
    slip = abs(slip_per_share)
    comm = abs(commission_round_trip)

    last_pnl = 0.0
    for index, price in enumerate(window):
        dte_left = max(dte0 - (index + 1), 0)
        if dte_left <= 0:
            pnl = _expiry_pnl_credit(
                float(price), short_k, long_k, entry_credit, opt_type, slip, comm
            )
            return index + 1, round(pnl, 2), "expiry_settle"

        mark_credit = mark_credit_vertical(
            float(price), short_k, long_k, dte_left, vol, opt_type
        )
        pnl = (entry_credit - mark_credit - slip) * 100.0 - comm
        last_pnl = pnl

        if pnl >= tp:
            return index + 1, round(pnl, 2), "50%_mtm_tp"
        if pnl <= -sl:
            return index + 1, round(pnl, 2), "mtm_stop"

    return len(window), round(last_pnl, 2), "time_exit_mtm"


def _expiry_pnl_credit(
    price: float,
    short_k: float,
    long_k: float,
    entry_credit: float,
    opt_type: str,
    slip: float,
    comm: float,
) -> float:
    """Credit vertical expiry: buy-back cost = spread intrinsic."""
    if opt_type == "call":
        # short lower call, long higher call
        if price <= short_k:
            value = 0.0
        elif price >= long_k:
            value = long_k - short_k
        else:
            value = price - short_k
    else:
        # short higher put, long lower put
        if price >= short_k:
            value = 0.0
        elif price <= long_k:
            value = short_k - long_k
        else:
            value = short_k - price
    return (entry_credit - value - slip) * 100.0 - comm


def _simulate_legacy(
    window: list[float],
    spread: dict,
    take_profit_pct: float,
) -> tuple[int, float, str]:
    max_profit = float(spread["max_profit"])
    max_loss = float(spread["max_loss"])
    breakeven = float(spread["breakeven"])
    strategy = str(spread.get("strategy") or "bull_put")
    for index, price in enumerate(window):
        if strategy == "bear_call":
            if price <= breakeven * 0.995:
                return index + 1, round(max_profit * take_profit_pct, 2), "50%_profit"
            if price >= breakeven * 1.03:
                return index + 1, round(-max_loss * 0.75, 2), "stop_loss"
        else:
            if price >= breakeven * 1.005:
                return index + 1, round(max_profit * take_profit_pct, 2), "50%_profit"
            if price <= breakeven * 0.97:
                return index + 1, round(-max_loss * 0.75, 2), "stop_loss"
    final_price = window[-1]
    win = final_price >= breakeven if strategy != "bear_call" else final_price <= breakeven
    if win:
        return len(window), round(max_profit * 0.65, 2), "time_exit_win"
    return len(window), round(-max_loss * 0.5, 2), "time_exit_loss"
