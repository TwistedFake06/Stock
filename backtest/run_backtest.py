"""Run a simple Bull Put Credit Spread backtest using daily Yahoo Finance prices."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine.data_loader import load_daily
from backtest.engine.metrics import summarize_trades
from backtest.engine.simulator import simulate_trade
from backtest.engine.spread_sim import estimate_bull_put

SYMBOL = "SPY"
START_DATE = "2022-01-01"
END_DATE = date.today().isoformat()
HOLD_DAYS = 14
OTM_PCT = 0.03
SPREAD_WIDTH = 5.0
MIN_CREDIT_WIDTH_RATIO = 0.25
VOLATILITY_WINDOW = 20


def rounded_strike(price: float, increment: float = 1.0) -> float:
    """Round a strike down to a tradable increment without moving it closer to spot."""
    return float(np.floor(price / increment) * increment)


def run_backtest() -> pd.DataFrame:
    """Load data, evaluate bullish entries, then save and return the trade log."""
    print(f"Downloading {SYMBOL} daily data from {START_DATE} to {END_DATE}...")
    prices = load_daily(SYMBOL, START_DATE, END_DATE)
    if prices.empty:
        raise RuntimeError(f"No daily data returned for {SYMBOL}.")

    prices["SMA20"] = prices["Close"].rolling(20).mean()
    prices["SMA50"] = prices["Close"].rolling(50).mean()
    prices["returns"] = prices["Close"].pct_change()
    prices["hist_vol"] = prices["returns"].rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)

    trades: list[dict[str, object]] = []
    print("Evaluating bullish Bull Put entries...")
    for index in range(50, len(prices) - HOLD_DAYS):
        row = prices.iloc[index]
        if pd.isna(row["hist_vol"]) or not (row["Close"] > row["SMA20"] > row["SMA50"]):
            continue

        spot = float(row["Close"])
        short_strike = rounded_strike(spot * (1 - OTM_PCT))
        long_strike = short_strike - SPREAD_WIDTH
        spread = estimate_bull_put(
            spot=spot,
            short_strike=short_strike,
            long_strike=long_strike,
            dte=HOLD_DAYS,
            volatility=float(row["hist_vol"]),
        )
        if spread["credit_width_ratio"] < MIN_CREDIT_WIDTH_RATIO or spread["max_loss"] <= 0:
            continue

        future_closes = prices["Close"].iloc[index + 1 : index + 1 + HOLD_DAYS].astype(float).tolist()
        days_held, pnl, exit_reason = simulate_trade(future_closes, spread, HOLD_DAYS)
        r_multiple = pnl / spread["max_loss"]
        trades.append(
            {
                "entry_date": row["Date"],
                "entry_price": round(spot, 2),
                "short_strike": short_strike,
                "long_strike": long_strike,
                "hist_vol": round(float(row["hist_vol"]), 4),
                "credit": spread["credit"],
                "max_profit": spread["max_profit"],
                "max_loss": spread["max_loss"],
                "breakeven": spread["breakeven"],
                "credit_width_ratio": spread["credit_width_ratio"],
                "days_held": days_held,
                "pnl": pnl,
                "r_multiple": round(r_multiple, 3),
                "is_win": pnl > 0,
                "exit_reason": exit_reason,
            }
        )

    trade_log = pd.DataFrame(trades)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"{SYMBOL.lower()}_bull_put_trades.csv"
    trade_log.to_csv(output_path, index=False)

    summary = summarize_trades(trade_log)
    print(f"Saved {len(trade_log)} trades to {output_path}")
    print(f"\n===== Backtest Result ({SYMBOL} Bull Put) =====")
    if "error" in summary:
        print(summary["error"])
    else:
        print(f"Total trades     : {summary['total_trades']}")
        print(f"Win rate         : {summary['win_rate']:.1f}%")
        print(f"Average R        : {summary['avg_r']:.3f}")
        print(f"Expectancy       : {summary['expectancy']:.3f}R")
        print(f"Profit Factor    : {summary['profit_factor']:.2f}")
        print(f"Average P&L      : ${summary['avg_pnl']:.2f}")
    return trade_log


if __name__ == "__main__":
    run_backtest()