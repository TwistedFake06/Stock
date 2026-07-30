"""Run a simple Bull Put Credit Spread backtest using daily Yahoo Finance prices."""

from __future__ import annotations

import argparse
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


def run_backtest(
    symbol: str = SYMBOL,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    hold_days: int = HOLD_DAYS,
    otm_pct: float = OTM_PCT,
    spread_width: float = SPREAD_WIDTH,
    min_credit_width_ratio: float = MIN_CREDIT_WIDTH_RATIO,
) -> pd.DataFrame:
    """Load data, evaluate bullish entries, then save and return the trade log."""
    symbol = symbol.upper().strip()
    print(f"Downloading {symbol} daily data from {start_date} to {end_date}...")
    prices = load_daily(symbol, start_date, end_date)
    if prices.empty:
        raise RuntimeError(f"No daily data returned for {symbol}.")

    prices["SMA20"] = prices["Close"].rolling(20).mean()
    prices["SMA50"] = prices["Close"].rolling(50).mean()
    prices["returns"] = prices["Close"].pct_change()
    prices["hist_vol"] = prices["returns"].rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)

    trades: list[dict[str, object]] = []
    print("Evaluating bullish Bull Put entries...")
    for index in range(50, len(prices) - hold_days):
        row = prices.iloc[index]
        if pd.isna(row["hist_vol"]) or not (row["Close"] > row["SMA20"] > row["SMA50"]):
            continue

        spot = float(row["Close"])
        short_strike = rounded_strike(spot * (1 - otm_pct))
        long_strike = short_strike - spread_width
        spread = estimate_bull_put(
            spot=spot,
            short_strike=short_strike,
            long_strike=long_strike,
            dte=hold_days,
            volatility=float(row["hist_vol"]),
        )
        if spread["credit_width_ratio"] < min_credit_width_ratio or spread["max_loss"] <= 0:
            continue

        future_closes = prices["Close"].iloc[index + 1 : index + 1 + hold_days].astype(float).tolist()
        days_held, pnl, exit_reason = simulate_trade(future_closes, spread, hold_days)
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
    output_path = results_dir / f"{symbol.lower()}_bull_put_trades.csv"
    trade_log.to_csv(output_path, index=False)

    summary = summarize_trades(trade_log)
    print(f"Saved {len(trade_log)} trades to {output_path}")
    print(f"\n===== Backtest Result ({symbol} Bull Put) =====")
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for quick symbol/parameter switching."""
    parser = argparse.ArgumentParser(description="Run Bull Put backtest with daily data.")
    parser.add_argument("--symbol", default=SYMBOL, help="Ticker symbol, e.g. SPY/VOO/QQQ")
    parser.add_argument("--start", default=START_DATE, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE, help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS, help="Max holding days")
    parser.add_argument("--otm-pct", type=float, default=OTM_PCT, help="OTM percentage, e.g. 0.03")
    parser.add_argument("--spread-width", type=float, default=SPREAD_WIDTH, help="Spread width in dollars")
    parser.add_argument(
        "--min-cw-ratio",
        type=float,
        default=MIN_CREDIT_WIDTH_RATIO,
        help="Minimum credit/width ratio",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        hold_days=args.hold_days,
        otm_pct=args.otm_pct,
        spread_width=args.spread_width,
        min_credit_width_ratio=args.min_cw_ratio,
    )