"""Run Bull Put / Bear Call credit-spread backtests with BS mark-to-market exits."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine.data_loader import load_daily
from backtest.engine.metrics import summarize_trades
from backtest.engine.simulator import simulate_trade
from backtest.engine.spread_sim import estimate_bear_call, estimate_bull_put

SYMBOL = "SPY"
START_DATE = "2022-01-01"
END_DATE = date.today().isoformat()
HOLD_DAYS = 14
OTM_PCT = 0.03
SPREAD_WIDTH = 5.0
MIN_CREDIT_WIDTH_RATIO = 0.25
VOLATILITY_WINDOW = 20
# Typical retail: ~$0.65/leg × open+close × 2 legs ≈ $2.6 RT per contract
DEFAULT_COMMISSION_RT = 2.6
DEFAULT_SLIP = 0.03
ENGINE = "bs_mtm_v3"


def rounded_strike(price: float, increment: float = 1.0) -> float:
    """Round a strike down to a tradable increment without moving it closer to spot."""
    return float(np.floor(price / increment) * increment)


def rounded_strike_up(price: float, increment: float = 1.0) -> float:
    return float(np.ceil(price / increment) * increment)


def run_backtest(
    symbol: str = SYMBOL,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    hold_days: int = HOLD_DAYS,
    otm_pct: float = OTM_PCT,
    spread_width: float = SPREAD_WIDTH,
    min_credit_width_ratio: float = MIN_CREDIT_WIDTH_RATIO,
    strategy: str = "bull_put",
    commission_rt: float = DEFAULT_COMMISSION_RT,
    slip_per_share: float = DEFAULT_SLIP,
) -> pd.DataFrame:
    """
    Load data, evaluate directional credit vertical entries, save trade log.

    strategy:
      - bull_put: trend up (Close > SMA20 > SMA50)
      - bear_call: trend down (Close < SMA20 < SMA50)

    Positions are non-overlapping. Engine uses BS daily mark-to-market exits.
    """
    symbol = symbol.upper().strip()
    strategy = strategy.strip().lower()
    if strategy not in ("bull_put", "bear_call"):
        raise ValueError("strategy must be bull_put or bear_call")

    print(
        f"Downloading {symbol} daily data from {start_date} to {end_date} "
        f"(strategy={strategy}, engine={ENGINE})..."
    )
    prices = load_daily(symbol, start_date, end_date)
    if prices.empty:
        raise RuntimeError(f"No daily data returned for {symbol}.")

    prices["SMA20"] = prices["Close"].rolling(20).mean()
    prices["SMA50"] = prices["Close"].rolling(50).mean()
    prices["returns"] = prices["Close"].pct_change()
    prices["hist_vol"] = prices["returns"].rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)

    trades: list[dict[str, object]] = []
    next_entry_index = 50
    print(f"Evaluating {strategy} entries (non-overlapping, MTM+commission)...")
    for index in range(50, len(prices) - hold_days):
        if index < next_entry_index:
            continue
        row = prices.iloc[index]
        if pd.isna(row["hist_vol"]):
            continue

        if strategy == "bull_put":
            if not (row["Close"] > row["SMA20"] > row["SMA50"]):
                continue
        else:
            if not (row["Close"] < row["SMA20"] < row["SMA50"]):
                continue

        spot = float(row["Close"])
        if strategy == "bull_put":
            short_strike = rounded_strike(spot * (1 - otm_pct))
            long_strike = short_strike - spread_width
            if long_strike <= 0:
                continue
            spread = estimate_bull_put(
                spot=spot,
                short_strike=short_strike,
                long_strike=long_strike,
                dte=hold_days,
                volatility=float(row["hist_vol"]),
            )
        else:
            short_strike = rounded_strike_up(spot * (1 + otm_pct))
            long_strike = short_strike + spread_width
            spread = estimate_bear_call(
                spot=spot,
                short_strike=short_strike,
                long_strike=long_strike,
                dte=hold_days,
                volatility=float(row["hist_vol"]),
            )

        if spread["credit_width_ratio"] < min_credit_width_ratio or spread["max_loss"] <= 0:
            continue

        future_closes = (
            prices["Close"].iloc[index + 1 : index + 1 + hold_days].astype(float).tolist()
        )
        spread_pkg = {
            **spread,
            "hist_vol": float(row["hist_vol"]),
            "dte": hold_days,
        }
        days_held, pnl, exit_reason = simulate_trade(
            future_closes,
            spread_pkg,
            hold_days=hold_days,
            take_profit_pct=0.5,
            stop_loss_pct=2.0,
            entry_vol=float(row["hist_vol"]),
            use_mtm=True,
            slip_per_share=slip_per_share,
            commission_round_trip=commission_rt,
        )
        if days_held <= 0:
            continue
        next_entry_index = index + days_held + 1
        r_multiple = pnl / spread["max_loss"] if spread["max_loss"] else 0.0
        trades.append(
            {
                "entry_date": row["Date"],
                "entry_price": round(spot, 2),
                "strategy": strategy,
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
                "commission_rt": commission_rt,
                "slip_per_share": slip_per_share,
                "engine": ENGINE,
            }
        )

    trade_log = pd.DataFrame(trades)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"{symbol.lower()}_{strategy}_trades.csv"
    trade_log.to_csv(output_path, index=False)

    summary = summarize_trades(trade_log)
    print(f"Saved {len(trade_log)} trades to {output_path}")
    print(f"\n===== Backtest ({symbol} {strategy} · {ENGINE}) =====")
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
    parser = argparse.ArgumentParser(description="Run credit vertical backtest (MTM v3).")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    parser.add_argument("--otm-pct", type=float, default=OTM_PCT)
    parser.add_argument("--spread-width", type=float, default=SPREAD_WIDTH)
    parser.add_argument("--min-cw-ratio", type=float, default=MIN_CREDIT_WIDTH_RATIO)
    parser.add_argument(
        "--strategy",
        default="bull_put",
        choices=["bull_put", "bear_call"],
        help="bull_put (uptrend) or bear_call (downtrend)",
    )
    parser.add_argument("--commission-rt", type=float, default=DEFAULT_COMMISSION_RT)
    parser.add_argument("--slip", type=float, default=DEFAULT_SLIP)
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
        strategy=args.strategy,
        commission_rt=args.commission_rt,
        slip_per_share=args.slip,
    )
