"""Estimate monthly SOP entry opportunity frequency from history."""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from analysis import analyze_bias
from entry_targets import analyze_entry, analyze_targets
from extra_analysis import analyze_risk, analyze_trend
from indicators import enrich
from stock_service import DEFAULT_WATCHLIST, fetch_history, normalize_symbol
from trade_sop import _enter_decision, _stability_score

EXTRA = [
    "SPY",
    "QQQ",
    "VOO",
    "META",
    "AMZN",
    "GOOGL",
    "AMD",
    "BABA",
    "3690.HK",
    "000001",
    "300750",
]
STEP = 5  # every 5 trading days (~weekly sample)


def classify_on_df(df: pd.DataFrame):
    if df is None or len(df) < 80:
        return None
    bias = analyze_bias(df)
    entry = analyze_entry(df)
    try:
        risk = analyze_risk(df)
        trend = analyze_trend(df)
        stab, _ = _stability_score(risk, trend, bias.score)
    except Exception:
        stab = 50.0
        risk = type("R", (), {"risk_level": "中", "win_rate_pct": 50})()

    wr = getattr(risk, "win_rate_pct", None)
    e_low, e_high = entry.suggested_entry_low, entry.suggested_entry_high
    stop = entry.stop_loss
    last = float(df["Close"].iloc[-1])
    plan = last
    if e_low and e_high and not (e_low <= last <= e_high):
        plan = (float(e_low) + float(e_high)) / 2
    try:
        targets = analyze_targets(df, info={}, entry=entry)
        t1 = getattr(getattr(targets, "short", None), "bull_target", None)
    except Exception:
        t1 = None
    rr = None
    if plan and stop and t1 and plan > stop and t1 > plan:
        rr = (t1 - plan) / (plan - stop)
    ok, score, _side = _enter_decision(
        entry.opportunity,
        entry.score,
        bias.bias,
        bias.score,
        stab,
        wr,
        rr,
        getattr(risk, "risk_level", "中") or "中",
    )
    return ok, score


def main() -> None:
    symbols = list(
        dict.fromkeys([normalize_symbol(s) for s in list(DEFAULT_WATCHLIST) + EXTRA])
    )
    all_events: list[tuple[str, pd.Timestamp, str, float]] = []
    per_sym: dict[str, dict] = {}

    print(f"Scanning {len(symbols)} symbols, step={STEP} bars (~weekly)...")
    for sym in symbols:
        hist = fetch_history(sym, period="2y", interval="1d")
        if hist is None or hist.empty or len(hist) < 120 or "Date" not in hist.columns:
            print(f"{sym:12} skip")
            continue
        hist = hist.reset_index(drop=True)
        events = []
        prev = None
        for i in range(100, len(hist), STEP):
            sub = enrich(hist.iloc[: i + 1].copy())
            res = classify_on_df(sub)
            if res is None:
                continue
            ok, score = res
            dt = pd.to_datetime(hist.iloc[i]["Date"])
            # New opportunity when entering 适合/谨慎 from outside
            if ok in ("适合入场", "谨慎试仓") and prev not in ("适合入场", "谨慎试仓"):
                events.append((dt, ok, score))
                all_events.append((sym, dt, ok, score))
            prev = ok

        d0 = pd.to_datetime(hist.iloc[100]["Date"])
        d1 = pd.to_datetime(hist.iloc[-1]["Date"])
        months = max((d1.to_period("M") - d0.to_period("M")).n + 1, 1)
        suit = sum(1 for e in events if e[1] == "适合入场")
        caut = sum(1 for e in events if e[1] == "谨慎试仓")
        per_sym[sym] = {
            "months": months,
            "events": len(events),
            "per_month": len(events) / months,
            "suit": suit,
            "caut": caut,
            "suit_pm": suit / months,
            "caut_pm": caut / months,
        }
        print(
            f"{sym:12} months={months:2d} events={len(events):2d} "
            f"per_mo={len(events)/months:.2f} suit={suit} caut={caut}"
        )

    if not per_sym:
        print("No data")
        return

    vals = [v["per_month"] for v in per_sym.values()]
    suit_pm = [v["suit_pm"] for v in per_sym.values()]
    caut_pm = [v["caut_pm"] for v in per_sym.values()]
    print()
    print("=== PER SYMBOL (new entry trigger into 适合/谨慎) ===")
    print(f"Avg opportunities/month/symbol (适合+谨慎): {np.mean(vals):.2f}")
    print(f"  适合入场 only: {np.mean(suit_pm):.2f}")
    print(f"  谨慎试仓 only: {np.mean(caut_pm):.2f}")
    print(f"Median: {np.median(vals):.2f}  P25-P75: {np.percentile(vals,25):.2f}-{np.percentile(vals,75):.2f}")

    if all_events:
        df_e = pd.DataFrame(all_events, columns=["sym", "dt", "ok", "score"])
        df_e["ym"] = df_e["dt"].dt.to_period("M")
        by_m = df_e.groupby("ym").size()
        print()
        print("=== PORTFOLIO (~19 names watched together) ===")
        print(f"Months: {len(by_m)}")
        print(
            f"Total entry events/month (sum across symbols): "
            f"mean={by_m.mean():.1f} median={by_m.median():.1f}"
        )
        suit = df_e[df_e["ok"] == "适合入场"]
        if len(suit):
            by_ms = suit.groupby("ym").size()
            print(
                f"适合入场 only /month: mean={by_ms.mean():.1f} median={by_ms.median():.1f}"
            )
        print("Last 6 months total events:")
        print(by_m.tail(6).to_string())

        # Active trading days: if only take 1 new trade at a time, how many/month?
        # Approximate: count distinct weeks with any 适合入场
        suit2 = suit.copy()
        if len(suit2):
            suit2["week"] = suit2["dt"].dt.to_period("W")
            weeks_pm = suit2.groupby("ym")["week"].nunique()
            print(
                f"Distinct weeks/month with 适合入场 (any symbol): "
                f"mean={weeks_pm.mean():.1f}"
            )


if __name__ == "__main__":
    main()
