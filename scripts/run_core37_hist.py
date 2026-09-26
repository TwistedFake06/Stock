# -*- coding: utf-8 -*-
"""CORE-37 ~2y daily hist screen using backtest_watchlist_swing engine."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import CORE_WATCHLIST
from backtest_watchlist_swing import backtest_symbol, summarize_history

OUT_CSV = ROOT / "data" / "watchlist_hist_core37.csv"
OUT_REPORT = ROOT / "data" / "watchlist_hist_core37_report.txt"
OUT_LOG = ROOT / "data" / "watchlist_hist_core37_log.txt"
PERIOD = "2y"
HORIZON = 10
MONTHS = 24
RISK = 500.0


def tier_of(row: dict) -> str:
    trades = int(row.get("trades") or 0)
    wr = row.get("win_rate")
    ar = row.get("avg_r")
    if trades < 5 or wr is None or ar is None:
        return "B"  # insufficient / watch
    # A: hist-usable for high WR — trades>=10 preferred, WR>=48 and avg_r>0
    # soft A: trades>=8, WR>=45, avg_r>0
    if trades >= 10 and wr >= 48.0 and ar > 0:
        return "A"
    if trades >= 8 and wr >= 45.0 and ar > 0.05:
        return "A"
    return "B"


def main() -> int:
    symbols = list(CORE_WATCHLIST)
    rows = []
    log_lines = []
    print(f"CORE-37 hist screen n={len(symbols)} period={PERIOD} horizon={HORIZON}d")
    for i, sym in enumerate(symbols, 1):
        msg = f"[{i}/{len(symbols)}] {sym}"
        print(msg, flush=True)
        log_lines.append(msg)
        try:
            trades = backtest_symbol(sym, period=PERIOD, horizon=HORIZON)
            summ = summarize_history(trades, risk_hkd=RISK, observed_months=MONTHS)
            row = {"symbol": sym, **summ}
            # confidence_tag
            n = int(summ.get("trades") or 0)
            row["confidence_tag"] = "ok" if n >= 10 else ("thin" if n >= 5 else "insuff")
            row["tier"] = tier_of(row)
            rows.append(row)
            detail = (
                f"  trades={row['trades']} wr={row['win_rate']} avg_r={row['avg_r']} "
                f"tag={row['confidence_tag']} tier={row['tier']}"
            )
            print(detail, flush=True)
            log_lines.append(detail)
        except Exception as exc:
            err = f"  ERROR: {exc}"
            print(err, flush=True)
            log_lines.append(err)
            log_lines.append(traceback.format_exc())
            rows.append(
                {
                    "symbol": sym,
                    "trades": 0,
                    "months": MONTHS,
                    "entries_per_month": 0.0,
                    "win_rate": None,
                    "avg_r": None,
                    "total_r": 0.0,
                    "profitable_month_pct": None,
                    "median_profit_month_hkd": None,
                    "profit_per_month_hkd": None,
                    "confidence": "error",
                    "confidence_tag": "error",
                    "tier": "B",
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # portfolio
    valid = df[df["trades"].fillna(0) >= 5]
    a = df[df["tier"] == "A"].sort_values(
        ["avg_r", "win_rate", "trades"], ascending=[False, False, False]
    )
    b = df[df["tier"] != "A"].sort_values(
        ["avg_r", "win_rate", "trades"], ascending=[False, False, False], na_position="last"
    )

    total_trades = int(df["trades"].fillna(0).sum())
    # reconstruct portfolio WR approx via weighted — we only have per-symbol avg
    # use trade-weighted avg_r / wr where available
    tw = df[df["trades"].fillna(0) > 0].copy()
    if len(tw):
        w = tw["trades"].astype(float)
        port_wr = float((tw["win_rate"].astype(float) * w).sum() / w.sum())
        port_ar = float((tw["avg_r"].astype(float) * w).sum() / w.sum())
        port_epm = float(tw["entries_per_month"].astype(float).sum())
        port_ppm = float(tw["profit_per_month_hkd"].astype(float).sum())
    else:
        port_wr = port_ar = port_epm = port_ppm = 0.0

    a_epm = float(a["entries_per_month"].fillna(0).sum()) if len(a) else 0.0
    # Prior wr_lift: strong+bias>=18+rr>=1.2 coverage ~27.5% of baseline enterable
    # strong_tier ~38.7%. Use 0.275 as Confirm-strict proxy on A-only hist entries.
    STRICT_FRAC = 0.275
    # first-2h haircut: assume ~40-50% of daily signals occur/usable in first 2h → use 0.45
    FIRST2H_FRAC = 0.45

    lines = []
    lines.append(
        f"Watchlist historical validation CORE-37 (simplified swing, NOT live guarantee)"
    )
    lines.append(f"list={len(symbols)} period={PERIOD} horizon={HORIZON}d risk_1R={RISK}HKD months={MONTHS}")
    lines.append("")
    lines.append("=== PORTFOLIO ===")
    lines.append(f"trades: {total_trades}")
    lines.append(f"months: {MONTHS}")
    lines.append(f"entries_per_month_full37: {port_epm}")
    lines.append(f"win_rate_trade_weighted: {port_wr}")
    lines.append(f"avg_r_trade_weighted: {port_ar}")
    lines.append(f"profit_per_month_hkd_sum: {port_ppm}")
    lines.append("")
    lines.append("=== TIERS ===")
    lines.append(f"A_count: {len(a)}")
    lines.append(f"B_count: {len(b)}")
    lines.append(f"A_entries_per_month_sum: {a_epm}")
    lines.append(
        f"est_Confirm_strict_A_pm (x{STRICT_FRAC}): {a_epm * STRICT_FRAC:.2f}"
    )
    lines.append(
        f"est_A+strict+first2h_pm (x{STRICT_FRAC}x{FIRST2H_FRAC}): {a_epm * STRICT_FRAC * FIRST2H_FRAC:.2f}"
    )
    lines.append(
        f"est_full37_strict_pm (x{STRICT_FRAC}): {port_epm * STRICT_FRAC:.2f}"
    )
    lines.append(
        f"est_full37_strict+first2h_pm: {port_epm * STRICT_FRAC * FIRST2H_FRAC:.2f}"
    )
    lines.append("")
    lines.append("=== A-TIER (hist-usable high WR) ===")
    if len(a):
        lines.append(
            a[
                [
                    "symbol",
                    "trades",
                    "confidence_tag",
                    "entries_per_month",
                    "win_rate",
                    "avg_r",
                    "profit_per_month_hkd",
                    "tier",
                ]
            ].to_string(index=False)
        )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("=== B-TIER (watch only) ===")
    if len(b):
        cols = [
            "symbol",
            "trades",
            "confidence_tag",
            "entries_per_month",
            "win_rate",
            "avg_r",
            "profit_per_month_hkd",
            "tier",
        ]
        lines.append(b[[c for c in cols if c in b.columns]].to_string(index=False))
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("=== WORKABLE CHECK ===")
    ok_n = int((valid["confidence_tag"] == "ok").sum()) if len(valid) else 0
    wr45 = int(((valid["win_rate"] >= 45) & (valid["trades"] >= 5)).sum()) if len(valid) else 0
    arpos = int(((valid["avg_r"] > 0) & (valid["trades"] >= 5)).sum()) if len(valid) else 0
    arneg = int(((valid["avg_r"] <= 0) & (valid["trades"] >= 5)).sum()) if len(valid) else 0
    lines.append(f"symbols_with_trades>=5: {len(valid)}/{len(symbols)}")
    lines.append(f"symbols_confidence_ok>=10trades: {ok_n}")
    lines.append(f"among_valid win_rate>=45%: {wr45}")
    lines.append(f"among_valid avg_r>0: {arpos}")
    lines.append(f"among_valid avg_r<=0: {arneg}")
    workable = len(a) >= 8 and a_epm >= 4
    lines.append(f"WORKABLE_A_POOL: {workable}")
    lines.append(
        "NOTE: Confirm live filters (lights + RR + first2h) cut hist enterable sharply; "
        "use STRICT_FRAC from prior wr_lift strong+bias>=18+rr>=1.2 ≈27.5%."
    )

    report = "\n".join(lines) + "\n"
    OUT_REPORT.write_text(report, encoding="utf-8")
    OUT_LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(report)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
