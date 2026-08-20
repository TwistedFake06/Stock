"""Estimate monthly entry opportunities using the current core three-light SOP."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import analyze_bias
from entry_targets import analyze_entry, analyze_targets
from exit_plan import DEFAULT_SLIP_PCT, apply_long_slippage
from indicators import enrich
from stock_service import fetch_history, normalize_symbol
from trade_sop import (
    MODE_THRESHOLDS,
    PATH_LOOKBACK_DEFAULT,
    decide_three_lights,
    plan_limit_from_zone,
    resolve_path_win_rate,
)

WATCHLIST = ROOT / "watchlist_scan.txt"
STEP = 5
WARMUP = 100


def load_symbols() -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for line in WATCHLIST.read_text(encoding="utf-8").splitlines():
        symbol = normalize_symbol(line.split("#", 1)[0].strip())
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def weekly_allows_long(history: pd.DataFrame) -> bool:
    dated = history[["Date", "Close"]].dropna().copy()
    dated["Date"] = pd.to_datetime(dated["Date"], utc=True).dt.tz_localize(None)
    weekly = dated.set_index("Date")["Close"].resample("W-FRI").last().dropna()
    if len(weekly) < 25:
        return True
    last = float(weekly.iloc[-1])
    sma20 = float(weekly.tail(20).mean())
    sma50 = float(weekly.tail(50).mean()) if len(weekly) >= 50 else None
    score = 50.0 + (15 if last > sma20 else -18)
    if sma50 is not None:
        score += 12 if last > sma50 else -12
        score += 8 if sma20 > sma50 else -8
    return score > 42


def classify(history: pd.DataFrame, mode: str) -> str:
    enriched = enrich(history.copy())
    entry = analyze_entry(enriched)
    bias = analyze_bias(enriched)
    targets = analyze_targets(enriched, info={}, entry=entry)
    entry_low = entry.suggested_entry_low
    entry_high = entry.suggested_entry_high
    stop = entry.stop_loss
    target = getattr(getattr(targets, "ultra", None), "bull_target", None)
    target = target or getattr(getattr(targets, "short", None), "bull_target", None)
    last = float(enriched["Close"].iloc[-1])
    plan = plan_limit_from_zone(entry_low, entry_high)
    if plan is None:
        plan = last
    if stop is None or target is None or plan <= float(stop) or float(target) <= plan:
        return "暫緩觀望"

    risk = plan - float(stop)
    reward = float(target) - plan
    wr, samples, _ = resolve_path_win_rate(
        enriched["Close"].astype(float),
        risk,
        reward,
        primary_horizon=10,
        lookback=PATH_LOOKBACK_DEFAULT,
        high=enriched["High"].astype(float),
        low=enriched["Low"].astype(float),
        ref_entry=plan,
    )
    slip = apply_long_slippage(
        plan,
        float(stop),
        float(target),
        win_rate_pct=wr,
        slip_pct=DEFAULT_SLIP_PCT,
    )
    result = decide_three_lights(
        thr=MODE_THRESHOLDS[mode],
        last=last,
        entry_low=entry_low,
        entry_high=entry_high,
        entry_plan=plan,
        stop=float(stop),
        target=float(target),
        wr=wr,
        wr_samples=samples,
        rr_net=slip.rr_net,
        rr_paper=reward / risk,
        price_far_chase=bool(entry_high and last > float(entry_high) * 1.03),
        entry_opp=entry.opportunity,
        bias_label=bias.bias,
        bias_score=float(bias.score),
        weekly_allow_long=weekly_allows_long(history),
        enforce_time_window=False,
    )
    return str(result["verdict"])


def summarize(events: pd.DataFrame, months: pd.PeriodIndex, mode: str) -> None:
    mode_events = events[events["mode"] == mode].copy()
    monthly = mode_events.groupby("month").size().reindex(months, fill_value=0)
    full = mode_events[mode_events["verdict"] == "可以入場"]
    full_monthly = full.groupby("month").size().reindex(months, fill_value=0)
    active_weeks = (
        mode_events.groupby("month")["week"].nunique().reindex(months, fill_value=0)
    )
    label = "A 防守版" if mode == "defensive" else "B 半風險版"
    print(f"\n=== {label} ===")
    print(
        f"每月訊號 mean={monthly.mean():.2f}, median={monthly.median():.1f}; "
        f"全綠 mean={full_monthly.mean():.2f}; "
        f"去除同周重複 mean={active_weeks.mean():.2f} 週/月"
    )
    print("最近 12 個月（訊號 / 可執行週）：")
    for month in months[-12:]:
        print(f"  {month}: {int(monthly[month])} / {int(active_weeks[month])}")


def main() -> None:
    symbols = load_symbols()
    modes = ("defensive", "aggressive")
    events: list[dict[str, object]] = []
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None
    print(f"掃描 {len(symbols)} 隻，2 年日線，每 {STEP} 個交易日評估一次…")
    for symbol in symbols:
        history = fetch_history(symbol, period="2y", interval="1d")
        if history.empty or len(history) < WARMUP + 20 or "Date" not in history:
            print(f"  {symbol}: 資料不足")
            continue
        history = history.reset_index(drop=True)
        dates = pd.to_datetime(history["Date"], utc=True).dt.tz_localize(None)
        first_date = dates.iloc[WARMUP] if first_date is None else min(first_date, dates.iloc[WARMUP])
        last_date = dates.iloc[-1] if last_date is None else max(last_date, dates.iloc[-1])
        previous = {mode: "" for mode in modes}
        counts = {mode: 0 for mode in modes}
        for index in range(WARMUP, len(history), STEP):
            snapshot = history.iloc[: index + 1].copy()
            event_date = dates.iloc[index]
            for mode in modes:
                verdict = classify(snapshot, mode)
                enterable = verdict in ("可以入場", "可以試倉")
                was_enterable = previous[mode] in ("可以入場", "可以試倉")
                if enterable and not was_enterable:
                    events.append(
                        {
                            "symbol": symbol,
                            "date": event_date,
                            "month": event_date.to_period("M"),
                            "week": event_date.to_period("W"),
                            "mode": mode,
                            "verdict": verdict,
                        }
                    )
                    counts[mode] += 1
                previous[mode] = verdict
        print(
            f"  {symbol}: A={counts['defensive']} · B={counts['aggressive']} 新機會"
        )

    if first_date is None or last_date is None:
        raise SystemExit("沒有足夠資料")
    months = pd.period_range(first_date.to_period("M"), last_date.to_period("M"), freq="M")
    frame = pd.DataFrame(events)
    if frame.empty:
        print("沒有符合條件的歷史機會")
        return
    for mode in modes:
        summarize(frame, months, mode)
    print("\n注意：這是核心三燈歷史估算，不含當時 1H、財報日曆及即時板塊資料。")


if __name__ == "__main__":
    main()