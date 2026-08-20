"""
Apply current short-term swing logic on historical daily data for each watchlist symbol.

For each past bar (warm-up after), if simplified "enterable" rules fire:
  - enter next bar open (or close if no open)
  - stop / target from then-available structure (same as SOP style)
  - walk forward up to horizon bars (10 = 0–2w, 20 = 2–4w)
  - success = hit target before stop; fail = hit stop first; timeout = neither

Usage:
  .venv\\Scripts\\python.exe scripts\\backtest_watchlist_swing.py
  .venv\\Scripts\\python.exe scripts\\backtest_watchlist_swing.py --horizon 10
  .venv\\Scripts\\python.exe scripts\\backtest_watchlist_swing.py --horizon both
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import analyze_bias
from entry_targets import analyze_entry, analyze_targets
from exit_plan import DEFAULT_SLIP_PCT, apply_long_slippage
from indicators import enrich
from stock_service import DEFAULT_WATCHLIST, fetch_history, normalize_symbol

DEFAULT_LIST = ROOT / "watchlist_scan.txt"


@dataclass
class HistTrade:
    symbol: str
    entry_date: str
    exit_date: str
    entry: float
    stop: float
    target: float
    exit_px: float
    result: str  # win | loss | timeout
    hold_days: int
    r_mult: float
    opportunity: str
    bias: str
    horizon: int


def summarize_history(
    trades: list[HistTrade],
    *,
    risk_hkd: float = 500.0,
    observed_months: int | None = None,
) -> dict[str, float | int | str]:
    """Summarize historical entries for the watchlist KPI panel."""
    if not trades:
        return {
            "trades": 0,
            "months": 0,
            "entries_per_month": 0.0,
            "win_rate": None,
            "avg_r": None,
            "total_r": 0.0,
            "profitable_month_pct": None,
            "median_profit_month_hkd": None,
            "profit_per_month_hkd": None,
            "confidence": "样本不足",
        }

    dates = pd.to_datetime([trade.entry_date for trade in trades])
    month_index = pd.period_range(dates.min().to_period("M"), dates.max().to_period("M"), freq="M")
    monthly_r = (
        pd.Series([float(trade.r_mult) for trade in trades], index=dates.to_period("M"))
        .groupby(level=0)
        .sum()
        .reindex(month_index, fill_value=0.0)
    )
    months = max(len(month_index), int(observed_months or 0), 1)
    if months > len(monthly_r):
        monthly_r = pd.concat(
            [monthly_r, pd.Series([0.0] * (months - len(monthly_r)), dtype=float)],
            ignore_index=True,
        )
    wins = sum(1 for trade in trades if trade.result == "win")
    total_r = sum(float(trade.r_mult) for trade in trades)
    confidence = "可参考" if len(trades) >= 10 else "低样本" if len(trades) >= 5 else "样本不足"
    return {
        "trades": len(trades),
        "months": months,
        "entries_per_month": len(trades) / months,
        "win_rate": 100.0 * wins / len(trades),
        "avg_r": total_r / len(trades),
        "total_r": total_r,
        "profitable_month_pct": 100.0 * float((monthly_r > 0).mean()),
        "median_profit_month_hkd": float(monthly_r.median()) * risk_hkd,
        "profit_per_month_hkd": total_r * risk_hkd / months,
        "confidence": confidence,
    }


def load_symbols(path: Path) -> list[str]:
    raw: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                raw.append(line)
    else:
        raw = list(DEFAULT_WATCHLIST)
    out, seen = [], set()
    for s in raw:
        n = normalize_symbol(s)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _enterable(entry_opp: str, bias_label: str, bias_score: float, rr: float | None) -> bool:
    """Mirror short-term spirit: allow try/enter-ish without full live edge stack."""
    if entry_opp in ("偏空回避",):
        return False
    if "强烈看空" in bias_label:
        return False
    if entry_opp == "不宜追高":
        return False
    if rr is not None and rr < 1.0:
        return False
    if entry_opp in ("较佳入场", "可关注") and bias_score >= -10:
        return True
    if entry_opp == "观望" and bias_score >= 20 and (rr is None or rr >= 1.2):
        return True
    return False


def simulate_forward(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    dates: list,
    i_entry: int,
    entry: float,
    stop: float,
    target: float,
    horizon: int,
) -> tuple[str, int, float, str, float]:
    """
    Returns result, hold_days, r_mult, exit_date, exit_px
    """
    risk = entry - stop
    if risk <= 0:
        return "invalid", 0, 0.0, str(dates[i_entry]), entry

    end = min(len(close) - 1, i_entry + horizon)
    for j in range(i_entry + 1, end + 1):
        # conservative: stop checked before target same bar
        if low[j] <= stop:
            r = (stop - entry) / risk  # ~ -1
            # slippage on stop
            fill = stop * (1 - DEFAULT_SLIP_PCT)
            r = (fill - entry) / risk
            return "loss", j - i_entry, round(r, 3), str(dates[j])[:10], round(fill, 4)
        if high[j] >= target:
            fill = target * (1 - DEFAULT_SLIP_PCT)
            r = (fill - entry) / risk
            return "win", j - i_entry, round(r, 3), str(dates[j])[:10], round(fill, 4)

    # timeout: exit at last close
    px = float(close[end])
    r = (px - entry) / risk
    return "timeout", end - i_entry, round(r, 3), str(dates[end])[:10], round(px, 4)


def backtest_symbol(
    symbol: str,
    *,
    period: str = "2y",
    horizon: int = 10,
    step: int = 1,
) -> list[HistTrade]:
    df = fetch_history(symbol, period=period, interval="1d")
    if df is None or df.empty or len(df) < 80:
        return []

    df = df.reset_index(drop=True)
    if "Date" not in df.columns:
        return []

    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    close = df["Close"].astype(float).values
    open_ = df["Open"].astype(float).values if "Open" in df.columns else close.copy()
    dates = [pd.Timestamp(d) for d in df["Date"]]

    trades: list[HistTrade] = []
    i = 60  # warm-up
    n = len(df)
    # leave room for forward path
    while i < n - horizon - 2:
        hist = df.iloc[: i + 1].copy()
        try:
            en = enrich(hist)
            entry_rep = analyze_entry(en)
            bias = analyze_bias(en)
            targets = analyze_targets(en, info={}, entry=entry_rep)
        except Exception:
            i += step
            continue

        e_lo = entry_rep.suggested_entry_low
        e_hi = entry_rep.suggested_entry_high
        stop = entry_rep.stop_loss
        last = float(close[i])

        # targets by horizon
        t_ultra = getattr(getattr(targets, "ultra", None), "bull_target", None)
        t_short = getattr(getattr(targets, "short", None), "bull_target", None)
        t_med = getattr(getattr(targets, "medium", None), "bull_target", None)
        if horizon <= 12:
            target = t_ultra or t_short
        else:
            target = t_short or t_med or t_ultra

        if stop is None or target is None or e_lo is None or e_hi is None:
            i += step
            continue

        mid = (float(e_lo) + float(e_hi)) / 2.0
        if mid <= float(stop) or float(target) <= mid:
            i += step
            continue

        rr = (float(target) - mid) / (mid - float(stop))
        # price should be near zone (not far chase)
        if last > float(e_hi) * 1.03:
            i += step
            continue

        if not _enterable(entry_rep.opportunity, bias.bias, bias.score, rr):
            i += step
            continue

        # Enter next bar open (more realistic)
        i_ent = i + 1
        if i_ent >= n - 1:
            break
        entry_px = float(open_[i_ent]) if open_[i_ent] > 0 else float(close[i_ent])
        # scale stop/target by same distances from mid, anchored to actual entry
        risk = mid - float(stop)
        reward = float(target) - mid
        stop_px = entry_px - risk
        target_px = entry_px + reward
        if stop_px <= 0 or target_px <= entry_px:
            i += step
            continue

        result, hold, r_mult, exit_d, exit_px = simulate_forward(
            high, low, close, dates, i_ent, entry_px, stop_px, target_px, horizon
        )
        if result == "invalid":
            i += step
            continue

        trades.append(
            HistTrade(
                symbol=symbol,
                entry_date=str(dates[i_ent])[:10],
                exit_date=exit_d,
                entry=round(entry_px, 4),
                stop=round(stop_px, 4),
                target=round(target_px, 4),
                exit_px=exit_px,
                result=result,
                hold_days=hold,
                r_mult=r_mult,
                opportunity=entry_rep.opportunity,
                bias=bias.bias,
                horizon=horizon,
            )
        )
        # no overlap: jump to after exit
        i = i_ent + max(hold, 1) + 1

    return trades


def summarize(trades: list[HistTrade], title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)
    if not trades:
        print("  (无交易样本)")
        return
    n = len(trades)
    wins = [t for t in trades if t.result == "win"]
    losses = [t for t in trades if t.result == "loss"]
    timeouts = [t for t in trades if t.result == "timeout"]
    wr = 100.0 * len(wins) / n
    avg_r = sum(t.r_mult for t in trades) / n
    avg_hold = sum(t.hold_days for t in trades) / n
    avg_hold_w = sum(t.hold_days for t in wins) / len(wins) if wins else 0
    avg_hold_l = sum(t.hold_days for t in losses) / len(losses) if losses else 0
    print(
        f"  交易数 {n} · 成功(先到目标) {len(wins)} · 失败(先止蚀) {len(losses)} · "
        f"超时 {len(timeouts)}"
    )
    print(
        f"  胜率 {wr:.1f}% · 平均R {avg_r:+.3f} · 平均持仓 {avg_hold:.1f} 交易日"
        f"（赢{avg_hold_w:.1f} / 亏{avg_hold_l:.1f}）"
    )
    # expectancy proxy
    print(f"  期望≈平均R（含超时按收盘算） {avg_r:+.3f}R / 笔")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_LIST))
    ap.add_argument("--period", default="2y", help="history length")
    ap.add_argument(
        "--horizon",
        default="both",
        choices=["10", "20", "both"],
        help="10=0-2w, 20=2-4w, both=run both",
    )
    ap.add_argument("--max-examples", type=int, default=3, help="print N example trades per symbol")
    args = ap.parse_args()

    symbols = load_symbols(Path(args.file))
    horizons = [10, 20] if args.horizon == "both" else [int(args.horizon)]

    print(
        f"历史回测 · {len(symbols)} 只 · period={args.period} · "
        f"horizons={horizons} · 规则≈当前短线入场(结构+方向+R:R≥1，无盘中IV/新闻)"
    )
    print("成功=持仓窗口内先触目标；失败=先触止蚀；超时=到期收盘离场")
    print("-" * 88)

    all_by_h: dict[int, list[HistTrade]] = {h: [] for h in horizons}

    for sym in symbols:
        print(f"\n>>> {sym}")
        for h in horizons:
            try:
                trades = backtest_symbol(sym, period=args.period, horizon=h)
            except Exception as exc:
                print(f"  ERROR {h}d: {exc}")
                continue
            all_by_h[h].extend(trades)
            n = len(trades)
            if n == 0:
                print(f"  周期{h}日: 无符合条件的历史入场")
                continue
            wins = sum(1 for t in trades if t.result == "win")
            losses = sum(1 for t in trades if t.result == "loss")
            timeouts = sum(1 for t in trades if t.result == "timeout")
            wr = 100.0 * wins / n
            avg_r = sum(t.r_mult for t in trades) / n
            avg_d = sum(t.hold_days for t in trades) / n
            label = "0–2周" if h <= 12 else "2–4周"
            print(
                f"  {label}({h}日): 笔数={n} 成功={wins} 失败={losses} 超时={timeouts} "
                f"胜率={wr:.0f}% 均R={avg_r:+.2f} 均持仓={avg_d:.1f}日"
            )
            # examples: mix win/loss
            shown = 0
            for t in trades:
                if shown >= args.max_examples:
                    break
                tag = {"win": "成功", "loss": "失败", "timeout": "超时"}.get(t.result, t.result)
                print(
                    f"    · {tag} {t.entry_date}→{t.exit_date} "
                    f"入{t.entry:.2f} 止{t.stop:.2f} 标{t.target:.2f} "
                    f"出{t.exit_px:.2f} 持{t.hold_days}日 R={t.r_mult:+.2f} "
                    f"[{t.opportunity}/{t.bias}]"
                )
                shown += 1

    for h in horizons:
        label = "0–2周（约10交易日）" if h <= 12 else "2–4周（约20交易日）"
        summarize(all_by_h[h], f"合计 · {label}")

    # per-symbol table for primary horizon 10
    h0 = horizons[0]
    print()
    print("=" * 88)
    print(f"分股票汇总 · horizon={h0} 交易日")
    print("=" * 88)
    print(f"{'代码':8} {'笔数':>4} {'成功':>4} {'失败':>4} {'超时':>4} {'胜率%':>7} {'均R':>7} {'均持仓日':>8}")
    by_sym: dict[str, list[HistTrade]] = {}
    for t in all_by_h[h0]:
        by_sym.setdefault(t.symbol, []).append(t)
    for sym in symbols:
        ts = by_sym.get(sym, [])
        if not ts:
            print(f"{sym:8} {'0':>4} {'—':>4} {'—':>4} {'—':>4} {'—':>7} {'—':>7} {'—':>8}")
            continue
        w = sum(1 for t in ts if t.result == "win")
        l = sum(1 for t in ts if t.result == "loss")
        o = sum(1 for t in ts if t.result == "timeout")
        n = len(ts)
        wr = 100.0 * w / n
        ar = sum(t.r_mult for t in ts) / n
        ad = sum(t.hold_days for t in ts) / n
        print(f"{sym:8} {n:4d} {w:4d} {l:4d} {o:4d} {wr:6.1f}% {ar:+7.2f} {ad:8.1f}")

    print()
    print("说明：")
    print("  · 用过去约2年日线，按「当时」可见数据做入场评估（结构+方向+R:R），再向前走目标/止蚀")
    print("  · 未逐日重跑完整线上SOP（板块实时/IV/新闻），故接近但不等于现在App每一项")
    print("  · 成功/失败含约0.15%单边滑点假设；超时=窗口内未触目标也未止蚀，按收盘离场算R")
    print("  · 仅供验证方法，不构成投资建议")


if __name__ == "__main__":
    main()
