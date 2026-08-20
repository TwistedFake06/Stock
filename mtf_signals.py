"""
Multi-timeframe + structure helpers for short-term swing (0–4 weeks):

- Weekly trend filter (big direction)
- 1H entry trigger (when to place limit)
- ADX regime (trend vs chop)
- Fibonacci retracement bands for entry zone

Uses yfinance via stock_service.fetch_history. Degrades gracefully offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from indicators import enrich
from stock_service import fetch_history, normalize_symbol


@dataclass
class WeeklyFilterReport:
    label: str = "—"  # 周线多头 | 周线中性 | 周线空头
    allow_long: bool = True
    score: float = 50.0
    close: float | None = None
    sma20: float | None = None
    sma50: float | None = None
    summary: str = ""
    available: bool = False


@dataclass
class H1TriggerReport:
    label: str = "—"  # 可掛單 | 等回踩 | 1H轉強待入 | 1H偏空 | 已遠離 | 无数据
    ready: bool = False  # True → can place limit now
    score: float = 50.0
    ema9: float | None = None
    ema21: float | None = None
    last: float | None = None
    in_entry_zone: bool = False
    bullets: list[str] = field(default_factory=list)
    summary: str = ""
    available: bool = False


@dataclass
class ADXReport:
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    label: str = "—"  # 强趋势 | 有趋势 | 震荡 | 无数据
    trending: bool = False
    score: float = 50.0  # higher = better for trend-follow longs when +DI>-DI
    summary: str = ""
    available: bool = False


@dataclass
class FibReport:
    swing_high: float | None = None
    swing_low: float | None = None
    level_382: float | None = None
    level_500: float | None = None
    level_618: float | None = None
    zone_low: float | None = None  # preferred buy band low (near 0.5–0.618)
    zone_high: float | None = None
    label: str = "—"
    summary: str = ""
    available: bool = False


def analyze_weekly_filter(symbol: str) -> WeeklyFilterReport:
    """
    Weekly SMA20/50 filter for short-term longs.
    周线空头 → 不鼓励做多波段（最多试仓由上层决定）.
    """
    df = fetch_history(normalize_symbol(symbol), period="3y", interval="1wk")
    if df.empty or len(df) < 25 or "Close" not in df.columns:
        return WeeklyFilterReport(
            summary="周线数据不足，跳过周线过滤。",
            allow_long=True,
            available=False,
        )
    close = df["Close"].astype(float)
    last = float(close.iloc[-1])
    sma20 = float(close.tail(20).mean()) if len(close) >= 20 else None
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    score = 50.0
    if sma20 is not None:
        if last > sma20:
            score += 15
        else:
            score -= 18
    if sma50 is not None:
        if last > sma50:
            score += 12
        else:
            score -= 12
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            score += 8
        else:
            score -= 8
    score = float(max(0.0, min(100.0, score)))
    if score >= 62:
        label = "周线多头"
        allow = True
    elif score <= 42:
        label = "周线空头"
        allow = False
    else:
        label = "周线中性"
        allow = True
    summary = f"**{label}**（{score:.0f}）· 周收 {last:.2f}"
    if sma20:
        summary += f" · SMA20 {sma20:.2f}"
    if sma50:
        summary += f" · SMA50 {sma50:.2f}"
    if not allow:
        summary += " · 大周期空头：短线做多降级/暂缓"
    summary += "。"
    return WeeklyFilterReport(
        label=label,
        allow_long=allow,
        score=round(score, 1),
        close=round(last, 4),
        sma20=round(sma20, 4) if sma20 else None,
        sma50=round(sma50, 4) if sma50 else None,
        summary=summary,
        available=True,
    )


def analyze_adx(df: pd.DataFrame) -> ADXReport:
    """Daily ADX regime from enriched OHLCV."""
    if df is None or df.empty:
        return ADXReport(summary="无数据。")
    data = df if "ADX" in df.columns else enrich(df)
    if "ADX" not in data.columns:
        return ADXReport(summary="无法计算 ADX。")
    adx = data["ADX"].iloc[-1]
    pdi = data["PLUS_DI"].iloc[-1] if "PLUS_DI" in data.columns else np.nan
    mdi = data["MINUS_DI"].iloc[-1] if "MINUS_DI" in data.columns else np.nan
    try:
        adx_v = float(adx) if pd.notna(adx) else None
        pdi_v = float(pdi) if pd.notna(pdi) else None
        mdi_v = float(mdi) if pd.notna(mdi) else None
    except Exception:
        return ADXReport(summary="ADX 无效。")
    if adx_v is None:
        return ADXReport(summary="ADX 无效。")

    score = 50.0
    if adx_v >= 30:
        label = "强趋势"
        trending = True
        score += 15
    elif adx_v >= 20:
        label = "有趋势"
        trending = True
        score += 8
    else:
        label = "震荡"
        trending = False
        score -= 12

    if pdi_v is not None and mdi_v is not None:
        if pdi_v > mdi_v:
            score += 10
            di_note = f"+DI {pdi_v:.0f} > -DI {mdi_v:.0f}（多头方向）"
        else:
            score -= 12
            di_note = f"+DI {pdi_v:.0f} < -DI {mdi_v:.0f}（空头方向）"
    else:
        di_note = ""

    score = float(max(0.0, min(100.0, score)))
    summary = f"ADX **{label}**（ADX={adx_v:.0f}·分{score:.0f}）"
    if di_note:
        summary += f" · {di_note}"
    if not trending:
        summary += " · 震荡市慎追突破"
    summary += "。"
    return ADXReport(
        adx=round(adx_v, 2),
        plus_di=round(pdi_v, 2) if pdi_v is not None else None,
        minus_di=round(mdi_v, 2) if mdi_v is not None else None,
        label=label,
        trending=trending,
        score=round(score, 1),
        summary=summary,
        available=True,
    )


def analyze_fib_levels(df: pd.DataFrame, lookback: int = 60) -> FibReport:
    """
    Swing high/low Fib retracement. Prefer buy zone between 0.382–0.618
    (center 0.5–0.618) of the last impulse up.
    """
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return FibReport(summary="数据不足，无 Fib。")
    work = df.tail(max(lookback, 30)).copy()
    if len(work) < 20:
        return FibReport(summary="K线不足。")
    # drop last incomplete daily if possible
    if len(work) >= 25:
        work = work.iloc[:-1]
    hi = float(work["High"].astype(float).max())
    lo = float(work["Low"].astype(float).min())
    if hi <= lo:
        return FibReport(summary="高低点无效。")
    span = hi - lo
    f382 = hi - 0.382 * span
    f500 = hi - 0.500 * span
    f618 = hi - 0.618 * span
    # buy band: 0.5 to 0.618 retracement (deeper value) with slight pad
    zone_lo = f618
    zone_hi = f382  # shallower end of pullback band
    if zone_lo > zone_hi:
        zone_lo, zone_hi = zone_hi, zone_lo
    last = float(work["Close"].astype(float).iloc[-1])
    if zone_lo <= last <= zone_hi:
        label = "价在Fib回撤区"
    elif last > zone_hi:
        label = "价在Fib上方(偏伸展)"
    else:
        label = "价在Fib下方(更深)"
    summary = (
        f"Fib **{label}** · 高{hi:.2f}/低{lo:.2f} · "
        f"0.382={f382:.2f} · 0.5={f500:.2f} · 0.618={f618:.2f} · "
        f"回撤买区≈{zone_lo:.2f}–{zone_hi:.2f}"
    )
    return FibReport(
        swing_high=round(hi, 4),
        swing_low=round(lo, 4),
        level_382=round(f382, 4),
        level_500=round(f500, 4),
        level_618=round(f618, 4),
        zone_low=round(zone_lo, 4),
        zone_high=round(zone_hi, 4),
        label=label,
        summary=summary,
        available=True,
    )


def merge_entry_with_fib(
    entry_low: float | None,
    entry_high: float | None,
    fib: FibReport | None,
    last: float | None = None,
) -> tuple[float | None, float | None, str]:
    """
    Blend structure entry zone with Fib 0.38–0.62 band (intersection preferred).
    Returns (new_low, new_high, note).
    """
    if entry_low is None or entry_high is None:
        if fib and fib.available and fib.zone_low and fib.zone_high:
            return fib.zone_low, fib.zone_high, "仅用 Fib 回撤区"
        return entry_low, entry_high, ""
    if not fib or not fib.available or fib.zone_low is None or fib.zone_high is None:
        return entry_low, entry_high, ""

    el, eh = float(entry_low), float(entry_high)
    fl, fh = float(fib.zone_low), float(fib.zone_high)
    # intersection
    lo = max(el, fl)
    hi = min(eh, fh)
    if lo < hi and (hi - lo) / max(eh - el, 1e-9) >= 0.25:
        return round(lo, 4), round(hi, 4), "结构区 ∩ Fib回撤区"
    # no good intersection: soft blend toward fib mid
    mid_s = (el + eh) / 2
    mid_f = (fl + fh) / 2
    mid = 0.55 * mid_s + 0.45 * mid_f
    width = max((eh - el) * 0.9, (fh - fl) * 0.5)
    return (
        round(mid - width / 2, 4),
        round(mid + width / 2, 4),
        "结构区与Fib融合（无清晰交集）",
    )


def analyze_h1_trigger(
    symbol: str,
    entry_low: float | None,
    entry_high: float | None,
) -> H1TriggerReport:
    """
    1H timing: EMA9/21 + price vs daily entry zone.
    """
    df = fetch_history(normalize_symbol(symbol), period="60d", interval="1h")
    if df.empty or len(df) < 30 or "Close" not in df.columns:
        return H1TriggerReport(
            summary="1小时数据不足（网络或标的限制）。",
            available=False,
        )
    close = df["Close"].astype(float)
    last = float(close.iloc[-1])
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema9_prev = float(close.ewm(span=9, adjust=False).mean().iloc[-2])
    ema21_prev = float(close.ewm(span=21, adjust=False).mean().iloc[-2])

    bullets: list[str] = []
    score = 50.0
    in_zone = False
    if entry_low is not None and entry_high is not None:
        el, eh = float(entry_low), float(entry_high)
        pad = (eh - el) * 0.05 + 1e-9
        in_zone = (el - pad) <= last <= (eh + pad)
        bullets.append(
            f"日线入场区 {el:.2f}–{eh:.2f} · 1H价 {last:.2f} · "
            + ("在区内" if in_zone else "在区外")
        )
        if last > eh * 1.02:
            score -= 15
            bullets.append("1H价明显高于入场区：不追")
        elif last < el * 0.98:
            score -= 5
            bullets.append("1H价在区下：等回到区或确认止跌")

    bull_cross = ema9_prev <= ema21_prev and ema9 > ema21
    bear_struct = ema9 < ema21 and last < ema21
    bull_struct = ema9 > ema21 and last > ema9

    if bull_cross:
        score += 14
        bullets.append("1H EMA9 上穿 EMA21（短线转强）")
    if bull_struct:
        score += 10
        bullets.append("1H 多头排列（价>EMA9>EMA21）")
    if bear_struct:
        score -= 14
        bullets.append("1H 偏空（价与EMA9在EMA21下）")

    # volume on 1H if present
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        ma = float(vol.tail(20).mean() or 0)
        if ma > 0:
            ratio = float(vol.iloc[-1] / ma)
            bullets.append(f"1H量比≈{ratio:.2f}x")
            if ratio >= 1.4 and bull_struct:
                score += 6

    score = float(max(0.0, min(100.0, score)))
    ready = False
    if bear_struct and not in_zone:
        label = "1H偏空"
    elif entry_low and entry_high and last > float(entry_high) * 1.025:
        label = "已遠離"
    elif in_zone and (bull_struct or bull_cross) and not bear_struct:
        label = "可掛單"
        ready = True
        score = max(score, 68)
    elif in_zone and not bear_struct:
        label = "等回踩"
        bullets.append("价在入场区但1H未转强：可挂限价等，不追阳线")
        score = max(score, 55)
    elif bull_cross or bull_struct:
        label = "1H轉強待入"
        bullets.append("1H已转强：等价格回到日线入场区再挂")
    else:
        label = "等回踩"

    summary = (
        f"1H触发 **{label}**（{score:.0f}）· "
        f"EMA9={ema9:.2f} EMA21={ema21:.2f} · 价{last:.2f}"
    )
    if ready:
        summary += " · 可按日线区限价"
    summary += "。"
    return H1TriggerReport(
        label=label,
        ready=ready,
        score=round(score, 1),
        ema9=round(ema9, 4),
        ema21=round(ema21, 4),
        last=round(last, 4),
        in_entry_zone=in_zone,
        bullets=bullets,
        summary=summary,
        available=True,
    )


def mtf_bundle(
    symbol: str,
    daily_df: pd.DataFrame,
    entry_low: float | None,
    entry_high: float | None,
    *,
    include_h1: bool = True,
) -> dict[str, Any]:
    """Weekly + ADX + Fib, with optional 1H timing confirmation."""
    weekly = analyze_weekly_filter(symbol)
    adx = analyze_adx(daily_df)
    fib = analyze_fib_levels(daily_df)
    # refine zone then 1H
    new_lo, new_hi, fib_note = merge_entry_with_fib(entry_low, entry_high, fib)
    h1 = analyze_h1_trigger(symbol, new_lo, new_hi) if include_h1 else None
    return {
        "weekly": weekly,
        "adx": adx,
        "fib": fib,
        "fib_note": fib_note,
        "entry_low": new_lo,
        "entry_high": new_hi,
        "h1": h1,
    }
