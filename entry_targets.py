"""
入场机会 + 超短/短/中期目标价分析。

超短：约 1 周内（≈3–5 个交易日）
短期：约 2 周–1 个月
中期：约 1–2 个月

方法综合：ATR 波动、支撑阻力、布林带、枢轴、动量外推、分析师目标价（若有）。
仅供学习参考，不构成投资建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from indicators import enrich


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _safe(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Entry opportunities
# ---------------------------------------------------------------------------

@dataclass
class EntrySignal:
    name: str
    side: str  # 做多机会 | 观望 | 谨慎/偏空
    score: float  # 0-10 quality
    urgency: str  # 高 | 中 | 低
    detail: str
    entry_zone_low: float | None = None
    entry_zone_high: float | None = None


@dataclass
class EntryReport:
    opportunity: str  # 较佳入场 | 可关注 | 观望 | 不宜追高 | 偏空回避
    score: float  # 0-100
    side_bias: str  # 偏多入场 | 中性 | 偏空
    current_price: float | None = None
    suggested_entry_low: float | None = None
    suggested_entry_high: float | None = None
    stop_loss: float | None = None
    risk_reward_short: float | None = None
    risk_reward_medium: float | None = None
    invalidation: str = ""
    signals: list[EntrySignal] = field(default_factory=list)
    checklist: list[str] = field(default_factory=list)
    summary: str = ""


def analyze_entry(df: pd.DataFrame) -> EntryReport:
    if df is None or df.empty or len(df) < 30:
        return EntryReport(
            opportunity="观望",
            score=0,
            side_bias="中性",
            summary="数据不足，无法评估入场机会。",
        )

    data = enrich(df)
    close = data["Close"].astype(float)
    last = float(close.iloc[-1])
    atr = _atr(data)
    atr_v = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else last * 0.02

    sma5 = _safe(data["SMA5"].iloc[-1]) if "SMA5" in data.columns else None
    sma20 = _safe(data["SMA20"].iloc[-1]) if "SMA20" in data.columns else None
    sma60 = _safe(data["SMA60"].iloc[-1]) if "SMA60" in data.columns else None
    rsi = _safe(data["RSI"].iloc[-1]) if "RSI" in data.columns else None
    macd = _safe(data["MACD"].iloc[-1]) if "MACD" in data.columns else None
    macd_sig = _safe(data["MACD_SIGNAL"].iloc[-1]) if "MACD_SIGNAL" in data.columns else None
    macd_hist = _safe(data["MACD_HIST"].iloc[-1]) if "MACD_HIST" in data.columns else None
    macd_hist_prev = (
        _safe(data["MACD_HIST"].iloc[-2]) if "MACD_HIST" in data.columns and len(data) > 1 else None
    )
    bb_u = _safe(data["BB_UPPER"].iloc[-1]) if "BB_UPPER" in data.columns else None
    bb_m = _safe(data["BB_MID"].iloc[-1]) if "BB_MID" in data.columns else None
    bb_l = _safe(data["BB_LOWER"].iloc[-1]) if "BB_LOWER" in data.columns else None

    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    recent_high = float(high.tail(20).max())
    recent_low = float(low.tail(20).min())

    vol = data["Volume"].astype(float) if "Volume" in data.columns else None
    vol_ratio = None
    if vol is not None and vol.tail(20).mean() > 0:
        vol_ratio = float(vol.iloc[-1] / vol.tail(20).mean())

    signals: list[EntrySignal] = []
    bull_pts = 0.0
    bear_pts = 0.0

    # 1) Trend context
    uptrend = (
        sma20 is not None
        and sma60 is not None
        and last > sma20
        and sma20 >= sma60 * 0.995
    )
    downtrend = (
        sma20 is not None
        and sma60 is not None
        and last < sma20
        and sma20 <= sma60 * 1.005
    )
    if uptrend:
        bull_pts += 2
        signals.append(
            EntrySignal(
                "趋势环境",
                "做多机会",
                7,
                "中",
                f"价格在 SMA20 上方且中期均线偏多（SMA20={sma20:.2f}, SMA60={sma60:.2f}）",
            )
        )
    elif downtrend:
        bear_pts += 2
        signals.append(
            EntrySignal(
                "趋势环境",
                "谨慎/偏空",
                6,
                "中",
                "价格在均线下方，中期偏空，反弹做多风险较高",
            )
        )
    else:
        signals.append(
            EntrySignal("趋势环境", "观望", 4, "低", "均线交织，趋势不明确")
        )

    # 2) Pullback to support in uptrend
    if uptrend and sma20 is not None:
        dist_sma20 = (last - sma20) / atr_v
        if -0.3 <= dist_sma20 <= 1.2:
            bull_pts += 2.5
            zone_lo = sma20 - 0.3 * atr_v
            zone_hi = sma20 + 0.5 * atr_v
            signals.append(
                EntrySignal(
                    "回踩支撑",
                    "做多机会",
                    8.5,
                    "高",
                    f"回踩/贴近 SMA20 支撑区，适合分批低吸（距均线 {dist_sma20:.1f}×ATR）",
                    zone_lo,
                    zone_hi,
                )
            )
        elif dist_sma20 > 2.5:
            bear_pts += 1.5
            signals.append(
                EntrySignal(
                    "乖离过大",
                    "谨慎/偏空",
                    7,
                    "中",
                    f"价格远离 SMA20 约 {dist_sma20:.1f}×ATR，短线追高风险大，宜等回踩",
                )
            )

    # 3) Breakout
    if last >= recent_high * 0.998 and (vol_ratio is None or vol_ratio >= 1.1):
        if uptrend or (sma20 and last > sma20):
            bull_pts += 2
            signals.append(
                EntrySignal(
                    "突破前高",
                    "做多机会",
                    7.5,
                    "高" if (vol_ratio or 0) >= 1.3 else "中",
                    f"逼近/突破近20日高点 {recent_high:.2f}"
                    + (f"，量比 {vol_ratio:.2f}" if vol_ratio else ""),
                    last * 0.995,
                    last * 1.005,
                )
            )
    elif last <= recent_low * 1.002:
        bear_pts += 1.5
        signals.append(
            EntrySignal(
                "跌破前低",
                "谨慎/偏空",
                7,
                "高",
                f"逼近/跌破近20日低点 {recent_low:.2f}，下破风险",
            )
        )

    # 4) RSI setups
    if rsi is not None:
        if 40 <= rsi <= 55 and uptrend:
            bull_pts += 1.5
            signals.append(
                EntrySignal(
                    "RSI 回落不破",
                    "做多机会",
                    7,
                    "中",
                    f"RSI={rsi:.1f} 回落至健康区且趋势仍多，常见二次上车点",
                )
            )
        elif rsi <= 32:
            bull_pts += 1.2
            signals.append(
                EntrySignal(
                    "RSI 超卖",
                    "做多机会",
                    6.5,
                    "中",
                    f"RSI={rsi:.1f} 超卖，短线反弹概率升（需确认止跌）",
                    last - 0.5 * atr_v,
                    last + 0.3 * atr_v,
                )
            )
        elif rsi >= 72:
            bear_pts += 1.5
            signals.append(
                EntrySignal(
                    "RSI 超买",
                    "谨慎/偏空",
                    7,
                    "中",
                    f"RSI={rsi:.1f} 超买，不宜追多，等冷却",
                )
            )

    # 5) MACD
    if macd is not None and macd_sig is not None and macd_hist is not None:
        if macd_hist_prev is not None and macd_hist_prev <= 0 < macd_hist:
            bull_pts += 2
            signals.append(
                EntrySignal(
                    "MACD 金叉",
                    "做多机会",
                    8,
                    "高",
                    "MACD 柱由负转正（金叉），动能转多",
                )
            )
        elif macd_hist_prev is not None and macd_hist_prev >= 0 > macd_hist:
            bear_pts += 2
            signals.append(
                EntrySignal(
                    "MACD 死叉",
                    "谨慎/偏空",
                    8,
                    "高",
                    "MACD 柱由正转负（死叉），动能转空",
                )
            )
        elif macd_hist > 0 and macd_hist_prev is not None and macd_hist > macd_hist_prev:
            bull_pts += 0.8
            signals.append(
                EntrySignal("MACD 动能增强", "做多机会", 5.5, "低", "MACD 红柱扩张")
            )

    # 6) Bollinger
    if bb_l is not None and bb_u is not None and bb_m is not None:
        if last <= bb_l * 1.01:
            bull_pts += 1.2
            signals.append(
                EntrySignal(
                    "触及布林下轨",
                    "做多机会",
                    6.5,
                    "中",
                    f"价格触及下轨 {bb_l:.2f}，短线超跌反弹区",
                    bb_l,
                    bb_m,
                )
            )
        elif last >= bb_u * 0.995:
            bear_pts += 1.2
            signals.append(
                EntrySignal(
                    "触及布林上轨",
                    "谨慎/偏空",
                    6.5,
                    "中",
                    f"价格触及上轨 {bb_u:.2f}，短线过热",
                )
            )

    # Aggregate opportunity
    raw = bull_pts - bear_pts
    score = max(0.0, min(100.0, 50 + raw * 10))

    entry_lows = [s.entry_zone_low for s in signals if s.entry_zone_low is not None]
    entry_highs = [s.entry_zone_high for s in signals if s.entry_zone_high is not None]
    sug_lo = min(entry_lows) if entry_lows else last - 0.8 * atr_v
    sug_hi = max(entry_highs) if entry_highs else last + 0.2 * atr_v
    # Clamp zone near price
    sug_lo = max(sug_lo, last - 2.5 * atr_v)
    sug_hi = min(max(sug_hi, sug_lo * 1.001), last + 1.5 * atr_v)

    # Stop: prefer tighter tactical stop (for RR), not the farthest historical low
    # Priority: SMA20-0.8ATR (if below price) → last-1.5ATR → recent_low buffer
    tactical = last - 1.5 * atr_v
    if sma20 is not None and sma20 < last:
        ma_stop = sma20 - 0.8 * atr_v
        if ma_stop < last:
            # Use the higher (tighter) of ma_stop and last-1.2ATR
            tactical = max(ma_stop, last - 1.8 * atr_v)
            tactical = min(tactical, last - 0.8 * atr_v)  # at least 0.8 ATR risk
    swing_stop = recent_low - 0.15 * atr_v
    # Final stop = tighter tactical, but not above last-0.6ATR; cap risk at 2.2ATR
    stop = max(tactical, last - 2.2 * atr_v)
    stop = min(stop, last - 0.6 * atr_v)
    # If swing low is close (within 2.5 ATR), prefer it as structure stop
    if last - 2.5 * atr_v < swing_stop < last - 0.5 * atr_v:
        stop = min(stop, swing_stop) if abs(last - swing_stop) < 2.0 * atr_v else stop
        # Prefer swing if it's a reasonable pullback stop
        if last - 2.0 * atr_v <= swing_stop <= last - 0.7 * atr_v:
            stop = swing_stop

    if score >= 68 and bull_pts > bear_pts + 1:
        opportunity = "较佳入场"
        side_bias = "偏多入场"
    elif score >= 55 and bull_pts >= bear_pts:
        opportunity = "可关注"
        side_bias = "偏多入场"
    elif score <= 35 or bear_pts >= bull_pts + 2:
        opportunity = "偏空回避" if downtrend or bear_pts > bull_pts + 1.5 else "不宜追高"
        side_bias = "偏空"
    elif any(s.name == "乖离过大" for s in signals) or (rsi is not None and rsi >= 70):
        opportunity = "不宜追高"
        side_bias = "中性"
    else:
        opportunity = "观望"
        side_bias = "中性"

    checklist = [
        f"现价 {last:.2f}，建议关注区间 {sug_lo:.2f} – {sug_hi:.2f}",
        f"建议止损参考 {stop:.2f}（约 {(stop/last-1)*100:+.1f}%）",
        "优先：趋势多 + 回踩支撑 / 金叉 / 放量突破，三者至少占其二更佳",
        "避免：RSI 严重超买、远离均线、放量跌破关键支撑时追多",
        "仓位：机会分「较佳」可计划仓，「可关注」轻仓试探，「观望/不宜」以等待为主",
    ]

    summary = (
        f"入场评估：**{opportunity}**（{score:.0f}分，{side_bias}）。"
        f"多头条件积分 {bull_pts:.1f} / 风险积分 {bear_pts:.1f}。"
        f"关注买区 {sug_lo:.2f}–{sug_hi:.2f}，止损参考 {stop:.2f}。"
    )

    return EntryReport(
        opportunity=opportunity,
        score=round(score, 1),
        side_bias=side_bias,
        current_price=last,
        suggested_entry_low=round(sug_lo, 2),
        suggested_entry_high=round(sug_hi, 2),
        stop_loss=round(stop, 2),
        invalidation=f"若收盘跌破 {stop:.2f} 或有效跌破近20日低点 {recent_low:.2f}，多头入场逻辑减弱",
        signals=signals,
        checklist=checklist,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Target prices
# ---------------------------------------------------------------------------

@dataclass
class TargetMethod:
    name: str
    price: float
    horizon: str  # 超短 | 短期 | 中期
    side: str  # 上 / 下
    weight: float
    detail: str


@dataclass
class HorizonTargets:
    horizon: str
    horizon_note: str
    bull_target: float | None  # consensus upside
    base_target: float | None
    bear_target: float | None
    upside_pct: float | None
    downside_pct: float | None
    methods: list[TargetMethod] = field(default_factory=list)
    summary: str = ""


@dataclass
class TargetReport:
    current_price: float
    atr: float
    ultra: HorizonTargets  # ≤1 week
    short: HorizonTargets
    medium: HorizonTargets
    analyst_target: float | None = None
    analyst_upside_pct: float | None = None
    entry_stop: float | None = None
    rr_ultra: float | None = None
    rr_short: float | None = None
    rr_medium: float | None = None
    summary: str = ""


def _weighted_median(prices: list[float], weights: list[float]) -> float | None:
    if not prices:
        return None
    pairs = sorted(zip(prices, weights), key=lambda x: x[0])
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return float(np.median(prices))
    acc = 0.0
    for p, w in pairs:
        acc += w
        if acc >= total_w / 2:
            return float(p)
    return float(pairs[-1][0])


def analyze_targets(
    df: pd.DataFrame,
    info: dict[str, Any] | None = None,
    entry: EntryReport | None = None,
) -> TargetReport:
    info = info or {}
    empty_h = HorizonTargets("—", "—", None, None, None, None, None, summary="数据不足")
    if df is None or df.empty or len(df) < 20:
        return TargetReport(
            0, 0, empty_h, empty_h, empty_h, summary="数据不足，无法估算目标价。"
        )

    data = enrich(df)
    close = data["Close"].astype(float)
    last = float(close.iloc[-1])
    atr_s = _atr(data)
    atr_v = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else last * 0.02

    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    h5, l5 = float(high.tail(5).max()), float(low.tail(5).min())
    h10, l10 = float(high.tail(10).max()), float(low.tail(10).min())
    h20, l20 = float(high.tail(20).max()), float(low.tail(20).min())
    h60, l60 = float(high.tail(60).max()), float(low.tail(min(60, len(high))).min())

    sma5 = _safe(data["SMA5"].iloc[-1]) if "SMA5" in data.columns else last
    sma20 = _safe(data["SMA20"].iloc[-1]) if "SMA20" in data.columns else last
    sma60 = _safe(data["SMA60"].iloc[-1]) if "SMA60" in data.columns else last
    bb_u = _safe(data["BB_UPPER"].iloc[-1]) if "BB_UPPER" in data.columns else last * 1.03
    bb_l = _safe(data["BB_LOWER"].iloc[-1]) if "BB_LOWER" in data.columns else last * 0.97
    # half-way band (mid to upper/lower) for ultra-short
    bb_m = _safe(data["BB_MID"].iloc[-1]) if "BB_MID" in data.columns else sma20

    # Pivot from previous bar
    h, l, c = float(high.iloc[-2]), float(low.iloc[-2]), float(close.iloc[-2])
    pivot = (h + l + c) / 3
    r1, s1 = 2 * pivot - l, 2 * pivot - h
    r2, s2 = pivot + (h - l), pivot - (h - l)

    # Momentum: average daily return * horizon days (capped)
    rets = close.pct_change().dropna().tail(20)
    avg_daily = float(rets.mean()) if len(rets) else 0.0
    avg_daily = float(np.clip(avg_daily, -0.015, 0.015))
    rets5 = close.pct_change().dropna().tail(5)
    avg_d5 = float(rets5.mean()) if len(rets5) else avg_daily
    avg_d5 = float(np.clip(avg_d5, -0.02, 0.02))

    ultra_methods: list[TargetMethod] = []
    short_methods: list[TargetMethod] = []
    med_methods: list[TargetMethod] = []

    # ---- Ultra-short (≈1 week / 3–5 trading days) ----
    ultra_methods.append(
        TargetMethod(
            "ATR×0.6 上",
            last + 0.6 * atr_v,
            "超短",
            "上",
            1.3,
            f"0.6×ATR({atr_v:.2f}) 一周常见波动上沿",
        )
    )
    ultra_methods.append(
        TargetMethod(
            "ATR×1.0 上",
            last + 1.0 * atr_v,
            "超短",
            "上",
            1.1,
            "1×ATR 一周偏强目标",
        )
    )
    ultra_methods.append(
        TargetMethod(
            "ATR×0.6 下",
            last - 0.6 * atr_v,
            "超短",
            "下",
            1.3,
            "0.6×ATR 一周常见波动下沿",
        )
    )
    ultra_methods.append(
        TargetMethod(
            "ATR×1.0 下",
            last - 1.0 * atr_v,
            "超短",
            "下",
            1.0,
            "1×ATR 一周偏弱下看",
        )
    )
    ultra_methods.append(
        TargetMethod("近5日高点", h5, "超短", "上", 1.4, "一周内最近波段高点/突破目标")
    )
    ultra_methods.append(
        TargetMethod("近5日低点", l5, "超短", "下", 1.4, "一周内最近波段低点/失守下看")
    )
    ultra_methods.append(
        TargetMethod("近10日高点", h10, "超短", "上", 0.9, "稍远阻力，强势周可能触及")
    )
    ultra_methods.append(
        TargetMethod("近10日低点", l10, "超短", "下", 0.9, "稍远支撑")
    )
    ultra_methods.append(
        TargetMethod("Pivot R1", r1, "超短", "上", 1.0, "隔日/当周枢轴阻力")
    )
    ultra_methods.append(
        TargetMethod("Pivot S1", s1, "超短", "下", 1.0, "隔日/当周枢轴支撑")
    )
    mom_u = last * (1 + avg_d5 * 5)
    ultra_methods.append(
        TargetMethod(
            "动量外推(5日)",
            mom_u,
            "超短",
            "上" if mom_u >= last else "下",
            1.0,
            f"近5日均日涨跌 {avg_d5*100:.2f}% ×5 个交易日",
        )
    )
    if sma5:
        ultra_methods.append(
            TargetMethod(
                "SMA5",
                sma5,
                "超短",
                "下" if last > sma5 else "上",
                0.9,
                "超短线生命线，回归或支撑/压力",
            )
        )
    if bb_m and bb_u:
        mid_up = (bb_m + bb_u) / 2
        ultra_methods.append(
            TargetMethod(
                "布林中上半轨",
                mid_up,
                "超短",
                "上",
                0.8,
                "布林中轨与上轨中点，一周常见阻力区",
            )
        )
    if bb_m and bb_l:
        mid_dn = (bb_m + bb_l) / 2
        ultra_methods.append(
            TargetMethod(
                "布林中下半轨",
                mid_dn,
                "超短",
                "下",
                0.8,
                "布林中轨与下轨中点，一周常见支撑区",
            )
        )

    # ---- Short-term methods (2w-1m ~ 10-22 trading days) ----
    short_methods.append(
        TargetMethod("ATR×1.5 上", last + 1.5 * atr_v, "短期", "上", 1.2, f"1.5×ATR({atr_v:.2f}) 短线波动上沿")
    )
    short_methods.append(
        TargetMethod("ATR×1.0 上", last + 1.0 * atr_v, "短期", "上", 1.0, "1×ATR 保守上行")
    )
    short_methods.append(
        TargetMethod("ATR×1.5 下", last - 1.5 * atr_v, "短期", "下", 1.2, "1.5×ATR 短线波动下沿")
    )
    short_methods.append(
        TargetMethod("布林上轨", bb_u, "短期", "上", 1.1, "布林带上轨常作短线阻力/目标")
    )
    short_methods.append(
        TargetMethod("布林下轨", bb_l, "短期", "下", 1.1, "布林带下轨常作短线支撑/下看")
    )
    short_methods.append(
        TargetMethod("近20日高点", h20, "短期", "上", 1.3, "突破后回踩确认的短线目标")
    )
    short_methods.append(
        TargetMethod("近20日低点", l20, "短期", "下", 1.3, "失守后的短线下看位")
    )
    short_methods.append(
        TargetMethod("Pivot R1", r1, "短期", "上", 0.9, "经典枢轴阻力 R1")
    )
    short_methods.append(
        TargetMethod("Pivot S1", s1, "短期", "下", 0.9, "经典枢轴支撑 S1")
    )
    mom_s = last * (1 + avg_daily * 15)
    short_methods.append(
        TargetMethod(
            "动量外推(15日)",
            mom_s,
            "短期",
            "上" if mom_s >= last else "下",
            0.8,
            f"近20日均日涨跌 {avg_daily*100:.2f}% ×15 日",
        )
    )
    if sma20:
        short_methods.append(
            TargetMethod(
                "回归SMA20",
                sma20,
                "短期",
                "下" if last > sma20 else "上",
                0.7,
                "乖离过大时向 SMA20 回归",
            )
        )

    # ---- Medium-term (1-2m ~ 22-44 trading days) ----
    med_methods.append(
        TargetMethod("ATR×2.5 上", last + 2.5 * atr_v, "中期", "上", 1.2, "2.5×ATR 中期波动上沿")
    )
    med_methods.append(
        TargetMethod("ATR×3.5 上", last + 3.5 * atr_v, "中期", "上", 0.9, "3.5×ATR 偏乐观上行")
    )
    med_methods.append(
        TargetMethod("ATR×2.5 下", last - 2.5 * atr_v, "中期", "下", 1.2, "2.5×ATR 中期波动下沿")
    )
    med_methods.append(
        TargetMethod("近60日高点", h60, "中期", "上", 1.3, "中期波段前高阻力/目标")
    )
    med_methods.append(
        TargetMethod("近60日低点", l60, "中期", "下", 1.3, "中期波段前低支撑/下看")
    )
    med_methods.append(
        TargetMethod("Pivot R2", r2, "中期", "上", 0.9, "枢轴 R2")
    )
    med_methods.append(
        TargetMethod("Pivot S2", s2, "中期", "下", 0.9, "枢轴 S2")
    )
    mom_m = last * (1 + avg_daily * 33)
    med_methods.append(
        TargetMethod(
            "动量外推(33日)",
            mom_m,
            "中期",
            "上" if mom_m >= last else "下",
            0.8,
            f"近20日均日涨跌外推约 1.5 个月",
        )
    )
    if sma60:
        # Channel: SMA60 + recent range
        range60 = h60 - l60
        med_methods.append(
            TargetMethod(
                "波段测算上",
                max(h60, last) + 0.5 * range60,
                "中期",
                "上",
                0.85,
                "以近60日波幅 50% 作测算目标",
            )
        )
        med_methods.append(
            TargetMethod(
                "波段测算下",
                min(l60, last) - 0.5 * range60,
                "中期",
                "下",
                0.85,
                "以近60日波幅 50% 作下看测算",
            )
        )
        med_methods.append(
            TargetMethod(
                "SMA60 中轴",
                sma60,
                "中期",
                "下" if last > sma60 else "上",
                0.7,
                "中期成本/中轴参考",
            )
        )

    # Analyst target (more medium/long, still show)
    analyst = _safe(info.get("targetMeanPrice"))
    analyst_up = None
    if analyst and analyst > 0:
        analyst_up = (analyst / last - 1) * 100
        med_methods.append(
            TargetMethod(
                "分析师均价",
                analyst,
                "中期",
                "上" if analyst >= last else "下",
                1.0,
                f"Yahoo 分析师目标均价（隐含 {analyst_up:+.1f}%）",
            )
        )

    def build_horizon(
        methods: list[TargetMethod],
        horizon: str,
        note: str,
    ) -> HorizonTargets:
        up = [(m.price, m.weight) for m in methods if m.side == "上" and m.price > last * 0.98]
        down = [(m.price, m.weight) for m in methods if m.side == "下" and m.price < last * 1.02]
        # Cap unrealistic moves by horizon
        if horizon == "超短":
            cap = 0.12
            default_mult = 0.7
        elif horizon == "短期":
            cap = 0.35
            default_mult = 1.2
        else:
            cap = 0.50
            default_mult = 2.5

        def filt(pairs):
            out = []
            for p, w in pairs:
                chg = abs(p / last - 1)
                if chg <= cap and p > 0:
                    out.append((p, w))
            return out

        up, down = filt(up), filt(down)
        bull = (
            _weighted_median([p for p, _ in up], [w for _, w in up])
            if up
            else last + default_mult * atr_v
        )
        bear = (
            _weighted_median([p for p, _ in down], [w for _, w in down])
            if down
            else last - default_mult * atr_v
        )
        up_w = sum(w for _, w in up) if up else 0
        dn_w = sum(w for _, w in down) if down else 0
        if bear is not None and bull is not None and bear > bull:
            bear, bull = min(bear, bull), max(bear, bull)

        if bull is not None and bear is not None:
            if up_w >= dn_w:
                base = last * 0.45 + bull * 0.55
            else:
                base = last * 0.45 + bear * 0.55
            lo, hi = min(bear, bull), max(bear, bull)
            base = min(max(base, lo), hi)
        else:
            base = bull or bear or last

        up_pct = (bull / last - 1) * 100 if bull else None
        dn_pct = (bear / last - 1) * 100 if bear else None
        summary = (
            f"{horizon}目标：看多区 **{bull:.2f}**（{up_pct:+.1f}%）· "
            f"中性 **{base:.2f}** · "
            f"看空区 **{bear:.2f}**（{dn_pct:+.1f}%）"
        )
        return HorizonTargets(
            horizon=horizon,
            horizon_note=note,
            bull_target=round(bull, 2) if bull else None,
            base_target=round(base, 2) if base else None,
            bear_target=round(bear, 2) if bear else None,
            upside_pct=round(up_pct, 2) if up_pct is not None else None,
            downside_pct=round(dn_pct, 2) if dn_pct is not None else None,
            methods=methods,
            summary=summary,
        )

    ultra = build_horizon(
        ultra_methods, "超短", "约 1 周内（≈3–5 个交易日）"
    )
    short = build_horizon(short_methods, "短期", "约 2 周 – 1 个月（≈10–22 个交易日）")
    medium = build_horizon(med_methods, "中期", "约 1 – 2 个月（≈22–44 个交易日）")

    stop = entry.stop_loss if entry and entry.stop_loss else last - 1.5 * atr_v
    risk = last - stop if stop < last else atr_v
    rr_u = rr_s = rr_m = None
    if risk > 0 and ultra.bull_target:
        rr_u = (ultra.bull_target - last) / risk
    if risk > 0 and short.bull_target:
        rr_s = (short.bull_target - last) / risk
    if risk > 0 and medium.bull_target:
        rr_m = (medium.bull_target - last) / risk

    summary = (
        f"现价 {last:.2f}（ATR≈{atr_v:.2f}）。"
        f"超短(1周)看多约 {ultra.bull_target}（{ultra.upside_pct:+.1f}%）；"
        f"短期约 {short.bull_target}（{short.upside_pct:+.1f}%）；"
        f"中期约 {medium.bull_target}（{medium.upside_pct:+.1f}%）。"
    )
    if analyst:
        summary += f" 分析师均价 {analyst:.2f}（{analyst_up:+.1f}%）。"

    return TargetReport(
        current_price=last,
        atr=round(atr_v, 2),
        ultra=ultra,
        short=short,
        medium=medium,
        analyst_target=round(analyst, 2) if analyst else None,
        analyst_upside_pct=round(analyst_up, 2) if analyst_up is not None else None,
        entry_stop=round(stop, 2),
        rr_ultra=round(rr_u, 2) if rr_u is not None else None,
        rr_short=round(rr_s, 2) if rr_s is not None else None,
        rr_medium=round(rr_m, 2) if rr_m is not None else None,
        summary=summary,
    )
