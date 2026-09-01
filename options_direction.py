"""Technical direction for choosing vertical side."""
from __future__ import annotations

import pandas as pd

from indicators import enrich
from options_models import DirectionReport

def analyze_direction(df: pd.DataFrame) -> DirectionReport:
    """Technical direction for choosing vertical side."""
    if df is None or df.empty or "Close" not in df.columns or len(df) < 30:
        return DirectionReport(
            direction="中性",
            strength="弱",
            score=0,
            preferred_verticals=["bull_put", "bear_call"],
            reasons=["K线不足，默认中性"],
            summary="数据不足，方向按中性处理。",
            style_hint="观望或极小仓信用价差",
            market_regime="未知",
        )

    data = enrich(df)
    close = data["Close"].astype(float)
    last = float(close.iloc[-1])
    sma5 = float(data["SMA5"].iloc[-1]) if "SMA5" in data.columns else last
    sma20 = float(data["SMA20"].iloc[-1]) if "SMA20" in data.columns else last
    sma60 = float(data["SMA60"].iloc[-1]) if "SMA60" in data.columns else last
    rsi = float(data["RSI"].iloc[-1]) if "RSI" in data.columns and pd.notna(data["RSI"].iloc[-1]) else 50.0
    macd_h = (
        float(data["MACD_HIST"].iloc[-1])
        if "MACD_HIST" in data.columns and pd.notna(data["MACD_HIST"].iloc[-1])
        else 0.0
    )
    macd_h_prev = (
        float(data["MACD_HIST"].iloc[-2])
        if "MACD_HIST" in data.columns and len(data) > 1 and pd.notna(data["MACD_HIST"].iloc[-2])
        else macd_h
    )

    score = 0.0
    reasons: list[str] = []

    # Trend / MA
    if last > sma20 > sma60:
        score += 28
        reasons.append(f"多头均线：价 > SMA20({sma20:.1f}) > SMA60({sma60:.1f})")
    elif last < sma20 < sma60:
        score -= 28
        reasons.append(f"空头均线：价 < SMA20({sma20:.1f}) < SMA60({sma60:.1f})")
    elif last > sma20:
        score += 12
        reasons.append("价格在 SMA20 上方")
    elif last < sma20:
        score -= 12
        reasons.append("价格在 SMA20 下方")

    if sma5 > sma20:
        score += 10
        reasons.append("SMA5 > SMA20（短线偏多）")
    elif sma5 < sma20:
        score -= 10
        reasons.append("SMA5 < SMA20（短线偏空）")
    else:
        reasons.append("SMA5 与 SMA20 持平（短线中性）")

    # Momentum 5/20d
    if len(close) >= 6:
        r5 = (last / float(close.iloc[-6]) - 1) * 100
        if r5 >= 2:
            score += 12
            reasons.append(f"近5日涨 {r5:+.1f}%")
        elif r5 <= -2:
            score -= 12
            reasons.append(f"近5日跌 {r5:+.1f}%")
    if len(close) >= 21:
        r20 = (last / float(close.iloc[-21]) - 1) * 100
        if r20 >= 5:
            score += 14
            reasons.append(f"近20日涨 {r20:+.1f}%")
        elif r20 <= -5:
            score -= 14
            reasons.append(f"近20日跌 {r20:+.1f}%")

    # RSI
    if rsi >= 70:
        score -= 8
        reasons.append(f"RSI={rsi:.0f} 超买，追多 debit 需谨慎")
    elif rsi >= 55:
        score += 10
        reasons.append(f"RSI={rsi:.0f} 偏强")
    elif rsi <= 30:
        score += 8
        reasons.append(f"RSI={rsi:.0f} 超卖，空头 debit 需谨慎")
    elif rsi <= 45:
        score -= 10
        reasons.append(f"RSI={rsi:.0f} 偏弱")
    else:
        reasons.append(f"RSI={rsi:.0f} 中性")

    # MACD hist
    if macd_h > 0 and macd_h >= macd_h_prev:
        score += 10
        reasons.append("MACD 柱为正且增强")
    elif macd_h < 0 and macd_h <= macd_h_prev:
        score -= 10
        reasons.append("MACD 柱为负且走弱")
    elif macd_h > 0:
        score += 5
        reasons.append("MACD 柱为正")
    elif macd_h < 0:
        score -= 5
        reasons.append("MACD 柱为负")

    score = float(max(-100.0, min(100.0, score)))

    # Use the latest 20 sessions as a practical range reference.  It does not
    # predict support/resistance; it only makes the OTM credit-side decision
    # explicit when the trend score is neutral.
    recent = data.tail(20)
    range_low = float(recent["Low"].min()) if "Low" in recent.columns else float(close.tail(20).min())
    range_high = float(recent["High"].max()) if "High" in recent.columns else float(close.tail(20).max())
    range_span = range_high - range_low
    range_position = (last - range_low) / range_span if range_span > 0 else 0.5
    range_position = float(max(0.0, min(1.0, range_position)))

    if score >= 25:
        direction = "看多"
        preferred = ["bull_put", "bull_call"]  # credit first if mild, debit if strong
        if score >= 45 and rsi < 72:
            preferred = ["bull_call", "bull_put"]
            style = "趋势偏强：可优先考虑 Bull Call Debit；或 Bull Put Credit 收租"
        else:
            style = "温和看多：优先 Bull Put Credit（胜率导向）；强突破再用 Bull Call"
    elif score <= -25:
        direction = "看空"
        preferred = ["bear_call", "bear_put"]
        if score <= -45 and rsi > 28:
            preferred = ["bear_put", "bear_call"]
            style = "趋势偏空：可优先 Bear Put Debit；或 Bear Call Credit 收租"
        else:
            style = "温和看空：优先 Bear Call Credit；下跌加速再用 Bear Put"
    else:
        direction = "中性"
        preferred = ["bull_put", "bear_call"]
        style = (
            f"震荡区间约 {range_low:.1f}–{range_high:.1f}：下缘有支撑才卖 OTM Put，"
            "上缘有压力才卖 OTM Call；不确定时观望，避免重仓 debit"
        )

    abs_s = abs(score)
    strength = "强" if abs_s >= 45 else "中" if abs_s >= 25 else "弱"

    summary = (
        f"方向 **{direction}**（强度 {strength}，得分 {score:+.0f}）。{style}"
    )
    return DirectionReport(
        direction=direction,
        strength=strength,
        score=round(max(-100.0, min(100.0, score)), 1),
        preferred_verticals=preferred,
        reasons=reasons,
        summary=summary,
        style_hint=style,
        market_regime="震荡" if direction == "中性" else "趋势",
        range_low=round(range_low, 2),
        range_high=round(range_high, 2),
        range_position=round(range_position, 2),
    )
