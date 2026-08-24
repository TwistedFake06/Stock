"""
技术面多空分析：综合均线、MACD、RSI、布林带、动量、量能等信号，
给出 看多 / 看空 / 中性 结论与分项依据。

仅供学习参考，不构成投资建议。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from indicators import enrich

# Label cutoffs for score ∈ [-100, 100]. Keep in sync with web/app.js BIAS_* constants.
BIAS_STRONG_THRESHOLD = 45
BIAS_MILD_THRESHOLD = 18


@dataclass
class Signal:
    name: str
    bias: str  # 看多 | 看空 | 中性
    score: float  # -2 .. +2  roughly
    detail: str


@dataclass
class BiasReport:
    bias: str  # 强烈看多 | 看多 | 中性 | 看空 | 强烈看空
    score: float  # -100 .. +100
    confidence: str  # 高 | 中 | 低
    summary: str
    signals: list[Signal] = field(default_factory=list)
    bull_count: int = 0
    bear_count: int = 0
    neutral_count: int = 0
    snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _safe_float(series: pd.Series, idx: int = -1) -> float | None:
    if series is None or len(series) == 0:
        return None
    try:
        val = series.iloc[idx]
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _bias_from_score(score: float) -> str:
    if score >= BIAS_STRONG_THRESHOLD:
        return "强烈看多"
    if score >= BIAS_MILD_THRESHOLD:
        return "看多"
    if score <= -BIAS_STRONG_THRESHOLD:
        return "强烈看空"
    if score <= -BIAS_MILD_THRESHOLD:
        return "看空"
    return "中性"


def _confidence(signals: list[Signal], score: float) -> str:
    """Higher when signals agree and absolute score is large."""
    if not signals:
        return "低"
    n = len(signals)
    bull = sum(1 for s in signals if s.bias == "看多")
    bear = sum(1 for s in signals if s.bias == "看空")
    agree = max(bull, bear) / n
    strength = min(abs(score) / 60.0, 1.0)
    conf = 0.55 * agree + 0.45 * strength
    if conf >= 0.72:
        return "高"
    if conf >= 0.45:
        return "中"
    return "低"


def analyze_bias(df: pd.DataFrame) -> BiasReport:
    """
    Analyze enriched (or raw OHLCV) history and return multi/空 report.
    Expects columns: Close, Open, High, Low, Volume; indicators auto-added if missing.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return BiasReport(
            bias="中性",
            score=0.0,
            confidence="低",
            summary="数据不足，无法判断多空。",
            signals=[],
        )

    data = df if "RSI" in df.columns else enrich(df)
    if len(data) < 5:
        return BiasReport(
            bias="中性",
            score=0.0,
            confidence="低",
            summary="K线数量过少，无法可靠判断。",
            signals=[],
        )

    signals: list[Signal] = []
    close = data["Close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else last

    # ---- 1. 均线排列 & 价格相对均线 ----
    sma5 = _safe_float(data.get("SMA5", pd.Series(dtype=float)))
    sma20 = _safe_float(data.get("SMA20", pd.Series(dtype=float)))
    sma60 = _safe_float(data.get("SMA60", pd.Series(dtype=float)))

    if sma5 is not None and sma20 is not None and sma60 is not None:
        if sma5 > sma20 > sma60 and last > sma20:
            signals.append(
                Signal(
                    "均线排列",
                    "看多",
                    2.0,
                    f"多头排列：SMA5({sma5:.2f}) > SMA20({sma20:.2f}) > SMA60({sma60:.2f})，价格在均线上方",
                )
            )
        elif sma5 < sma20 < sma60 and last < sma20:
            signals.append(
                Signal(
                    "均线排列",
                    "看空",
                    -2.0,
                    f"空头排列：SMA5({sma5:.2f}) < SMA20({sma20:.2f}) < SMA60({sma60:.2f})，价格在均线下方",
                )
            )
        elif last > sma20 and sma5 >= sma20:
            signals.append(
                Signal(
                    "均线排列",
                    "看多",
                    1.0,
                    f"价格站上 SMA20({sma20:.2f})，短期均线偏强",
                )
            )
        elif last < sma20 and sma5 <= sma20:
            signals.append(
                Signal(
                    "均线排列",
                    "看空",
                    -1.0,
                    f"价格跌破 SMA20({sma20:.2f})，短期均线偏弱",
                )
            )
        else:
            signals.append(
                Signal(
                    "均线排列",
                    "中性",
                    0.0,
                    f"均线交织，方向不明（SMA5={sma5:.2f}, SMA20={sma20:.2f}, SMA60={sma60:.2f}）",
                )
            )

    # ---- 2. 均线金叉 / 死叉（SMA5 vs SMA20）----
    if "SMA5" in data.columns and "SMA20" in data.columns and len(data) >= 3:
        s5_now = _safe_float(data["SMA5"])
        s20_now = _safe_float(data["SMA20"])
        s5_prev = _safe_float(data["SMA5"], -2)
        s20_prev = _safe_float(data["SMA20"], -2)
        if None not in (s5_now, s20_now, s5_prev, s20_prev):
            if s5_prev <= s20_prev and s5_now > s20_now:
                signals.append(
                    Signal("均线交叉", "看多", 1.5, "SMA5 上穿 SMA20（金叉）")
                )
            elif s5_prev >= s20_prev and s5_now < s20_now:
                signals.append(
                    Signal("均线交叉", "看空", -1.5, "SMA5 下穿 SMA20（死叉）")
                )
            else:
                side = "多头" if s5_now > s20_now else "空头"
                signals.append(
                    Signal(
                        "均线交叉",
                        "看多" if s5_now > s20_now else "看空",
                        0.5 if s5_now > s20_now else -0.5,
                        f"SMA5 位于 SMA20 {'上方' if s5_now > s20_now else '下方'}（维持{side}结构）",
                    )
                )

    # ---- 3. MACD ----
    macd = _safe_float(data.get("MACD", pd.Series(dtype=float)))
    signal_line = _safe_float(data.get("MACD_SIGNAL", pd.Series(dtype=float)))
    hist = _safe_float(data.get("MACD_HIST", pd.Series(dtype=float)))
    hist_prev = _safe_float(data.get("MACD_HIST", pd.Series(dtype=float)), -2)

    if macd is not None and signal_line is not None and hist is not None:
        macd_prev = _safe_float(data["MACD"], -2)
        sig_prev = _safe_float(data["MACD_SIGNAL"], -2)
        if (
            macd_prev is not None
            and sig_prev is not None
            and macd_prev <= sig_prev
            and macd > signal_line
        ):
            signals.append(
                Signal(
                    "MACD",
                    "看多",
                    2.0,
                    f"MACD 金叉（柱={hist:.4f}）",
                )
            )
        elif (
            macd_prev is not None
            and sig_prev is not None
            and macd_prev >= sig_prev
            and macd < signal_line
        ):
            signals.append(
                Signal(
                    "MACD",
                    "看空",
                    -2.0,
                    f"MACD 死叉（柱={hist:.4f}）",
                )
            )
        elif hist > 0 and (hist_prev is None or hist >= hist_prev):
            signals.append(
                Signal(
                    "MACD",
                    "看多",
                    1.0,
                    f"MACD 柱为正且动能未衰减（柱={hist:.4f}）",
                )
            )
        elif hist < 0 and (hist_prev is None or hist <= hist_prev):
            signals.append(
                Signal(
                    "MACD",
                    "看空",
                    -1.0,
                    f"MACD 柱为负且动能偏弱（柱={hist:.4f}）",
                )
            )
        else:
            signals.append(
                Signal(
                    "MACD",
                    "中性",
                    0.0,
                    f"MACD 信号混杂（MACD={macd:.4f}, Signal={signal_line:.4f}）",
                )
            )

    # ---- 4. RSI ----
    rsi = _safe_float(data.get("RSI", pd.Series(dtype=float)))
    if rsi is not None:
        if rsi >= 70:
            # Overbought: caution / short-term bearish pressure
            signals.append(
                Signal(
                    "RSI",
                    "看空",
                    -1.2,
                    f"RSI={rsi:.1f} 超买区，短线回调风险升高",
                )
            )
        elif rsi <= 30:
            signals.append(
                Signal(
                    "RSI",
                    "看多",
                    1.2,
                    f"RSI={rsi:.1f} 超卖区，短线反弹概率上升",
                )
            )
        elif rsi >= 55:
            signals.append(
                Signal(
                    "RSI",
                    "看多",
                    0.8,
                    f"RSI={rsi:.1f} 偏强（多头动能区）",
                )
            )
        elif rsi <= 45:
            signals.append(
                Signal(
                    "RSI",
                    "看空",
                    -0.8,
                    f"RSI={rsi:.1f} 偏弱（空头动能区）",
                )
            )
        else:
            signals.append(
                Signal("RSI", "中性", 0.0, f"RSI={rsi:.1f} 中性区间")
            )

    # ---- 5. 布林带位置 ----
    bb_u = _safe_float(data.get("BB_UPPER", pd.Series(dtype=float)))
    bb_m = _safe_float(data.get("BB_MID", pd.Series(dtype=float)))
    bb_l = _safe_float(data.get("BB_LOWER", pd.Series(dtype=float)))
    if bb_u is not None and bb_m is not None and bb_l is not None and bb_u != bb_l:
        pos = (last - bb_l) / (bb_u - bb_l)
        if last >= bb_u:
            signals.append(
                Signal(
                    "布林带",
                    "看空",
                    -1.0,
                    f"价格触及/突破上轨（位置 {pos:.0%}），短线过热",
                )
            )
        elif last <= bb_l:
            signals.append(
                Signal(
                    "布林带",
                    "看多",
                    1.0,
                    f"价格触及/跌破下轨（位置 {pos:.0%}），短线超跌",
                )
            )
        elif last > bb_m:
            signals.append(
                Signal(
                    "布林带",
                    "看多",
                    0.6,
                    f"价格位于中轨上方（位置 {pos:.0%}）",
                )
            )
        elif last < bb_m:
            signals.append(
                Signal(
                    "布林带",
                    "看空",
                    -0.6,
                    f"价格位于中轨下方（位置 {pos:.0%}）",
                )
            )
        else:
            signals.append(Signal("布林带", "中性", 0.0, "价格贴近中轨"))

    # ---- 6. 短期动量（近 5 / 20 日涨跌）----
    if len(close) >= 6:
        ret5 = (last / float(close.iloc[-6]) - 1) * 100
        if ret5 >= 3:
            signals.append(
                Signal("短期动量", "看多", 1.0, f"近5根K线涨幅 {ret5:+.2f}%")
            )
        elif ret5 <= -3:
            signals.append(
                Signal("短期动量", "看空", -1.0, f"近5根K线跌幅 {ret5:+.2f}%")
            )
        else:
            signals.append(
                Signal("短期动量", "中性", 0.2 if ret5 > 0 else -0.2 if ret5 < 0 else 0.0, f"近5根K线变动 {ret5:+.2f}%")
            )

    if len(close) >= 21:
        ret20 = (last / float(close.iloc[-21]) - 1) * 100
        if ret20 >= 8:
            signals.append(
                Signal("中期动量", "看多", 1.2, f"近20根K线涨幅 {ret20:+.2f}%")
            )
        elif ret20 <= -8:
            signals.append(
                Signal("中期动量", "看空", -1.2, f"近20根K线跌幅 {ret20:+.2f}%")
            )
        else:
            signals.append(
                Signal(
                    "中期动量",
                    "看多" if ret20 > 1 else "看空" if ret20 < -1 else "中性",
                    0.4 if ret20 > 1 else -0.4 if ret20 < -1 else 0.0,
                    f"近20根K线变动 {ret20:+.2f}%",
                )
            )

    # ---- 7. 量能（若有成交量）----
    if "Volume" in data.columns and data["Volume"].notna().sum() >= 10:
        vol = data["Volume"].astype(float)
        vol_ma = vol.rolling(10, min_periods=5).mean()
        v_last = _safe_float(vol)
        v_ma = _safe_float(vol_ma)
        day_up = last >= prev
        if v_last is not None and v_ma is not None and v_ma > 0:
            ratio = v_last / v_ma
            if ratio >= 1.4 and day_up:
                signals.append(
                    Signal(
                        "量能",
                        "看多",
                        1.0,
                        f"放量上涨（量/10日均量={ratio:.2f}x）",
                    )
                )
            elif ratio >= 1.4 and not day_up:
                signals.append(
                    Signal(
                        "量能",
                        "看空",
                        -1.0,
                        f"放量下跌（量/10日均量={ratio:.2f}x）",
                    )
                )
            elif ratio < 0.6:
                signals.append(
                    Signal(
                        "量能",
                        "中性",
                        0.0,
                        f"缩量整理（量/10日均量={ratio:.2f}x）",
                    )
                )
            else:
                signals.append(
                    Signal(
                        "量能",
                        "中性",
                        0.2 if day_up else -0.2,
                        f"量能正常（量/10日均量={ratio:.2f}x，{'阳线' if day_up else '阴线'}）",
                    )
                )

    # ---- Aggregate score ----
    raw = sum(s.score for s in signals)
    # Normalize roughly to -100..100 (typical |raw| max ~10)
    score = max(-100.0, min(100.0, raw / 10.0 * 100.0))
    bias = _bias_from_score(score)
    bull_count = sum(1 for s in signals if s.bias == "看多")
    bear_count = sum(1 for s in signals if s.bias == "看空")
    neutral_count = sum(1 for s in signals if s.bias == "中性")
    conf = _confidence(signals, score)

    if bias in ("看多", "强烈看多"):
        summary = (
            f"综合 **{bias}**（得分 {score:+.0f}）。"
            f"多头信号 {bull_count} 项，空头 {bear_count} 项。"
            f"信号一致度：{conf}。"
        )
    elif bias in ("看空", "强烈看空"):
        summary = (
            f"综合 **{bias}**（得分 {score:+.0f}）。"
            f"空头信号 {bear_count} 项，多头 {bull_count} 项。"
            f"信号一致度：{conf}。"
        )
    else:
        summary = (
            f"综合 **中性**（得分 {score:+.0f}），多空力量接近。"
            f"多头 {bull_count} / 空头 {bear_count} / 中性 {neutral_count}。"
            f"信号一致度：{conf}。"
        )

    snapshot = {
        "close": last,
        "rsi": rsi,
        "macd": macd,
        "macd_hist": hist,
        "sma5": sma5,
        "sma20": sma20,
        "sma60": sma60,
        "bb_position": (
            (last - bb_l) / (bb_u - bb_l)
            if bb_u is not None and bb_l is not None and bb_u != bb_l
            else None
        ),
    }

    return BiasReport(
        bias=bias,
        score=round(score, 1),
        confidence=conf,
        summary=summary,
        signals=signals,
        bull_count=bull_count,
        bear_count=bear_count,
        neutral_count=neutral_count,
        snapshot=snapshot,
    )


def bias_emoji(bias: str) -> str:
    return {
        "强烈看多": "🚀",
        "看多": "📈",
        "中性": "⚖️",
        "看空": "📉",
        "强烈看空": "💥",
    }.get(bias, "⚖️")


def bias_color(bias: str) -> str:
    """CSS color for bias label (A-share style: red up / green down)."""
    if bias in ("看多", "强烈看多"):
        return "#ef5350"
    if bias in ("看空", "强烈看空"):
        return "#26a69a"
    return "#78909c"
