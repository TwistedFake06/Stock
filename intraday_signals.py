"""Rule-based opening-range setups for liquid US and Hong Kong equities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from indicators import add_bollinger, add_macd, add_rsi


@dataclass
class IntradaySetup:
    """A long-only opening-range plan; values are trigger levels, not orders."""

    symbol: str = ""
    market: str = "US"
    session_label: str = ""
    verdict: str = "资料不足"  # 可做 | 等待 | 不做 | 资料不足
    score: int = 0
    last_price: float | None = None
    entry: float | None = None
    stop: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    opening_range_high: float | None = None
    vwap: float | None = None
    risk_per_share: float | None = None
    relative_volume: float | None = None
    divergence_warning: bool = False
    reasons: list[str] = field(default_factory=list)


def intraday_market(symbol: str) -> str:
    """Return the supported intraday market inferred from a Yahoo symbol."""
    return "HK" if str(symbol or "").strip().upper().endswith(".HK") else "US"


def market_label(symbol: str) -> str:
    return "港股" if intraday_market(symbol) == "HK" else "美股"


def is_intraday_alert_window(symbol: str, now: datetime | None = None) -> bool:
    """True in a supported market's liquid opening windows, excluding Hong Kong lunch."""
    try:
        from zoneinfo import ZoneInfo

        market = intraday_market(symbol)
        zone = ZoneInfo("Asia/Hong_Kong" if market == "HK" else "America/New_York")
        local_now = now or datetime.now(zone)
        if local_now.tzinfo is None:
            local_now = local_now.replace(tzinfo=zone)
        else:
            local_now = local_now.astimezone(zone)
    except Exception:
        return False

    if local_now.weekday() >= 5:
        return False
    minute = local_now.hour * 60 + local_now.minute
    if intraday_market(symbol) == "HK":
        return (9 * 60 + 45 <= minute < 12 * 60) or (13 * 60 + 15 <= minute < 16 * 60)
    return 9 * 60 + 45 <= minute < 12 * 60


def _rth_bars(symbol: str, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Keep the latest supported regular-session block from Yahoo intraday bars."""
    needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if data is None or data.empty or not needed.issubset(data.columns):
        return pd.DataFrame(), ""

    work = data.copy()
    dates = pd.to_datetime(work["Date"], errors="coerce")
    if dates.isna().all():
        return pd.DataFrame(), ""
    market = intraday_market(symbol)
    timezone = "Asia/Hong_Kong" if market == "HK" else "America/New_York"
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(timezone)
    else:
        dates = dates.dt.tz_localize(timezone)
    work["_local"] = dates
    work = work[work["_local"].dt.weekday < 5]
    minutes = work["_local"].dt.hour * 60 + work["_local"].dt.minute
    if market == "HK":
        morning = (minutes >= 9 * 60 + 30) & (minutes < 12 * 60)
        afternoon = (minutes >= 13 * 60) & (minutes < 16 * 60)
        work = work[morning | afternoon]
    else:
        work = work[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)]
    if work.empty:
        return pd.DataFrame(), ""
    latest_day = work["_local"].dt.normalize().max()
    work = work[work["_local"].dt.normalize() == latest_day]
    if market == "HK" and (work["_local"].dt.hour * 60 + work["_local"].dt.minute >= 13 * 60).any():
        work = work[work["_local"].dt.hour * 60 + work["_local"].dt.minute >= 13 * 60]
        session = "下午开市"
    else:
        session = "上午开市" if market == "HK" else "开市"
    return work.sort_values("_local").reset_index(drop=True), session


def _add_vwap_and_kdj(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    typical = (out["High"].astype(float) + out["Low"].astype(float) + out["Close"].astype(float)) / 3
    volume = out["Volume"].astype(float).clip(lower=0)
    cumulative_volume = volume.cumsum().replace(0, pd.NA)
    out["VWAP"] = (typical * volume).cumsum() / cumulative_volume

    low_9 = out["Low"].astype(float).rolling(9, min_periods=9).min()
    high_9 = out["High"].astype(float).rolling(9, min_periods=9).max()
    rsv = 100 * (out["Close"].astype(float) - low_9) / (high_9 - low_9).replace(0, pd.NA)
    out["KDJ_K"] = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
    out["KDJ_D"] = out["KDJ_K"].ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
    return out


def analyze_opening_range_setup(symbol: str, data: pd.DataFrame) -> IntradaySetup:
    """Build a 5-minute opening-range long plan from one symbol's intraday bars."""
    market = intraday_market(symbol)
    bars, session_label = _rth_bars(symbol, data)
    if len(bars) < 20:
        market_name = "港股" if market == "HK" else "美股"
        return IntradaySetup(
            symbol=symbol,
            market=market,
            session_label=session_label,
            reasons=[f"本交易时段 5 分钟 K线不足，{market_name}开市后再检查。"],
        )

    bars = add_bollinger(bars)
    bars = add_rsi(bars)
    bars = add_macd(bars)
    bars = _add_vwap_and_kdj(bars)
    high = bars["High"].astype(float)
    low = bars["Low"].astype(float)
    close = bars["Close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(true_range.rolling(14, min_periods=14).mean().iloc[-1])
    if not pd.notna(atr) or atr <= 0:
        return IntradaySetup(symbol=symbol, market=market, session_label=session_label, reasons=["盘中 ATR 不足，无法计算止损。"])

    current = bars.iloc[-1]
    opening = bars.iloc[:3]
    opening_high = float(opening["High"].max())
    last_price = float(current["Close"])
    vwap = float(current["VWAP"])
    entry = max(opening_high * 1.0005, vwap)
    recent_low = float(bars["Low"].tail(3).min())
    stop = min(vwap, recent_low) - atr * 0.15
    risk = entry - stop
    if risk <= 0:
        return IntradaySetup(symbol=symbol, market=market, session_label=session_label, reasons=["入场与止损价无有效距离，等待结构形成。"])

    score = 0
    reasons: list[str] = []
    if last_price > vwap:
        score += 20
        reasons.append("价格在 VWAP 上方")
    else:
        reasons.append("价格在 VWAP 下方")
    if last_price >= opening_high * 1.0005:
        score += 22
        reasons.append("5分钟收市确认突破首15分钟高位")
    else:
        reasons.append("仍未确认突破首15分钟高位")

    bb_upper = current["BB_UPPER"]
    if pd.notna(bb_upper) and last_price >= float(bb_upper):
        score += 8
        reasons.append("触及/站上 Boll 上轨，波动扩张")
    macd_hist = current["MACD_HIST"]
    prev_macd_hist = bars["MACD_HIST"].iloc[-2]
    if pd.notna(macd_hist) and pd.notna(prev_macd_hist) and macd_hist > 0 and macd_hist >= prev_macd_hist:
        score += 12
        reasons.append("MACD 柱为正且扩大")
    rsi = current["RSI"]
    if pd.notna(rsi) and 52 <= float(rsi) <= 78:
        score += 8
        reasons.append("RSI 位于可持续的强势区")
    elif pd.notna(rsi) and float(rsi) > 82:
        score -= 8
        reasons.append("RSI 过热，禁止追价")
    k_value = current["KDJ_K"]
    d_value = current["KDJ_D"]
    if pd.notna(k_value) and pd.notna(d_value) and k_value > d_value and 50 <= k_value <= 90:
        score += 5
        reasons.append("KDJ 多头且未极端钝化")

    prior_volume = bars["Volume"].iloc[-13:-1].astype(float)
    average_volume = float(prior_volume.mean()) if not prior_volume.empty else 0.0
    relative_volume = float(current["Volume"]) / average_volume if average_volume > 0 else None
    if relative_volume is not None and relative_volume >= 1.2:
        score += 15
        reasons.append(f"当前5分钟量能为近期均量 {relative_volume:.1f}x")
    else:
        reasons.append("突破量能未达近期均量 1.2x")

    recent = bars.tail(10)
    divergence = bool(
        last_price >= float(recent["High"].max()) * 0.998
        and pd.notna(macd_hist)
        and float(macd_hist) < float(recent["MACD_HIST"].max())
    )
    if divergence:
        score -= 12
        reasons.append("价格近高但 MACD 柱未创新高：顶背离警报，不加仓")

    chase_distance = last_price - entry
    chasing = chase_distance > max(atr * 0.8, entry * 0.004)
    if chasing:
        score -= 15
        reasons.append("现价远离计划入场，等待回踩而非追价")
    risk_pct = risk / entry * 100
    if risk_pct > 1.5:
        score -= 15
        reasons.append("止损距离超过入场价 1.5%，风险不合格")

    if score >= 70 and not chasing and 0.1 <= risk_pct <= 1.5:
        verdict = "可做"
    elif score >= 45:
        verdict = "等待"
    else:
        verdict = "不做"

    return IntradaySetup(
        symbol=symbol,
        market=market,
        session_label=session_label,
        verdict=verdict,
        score=max(0, min(100, score)),
        last_price=last_price,
        entry=entry,
        stop=stop,
        target_1=entry + risk,
        target_2=entry + risk * 2,
        opening_range_high=opening_high,
        vwap=vwap,
        risk_per_share=risk,
        relative_volume=relative_volume,
        divergence_warning=divergence,
        reasons=reasons,
    )