"""Multi-strategy watchlist scanner — parallel entry lenses.

Aggregates independent strategies (bullish bias, oversold bounce, entry zone,
volume pullback, trend align, MACD turn) so screening is not SOP-Confirm-only.

Research only; not investment advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analysis import BIAS_MILD_THRESHOLD, analyze_bias
from edge_signals import analyze_trend_align, analyze_volume_confirm
from entry_targets import analyze_entry
from indicators import enrich
from stock_service import fetch_history, normalize_symbol

try:
    from config import CORE_WATCHLIST
except Exception:  # pragma: no cover
    CORE_WATCHLIST = [
        "GOOGL",
        "NVDA",
        "MSFT",
        "TSLA",
        "META",
        "SNDK",
        "MU",
        "INTC",
        "ORCL",
        "AMD",
        "AAPL",
        "AMZN",
        "SMCI",
        "IONQ",
        "RGTI",
        "QUBT",
        "ONDS",
        "QCOM",
        "WDC",
        "LITE",
        "VRT",
        "MUR",
        "NFLX",
        "NBIS",
        "UNH",
        "PLTR",
        "RXRX",
        "SMR",
        "BE",
        "COHR",
        "QQQ",
        "SOXX",
        "SPCX",
        "RKLB",
        "WMT",
        "DELL",
    ]


@dataclass
class StrategyHit:
    id: str
    label: str
    fired: bool
    strength: float  # 0-100
    reason: str


@dataclass
class SymbolMultiResult:
    symbol: str
    name: str
    last_price: float | None
    hits: list[StrategyHit] = field(default_factory=list)
    hit_count: int = 0
    hit_labels: str = ""
    primary_reason: str = ""
    suggest_tier: str = "不入場"  # 可入場 | 可考慮 | 不入場
    entry_opportunity: str = ""
    bias: str = ""
    error: str = ""


def _rsi_last(data: pd.DataFrame) -> float | None:
    if "RSI" not in data.columns or data.empty:
        return None
    try:
        v = float(data["RSI"].iloc[-1])
        return v if v == v else None
    except Exception:
        return None


def _macd_turn_up(data: pd.DataFrame) -> tuple[bool, str, float]:
    if "MACD_HIST" not in data.columns or len(data) < 3:
        return False, "MACD 數據不足", 0.0
    h0 = float(data["MACD_HIST"].iloc[-1])
    h1 = float(data["MACD_HIST"].iloc[-2])
    if h0 != h0 or h1 != h1:
        return False, "MACD 無效", 0.0
    fired = h0 > h1 and h0 > -1e9
    # Prefer crossing up through zero or rising from negative
    if h1 <= 0 <= h0:
        return True, f"MACD柱由負轉正（{h1:.3f}→{h0:.3f}）", 75.0
    if h0 > h1 and h0 > 0:
        return True, f"MACD柱上升且為正（{h1:.3f}→{h0:.3f}）", 60.0
    if h0 > h1:
        return True, f"MACD柱走高（{h1:.3f}→{h0:.3f}）", 45.0
    return False, f"MACD柱未轉強（{h0:.3f}）", 0.0


def _in_entry_zone(price: float | None, lo: float | None, hi: float | None) -> bool:
    if price is None or lo is None or hi is None:
        return False
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    return a <= price <= b


def evaluate_strategies(
    symbol: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    info: dict[str, Any] | None = None,
) -> SymbolMultiResult:
    """Run all strategies for one symbol (network: history fetch)."""
    sym = normalize_symbol(symbol)
    out = SymbolMultiResult(symbol=sym, name=sym, last_price=None)
    try:
        hist = fetch_history(sym, period=period, interval=interval)
        if hist is None or hist.empty or len(hist) < 40:
            out.error = "行情不足"
            out.suggest_tier = "不入場"
            return out
        data = enrich(hist)
        last = float(data["Close"].iloc[-1])
        out.last_price = last
        info = info or {}
        name = str(info.get("shortName") or info.get("longName") or "").strip()
        if name:
            out.name = name

        bias = analyze_bias(data)
        entry = analyze_entry(data)
        out.bias = bias.bias
        out.entry_opportunity = entry.opportunity
        rsi = _rsi_last(data)

        hits: list[StrategyHit] = []

        # 1) 看多
        bull = bias.bias in ("看多", "强烈看多")
        bull_str = min(100.0, max(0.0, (bias.score + 100) / 2))
        hits.append(
            StrategyHit(
                id="bullish",
                label="看多",
                fired=bull,
                strength=bull_str if bull else 0.0,
                reason=(
                    f"{bias.bias}（分 {bias.score:+.0f}，一致度{bias.confidence}）"
                    if bull
                    else f"非看多：{bias.bias}（{bias.score:+.0f}）"
                ),
            )
        )

        # 2) 超賣反彈（RSI 低 + 非強烈看空）
        oversold = rsi is not None and rsi <= 35 and bias.score > -BIAS_MILD_THRESHOLD
        hits.append(
            StrategyHit(
                id="oversold",
                label="超賣",
                fired=bool(oversold),
                strength=(max(0.0, 35 - rsi) / 35 * 80 + 20) if oversold and rsi is not None else 0.0,
                reason=(
                    f"RSI={rsi:.1f} 超賣區，且未強烈看空"
                    if oversold and rsi is not None
                    else (f"RSI={rsi:.1f} 未達超賣條件" if rsi is not None else "無 RSI")
                ),
            )
        )

        # 3) 入場區／結構
        zone = _in_entry_zone(
            entry.current_price or last,
            entry.suggested_entry_low,
            entry.suggested_entry_high,
        )
        # Only count zone when entry assessment is not an explicit avoid.
        avoid = entry.opportunity in ("偏空回避", "回避", "不宜入场", "观望")
        entry_ok = entry.opportunity in ("较佳入场", "可关注") or (zone and not avoid)
        hits.append(
            StrategyHit(
                id="entry_zone",
                label="入場區",
                fired=bool(entry_ok),
                strength=float(entry.score) if entry_ok else 0.0,
                reason=(
                    f"{entry.opportunity}；"
                    + (
                        "現價在建議區內"
                        if zone and entry_ok
                        else (
                            "評估偏回避，唔計入場區"
                            if avoid
                            else f"區 {entry.suggested_entry_low}–{entry.suggested_entry_high}"
                        )
                    )
                ),
            )
        )

        # 4) 量能（縮量回踩／放量確認，排除放量下跌）
        vol = analyze_volume_confirm(data)
        vol_fire = vol.available and vol.label in ("缩量回踩", "放量确认", "縮量回踩", "放量確認")
        # also accept score high without crash label
        if vol.available and vol.label == "放量下跌":
            vol_fire = False
        elif vol.available and vol.score >= 65:
            vol_fire = True
        hits.append(
            StrategyHit(
                id="volume",
                label="量能",
                fired=bool(vol_fire),
                strength=float(vol.score) if vol_fire else 0.0,
                reason=vol.summary or vol.label or "量能不足",
            )
        )

        # 5) 趨勢對齊（股+板塊+大盤）
        sector = str(info.get("sector") or "")
        industry = str(info.get("industry") or "")
        trend = analyze_trend_align(
            data, sector=sector, industry=industry, period=period
        )
        trend_fire = bool(getattr(trend, "available", False)) and (
            float(getattr(trend, "score", 0) or 0) >= 60
            or "顺" in str(getattr(trend, "label", ""))
            or "順" in str(getattr(trend, "label", ""))
            or "对齐" in str(getattr(trend, "label", ""))
            or "對齊" in str(getattr(trend, "label", ""))
        )
        hits.append(
            StrategyHit(
                id="trend_align",
                label="趨勢對齊",
                fired=bool(trend_fire),
                strength=float(getattr(trend, "score", 0) or 0) if trend_fire else 0.0,
                reason=getattr(trend, "summary", None)
                or getattr(trend, "label", None)
                or "趨勢對齊不足",
            )
        )

        # 6) MACD 轉強
        macd_fire, macd_reason, macd_str = _macd_turn_up(data)
        hits.append(
            StrategyHit(
                id="macd_turn",
                label="MACD轉強",
                fired=macd_fire,
                strength=macd_str if macd_fire else 0.0,
                reason=macd_reason,
            )
        )

        fired = [h for h in hits if h.fired]
        out.hits = hits
        out.hit_count = len(fired)
        out.hit_labels = "、".join(h.label for h in fired) if fired else "—"
        if fired:
            best = max(fired, key=lambda h: h.strength)
            out.primary_reason = best.reason
        else:
            out.primary_reason = "各策略均未明顯觸發"

        # Tier: multi-hit preferred
        if out.hit_count >= 3 and bull:
            out.suggest_tier = "可入場"
        elif out.hit_count >= 2:
            out.suggest_tier = "可考慮"
        elif out.hit_count == 1:
            out.suggest_tier = "不入場"
        else:
            out.suggest_tier = "不入場"
        return out
    except Exception as exc:  # pragma: no cover
        out.error = f"{type(exc).__name__}: {exc}"
        return out


def scan_symbols(
    symbols: list[str],
    *,
    period: str = "1y",
    interval: str = "1d",
    core_only: bool = True,
) -> list[SymbolMultiResult]:
    """Scan a list; optional CORE intersection."""
    core_set = {normalize_symbol(s) for s in CORE_WATCHLIST}
    cleaned: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        n = normalize_symbol(s)
        if not n or n in seen:
            continue
        if core_only and n not in core_set:
            continue
        seen.add(n)
        cleaned.append(n)
    if core_only and not cleaned:
        cleaned = [normalize_symbol(s) for s in CORE_WATCHLIST]

    results: list[SymbolMultiResult] = []
    for sym in cleaned:
        results.append(evaluate_strategies(sym, period=period, interval=interval))
    results.sort(
        key=lambda r: (
            {"可入場": 0, "可考慮": 1, "不入場": 2}.get(r.suggest_tier, 9),
            -r.hit_count,
            -(max((h.strength for h in r.hits if h.fired), default=0)),
        )
    )
    return results
