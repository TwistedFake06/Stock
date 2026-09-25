"""Multi-strategy watchlist scanner — parallel bullish + bearish lenses.

Bullish lenses (hits / hit_count / suggest_tier):
  看多, 超賣, 入場區, 量能, 趨勢對齊, MACD轉強
  Tier: >=3 + bull bias -> 可入場; >=2 -> 可考慮; else 不入場.

Bearish lenses (bear_hits / bear_hit_count / bear_tier) — kept separate so
digests that gate on bullish multi hits are not polluted:
  看空, 超買, 壓力／回避區, 量能偏空, 趨勢逆勢, MACD轉弱
  Tier: >=3 + bear bias -> 偏空強; >=2 -> 偏空留意; else 無偏空訊號.

Research only; not investment advice. Multi hits alone are NEVER the real
entry gate in digests (Confirm SOP is).
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
        "CRWV",
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

# RSI thresholds (mirror pair)
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Bearish tier labels (short, clear)
BEAR_TIER_STRONG = "偏空強"
BEAR_TIER_WATCH = "偏空留意"
BEAR_TIER_NONE = "無偏空訊號"

# Avoid / chase opportunity labels used by entry_targets
_AVOID_OPPS = ("偏空回避", "回避", "不宜入场", "不宜入場", "观望", "觀望", "不宜追高")
_CHASE_AVOID_OPPS = ("不宜追高", "偏空回避", "回避", "不宜入场", "不宜入場")


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
    # --- bullish side (legacy field names; digests rely on these) ---
    hits: list[StrategyHit] = field(default_factory=list)
    hit_count: int = 0
    hit_labels: str = ""
    primary_reason: str = ""
    suggest_tier: str = "不入場"  # 可入場 | 可考慮 | 不入場
    # --- bearish side (parallel; do NOT mix into hit_count) ---
    bear_hits: list[StrategyHit] = field(default_factory=list)
    bear_hit_count: int = 0
    bear_hit_labels: str = ""
    bear_primary_reason: str = ""
    bear_tier: str = BEAR_TIER_NONE  # 偏空強 | 偏空留意 | 無偏空訊號
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
    # Prefer crossing up through zero or rising from negative
    if h1 <= 0 <= h0:
        return True, f"MACD柱由負轉正（{h1:.3f}→{h0:.3f}）", 75.0
    if h0 > h1 and h0 > 0:
        return True, f"MACD柱上升且為正（{h1:.3f}→{h0:.3f}）", 60.0
    if h0 > h1:
        return True, f"MACD柱走高（{h1:.3f}→{h0:.3f}）", 45.0
    return False, f"MACD柱未轉強（{h0:.3f}）", 0.0


def _macd_turn_down(data: pd.DataFrame) -> tuple[bool, str, float]:
    """Mirror of _macd_turn_up: hist falling / cross down through zero."""
    if "MACD_HIST" not in data.columns or len(data) < 3:
        return False, "MACD 數據不足", 0.0
    h0 = float(data["MACD_HIST"].iloc[-1])
    h1 = float(data["MACD_HIST"].iloc[-2])
    if h0 != h0 or h1 != h1:
        return False, "MACD 無效", 0.0
    if h1 >= 0 >= h0:
        return True, f"MACD柱由正轉負（{h1:.3f}→{h0:.3f}）", 75.0
    if h0 < h1 and h0 < 0:
        return True, f"MACD柱下降且為負（{h1:.3f}→{h0:.3f}）", 60.0
    if h0 < h1:
        return True, f"MACD柱走弱（{h1:.3f}→{h0:.3f}）", 45.0
    return False, f"MACD柱未轉弱（{h0:.3f}）", 0.0


def _in_entry_zone(price: float | None, lo: float | None, hi: float | None) -> bool:
    if price is None or lo is None or hi is None:
        return False
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    return a <= price <= b


def _above_entry_zone(price: float | None, lo: float | None, hi: float | None) -> bool:
    if price is None or hi is None:
        return False
    top = hi if lo is None else max(lo, hi)
    return price > top


def evaluate_strategies(
    symbol: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    info: dict[str, Any] | None = None,
) -> SymbolMultiResult:
    """Run bullish + bearish strategy lenses for one symbol (network: history fetch)."""
    sym = normalize_symbol(symbol)
    out = SymbolMultiResult(symbol=sym, name=sym, last_price=None)
    try:
        hist = fetch_history(sym, period=period, interval=interval)
        if hist is None or hist.empty or len(hist) < 40:
            out.error = "行情不足"
            out.suggest_tier = "不入場"
            out.bear_tier = BEAR_TIER_NONE
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
        bear_hits: list[StrategyHit] = []

        # ========== BULLISH LENSES ==========
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
        oversold = (
            rsi is not None
            and rsi <= RSI_OVERSOLD
            and bias.score > -BIAS_MILD_THRESHOLD
        )
        hits.append(
            StrategyHit(
                id="oversold",
                label="超賣",
                fired=bool(oversold),
                strength=(
                    max(0.0, RSI_OVERSOLD - rsi) / RSI_OVERSOLD * 80 + 20
                    if oversold and rsi is not None
                    else 0.0
                ),
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
        avoid = entry.opportunity in _AVOID_OPPS
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
        vol_fire = vol.available and vol.label in (
            "缩量回踩",
            "放量确认",
            "縮量回踩",
            "放量確認",
            "偏多量能",
        )
        if vol.available and vol.label == "放量下跌":
            vol_fire = False
        elif vol.available and vol.score >= 65 and vol.label not in (
            "放量下跌",
            "量价背离",
            "量價背離",
            "偏空量能",
        ):
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
        trend_label = str(getattr(trend, "label", "") or "")
        trend_fire = bool(getattr(trend, "available", False)) and (
            float(getattr(trend, "score", 0) or 0) >= 60
            or "顺" in trend_label
            or "順" in trend_label
            or "跟势" in trend_label
            or "跟勢" in trend_label
            or "对齐" in trend_label
            or "對齊" in trend_label
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

        # Bullish tier: multi-hit preferred
        if out.hit_count >= 3 and bull:
            out.suggest_tier = "可入場"
        elif out.hit_count >= 2:
            out.suggest_tier = "可考慮"
        else:
            out.suggest_tier = "不入場"

        # ========== BEARISH LENSES (parallel; separate counters) ==========
        # 1) 看空
        bear = bias.bias in ("看空", "强烈看空")
        bear_str = min(100.0, max(0.0, (100 - bias.score) / 2))
        bear_hits.append(
            StrategyHit(
                id="bearish",
                label="看空",
                fired=bear,
                strength=bear_str if bear else 0.0,
                reason=(
                    f"{bias.bias}（分 {bias.score:+.0f}，一致度{bias.confidence}）"
                    if bear
                    else f"非看空：{bias.bias}（{bias.score:+.0f}）"
                ),
            )
        )

        # 2) 超買（RSI 高 + 非強烈看多）— mirror of 超賣
        overbought = (
            rsi is not None
            and rsi >= RSI_OVERBOUGHT
            and bias.score < BIAS_MILD_THRESHOLD
        )
        bear_hits.append(
            StrategyHit(
                id="overbought",
                label="超買",
                fired=bool(overbought),
                strength=(
                    max(0.0, rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 80 + 20
                    if overbought and rsi is not None
                    else 0.0
                ),
                reason=(
                    f"RSI={rsi:.1f} 超買區，且未強烈看多"
                    if overbought and rsi is not None
                    else (f"RSI={rsi:.1f} 未達超買條件" if rsi is not None else "無 RSI")
                ),
            )
        )

        # 3) 壓力／回避區 — avoid/chase opportunity OR price above zone while chase/avoid
        above = _above_entry_zone(
            entry.current_price or last,
            entry.suggested_entry_low,
            entry.suggested_entry_high,
        )
        # opportunity explicitly avoid/chase, OR extended above buy-zone while wait/avoid
        pressure_fire = entry.opportunity in _CHASE_AVOID_OPPS or (
            above
            and entry.opportunity
            in (
                "观望",
                "觀望",
                "不宜追高",
                "偏空回避",
                "回避",
                "不宜入场",
                "不宜入場",
            )
        )
        bear_hits.append(
            StrategyHit(
                id="pressure_zone",
                label="壓力／回避區",
                fired=bool(pressure_fire),
                strength=(
                    max(0.0, 100.0 - float(entry.score)) if pressure_fire else 0.0
                ),
                reason=(
                    f"{entry.opportunity}"
                    + ("；現價高於建議買區" if above and pressure_fire else "")
                    if pressure_fire
                    else f"{entry.opportunity}；未觸發回避／追高壓力"
                ),
            )
        )

        # 4) 量能偏空 — 放量下跌 / 偏空量能 / 量价背离 (not 缩量回踩/放量确认)
        vol_bear_labels = (
            "放量下跌",
            "偏空量能",
            "量价背离",
            "量價背離",
        )
        vol_bull_block = (
            "缩量回踩",
            "放量确认",
            "縮量回踩",
            "放量確認",
            "偏多量能",
        )
        vol_bear_fire = False
        if vol.available:
            if vol.label in vol_bear_labels:
                vol_bear_fire = True
            elif vol.score <= 35 and vol.label not in vol_bull_block:
                vol_bear_fire = True
        bear_hits.append(
            StrategyHit(
                id="volume_bear",
                label="量能偏空",
                fired=bool(vol_bear_fire),
                strength=(
                    max(0.0, 100.0 - float(vol.score)) if vol_bear_fire else 0.0
                ),
                reason=vol.summary or vol.label or "量能不足",
            )
        )

        # 5) 趨勢逆勢 — low score / 逆 / against_trend / 背离
        against = bool(getattr(trend, "against_trend", False))
        trend_score = float(getattr(trend, "score", 0) or 0)
        trend_bear_fire = bool(getattr(trend, "available", False)) and (
            against
            or trend_score <= 40
            or "逆" in trend_label
            or "背离" in trend_label
            or "背離" in trend_label
        )
        bear_hits.append(
            StrategyHit(
                id="trend_against",
                label="趨勢逆勢",
                fired=bool(trend_bear_fire),
                strength=(
                    max(0.0, 100.0 - trend_score) if trend_bear_fire else 0.0
                ),
                reason=getattr(trend, "summary", None)
                or getattr(trend, "label", None)
                or "趨勢逆勢不足",
            )
        )

        # 6) MACD 轉弱
        macd_dn, macd_dn_reason, macd_dn_str = _macd_turn_down(data)
        bear_hits.append(
            StrategyHit(
                id="macd_turn_down",
                label="MACD轉弱",
                fired=macd_dn,
                strength=macd_dn_str if macd_dn else 0.0,
                reason=macd_dn_reason,
            )
        )

        bear_fired = [h for h in bear_hits if h.fired]
        out.bear_hits = bear_hits
        out.bear_hit_count = len(bear_fired)
        out.bear_hit_labels = (
            "、".join(h.label for h in bear_fired) if bear_fired else "—"
        )
        if bear_fired:
            best_b = max(bear_fired, key=lambda h: h.strength)
            out.bear_primary_reason = best_b.reason
        else:
            out.bear_primary_reason = "各偏空策略均未明顯觸發"

        if out.bear_hit_count >= 3 and bear:
            out.bear_tier = BEAR_TIER_STRONG
        elif out.bear_hit_count >= 2:
            out.bear_tier = BEAR_TIER_WATCH
        else:
            out.bear_tier = BEAR_TIER_NONE

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
    """Scan a list; optional CORE intersection. Sorted by bullish tier then hits."""
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
