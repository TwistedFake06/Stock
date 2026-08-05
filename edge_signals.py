"""
Edge signals that improve long-entry accuracy:

1) Sector ETF relative strength (stock vs its sector proxy)
2) Volume confirmation (breakout volume / healthy pullback shrink)
3) Rough IV regime (ATM IV vs realized HV) for event-risk filter
4) False-breakout filter (failed breakouts / no close confirmation)
5) Trend-follow score (stock + sector ETF + SPY alignment)

All free via yfinance. Degrade gracefully when data missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from free_data import multi_horizon_rs
from stock_service import cache_bucket, fetch_history, normalize_symbol


# GICS-ish sector (yfinance) → liquid sector ETF
SECTOR_TO_ETF: dict[str, tuple[str, str]] = {
    "technology": ("XLK", "科技 XLK"),
    "communication services": ("XLC", "通讯 XLC"),
    "consumer cyclical": ("XLY", "可选消费 XLY"),
    "consumer defensive": ("XLP", "必选消费 XLP"),
    "financial services": ("XLF", "金融 XLF"),
    "financials": ("XLF", "金融 XLF"),
    "healthcare": ("XLV", "医疗 XLV"),
    "industrials": ("XLI", "工业 XLI"),
    "basic materials": ("XLB", "材料 XLB"),
    "energy": ("XLE", "能源 XLE"),
    "utilities": ("XLU", "公用 XLU"),
    "real estate": ("XLRE", "地产 XLRE"),
}

# Industry substring → more precise proxy when useful
INDUSTRY_TO_ETF: list[tuple[str, str, str]] = [
    ("semiconductor", "SMH", "半导体 SMH"),
    ("software", "IGV", "软件 IGV"),
    ("biotechnology", "XBI", "生物科技 XBI"),
    ("banks", "KBE", "银行 KBE"),
    ("oil", "XLE", "能源 XLE"),
    ("gold", "GLD", "黄金 GLD"),
    ("retail", "XRT", "零售 XRT"),
    ("aerospace", "ITA", "航天 ITA"),
]


@dataclass
class SectorRSReport:
    sector: str = ""
    industry: str = ""
    etf: str = ""
    etf_label: str = ""
    score: float | None = None  # 0-100 stock vs sector
    label: str = "—"  # 强于板块 | 同步 | 弱于板块
    alphas: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    available: bool = False


@dataclass
class VolumeConfirmReport:
    """Long-entry volume quality."""

    score: float = 50.0  # 0-100 higher = better long confirmation
    label: str = "—"  # 放量确认 | 缩量回踩 | 量价背离 | 放量下跌 | 正常
    vol_ratio: float | None = None  # last / MA20
    price_chg_5d_pct: float | None = None
    bullets: list[str] = field(default_factory=list)
    summary: str = ""
    available: bool = False


@dataclass
class IVRegimeReport:
    """Rough IV vs HV for US names with options."""

    iv_atm: float | None = None  # decimal e.g. 0.28
    hv_30: float | None = None
    iv_hv_ratio: float | None = None
    label: str = "—"  # 低IV | 正常 | 偏高 | 极高 | 无数据
    score: float = 50.0  # for long equity: high IV slightly worse
    high_event_risk: bool = False
    summary: str = ""
    available: bool = False
    notes: list[str] = field(default_factory=list)


def map_sector_etf(sector: str = "", industry: str = "") -> tuple[str, str]:
    """Return (etf_ticker, label). Default SPY if unknown."""
    ind = (industry or "").lower()
    for key, etf, lab in INDUSTRY_TO_ETF:
        if key in ind:
            return etf, lab
    sec = (sector or "").strip().lower()
    if sec in SECTOR_TO_ETF:
        return SECTOR_TO_ETF[sec]
    # partial match
    for k, v in SECTOR_TO_ETF.items():
        if k in sec or sec in k:
            return v
    return "SPY", "大盘 SPY（未识别板块）"


@lru_cache(maxsize=64)
def _cached_hist(symbol: str, period: str, _bucket: str) -> tuple[tuple, ...]:
    """Hashable cache of close series for RS joins."""
    df = fetch_history(symbol, period=period, interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return tuple()
    out = []
    for _, row in df[["Date", "Close"]].dropna().iterrows():
        try:
            out.append((str(pd.Timestamp(row["Date"]).date()), float(row["Close"])))
        except Exception:
            continue
    return tuple(out)


def _hist_df(symbol: str, period: str = "1y") -> pd.DataFrame:
    rows = _cached_hist(normalize_symbol(symbol), period, cache_bucket(15))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame({"Date": pd.to_datetime([r[0] for r in rows]), "Close": [r[1] for r in rows]})


def analyze_sector_rs(
    stock_df: pd.DataFrame,
    *,
    sector: str = "",
    industry: str = "",
    period: str = "1y",
) -> SectorRSReport:
    """Multi-horizon excess return of stock vs its sector ETF."""
    etf, lab = map_sector_etf(sector, industry)
    if stock_df is None or stock_df.empty:
        return SectorRSReport(
            sector=sector,
            industry=industry,
            etf=etf,
            etf_label=lab,
            summary="个股数据不足，无法做板块相对强弱。",
        )
    try:
        etf_df = _hist_df(etf, period=period)
        if etf_df.empty:
            # fallback SPY
            if etf != "SPY":
                etf, lab = "SPY", "大盘 SPY（板块ETF失败）"
                etf_df = _hist_df("SPY", period=period)
        if etf_df.empty:
            return SectorRSReport(
                sector=sector,
                industry=industry,
                etf=etf,
                etf_label=lab,
                summary="板块 ETF 行情拉取失败。",
            )
        # Align columns
        stock = stock_df.copy()
        if "Date" not in stock.columns and stock.index.name:
            stock = stock.reset_index()
        mh = multi_horizon_rs(stock, etf_df, windows=(21, 63, 126))
        score = mh.get("score")
        label = mh.get("label") or "—"
        # remap labels to 板块 wording
        if label == "强于大盘":
            label = "强于板块"
        elif label == "弱于大盘":
            label = "弱于板块"
        elif label == "同步":
            label = "与板块同步"
        summary = (
            f"板块 **{lab}**（{etf}）· "
            + (mh.get("summary") or "").replace("大盘", "板块")
        )
        return SectorRSReport(
            sector=sector or "—",
            industry=industry or "—",
            etf=etf,
            etf_label=lab,
            score=float(score) if score is not None else None,
            label=label,
            alphas=dict(mh.get("alphas") or {}),
            summary=summary,
            available=score is not None,
        )
    except Exception as exc:
        return SectorRSReport(
            sector=sector,
            industry=industry,
            etf=etf,
            etf_label=lab,
            summary=f"板块 RS 计算失败：{exc}",
        )


def analyze_volume_confirm(df: pd.DataFrame) -> VolumeConfirmReport:
    """
    Score volume behaviour for *long* entries.

    Good:
      - Pullback / mild down day with shrink volume (healthy)
      - Up day / breakout with expand volume (confirmation)
    Bad:
      - Selloff with expand volume (distribution)
      - Rally with very thin volume (weak)
    """
    if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
        return VolumeConfirmReport(summary="无量价数据。")

    work = df.dropna(subset=["Close", "Volume"]).copy()
    if len(work) < 25:
        return VolumeConfirmReport(summary="K线不足，量能确认不可用。")

    # 默认丢掉最后一根（盘中未完成日K），避免 10 分钟内量比/涨跌乱跳
    # 若只有收盘后完整数据，iloc[:-1] 仍是「上一完整交易日」，更稳定
    if len(work) >= 26:
        work = work.iloc[:-1].copy()

    close = work["Close"].astype(float)
    vol = work["Volume"].astype(float)
    if float(vol.tail(20).mean() or 0) <= 0:
        return VolumeConfirmReport(summary="成交量无效。")

    ma20 = float(vol.tail(20).mean())
    last_v = float(vol.iloc[-1])
    ratio = last_v / ma20 if ma20 > 0 else None
    chg1 = float(close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0.0
    chg5 = (
        float(close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else None
    )

    # recent high proximity (breakout context)
    hi20 = float(close.tail(20).max())
    last = float(close.iloc[-1])
    near_high = hi20 > 0 and last >= hi20 * 0.985

    score = 50.0
    bullets: list[str] = []
    label = "正常"

    if ratio is not None:
        bullets.append(f"量比(最近完整日/20日均) ≈ {ratio:.2f}x")

    if ratio is not None and chg1 is not None:
        if chg1 >= 0.3 and ratio >= 1.4:
            score += 18
            label = "放量确认"
            bullets.append("上涨且放量：买盘确认较好")
            if near_high and ratio >= 1.6:
                score += 6
                bullets.append("接近/突破近高 + 放量：突破可信度↑")
        elif chg1 >= 0.3 and ratio < 0.75:
            score -= 12
            label = "量价背离"
            bullets.append("上涨缩量：动能不足，假突破风险↑")
        elif chg1 <= -0.3 and ratio >= 1.5:
            score -= 20
            label = "放量下跌"
            bullets.append("下跌放量：抛压确认，不宜抄底接刀")
        elif chg1 <= -0.15 and ratio <= 0.85:
            score += 12
            label = "缩量回踩"
            bullets.append("回调缩量：更像健康回踩而非崩盘")
        elif abs(chg1) < 0.3 and 0.85 <= ratio <= 1.25:
            label = "正常"
            bullets.append("量价平稳")
        elif chg1 <= -0.3 and ratio < 0.7:
            score += 4
            label = "缩量回踩"
            bullets.append("跌但极度缩量：抛压暂缓")

    # 5-day context
    if chg5 is not None:
        bullets.append(f"近5日涨跌 {chg5:+.1f}%")
        if chg5 <= -6 and ratio is not None and ratio >= 1.6:
            score -= 8
            bullets.append("近5日大跌且仍放量：趋势破坏风险")
        if chg5 >= 8 and ratio is not None and ratio < 0.8:
            score -= 6
            bullets.append("近5日大涨但量能跟不上")

    score = float(max(0.0, min(100.0, score)))
    if score >= 68 and label == "正常":
        label = "偏多量能"
    elif score <= 35 and label == "正常":
        label = "偏空量能"

    summary = f"量能确认 **{label}**（{score:.0f}/100）"
    if ratio is not None:
        summary += f" · 量比 {ratio:.2f}x"
    summary += "。"
    return VolumeConfirmReport(
        score=round(score, 1),
        label=label,
        vol_ratio=round(ratio, 3) if ratio is not None else None,
        price_chg_5d_pct=round(chg5, 2) if chg5 is not None else None,
        bullets=bullets,
        summary=summary,
        available=True,
    )


@lru_cache(maxsize=48)
def _fetch_iv_hv(symbol: str, _bucket: str) -> tuple[float | None, float | None]:
    """ATM IV (nearest expiry) and 30d HV. Returns (iv, hv) as decimals."""
    sym = normalize_symbol(symbol)
    iv = None
    hv = None
    try:
        t = yf.Ticker(sym)
        # HV
        try:
            h = t.history(period="3mo")
            if h is not None and not h.empty and "Close" in h.columns:
                rets = h["Close"].pct_change().dropna()
                if len(rets) >= 15:
                    hv = float(rets.tail(30).std() * np.sqrt(252))
        except Exception:
            hv = None
        # IV from nearest option expiry ATM
        try:
            exps = list(t.options or [])
            if exps:
                chain = t.option_chain(exps[0])
                calls = chain.calls
                puts = chain.puts
                spot = None
                try:
                    fi = t.fast_info
                    spot = float(getattr(fi, "last_price", None) or 0) or None
                except Exception:
                    spot = None
                if spot is None and hv is not None and h is not None and not h.empty:
                    spot = float(h["Close"].iloc[-1])
                if spot and spot > 0:
                    frames = []
                    for df in (calls, puts):
                        if df is None or df.empty:
                            continue
                        d = df.copy()
                        if "impliedVolatility" not in d.columns or "strike" not in d.columns:
                            continue
                        d = d.dropna(subset=["impliedVolatility", "strike"])
                        if d.empty:
                            continue
                        d["_dist"] = (d["strike"].astype(float) - spot).abs()
                        frames.append(d)
                    if frames:
                        all_o = pd.concat(frames, ignore_index=True).sort_values("_dist")
                        v = float(all_o.iloc[0]["impliedVolatility"])
                        if v > 3:
                            v = v / 100.0
                        if 0.03 <= v < 3:
                            iv = v
        except Exception:
            iv = None
    except Exception:
        return None, None
    return iv, hv


def analyze_iv_regime(symbol: str) -> IVRegimeReport:
    """
    Rough IV regime for long equity risk control.

    High IV vs HV → event/uncertainty premium → cut full-size longs.
    """
    sym = normalize_symbol(symbol)
    # Skip clearly non-US for options
    if sym.endswith((".HK", ".SS", ".SZ", ".BJ", ".SH")) or (sym.isdigit() and len(sym) == 6):
        return IVRegimeReport(
            label="无数据",
            summary="非美股或无美股期权，跳过 IV 分析。",
            available=False,
        )

    iv, hv = _fetch_iv_hv(sym, cache_bucket(30))
    if iv is None and hv is None:
        return IVRegimeReport(
            label="无数据",
            summary="无法获取 ATM IV / 历史波动（可能无期权或限流）。",
            available=False,
        )

    ratio = None
    if iv is not None and hv is not None and hv > 0.03:
        ratio = iv / hv

    score = 50.0
    notes: list[str] = []
    high_event = False
    label = "正常"

    if iv is not None:
        iv_pct = iv * 100
        notes.append(f"ATM IV ≈ {iv_pct:.0f}%")
        if iv_pct < 18:
            score += 8
            label = "低IV"
            notes.append("隐含波动偏低：期权偏贵风险小，事件溢价不高")
        elif iv_pct < 35:
            score += 2
            label = "正常"
        elif iv_pct < 55:
            score -= 10
            label = "偏高"
            notes.append("IV 偏高：常伴财报/事件，股票隔夜波动↑")
            high_event = True
        else:
            score -= 18
            label = "极高"
            notes.append("IV 极高：事件风险大，不宜满仓追多")
            high_event = True

    if hv is not None:
        notes.append(f"30日 HV ≈ {hv * 100:.0f}%")

    if ratio is not None:
        notes.append(f"IV/HV ≈ {ratio:.2f}x")
        if ratio >= 1.8:
            score -= 12
            high_event = True
            label = "极高" if label != "极高" else label
            notes.append("IV 显著高于已实现波动：市场在定价大波动")
        elif ratio >= 1.35:
            score -= 6
            if label == "正常":
                label = "偏高"
            notes.append("IV 高于 HV：谨慎满仓")
        elif ratio <= 0.85:
            score += 5
            if label == "正常":
                label = "低IV"
            notes.append("IV 低于 HV：波动溢价不高")

    score = float(max(0.0, min(100.0, score)))
    summary = f"IV 环境 **{label}**（{score:.0f}/100）"
    if iv is not None:
        summary += f" · IV {iv * 100:.0f}%"
    if hv is not None:
        summary += f" · HV {hv * 100:.0f}%"
    if high_event:
        summary += " · 事件风险提示"
    summary += "。"

    return IVRegimeReport(
        iv_atm=round(iv, 4) if iv is not None else None,
        hv_30=round(hv, 4) if hv is not None else None,
        iv_hv_ratio=round(ratio, 3) if ratio is not None else None,
        label=label,
        score=round(score, 1),
        high_event_risk=high_event,
        summary=summary,
        available=True,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# False breakout filter (short-term long entries)
# ---------------------------------------------------------------------------


@dataclass
class FalseBreakoutReport:
    """Detect failed / unconfirmed breakouts that trap late longs."""

    score: float = 50.0  # 0-100 higher = safer for long (less false-break risk)
    label: str = "—"  # 突破有效 | 待确认 | 假突破风险 | 正常 | 无数据
    false_break_risk: bool = False  # hard: block 可以入場
    block_breakout_chase: bool = False  # hard: don't chase highs
    level: float | None = None  # recent resistance tested
    bullets: list[str] = field(default_factory=list)
    summary: str = ""
    available: bool = False


def analyze_false_breakout(df: pd.DataFrame) -> FalseBreakoutReport:
    """
    Short-term false breakout heuristics on daily bars (exclude incomplete last bar).

    Flags:
      - Pierce 20d high then close back below (failed breakout)
      - Close above prior high but with long upper wick + next day fails
      - Breakout day thin volume (unconfirmed)
      - 2+ closes above level = healthier confirmation
    """
    if df is None or df.empty or "Close" not in df.columns:
        return FalseBreakoutReport(summary="无价格数据，跳过假突破检测。")

    need = ["High", "Low", "Close"]
    if any(c not in df.columns for c in need):
        return FalseBreakoutReport(summary="缺 High/Low，跳过假突破检测。")

    work = df.dropna(subset=need).copy()
    if len(work) < 30:
        return FalseBreakoutReport(summary="K线不足，假突破检测不可用。")
    # drop incomplete session bar
    if len(work) >= 31:
        work = work.iloc[:-1].copy()

    high = work["High"].astype(float)
    low = work["Low"].astype(float)
    close = work["Close"].astype(float)
    vol = work["Volume"].astype(float) if "Volume" in work.columns else None

    # Prior 20-day high excluding last 3 bars (the breakout window)
    look = 20
    if len(close) < look + 5:
        return FalseBreakoutReport(summary="样本不足。")

    window = work.iloc[-(look + 3) : -3]
    level = float(window["High"].max())
    last3 = work.iloc[-3:]
    c0, c1, c2 = float(close.iloc[-3]), float(close.iloc[-2]), float(close.iloc[-1])
    h0, h1, h2 = float(high.iloc[-3]), float(high.iloc[-2]), float(high.iloc[-1])
    l2 = float(low.iloc[-1])

    score = 55.0
    bullets: list[str] = [f"近端阻力参考（前{look}日高）≈ {level:.2f}"]
    false_risk = False
    block_chase = False
    label = "正常"

    # --- A) Intraday pierce then close back (classic false BO) ---
    pierced = max(h0, h1, h2) > level * 1.001
    closed_back = c2 < level * 0.998
    if pierced and closed_back:
        score -= 28
        false_risk = True
        block_chase = True
        label = "假突破风险"
        bullets.append("近日刺破前高后收盘回到下方 → 假突破/多头陷阱")

    # --- B) Only wick above level, body closes below ---
    upper_wick = h2 - max(c2, float(work["Open"].iloc[-1]) if "Open" in work.columns else c2)
    body = abs(c2 - float(work["Open"].iloc[-1])) if "Open" in work.columns else 0.0
    if h2 > level * 1.002 and c2 < level and upper_wick > max(body * 1.5, level * 0.008):
        score -= 14
        block_chase = True
        if label == "正常":
            label = "待确认"
        bullets.append("长上影刺破阻力后回落 → 不宜追高")

    # --- C) Close above level but no follow-through next day ---
    # if day-2 closed above and day-1 closed back below
    if c1 > level * 1.001 and c2 < level * 0.999:
        score -= 18
        false_risk = True
        block_chase = True
        label = "假突破风险"
        bullets.append("收盘突破后次日回落破位 → 突破失败")

    # --- D) Confirmed breakout: 2 closes above level ---
    closes_above = sum(1 for x in (c0, c1, c2) if x > level * 1.001)
    if closes_above >= 2 and c2 > level:
        score += 16
        label = "突破有效"
        bullets.append(f"近3日有 {closes_above} 根收在前高之上 → 突破较可靠")
        # volume confirm on breakout day (use highest volume among closes above)
        if vol is not None and float(vol.tail(20).mean() or 0) > 0:
            ma = float(vol.tail(20).mean())
            # volume on last close-above bar
            idx_above = [i for i, x in enumerate((c0, c1, c2)) if x > level * 1.001]
            if idx_above:
                # map to last3 rows
                v_break = float(vol.iloc[-3 + idx_above[-1]])
                if v_break >= ma * 1.25:
                    score += 8
                    bullets.append("突破日放量 → 确认加分")
                elif v_break < ma * 0.75:
                    score -= 12
                    block_chase = True
                    if label == "突破有效":
                        label = "待确认"
                    bullets.append("突破日缩量 → 假突破概率↑，等回踩确认")

    # --- E) Single close above only (待确认，可试仓不可猛追) ---
    elif closes_above == 1 and c2 > level:
        score += 4
        label = "待确认"
        block_chase = True  # don't FOMO; wait pullback
        bullets.append("仅1根收在前高上 → 等回踩/再收确认，禁止追高")

    # --- F) Clean pullback under level (not breakout context) ---
    if c2 < level * 0.97 and not pierced:
        score += 4
        if label == "正常":
            label = "正常"
        bullets.append("现价在阻力下方较远处：属回踩/整理，非追突破")

    # Distance to level for chase warning
    dist_pct = (c2 / level - 1.0) * 100 if level > 0 else 0.0
    if dist_pct > 3.0 and not (closes_above >= 2 and c2 > level):
        score -= 8
        block_chase = True
        bullets.append(f"现价相对前高 {dist_pct:+.1f}%：远离结构，追涨风险")

    score = float(max(0.0, min(100.0, score)))
    if false_risk:
        label = "假突破风险"
    elif score >= 68 and label == "正常":
        label = "结构健康"

    summary = f"假突破过滤 **{label}**（{score:.0f}/100）· 阻力≈{level:.2f}"
    if false_risk:
        summary += " · 不宜追多"
    summary += "。"
    return FalseBreakoutReport(
        score=round(score, 1),
        label=label,
        false_break_risk=false_risk,
        block_breakout_chase=block_chase,
        level=round(level, 4),
        bullets=bullets,
        summary=summary,
        available=True,
    )


# ---------------------------------------------------------------------------
# Trend-follow: stock + sector + SPY
# ---------------------------------------------------------------------------


@dataclass
class TrendAlignReport:
    """Follow-the-trend score for short-term longs."""

    score: float = 50.0  # 0-100
    label: str = "—"  # 强跟势 | 跟势 | 中性 | 逆势
    spy_trend: str = "—"  # 多 | 中 | 空
    sector_trend: str = "—"
    stock_trend: str = "—"
    align_count: int = 0  # how many of 3 are bullish
    against_trend: bool = False  # hard: block full entry / prefer 暂缓
    bullets: list[str] = field(default_factory=list)
    summary: str = ""
    available: bool = False
    sector_etf: str = ""


def _series_trend(close: pd.Series) -> tuple[str, float]:
    """
    Return (多|中|空, sub_score 0-100) from SMA20/50 + 10d slope.
    """
    c = close.astype(float).dropna()
    if len(c) < 55:
        if len(c) < 25:
            return "中", 50.0
        sma20 = float(c.tail(20).mean())
        last = float(c.iloc[-1])
        slope = float(c.iloc[-1] / c.iloc[-10] - 1) * 100 if len(c) >= 10 else 0.0
        if last > sma20 and slope > 0:
            return "多", 62.0
        if last < sma20 and slope < 0:
            return "空", 38.0
        return "中", 50.0

    last = float(c.iloc[-1])
    sma20 = float(c.tail(20).mean())
    sma50 = float(c.tail(50).mean())
    slope10 = float(c.iloc[-1] / c.iloc[-11] - 1) * 100
    score = 50.0
    if last > sma20:
        score += 12
    else:
        score -= 12
    if last > sma50:
        score += 12
    else:
        score -= 12
    if sma20 > sma50:
        score += 8
    else:
        score -= 8
    if slope10 > 1.0:
        score += 8
    elif slope10 < -1.0:
        score -= 8
    score = float(max(0.0, min(100.0, score)))
    if score >= 62:
        return "多", score
    if score <= 42:
        return "空", score
    return "中", score


def analyze_trend_align(
    stock_df: pd.DataFrame,
    *,
    sector: str = "",
    industry: str = "",
    period: str = "1y",
) -> TrendAlignReport:
    """
    跟势分：SPY + 板块ETF + 个股 三者趋势是否同向做多。

    短线优先：三者偏多 → 强跟势；SPY/板块空而个股多 → 逆势（暂缓追多）。
    """
    if stock_df is None or stock_df.empty or "Close" not in stock_df.columns:
        return TrendAlignReport(summary="个股数据不足，无法计算跟势。")

    etf, etf_lab = map_sector_etf(sector, industry)
    stock_close = stock_df["Close"].astype(float)
    # drop incomplete last daily bar if very short change? keep full for trend SMAs
    if len(stock_close) >= 5:
        # still use all available; SMA stable enough
        pass

    st_lab, st_sc = _series_trend(stock_close)

    spy_df = _hist_df("SPY", period=period)
    if spy_df.empty:
        spy_lab, spy_sc = "中", 50.0
    else:
        spy_lab, spy_sc = _series_trend(spy_df["Close"])

    sec_df = _hist_df(etf, period=period)
    if sec_df.empty and etf != "SPY":
        sec_df = spy_df
        etf, etf_lab = "SPY", "大盘 SPY（板块ETF失败）"
    if sec_df.empty:
        sec_lab, sec_sc = "中", 50.0
    else:
        sec_lab, sec_sc = _series_trend(sec_df["Close"])

    bullets = [
        f"SPY 趋势 **{spy_lab}**（{spy_sc:.0f}）",
        f"板块 {etf_lab} **{sec_lab}**（{sec_sc:.0f}）",
        f"个股 **{st_lab}**（{st_sc:.0f}）",
    ]

    bull = sum(1 for x in (spy_lab, sec_lab, st_lab) if x == "多")
    bear = sum(1 for x in (spy_lab, sec_lab, st_lab) if x == "空")

    # Weighted: SPY 35% + sector 35% + stock 30%
    score = 0.35 * spy_sc + 0.35 * sec_sc + 0.30 * st_sc
    against = False

    if bull >= 3:
        label = "强跟势"
        score = max(score, 72)
        bullets.append("大盘+板块+个股同向偏多 → 短线做多环境佳")
    elif bull == 2 and bear == 0:
        label = "跟势"
        score = max(score, 62)
        bullets.append("三者至少两者偏多、无空头 → 可跟势做")
    elif st_lab == "多" and (spy_lab == "空" or sec_lab == "空"):
        label = "逆势"
        against = True
        score = min(score, 42)
        bullets.append("个股偏多但大盘/板块偏空 → 逆势反弹，假突破与回撤风险大")
    elif bear >= 2:
        label = "逆势"
        against = True
        score = min(score, 38)
        bullets.append("多数空头结构 → 短线优先观望或只做空头策略")
    else:
        label = "中性"
        bullets.append("多空混杂 → 减小仓位或等方向清晰")

    # Extra: stock making new thrust while SPY weak
    if st_lab == "多" and spy_lab == "空":
        against = True
        score -= 6

    score = float(max(0.0, min(100.0, score)))
    summary = (
        f"跟势 **{label}**（{score:.0f}/100）· "
        f"SPY {spy_lab} / 板块 {sec_lab} / 个股 {st_lab}"
    )
    if against:
        summary += " · 逆势警告"
    summary += "。"

    return TrendAlignReport(
        score=round(score, 1),
        label=label,
        spy_trend=spy_lab,
        sector_trend=sec_lab,
        stock_trend=st_lab,
        align_count=bull,
        against_trend=against,
        bullets=bullets,
        summary=summary,
        available=True,
        sector_etf=etf,
    )


def edge_bundle(
    symbol: str,
    stock_df: pd.DataFrame,
    info: dict[str, Any] | None = None,
    *,
    period: str = "1y",
) -> dict[str, Any]:
    """One-shot helper for SOP: sector RS + volume + IV + false BO + trend align."""
    info = info or {}
    sector = str(info.get("sector") or "")
    industry = str(info.get("industry") or "")
    sec = analyze_sector_rs(stock_df, sector=sector, industry=industry, period=period)
    vol = analyze_volume_confirm(stock_df)
    iv = analyze_iv_regime(symbol)
    fbo = analyze_false_breakout(stock_df)
    trend = analyze_trend_align(
        stock_df, sector=sector, industry=industry, period=period
    )
    return {
        "sector_rs": sec,
        "volume": vol,
        "iv": iv,
        "false_break": fbo,
        "trend_align": trend,
    }
