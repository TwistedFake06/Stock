"""
扩展分析模块：
- 风险（回撤、波动、夏普、VaR）
- 支撑 / 阻力
- 趋势结构
- 基本面评分
- 相对强弱（vs 基准）
- 量价分析
- 综合评分卡

仅供学习参考，不构成投资建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from indicators import enrich


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(val: Any) -> float | None:
    if val is None:
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _grade(score: float | None) -> str:
    """Map 0-100 score to letter-ish grade in Chinese."""
    if score is None:
        return "—"
    if score >= 80:
        return "优秀"
    if score >= 65:
        return "良好"
    if score >= 50:
        return "一般"
    if score >= 35:
        return "偏弱"
    return "较差"


def default_benchmark(symbol: str) -> tuple[str, str]:
    """Pick a reasonable benchmark by market. Returns (ticker, label)."""
    s = symbol.upper()
    if s.endswith(".SS") or s.endswith(".SZ") or (s.isdigit() and len(s) == 6):
        return "000001.SS", "上证指数"
    if s.endswith(".HK"):
        return "^HSI", "恒生指数"
    return "SPY", "标普500 ETF (SPY)"


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@dataclass
class RiskReport:
    total_return_pct: float | None = None
    ann_return_pct: float | None = None
    ann_vol_pct: float | None = None
    sharpe: float | None = None
    max_drawdown_pct: float | None = None
    max_dd_start: str | None = None
    max_dd_end: str | None = None
    var_95_pct: float | None = None  # daily historical VaR
    calmar: float | None = None
    win_rate_pct: float | None = None
    avg_up_pct: float | None = None
    avg_down_pct: float | None = None
    risk_level: str = "—"  # 低 | 中 | 高 | 极高
    summary: str = ""
    equity_curve: pd.Series | None = field(default=None, repr=False)
    drawdown_curve: pd.Series | None = field(default=None, repr=False)


def analyze_risk(df: pd.DataFrame, periods_per_year: float = 252.0) -> RiskReport:
    if df is None or df.empty or "Close" not in df.columns:
        return RiskReport(summary="数据不足，无法做风险分析。")

    work = df[["Close"] + (["Date"] if "Date" in df.columns else [])].copy()
    work = work.dropna(subset=["Close"]).reset_index(drop=True)
    close = work["Close"].astype(float)
    if len(close) < 5:
        return RiskReport(summary="K线过少，无法做风险分析。")

    if "Date" in work.columns:
        dates = pd.to_datetime(work["Date"])
        try:
            dates = dates.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    else:
        dates = pd.RangeIndex(len(close))

    rets = close.pct_change().dropna()
    total = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
    n = len(rets)
    years = max(n / periods_per_year, 1e-9)
    ann_ret = float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100
    ann_vol = float(rets.std() * np.sqrt(periods_per_year) * 100) if n > 1 else None

    rf = 0.02
    sharpe = None
    if ann_vol and ann_vol > 0:
        sharpe = (ann_ret / 100 - rf) / (ann_vol / 100)

    peak = close.cummax()
    dd = (close / peak - 1.0) * 100
    max_dd = float(dd.min())
    end_pos = int(dd.values.argmin())
    peak_pos = int(close.iloc[: end_pos + 1].values.argmax())
    max_dd_start = max_dd_end = None
    try:
        max_dd_start = pd.Timestamp(dates.iloc[peak_pos]).strftime("%Y-%m-%d")
        max_dd_end = pd.Timestamp(dates.iloc[end_pos]).strftime("%Y-%m-%d")
    except Exception:
        pass

    equity = close / close.iloc[0] * 100
    equity.index = dates
    dd.index = dates

    var_95 = float(np.percentile(rets, 5) * 100) if n >= 10 else None
    calmar = (ann_ret / 100) / abs(max_dd / 100) if max_dd < 0 else None

    up = rets[rets > 0]
    down = rets[rets < 0]
    win_rate = float(len(up) / len(rets) * 100) if len(rets) else None
    avg_up = float(up.mean() * 100) if len(up) else None
    avg_down = float(down.mean() * 100) if len(down) else None

    if ann_vol is not None and (ann_vol >= 45 or max_dd <= -40):
        risk_level = "极高"
    elif ann_vol is not None and (ann_vol >= 30 or max_dd <= -25):
        risk_level = "高"
    elif ann_vol is not None and ann_vol <= 15 and max_dd >= -12:
        risk_level = "低"
    else:
        risk_level = "中"

    parts = [
        f"区间收益 {total:+.1f}%",
        f"年化波动约 {ann_vol:.1f}%" if ann_vol is not None else None,
        f"最大回撤 {max_dd:.1f}%",
        f"风险等级 **{risk_level}**",
    ]
    summary = "；".join(p for p in parts if p) + "。"

    return RiskReport(
        total_return_pct=total,
        ann_return_pct=ann_ret,
        ann_vol_pct=ann_vol,
        sharpe=float(sharpe) if sharpe is not None else None,
        max_drawdown_pct=max_dd,
        max_dd_start=max_dd_start,
        max_dd_end=max_dd_end,
        var_95_pct=var_95,
        calmar=float(calmar) if calmar is not None else None,
        win_rate_pct=win_rate,
        avg_up_pct=avg_up,
        avg_down_pct=avg_down,
        risk_level=risk_level,
        summary=summary,
        equity_curve=equity,
        drawdown_curve=dd,
    )


# ---------------------------------------------------------------------------
# Support / Resistance
# ---------------------------------------------------------------------------

@dataclass
class Level:
    price: float
    kind: str  # 支撑 | 阻力 | 枢轴
    strength: str  # 强 | 中 | 弱
    detail: str


@dataclass
class SRReport:
    last_price: float | None = None
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    upside_pct: float | None = None
    downside_pct: float | None = None
    position_in_range: float | None = None  # 0-1 in recent high-low
    pivot: float | None = None
    r1: float | None = None
    s1: float | None = None
    r2: float | None = None
    s2: float | None = None
    levels: list[Level] = field(default_factory=list)
    summary: str = ""


def analyze_support_resistance(df: pd.DataFrame, lookback: int = 60) -> SRReport:
    if df is None or df.empty or "Close" not in df.columns:
        return SRReport(summary="数据不足。")

    data = df.tail(max(lookback, 20)).copy()
    last = float(data["Close"].iloc[-1])
    high = data["High"].astype(float) if "High" in data.columns else data["Close"].astype(float)
    low = data["Low"].astype(float) if "Low" in data.columns else data["Close"].astype(float)
    close = data["Close"].astype(float)

    recent_high = float(high.max())
    recent_low = float(low.min())
    rng = recent_high - recent_low
    pos = (last - recent_low) / rng if rng > 0 else 0.5

    # Classic pivot from last bar's OHLC (or previous day style: use last completed)
    h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
    if len(data) >= 2:
        h, l, c = float(high.iloc[-2]), float(low.iloc[-2]), float(close.iloc[-2])
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)

    levels: list[Level] = []

    # Swing highs/lows (simple local extrema)
    window = 3
    highs_arr = high.values
    lows_arr = low.values
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(window, len(highs_arr) - window):
        if highs_arr[i] == max(highs_arr[i - window : i + window + 1]):
            swing_highs.append(float(highs_arr[i]))
        if lows_arr[i] == min(lows_arr[i - window : i + window + 1]):
            swing_lows.append(float(lows_arr[i]))

    def _cluster(prices: list[float], tol_pct: float = 0.015) -> list[tuple[float, int]]:
        if not prices:
            return []
        prices = sorted(prices)
        clusters: list[list[float]] = [[prices[0]]]
        for p in prices[1:]:
            if abs(p - clusters[-1][-1]) / clusters[-1][-1] <= tol_pct:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        out = [(float(np.mean(c)), len(c)) for c in clusters]
        return out

    for price, cnt in _cluster(swing_lows):
        if price < last:
            strength = "强" if cnt >= 3 else "中" if cnt >= 2 else "弱"
            levels.append(Level(price, "支撑", strength, f"波段低点聚类 ×{cnt}"))
    for price, cnt in _cluster(swing_highs):
        if price > last:
            strength = "强" if cnt >= 3 else "中" if cnt >= 2 else "弱"
            levels.append(Level(price, "阻力", strength, f"波段高点聚类 ×{cnt}"))

    # SMA as dynamic S/R
    if len(df) >= 20:
        en = enrich(df)
        for col, name in (("SMA20", "SMA20"), ("SMA60", "SMA60")):
            if col in en.columns:
                v = float(en[col].iloc[-1])
                kind = "支撑" if last >= v else "阻力"
                levels.append(Level(v, kind, "中", f"动态均线 {name}"))

    levels.append(Level(pivot, "枢轴", "中", "经典枢轴点 Pivot"))
    levels.append(Level(r1, "阻力", "中", "Pivot R1"))
    levels.append(Level(s1, "支撑", "中", "Pivot S1"))
    levels.append(Level(r2, "阻力", "弱", "Pivot R2"))
    levels.append(Level(s2, "支撑", "弱", "Pivot S2"))
    levels.append(Level(recent_high, "阻力", "强", f"近{lookback}根高点"))
    levels.append(Level(recent_low, "支撑", "强", f"近{lookback}根低点"))

    # Deduplicate near levels
    levels = sorted(levels, key=lambda x: x.price)
    deduped: list[Level] = []
    for lv in levels:
        if not deduped or abs(lv.price - deduped[-1].price) / last > 0.008:
            deduped.append(lv)
        else:
            # keep stronger
            rank = {"强": 3, "中": 2, "弱": 1}
            if rank.get(lv.strength, 0) > rank.get(deduped[-1].strength, 0):
                deduped[-1] = lv
    levels = deduped

    supports = [lv.price for lv in levels if lv.kind == "支撑" and lv.price < last]
    resists = [lv.price for lv in levels if lv.kind == "阻力" and lv.price > last]
    ns = max(supports) if supports else recent_low
    nr = min(resists) if resists else recent_high
    up = (nr / last - 1) * 100
    down = (ns / last - 1) * 100

    summary = (
        f"现价 {last:.2f}，位于近区间 {pos:.0%} 位置；"
        f"最近支撑约 {ns:.2f}（{down:+.1f}%），"
        f"最近阻力约 {nr:.2f}（{up:+.1f}%）。"
    )

    return SRReport(
        last_price=last,
        nearest_support=ns,
        nearest_resistance=nr,
        upside_pct=up,
        downside_pct=down,
        position_in_range=pos,
        pivot=pivot,
        r1=r1,
        s1=s1,
        r2=r2,
        s2=s2,
        levels=levels,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Trend structure
# ---------------------------------------------------------------------------

@dataclass
class TrendReport:
    short_trend: str = "—"  # 上升 | 下降 | 震荡
    medium_trend: str = "—"
    structure: str = "—"  # 高高低低 / 低低高高 / 混乱
    higher_highs: bool | None = None
    higher_lows: bool | None = None
    adx_proxy: float | None = None  # simplified trend strength 0-100
    strength_label: str = "—"  # 强趋势 | 中等 | 弱/震荡
    summary: str = ""


def analyze_trend(df: pd.DataFrame) -> TrendReport:
    if df is None or df.empty or "Close" not in df.columns or len(df) < 30:
        return TrendReport(summary="数据不足，无法判断趋势结构。")

    data = enrich(df)
    close = data["Close"].astype(float)
    last = float(close.iloc[-1])
    sma20 = float(data["SMA20"].iloc[-1]) if "SMA20" in data.columns else last
    sma60 = float(data["SMA60"].iloc[-1]) if "SMA60" in data.columns else last

    # Short: last 20 vs prior 20 slope
    def _slope_label(series: pd.Series) -> str:
        if len(series) < 5:
            return "震荡"
        chg = series.iloc[-1] / series.iloc[0] - 1
        if chg > 0.03:
            return "上升"
        if chg < -0.03:
            return "下降"
        return "震荡"

    short = _slope_label(close.iloc[-20:])
    medium = _slope_label(close.iloc[-60:]) if len(close) >= 60 else _slope_label(close)

    # Swing structure last ~40 bars
    window = 2
    highs = data["High"].astype(float).values if "High" in data.columns else close.values
    lows = data["Low"].astype(float).values if "Low" in data.columns else close.values
    sh, sl = [], []
    seg = highs[-50:] if len(highs) >= 50 else highs
    segl = lows[-50:] if len(lows) >= 50 else lows
    for i in range(window, len(seg) - window):
        if seg[i] == max(seg[i - window : i + window + 1]):
            sh.append(seg[i])
        if segl[i] == min(segl[i - window : i + window + 1]):
            sl.append(segl[i])

    hh = hl = None
    if len(sh) >= 2:
        hh = sh[-1] > sh[-2]
    if len(sl) >= 2:
        hl = sl[-1] > sl[-2]

    if hh and hl:
        structure = "上升结构（更高高点 + 更高低点）"
    elif hh is False and hl is False:
        structure = "下降结构（更低高点 + 更低低点）"
    else:
        structure = "结构混乱 / 转换中"

    # ADX-like: normalized |SMA20-SMA60| / ATR proxy
    tr = (data["High"] - data["Low"]).astype(float) if "High" in data.columns else close.diff().abs()
    atr = float(tr.tail(14).mean()) if len(tr) else 0
    spread = abs(sma20 - sma60)
    adx_proxy = min(100.0, (spread / atr) * 25) if atr > 0 else 0.0
    if adx_proxy >= 40 and short == medium and short != "震荡":
        strength = "强趋势"
    elif adx_proxy >= 20:
        strength = "中等"
    else:
        strength = "弱/震荡"

    # Price vs MAs refine short
    if last > sma20 > sma60 and short != "下降":
        short = "上升"
    elif last < sma20 < sma60 and short != "上升":
        short = "下降"

    summary = (
        f"短期趋势 **{short}**，中期 **{medium}**；{structure}；"
        f"趋势强度 {strength}（强度指数约 {adx_proxy:.0f}）。"
    )
    return TrendReport(
        short_trend=short,
        medium_trend=medium,
        structure=structure,
        higher_highs=hh,
        higher_lows=hl,
        adx_proxy=adx_proxy,
        strength_label=strength,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

@dataclass
class FundaItem:
    name: str
    value: Any
    display: str
    verdict: str  # 偏多 | 偏空 | 中性 | —
    note: str


@dataclass
class FundaReport:
    score: float | None = None  # 0-100
    grade: str = "—"
    items: list[FundaItem] = field(default_factory=list)
    summary: str = ""
    available: bool = False


def analyze_fundamentals(info: dict[str, Any]) -> FundaReport:
    if not info:
        return FundaReport(summary="无基本面数据。")

    items: list[FundaItem] = []
    scores: list[float] = []

    def add(name, raw, display, verdict, note, sc: float | None = None):
        items.append(FundaItem(name, raw, display, verdict, note))
        if sc is not None:
            scores.append(sc)

    pe = _f(info.get("trailingPE") or info.get("forwardPE"))
    if pe is not None:
        if 0 < pe < 15:
            v, sc, note = "偏多", 80, "估值偏低（相对常见区间）"
        elif pe < 25:
            v, sc, note = "中性", 60, "估值适中"
        elif pe < 40:
            v, sc, note = "偏空", 40, "估值偏高"
        else:
            v, sc, note = "偏空", 25, "估值较高，需成长匹配"
        if pe <= 0:
            v, sc, note = "中性", 45, "PE 异常/亏损"
        add("市盈率 PE", pe, f"{pe:.2f}", v, note, sc)

    pb = _f(info.get("priceToBook"))
    if pb is not None:
        if 0 < pb < 1.5:
            v, sc, note = "偏多", 75, "市净率较低"
        elif pb < 4:
            v, sc, note = "中性", 55, "市净率正常"
        else:
            v, sc, note = "偏空", 35, "市净率偏高"
        add("市净率 PB", pb, f"{pb:.2f}", v, note, sc)

    ps = _f(info.get("priceToSalesTrailing12Months"))
    if ps is not None:
        if 0 < ps < 2:
            v, sc, note = "偏多", 70, "市销率偏低"
        elif ps < 6:
            v, sc, note = "中性", 55, "市销率适中"
        else:
            v, sc, note = "偏空", 35, "市销率偏高"
        add("市销率 PS", ps, f"{ps:.2f}", v, note, sc)

    peg = _f(info.get("pegRatio"))
    if peg is not None and peg > 0:
        if peg < 1:
            v, sc, note = "偏多", 80, "PEG<1，成长相对便宜"
        elif peg < 2:
            v, sc, note = "中性", 55, "PEG 尚可"
        else:
            v, sc, note = "偏空", 35, "PEG 偏高"
        add("PEG", peg, f"{peg:.2f}", v, note, sc)

    roe = _f(info.get("returnOnEquity"))
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) <= 1.5 else roe
        if roe_pct >= 15:
            v, sc, note = "偏多", 85, "盈利能力强"
        elif roe_pct >= 8:
            v, sc, note = "中性", 60, "盈利能力一般"
        else:
            v, sc, note = "偏空", 35, "盈利能力偏弱"
        add("ROE", roe_pct, f"{roe_pct:.1f}%", v, note, sc)

    pm = _f(info.get("profitMargins"))
    if pm is not None:
        pm_pct = pm * 100 if abs(pm) <= 1.5 else pm
        if pm_pct >= 15:
            v, sc, note = "偏多", 80, "净利率优秀"
        elif pm_pct >= 5:
            v, sc, note = "中性", 55, "净利率尚可"
        else:
            v, sc, note = "偏空", 35, "净利率偏低"
        add("净利率", pm_pct, f"{pm_pct:.1f}%", v, note, sc)

    rg = _f(info.get("revenueGrowth"))
    if rg is not None:
        rg_pct = rg * 100 if abs(rg) <= 2 else rg
        if rg_pct >= 15:
            v, sc, note = "偏多", 85, "营收高增长"
        elif rg_pct >= 0:
            v, sc, note = "中性", 55, "营收正增长"
        else:
            v, sc, note = "偏空", 30, "营收下滑"
        add("营收增长", rg_pct, f"{rg_pct:+.1f}%", v, note, sc)

    eg = _f(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))
    if eg is not None:
        eg_pct = eg * 100 if abs(eg) <= 2 else eg
        if eg_pct >= 15:
            v, sc, note = "偏多", 85, "盈利高增长"
        elif eg_pct >= 0:
            v, sc, note = "中性", 55, "盈利正增长"
        else:
            v, sc, note = "偏空", 30, "盈利下滑"
        add("盈利增长", eg_pct, f"{eg_pct:+.1f}%", v, note, sc)

    de = _f(info.get("debtToEquity"))
    if de is not None:
        # yfinance often gives percent-like 50 = 0.5
        de_v = de / 100 if de > 10 else de
        if de_v < 0.5:
            v, sc, note = "偏多", 75, "负债率较低"
        elif de_v < 1.5:
            v, sc, note = "中性", 55, "负债可控"
        else:
            v, sc, note = "偏空", 35, "负债偏高"
        add("资产负债(D/E)", de, f"{de:.2f}", v, note, sc)

    dy = _f(info.get("dividendYield"))
    if dy is not None:
        dy_pct = dy * 100 if dy < 1 else dy
        if dy_pct >= 2:
            v, sc, note = "偏多", 70, "股息率有吸引力"
        elif dy_pct > 0:
            v, sc, note = "中性", 55, "有分红"
        else:
            v, sc, note = "中性", 50, "几乎无股息"
        add("股息率", dy_pct, f"{dy_pct:.2f}%", v, note, sc)

    # Soft factors
    target = _f(info.get("targetMeanPrice"))
    price = _f(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("last_price"))
    if target and price and price > 0:
        upside = (target / price - 1) * 100
        if upside >= 15:
            v, sc, note = "偏多", 75, f"分析师目标价隐含 {upside:+.1f}%"
        elif upside >= 0:
            v, sc, note = "中性", 55, f"目标价隐含 {upside:+.1f}%"
        else:
            v, sc, note = "偏空", 35, f"目标价隐含 {upside:+.1f}%"
        add("目标价空间", upside, f"{upside:+.1f}%", v, note, sc)

    rec = info.get("recommendationKey") or info.get("recommendationMean")
    if rec is not None:
        if isinstance(rec, str):
            mapping = {
                "strong_buy": ("偏多", 90, "评级 strong_buy"),
                "buy": ("偏多", 75, "评级 buy"),
                "hold": ("中性", 50, "评级 hold"),
                "underperform": ("偏空", 30, "评级 underperform"),
                "sell": ("偏空", 20, "评级 sell"),
            }
            v, sc, note = mapping.get(rec.lower(), ("中性", 50, f"评级 {rec}"))
            add("分析师评级", rec, str(rec), v, note, sc)
        else:
            # 1=strong buy ... 5=sell
            mean = float(rec)
            if mean <= 2.0:
                v, sc, note = "偏多", 80, f"评级均值 {mean:.2f}（越低越好）"
            elif mean <= 3.0:
                v, sc, note = "中性", 55, f"评级均值 {mean:.2f}"
            else:
                v, sc, note = "偏空", 30, f"评级均值 {mean:.2f}"
            add("分析师评级", mean, f"{mean:.2f}", v, note, sc)

    if not items:
        return FundaReport(
            summary="该标的基本面字段较少（常见于部分 A股/港股），请结合其他分析。",
            available=False,
        )

    score = float(np.mean(scores)) if scores else None
    grade = _grade(score)
    bull = sum(1 for i in items if i.verdict == "偏多")
    bear = sum(1 for i in items if i.verdict == "偏空")
    summary = (
        f"基本面综合 **{grade}**"
        + (f"（{score:.0f}分）" if score is not None else "")
        + f"；偏多 {bull} 项 / 偏空 {bear} 项。"
    )
    return FundaReport(score=score, grade=grade, items=items, summary=summary, available=True)


# ---------------------------------------------------------------------------
# Relative strength
# ---------------------------------------------------------------------------

@dataclass
class RSReport:
    benchmark: str = ""
    bench_label: str = ""
    stock_return_pct: float | None = None
    bench_return_pct: float | None = None
    alpha_pct: float | None = None  # excess return
    beta: float | None = None
    corr: float | None = None
    rs_rating: str = "—"  # 强于大盘 | 同步 | 弱于大盘
    summary: str = ""
    relative_curve: pd.DataFrame | None = field(default=None, repr=False)


def analyze_relative_strength(
    stock_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    benchmark: str = "",
    bench_label: str = "",
) -> RSReport:
    if stock_df is None or stock_df.empty or bench_df is None or bench_df.empty:
        return RSReport(
            benchmark=benchmark,
            bench_label=bench_label,
            summary="无法计算相对强弱（缺少股票或基准数据）。",
        )

    s = stock_df.set_index("Date")["Close"].astype(float)
    b = bench_df.set_index("Date")["Close"].astype(float)
    # timezone-naive align
    s.index = pd.to_datetime(s.index).tz_localize(None)
    b.index = pd.to_datetime(b.index).tz_localize(None)
    joined = pd.concat([s.rename("stock"), b.rename("bench")], axis=1).dropna()
    if len(joined) < 10:
        return RSReport(
            benchmark=benchmark,
            bench_label=bench_label,
            summary="重叠交易日过少，无法计算相对强弱。",
        )

    stock_ret = (joined["stock"].iloc[-1] / joined["stock"].iloc[0] - 1) * 100
    bench_ret = (joined["bench"].iloc[-1] / joined["bench"].iloc[0] - 1) * 100
    alpha = stock_ret - bench_ret

    rs = joined["stock"].pct_change().dropna()
    rb = joined["bench"].pct_change().dropna()
    aligned = pd.concat([rs, rb], axis=1).dropna()
    beta = corr = None
    if len(aligned) > 5 and aligned.iloc[:, 1].var() > 0:
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        beta = float(cov[0, 1] / cov[1, 1])
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    if alpha >= 5:
        rating = "强于大盘"
    elif alpha <= -5:
        rating = "弱于大盘"
    else:
        rating = "同步大盘"

    rel = pd.DataFrame(
        {
            "股票": joined["stock"] / joined["stock"].iloc[0] * 100,
            "基准": joined["bench"] / joined["bench"].iloc[0] * 100,
            "相对强度": (joined["stock"] / joined["bench"])
            / (joined["stock"].iloc[0] / joined["bench"].iloc[0])
            * 100,
        }
    )

    summary = (
        f"相对 **{bench_label or benchmark}**：股票 {stock_ret:+.1f}% / "
        f"基准 {bench_ret:+.1f}% / 超额 {alpha:+.1f}% → **{rating}**"
        + (f"；β≈{beta:.2f}" if beta is not None else "")
        + (f"；相关 {corr:.2f}" if corr is not None else "")
        + "。"
    )
    return RSReport(
        benchmark=benchmark,
        bench_label=bench_label,
        stock_return_pct=float(stock_ret),
        bench_return_pct=float(bench_ret),
        alpha_pct=float(alpha),
        beta=beta,
        corr=corr,
        rs_rating=rating,
        summary=summary,
        relative_curve=rel,
    )


# ---------------------------------------------------------------------------
# Volume / price action extras
# ---------------------------------------------------------------------------

@dataclass
class VolumeReport:
    vol_ratio: float | None = None  # last / ma20
    trend: str = "—"  # 放量 | 缩量 | 正常
    price_volume: str = "—"  # 量价齐升 等
    obv_trend: str = "—"
    summary: str = ""


def analyze_volume(df: pd.DataFrame) -> VolumeReport:
    if df is None or df.empty or "Volume" not in df.columns:
        return VolumeReport(summary="无成交量数据。")

    vol = df["Volume"].astype(float)
    close = df["Close"].astype(float)
    if vol.sum() <= 0 or len(vol) < 5:
        return VolumeReport(summary="成交量数据无效。")

    ma = vol.rolling(20, min_periods=5).mean()
    v_last = float(vol.iloc[-1])
    v_ma = float(ma.iloc[-1]) if not np.isnan(ma.iloc[-1]) else None
    ratio = v_last / v_ma if v_ma and v_ma > 0 else None

    if ratio is None:
        trend = "—"
    elif ratio >= 1.5:
        trend = "放量"
    elif ratio <= 0.6:
        trend = "缩量"
    else:
        trend = "正常"

    ret5 = close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else 0
    vol5 = vol.iloc[-5:].mean() / vol.iloc[-25:-5].mean() if len(vol) >= 25 else 1

    if ret5 > 0.02 and vol5 > 1.2:
        pv = "量价齐升"
    elif ret5 < -0.02 and vol5 > 1.2:
        pv = "放量下跌"
    elif ret5 > 0.02 and vol5 < 0.8:
        pv = "缩量上涨（动能存疑）"
    elif ret5 < -0.02 and vol5 < 0.8:
        pv = "缩量下跌（抛压减缓）"
    else:
        pv = "量价平衡"

    # OBV simple trend
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * vol).cumsum()
    if len(obv) >= 20:
        if obv.iloc[-1] > obv.iloc[-20]:
            obv_t = "OBV 上升（资金偏流入）"
        elif obv.iloc[-1] < obv.iloc[-20]:
            obv_t = "OBV 下降（资金偏流出）"
        else:
            obv_t = "OBV 走平"
    else:
        obv_t = "—"

    summary = (
        f"量能 **{trend}**"
        + (f"（量比≈{ratio:.2f}）" if ratio else "")
        + f"；{pv}；{obv_t}。"
    )
    return VolumeReport(
        vol_ratio=ratio,
        trend=trend,
        price_volume=pv,
        obv_trend=obv_t,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Scorecard (combine)
# ---------------------------------------------------------------------------

@dataclass
class Scorecard:
    technical_score: float | None = None  # 0-100 from bias
    funda_score: float | None = None
    risk_score: float | None = None  # higher = safer / better risk-adjusted
    rs_score: float | None = None
    total_score: float | None = None
    total_grade: str = "—"
    stance: str = "—"  # 综合偏多 | 中性 | 偏空
    bullets: list[str] = field(default_factory=list)
    summary: str = ""


def build_scorecard(
    bias_score: float | None,
    funda: FundaReport | None,
    risk: RiskReport | None,
    rs: RSReport | None,
) -> Scorecard:
    tech = None
    if bias_score is not None:
        tech = max(0.0, min(100.0, (bias_score + 100) / 2))

    funda_s = funda.score if funda and funda.available else None

    risk_s = None
    if risk and risk.sharpe is not None:
        # Map sharpe -1..2 -> 0..100
        risk_s = max(0.0, min(100.0, (risk.sharpe + 1) / 3 * 100))
        if risk.max_drawdown_pct is not None:
            # penalize deep drawdown
            risk_s = max(0.0, risk_s + risk.max_drawdown_pct / 2)  # dd negative

    rs_s = None
    if rs and rs.alpha_pct is not None:
        rs_s = max(0.0, min(100.0, 50 + rs.alpha_pct * 2))

    parts = [x for x in (tech, funda_s, risk_s, rs_s) if x is not None]
    total = float(np.mean(parts)) if parts else None
    grade = _grade(total)

    if total is None:
        stance = "—"
    elif total >= 60:
        stance = "综合偏多"
    elif total <= 40:
        stance = "综合偏空"
    else:
        stance = "综合中性"

    bullets = []
    if tech is not None:
        bullets.append(f"技术面 {tech:.0f} 分")
    if funda_s is not None:
        bullets.append(f"基本面 {funda_s:.0f} 分（{funda.grade}）")
    if risk_s is not None:
        bullets.append(f"风险调整 {risk_s:.0f} 分（风险{risk.risk_level}）")
    if rs_s is not None:
        bullets.append(f"相对强弱 {rs_s:.0f} 分（{rs.rs_rating}）")

    summary = (
        f"**{stance}** · 综合评分 **{total:.0f}**（{grade}）。" if total is not None else "评分数据不足。"
    )
    if bullets:
        summary += " " + "；".join(bullets) + "。"

    return Scorecard(
        technical_score=tech,
        funda_score=funda_s,
        risk_score=risk_s,
        rs_score=rs_s,
        total_score=total,
        total_grade=grade,
        stance=stance,
        bullets=bullets,
        summary=summary,
    )
