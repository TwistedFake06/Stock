"""
大盘 ETF · 仅 Vertical Spread 分析。

1) 判断当前方向：看多 / 看空 / 中性
2) 扫描垂直价差候选，按方向推荐「最佳买入/开仓」结构

四种 vertical：
- Bull Put Credit  卖高Put + 买低Put   （偏多收权利金）
- Bull Call Debit  买低Call + 卖高Call （看多付权利金）
- Bear Call Credit 卖低Call + 买高Call （偏空收权利金）
- Bear Put Debit   买高Put + 卖低Put   （看空付权利金）

仅限 QQQ/VOO/SPY 等白名单；数据 Yahoo 延迟；仅供学习。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import enrich

INDEX_ETF_WHITELIST = {
    "QQQ": "纳斯达克100 ETF",
    "VOO": "标普500 ETF (Vanguard)",
    "SPY": "标普500 ETF (SPDR)",
    "IVV": "标普500 ETF (iShares)",
    "DIA": "道琼斯 ETF",
    "IWM": "罗素2000 ETF",
    "VTI": "全美股市 ETF",
    "VT": "全球股市 ETF",
    "EFA": "发达市场 ETF",
}

LEVERAGED = {"QLD", "SSO", "TQQQ", "UPRO", "SPXL"}


def is_options_eligible(symbol: str) -> bool:
    return options_symbol(symbol) in INDEX_ETF_WHITELIST


def options_symbol(symbol: str) -> str:
    return symbol.strip().upper().split(".")[0]


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

@dataclass
class DirectionReport:
    direction: str  # 看多 | 看空 | 中性
    strength: str  # 强 | 中 | 弱
    score: float  # -100 .. +100
    preferred_verticals: list[str]  # codes ordered
    reasons: list[str] = field(default_factory=list)
    summary: str = ""
    style_hint: str = ""  # 偏信用价差 | 偏借方价差 | 观望


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
    else:
        score -= 10
        reasons.append("SMA5 < SMA20（短线偏空）")

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

    score = float(np.clip(score, -100, 100))

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
        style = "方向不明：若交易，两侧信用价差均可，仓位宜小；避免重仓 debit"

    abs_s = abs(score)
    strength = "强" if abs_s >= 45 else "中" if abs_s >= 25 else "弱"

    summary = (
        f"方向 **{direction}**（强度 {strength}，得分 {score:+.0f}）。{style}"
    )
    return DirectionReport(
        direction=direction,
        strength=strength,
        score=round(score, 1),
        preferred_verticals=preferred,
        reasons=reasons,
        summary=summary,
        style_hint=style,
    )


# ---------------------------------------------------------------------------
# Legs / chain helpers
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    right: str
    strike: float
    side: str
    mid: float
    bid: float
    ask: float
    iv: float | None
    oi: float | None
    volume: float | None
    delta_proxy: float | None = None
    # 成交假设：卖用 bid、买用 ask（自然成交）；无报价时回退 mid
    fill: float = 0.0
    fill_source: str = "mid"  # bid | ask | mid


@dataclass
class SpreadIdea:
    name: str
    code: str  # bull_put | bull_call | bear_call | bear_put
    structure: str  # Credit Vertical | Debit Vertical
    thesis: str
    net_credit: float | None
    net_debit: float | None
    max_profit: float
    max_loss: float
    breakevens: list[float]
    width: float
    pop_est: float | None  # = win_rate_profit（兼容旧字段）
    score: float
    dte: int
    expiry: str
    legs: list[Leg]
    notes: list[str] = field(default_factory=list)
    risk_reward: float | None = None
    otm_label: str = ""  # e.g. 约2% OTM
    rank_reason: str = ""
    # 胜率（对数正态 + IV/HV 估算）
    win_rate_profit: float | None = None  # 到期有利润的概率 %
    win_rate_max: float | None = None  # 拿到最大利润的概率 %（信用短腿OTM）
    win_rate_method: str = ""  # 说明
    expected_value: float | None = None  # 粗算期望 $/张
    # 流动性
    liquidity_score: float | None = None  # 0-100
    liquidity_label: str = ""  # 高 / 中 / 低 / 未知
    liquidity_detail: str = ""  # 白话说明
    # 定价：自然成交（bid/ask）还是 mid 回退
    pricing_mode: str = "natural"  # natural | mid_mixed | mid_only
    # 50% 止盈：买回/卖出目标价（$/股）与浮盈（$/张）
    metric_half_buyback: float | None = None
    metric_half_profit: float | None = None

    def __post_init__(self) -> None:
        # 兼容旧缓存 / 不完整构造，保证属性始终存在
        if not hasattr(self, "win_rate_profit"):
            self.win_rate_profit = None
        if not hasattr(self, "win_rate_max"):
            self.win_rate_max = None
        if not hasattr(self, "win_rate_method"):
            self.win_rate_method = ""
        if not hasattr(self, "expected_value"):
            self.expected_value = None
        if not hasattr(self, "liquidity_score"):
            self.liquidity_score = None
        if not hasattr(self, "liquidity_label"):
            self.liquidity_label = ""
        if not hasattr(self, "liquidity_detail"):
            self.liquidity_detail = ""
        if not hasattr(self, "pricing_mode"):
            self.pricing_mode = "natural"
        if not hasattr(self, "metric_half_buyback"):
            self.metric_half_buyback = None
        if not hasattr(self, "metric_half_profit"):
            self.metric_half_profit = None
        if self.pop_est is None and self.win_rate_profit is not None:
            self.pop_est = self.win_rate_profit


@dataclass
class OptionsReport:
    symbol: str
    label: str
    spot: float
    eligible: bool
    message: str
    direction: DirectionReport | None = None
    expiries: list[str] = field(default_factory=list)
    selected_expiry: str | None = None
    dte: int | None = None
    iv_atm: float | None = None
    ideas: list[SpreadIdea] = field(default_factory=list)
    best: SpreadIdea | None = None
    best_alt: SpreadIdea | None = None
    best_winrate: SpreadIdea | None = None  # 全体最高「有利润」胜率
    best_winrate_aligned: SpreadIdea | None = None  # 与方向契合的最高胜率
    best_playbook: SpreadIdea | None = None  # 贴合常见实战策略
    best_playbook_wr: SpreadIdea | None = None  # 实战规则下最高赢面
    playbook_table: list = field(default_factory=list)
    regime: str = "—"
    summary: str = ""
    action_plan: list[str] = field(default_factory=list)
    # 报价质量
    after_hours: bool = False
    pricing_note: str = ""
    quote_warning: str = ""
    filtered_out: int = 0  # 硬过滤剔除数


# ---------------------------------------------------------------------------
# Pricing helpers (bid/ask natural fill)
# ---------------------------------------------------------------------------

# Hard filter thresholds (credit fill = credit / width)
HARD_MIN_LIQ_SCORE = 28.0
HARD_CREDIT_FILL_LO = 0.12
HARD_CREDIT_FILL_HI = 0.45
HARD_DEBIT_FILL_HI = 0.78
HARD_MIN_CREDIT = 0.08
HARD_MIN_AVG_OI = 15.0


def _mid(bid: float, ask: float, last: float) -> float:
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last and last > 0:
        return float(last)
    if ask and ask > 0:
        return float(ask)
    if bid and bid > 0:
        return float(bid)
    return 0.0


def _fill_price(side: str, bid: float, ask: float, mid: float) -> tuple[float, str]:
    """
    自然成交假设：卖腿吃 bid（别人出价买你的），买腿付 ask（你按卖价买）。
    无有效 bid/ask 时回退 mid（盘后常见）。
    """
    b = float(bid or 0)
    a = float(ask or 0)
    m = float(mid or 0)
    if side == "sell":
        if b > 0:
            return b, "bid"
        return m, "mid"
    # buy
    if a > 0:
        return a, "ask"
    return m, "mid"


def _pricing_mode_for_legs(legs: list[Leg]) -> str:
    srcs = {getattr(lg, "fill_source", "mid") or "mid" for lg in legs}
    if srcs <= {"bid", "ask"}:
        return "natural"
    if "bid" in srcs or "ask" in srcs:
        return "mid_mixed"
    return "mid_only"


def _is_us_rth() -> bool:
    """美东常规交易时段 Mon–Fri 09:30–16:00。"""
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # 粗略按 UTC-4（夏令）回退
        now = datetime.utcnow() - timedelta(hours=4)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return (9 * 60 + 30) <= mins < (16 * 60)


def half_profit_close_price(idea: SpreadIdea) -> float | None:
    """
    50% 最大利润时的平仓目标价（$/股）。
    信用：开仓收 C → 买回约 C*0.5 即赚一半。
    借方：开仓付 D → 卖出约 D + 0.5*(W-D) 即赚一半。
    一律用已入账的 net_credit/net_debit（两位小数）计算，避免舍入不一致。
    """
    if idea.net_credit is not None:
        c = round(float(idea.net_credit), 2)
        if c <= 0:
            return None
        return round(c * 0.5, 2)
    if idea.net_debit is not None:
        d = round(float(idea.net_debit), 2)
        w = float(idea.width or 0)
        if d <= 0:
            return None
        if w > d:
            return round(d + 0.5 * (w - d), 2)
        half_ps = float(idea.max_profit or 0) / 200.0
        return round(d + half_ps, 2)
    return None


def passes_hard_filters(idea: SpreadIdea) -> bool:
    """流动性 + 权利金/宽度硬门槛；不过滤掉则不应出现在推荐列表。"""
    liq = getattr(idea, "liquidity_score", None)
    label = getattr(idea, "liquidity_label", "") or ""
    if label == "很差":
        return False
    if liq is not None and float(liq) < HARD_MIN_LIQ_SCORE:
        return False

    w = float(idea.width or 0)
    if w <= 0:
        return False

    if idea.net_credit is not None:
        c = float(idea.net_credit)
        if c < HARD_MIN_CREDIT:
            return False
        fill = c / w
        if fill < HARD_CREDIT_FILL_LO or fill > HARD_CREDIT_FILL_HI:
            return False
    elif idea.net_debit is not None:
        d = float(idea.net_debit)
        if d < 0.05:
            return False
        fill = d / w
        if fill > HARD_DEBIT_FILL_HI:
            return False
    else:
        return False

    # 两腿平均 OI 过低且流动性一般 → 丢弃
    ois = [float(lg.oi or 0) for lg in idea.legs]
    avg_oi = sum(ois) / max(len(ois), 1)
    if avg_oi < HARD_MIN_AVG_OI and (liq is None or float(liq) < 45):
        return False

    return True


def _sane_option_mid(
    mid: float,
    bid: float,
    ask: float,
    last: float,
    strike: float,
    spot: float,
    right: str,
) -> float:
    """Drop clearly stale lastPrice (common after hours on Yahoo)."""
    if mid <= 0:
        return 0.0
    has_nbbo = bid > 0 and ask > 0
    if has_nbbo:
        return mid
    # last-only: OTM premium cannot be huge (Yahoo last often stale)
    if right == "call" and strike >= spot:
        otm = strike - spot
        cap = max(spot * 0.012, otm * 0.20 + spot * 0.004)
        if mid > cap:
            return 0.0
    if right == "put" and strike <= spot:
        otm = spot - strike
        cap = max(spot * 0.012, otm * 0.20 + spot * 0.004)
        if mid > cap:
            return 0.0
    # ITM last should be near intrinsic
    if right == "call" and strike < spot:
        intrinsic = spot - strike
        if mid > intrinsic * 1.25 + spot * 0.008 or mid < intrinsic * 0.5:
            return 0.0
    if right == "put" and strike > spot:
        intrinsic = strike - spot
        if mid > intrinsic * 1.25 + spot * 0.008 or mid < intrinsic * 0.5:
            return 0.0
    return mid

def _parse_chain(df: pd.DataFrame, spot: float, right: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ("bid", "ask", "lastPrice", "strike", "impliedVolatility", "volume", "openInterest"):
        if col not in out.columns:
            out[col] = np.nan
    mids = []
    for b, a, lp, k in zip(out["bid"], out["ask"], out["lastPrice"], out["strike"]):
        raw = _mid(float(b or 0), float(a or 0), float(lp or 0))
        mids.append(
            _sane_option_mid(
                raw,
                float(b or 0),
                float(a or 0),
                float(lp or 0),
                float(k),
                spot,
                right,
            )
        )
    out["mid"] = mids
    out["spread"] = (out["ask"].fillna(0) - out["bid"].fillna(0)).clip(lower=0)
    out["spread_pct"] = np.where(out["mid"] > 0, out["spread"] / out["mid"], 9.9)
    out["oi"] = out["openInterest"].fillna(0)
    out["vol"] = out["volume"].fillna(0)
    out = out[out["mid"] > 0].copy()
    return out

def _pick_expiry(exps: list[str], target_dte: int = 30) -> str | None:
    if not exps:
        return None
    today = date.today()
    best, best_diff = None, 10**9
    for e in exps:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (d - today).days
        if dte < 7:
            continue
        diff = abs(dte - target_dte)
        if diff < best_diff:
            best_diff, best = diff, e
    if best:
        return best
    for e in exps:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue
        if (d - today).days >= 5:
            return e
    return exps[min(2, len(exps) - 1)]


def _dte(expiry: str) -> int:
    try:
        return max((datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days, 0)
    except ValueError:
        return 0


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float | None:
    frames = [x for x in (calls, puts) if x is not None and not x.empty]
    if not frames:
        return None
    all_opts = pd.concat(frames, ignore_index=True)
    all_opts = all_opts[all_opts["mid"] > 0].copy()
    if all_opts.empty:
        return None
    all_opts["_dist"] = (all_opts["strike"] - spot).abs()
    all_opts = all_opts.sort_values("_dist")
    try:
        v = float(all_opts.iloc[0].get("impliedVolatility"))
        if v > 3:
            v /= 100.0
        return v if 0.05 <= v < 3 else None
    except Exception:
        return None


def _hist_vol(symbol: str, days: int = 30) -> float | None:
    try:
        h = yf.Ticker(symbol).history(period=f"{max(days + 10, 40)}d")
        if h is None or h.empty:
            return None
        rets = h["Close"].pct_change().dropna()
        if len(rets) < 10:
            return None
        return float(rets.tail(days).std() * np.sqrt(252))
    except Exception:
        return None


def _liquid(df: pd.DataFrame, max_spread_pct: float = 0.4) -> pd.DataFrame:
    if df.empty:
        return df
    q = df[df["mid"] > 0.05].copy()
    has_quote = (q["bid"].fillna(0) > 0) & (q["ask"].fillna(0) > 0)
    tight = q[has_quote & (q["spread_pct"] <= max_spread_pct)]
    if len(tight) >= 8:
        q = tight
    active = q[(q["oi"] > 0) | (q["vol"] > 0)]
    if len(active) >= 6:
        q = active
    return q.sort_values("strike").reset_index(drop=True)


def _nearest_strike(df: pd.DataFrame, target: float, side: str = "any") -> pd.Series | None:
    if df is None or df.empty:
        return None
    if side == "below":
        sub = df[df["strike"] <= target + 1e-9]
        return None if sub.empty else sub.iloc[-1]
    if side == "above":
        sub = df[df["strike"] >= target - 1e-9]
        return None if sub.empty else sub.iloc[0]
    idx = (df["strike"] - target).abs().idxmin()
    return df.loc[idx]


def _leg_from_row(row: pd.Series, right: str, side: str, spot: float) -> Leg:
    strike = float(row["strike"])
    mid = float(row["mid"])
    bid = float(row.get("bid") or 0)
    ask = float(row.get("ask") or 0)
    fill, fill_src = _fill_price(side, bid, ask, mid)
    if right == "put":
        if strike < spot:
            otm = (spot - strike) / spot
            delta_proxy = max(0.05, 0.5 * np.exp(-8 * otm))
        else:
            delta_proxy = min(0.7, 0.5 + (strike - spot) / spot)
    else:
        if strike > spot:
            otm = (strike - spot) / spot
            delta_proxy = max(0.05, 0.5 * np.exp(-8 * otm))
        else:
            delta_proxy = min(0.7, 0.5 + (spot - strike) / spot * 0.5)
    iv = None
    try:
        iv = float(row.get("impliedVolatility"))
        if iv > 3:
            iv /= 100.0
        if iv < 0.05:
            iv = None
    except Exception:
        pass
    return Leg(
        right=right,
        strike=strike,
        side=side,
        mid=mid,
        bid=bid,
        ask=ask,
        iv=iv,
        oi=float(row.get("oi") or 0),
        volume=float(row.get("vol") or 0),
        delta_proxy=float(delta_proxy),
        fill=float(fill),
        fill_source=fill_src,
    )


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prob_above(spot: float, strike: float, sigma: float, t_years: float, mu: float = 0.0) -> float:
    """P(S_T > K) under lognormal drift mu (default 0)."""
    if spot <= 0 or strike <= 0 or sigma <= 0 or t_years <= 0:
        return 0.5
    vol_sq = sigma * math.sqrt(t_years)
    if vol_sq < 1e-9:
        return 1.0 if spot > strike else 0.0
    d2 = (math.log(spot / strike) + (mu - 0.5 * sigma * sigma) * t_years) / vol_sq
    return float(_norm_cdf(d2))


def _prob_below(spot: float, strike: float, sigma: float, t_years: float, mu: float = 0.0) -> float:
    return 1.0 - _prob_above(spot, strike, sigma, t_years, mu)


def estimate_vertical_win_rates(
    code: str,
    spot: float,
    short_strike: float | None,
    long_strike: float | None,
    breakeven: float | None,
    sigma: float | None,
    dte: int,
    max_profit: float,
    max_loss: float,
) -> tuple[float | None, float | None, str, float | None]:
    """
    Returns (win_rate_profit%, win_rate_max%, method, expected_value$).

    Credit bull put:  profit if S>BE; max if S>short put
    Credit bear call: profit if S<BE; max if S<short call
    Debit bull call:  profit if S>BE; max if S>short call
    Debit bear put:   profit if S<BE; max if S<short put
    """
    del long_strike  # reserved for multi-leg refinements
    if not sigma or sigma <= 0 or dte <= 0 or spot <= 0:
        return None, None, "波动率/天数不足，无法估胜率", None

    t = max(dte, 1) / 365.0
    sig = max(0.08, min(float(sigma), 0.80))
    method = f"对数正态 · σ≈{sig * 100:.1f}% · {dte}日 · μ=0（中性漂移）"

    wr_p = wr_m = None
    if code == "bull_put":
        if breakeven:
            wr_p = _prob_above(spot, breakeven, sig, t) * 100
        if short_strike:
            wr_m = _prob_above(spot, short_strike, sig, t) * 100
    elif code == "bear_call":
        if breakeven:
            wr_p = _prob_below(spot, breakeven, sig, t) * 100
        if short_strike:
            wr_m = _prob_below(spot, short_strike, sig, t) * 100
    elif code == "bull_call":
        if breakeven:
            wr_p = _prob_above(spot, breakeven, sig, t) * 100
        if short_strike:
            wr_m = _prob_above(spot, short_strike, sig, t) * 100
    elif code == "bear_put":
        if breakeven:
            wr_p = _prob_below(spot, breakeven, sig, t) * 100
        if short_strike:
            wr_m = _prob_below(spot, short_strike, sig, t) * 100

    if wr_p is not None:
        wr_p = float(np.clip(wr_p, 1.0, 99.0))
    if wr_m is not None:
        wr_m = float(np.clip(wr_m, 1.0, 99.0))

    ev = None
    if wr_p is not None:
        p = wr_p / 100.0
        # 有利润时按 55% 满盈近似（路径非总能到 max）
        ev = p * (0.55 * max_profit) - (1 - p) * max_loss

    return (
        round(wr_p, 1) if wr_p is not None else None,
        round(wr_m, 1) if wr_m is not None else None,
        method,
        round(ev, 1) if ev is not None else None,
    )


def score_liquidity(idea: SpreadIdea) -> SpreadIdea:
    """
    用两腿的 成交量、未平仓(OI)、买卖价差 估流动性。
    盘后 bid/ask 常为 0，会更多依赖 OI/量，并标明「盘后参考」。
    """
    if not idea.legs:
        idea.liquidity_score = 0.0
        idea.liquidity_label = "未知"
        idea.liquidity_detail = "没有腿数据"
        return idea

    oi_sum = 0.0
    vol_sum = 0.0
    spread_pcts: list[float] = []
    has_quote = 0
    n = len(idea.legs)

    for leg in idea.legs:
        oi_sum += float(leg.oi or 0)
        vol_sum += float(leg.volume or 0)
        bid = float(leg.bid or 0)
        ask = float(leg.ask or 0)
        mid = float(leg.mid or 0)
        if bid > 0 and ask > 0 and mid > 0:
            has_quote += 1
            spread_pcts.append((ask - bid) / mid)

    # OI：单腿均 300+ 算可用，800+ 较好，2000+ 很好
    avg_oi = oi_sum / max(n, 1)
    oi_score = min(100.0, avg_oi / 8.0)  # 800 avg -> 100
    # Volume：盘中更有用；均 50+ 可用
    avg_vol = vol_sum / max(n, 1)
    vol_score = min(100.0, avg_vol / 1.5)  # 150 avg -> 100
    # Bid-ask tightness
    if spread_pcts:
        avg_sp = sum(spread_pcts) / len(spread_pcts)
        # 2% mid -> ~92, 10% -> ~60, 25%+ -> 差
        ba_score = float(max(0.0, min(100.0, 100.0 - avg_sp * 400.0)))
    else:
        ba_score = 40.0  # 无报价时给中性
        avg_sp = None

    # 权重：有报价时更看买卖差；无报价更看 OI
    if has_quote == n:
        score = 0.45 * ba_score + 0.30 * oi_score + 0.25 * vol_score
    elif has_quote > 0:
        score = 0.30 * ba_score + 0.40 * oi_score + 0.30 * vol_score
    else:
        score = 0.55 * oi_score + 0.45 * vol_score
        if oi_sum < 50 and vol_sum < 20:
            score = min(score, 28.0)

    score = float(np.clip(score, 0, 100))
    if score >= 65:
        label = "高"
    elif score >= 42:
        label = "中"
    elif score >= 25:
        label = "低"
    else:
        label = "很差"

    parts = [
        f"未平仓合计约 {oi_sum:.0f}",
        f"成交量合计约 {vol_sum:.0f}",
    ]
    if avg_sp is not None:
        parts.append(f"买卖价差约中间价的 {avg_sp * 100:.0f}%")
    else:
        parts.append("盘后可能无买卖价（用未平仓/量估算）")
    parts.append(f"流动性：{label}（{score:.0f}分）")

    idea.liquidity_score = round(score, 1)
    idea.liquidity_label = label
    idea.liquidity_detail = "；".join(parts)
    # 原评分略混入流动性
    idea.score = round(float(np.clip(idea.score * 0.85 + score * 0.15, 0, 100)), 1)
    return idea


def _attach_win_rates(idea: SpreadIdea, spot: float, sigma: float | None) -> SpreadIdea:
    short_k = long_k = None
    for leg in idea.legs:
        if leg.side == "sell":
            short_k = leg.strike
        else:
            long_k = leg.strike
    be = idea.breakevens[0] if idea.breakevens else None
    wr_p, wr_m, method, ev = estimate_vertical_win_rates(
        idea.code,
        spot,
        short_k,
        long_k,
        be,
        sigma,
        idea.dte,
        idea.max_profit,
        idea.max_loss,
    )
    idea.win_rate_profit = wr_p
    idea.win_rate_max = wr_m
    idea.win_rate_method = method
    idea.expected_value = ev
    idea.pop_est = wr_p
    if wr_p is not None:
        # 评分混入胜率
        idea.score = round(float(np.clip(idea.score * 0.72 + wr_p * 0.28, 0, 100)), 1)
        idea.notes = list(idea.notes) + [
            f"预估胜率（有利润）≈ {wr_p:.1f}%"
            + (f"；满盈概率≈ {wr_m:.1f}%" if wr_m is not None else ""),
            f"粗算期望 ≈ ${ev:.0f}/张" if ev is not None else method,
        ]
    return idea


def suggest_width(spot: float) -> float:
    if spot >= 400:
        return 10.0
    if spot >= 200:
        return 5.0
    if spot >= 50:
        return 2.0
    return 1.0


def _width_ok(actual: float, target: float) -> bool:
    """Reject legs that landed far from requested vertical width."""
    if actual <= 0:
        return False
    return abs(actual - target) <= max(target * 0.6, 2.5)


# ---------------------------------------------------------------------------
# Vertical builders
# ---------------------------------------------------------------------------

def build_bull_put(
    puts: pd.DataFrame,
    spot: float,
    width: float,
    otm_pct: float = 0.03,
    expiry: str = "",
    dte: int = 0,
) -> SpreadIdea | None:
    """Credit: sell higher put, buy lower put."""
    liq = _liquid(puts)
    if liq.empty:
        return None
    otm = liq[liq["strike"] < spot * 0.999]
    if otm.empty:
        return None
    target = spot * (1 - otm_pct)
    short_row = _nearest_strike(otm, target, "below")
    if short_row is None:
        return None
    short_k = float(short_row["strike"])
    lower = liq[liq["strike"] < short_k]
    if lower.empty:
        return None
    long_row = lower.iloc[(lower["strike"] - (short_k - width)).abs().argmin()]
    long_k = float(long_row["strike"])
    if long_k >= short_k:
        return None

    short_leg = _leg_from_row(short_row, "put", "sell", spot)
    long_leg = _leg_from_row(long_row, "put", "buy", spot)
    # 自然成交：卖腿 bid − 买腿 ask（比 mid 更保守）
    credit = short_leg.fill - long_leg.fill
    if credit <= HARD_MIN_CREDIT:
        return None
    w = short_k - long_k
    if not _width_ok(w, width) or credit / w > HARD_CREDIT_FILL_HI:
        return None
    max_profit = credit * 100
    max_loss = (w - credit) * 100
    if max_loss <= 0:
        return None
    be = short_k - credit
    fill_ratio = credit / w
    rr = max_profit / max_loss
    score = min(100.0, fill_ratio * 140 + min(short_leg.oi or 0, 3000) / 3000 * 12 + 40)
    actual_otm = (spot - short_k) / spot * 100
    credit_r = round(credit, 2)
    half_bb = round(credit_r * 0.5, 2)
    max_profit_r = round(max_profit, 2)
    return SpreadIdea(
        name=f"Bull Put Credit · 卖{short_k:.0f}/买{long_k:.0f}",
        code="bull_put",
        structure="Credit Vertical",
        thesis="看多/偏多",
        net_credit=credit_r,
        net_debit=None,
        max_profit=max_profit_r,
        max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)],
        width=round(w, 2),
        pop_est=None,
        score=round(score, 1),
        dte=dte,
        expiry=expiry,
        legs=[short_leg, long_leg],
        notes=[
            f"开仓：卖 {short_k:.0f} Put，买 {long_k:.0f} Put（同一到期）",
            f"净收约 ${credit_r:.2f}/股（卖腿按买价bid、买腿按卖价ask）→ 最大盈 ${max_profit_r:.0f}，最大亏 ${max_loss:.0f}",
            f"打和点 {be:.2f}；到期收盘 > {be:.2f} 有利",
            f"短腿约 {actual_otm:.1f}% OTM",
            f"50%止盈：价差买回约 ${half_bb:.2f}/股（约赚 ${max_profit_r * 0.5:.0f}/张）",
        ],
        risk_reward=round(rr, 2),
        otm_label=f"短腿约 {actual_otm:.1f}% OTM",
        pricing_mode=_pricing_mode_for_legs([short_leg, long_leg]),
        metric_half_buyback=half_bb,
        metric_half_profit=round(max_profit_r * 0.5, 1),
    )


def build_bear_call(
    calls: pd.DataFrame,
    spot: float,
    width: float,
    otm_pct: float = 0.03,
    expiry: str = "",
    dte: int = 0,
) -> SpreadIdea | None:
    """Credit: sell lower call, buy higher call."""
    liq = _liquid(calls)
    if liq.empty:
        return None
    otm = liq[liq["strike"] > spot * 1.001]
    if otm.empty:
        return None
    target = spot * (1 + otm_pct)
    short_row = _nearest_strike(otm, target, "above")
    if short_row is None:
        return None
    short_k = float(short_row["strike"])
    upper = liq[liq["strike"] > short_k]
    if upper.empty:
        return None
    long_row = upper.iloc[(upper["strike"] - (short_k + width)).abs().argmin()]
    long_k = float(long_row["strike"])
    if long_k <= short_k:
        return None

    short_leg = _leg_from_row(short_row, "call", "sell", spot)
    long_leg = _leg_from_row(long_row, "call", "buy", spot)
    credit = short_leg.fill - long_leg.fill
    if credit <= HARD_MIN_CREDIT:
        return None
    w = long_k - short_k
    if not _width_ok(w, width) or credit / w > HARD_CREDIT_FILL_HI:
        return None
    max_profit = credit * 100
    max_loss = (w - credit) * 100
    if max_loss <= 0:
        return None
    be = short_k + credit
    fill_ratio = credit / w
    rr = max_profit / max_loss
    score = min(100.0, fill_ratio * 140 + min(short_leg.oi or 0, 3000) / 3000 * 12 + 40)
    actual_otm = (short_k - spot) / spot * 100
    credit_r = round(credit, 2)
    half_bb = round(credit_r * 0.5, 2)
    max_profit_r = round(max_profit, 2)
    return SpreadIdea(
        name=f"Bear Call Credit · 卖{short_k:.0f}/买{long_k:.0f}",
        code="bear_call",
        structure="Credit Vertical",
        thesis="看空/偏空",
        net_credit=credit_r,
        net_debit=None,
        max_profit=max_profit_r,
        max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)],
        width=round(w, 2),
        pop_est=None,
        score=round(score, 1),
        dte=dte,
        expiry=expiry,
        legs=[short_leg, long_leg],
        notes=[
            f"开仓：卖 {short_k:.0f} Call，买 {long_k:.0f} Call",
            f"净收约 ${credit_r:.2f}/股（卖腿bid / 买腿ask）→ 最大盈 ${max_profit_r:.0f}，最大亏 ${max_loss:.0f}",
            f"打和点 {be:.2f}；到期收盘 < {be:.2f} 有利",
            f"短腿约 {actual_otm:.1f}% OTM",
            f"50%止盈：价差买回约 ${half_bb:.2f}/股（约赚 ${max_profit_r * 0.5:.0f}/张）",
        ],
        risk_reward=round(rr, 2),
        otm_label=f"短腿约 {actual_otm:.1f}% OTM",
        pricing_mode=_pricing_mode_for_legs([short_leg, long_leg]),
        metric_half_buyback=half_bb,
        metric_half_profit=round(max_profit_r * 0.5, 1),
    )


def build_bull_call(
    calls: pd.DataFrame,
    spot: float,
    width: float,
    long_offset_pct: float = 0.0,
    expiry: str = "",
    dte: int = 0,
) -> SpreadIdea | None:
    """Debit: buy lower call, sell higher call."""
    liq = _liquid(calls)
    if liq.empty:
        return None
    target = spot * (1 + long_offset_pct)
    long_row = _nearest_strike(liq, target, "any")
    if long_row is None:
        return None
    long_k = float(long_row["strike"])
    upper = liq[liq["strike"] > long_k]
    if upper.empty:
        return None
    short_row = upper.iloc[(upper["strike"] - (long_k + width)).abs().argmin()]
    short_k = float(short_row["strike"])
    if short_k <= long_k:
        return None

    long_leg = _leg_from_row(long_row, "call", "buy", spot)
    short_leg = _leg_from_row(short_row, "call", "sell", spot)
    # 自然成交：买腿 ask − 卖腿 bid（借方更贵、更保守）
    debit = long_leg.fill - short_leg.fill
    if debit <= 0.05:
        return None
    w = short_k - long_k
    if not _width_ok(w, width) or debit / w > HARD_DEBIT_FILL_HI:
        return None
    max_profit = (w - debit) * 100
    max_loss = debit * 100
    if max_profit <= 0:
        return None
    be = long_k + debit
    rr = max_profit / max_loss
    fill = 1 - debit / w
    score = min(100.0, fill * 90 + rr * 35 + min(long_leg.oi or 0, 2000) / 2000 * 10)
    debit_r = round(debit, 2)
    w_r = round(w, 2)
    half_bb = round(debit_r + 0.5 * (w_r - debit_r), 2)
    max_profit_r = round(max_profit, 2)
    return SpreadIdea(
        name=f"Bull Call Debit · 买{long_k:.0f}/卖{short_k:.0f}",
        code="bull_call",
        structure="Debit Vertical",
        thesis="看多",
        net_credit=None,
        net_debit=debit_r,
        max_profit=max_profit_r,
        max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)],
        width=w_r,
        pop_est=None,
        score=round(score, 1),
        dte=dte,
        expiry=expiry,
        legs=[long_leg, short_leg],
        pricing_mode=_pricing_mode_for_legs([long_leg, short_leg]),
        metric_half_buyback=half_bb,
        metric_half_profit=round(max_profit_r * 0.5, 1),
        notes=[
            f"开仓：买 {long_k:.0f} Call，卖 {short_k:.0f} Call",
            f"净付约 ${debit_r:.2f}/股（买腿ask / 卖腿bid）→ 最大盈 ${max_profit_r:.0f}，最大亏 ${max_loss:.0f}",
            f"打和点 {be:.2f}；越接近/超过 {short_k:.0f} 利润越大",
            "适合明确看多；比单买 Call 更便宜但封顶盈利",
            f"50%止盈：价差卖出约 ${half_bb:.2f}/股（约赚 ${max_profit_r * 0.5:.0f}/张）",
        ],
        risk_reward=round(rr, 2),
        otm_label=f"长腿行权价 {long_k:.0f}",
    )


def build_bear_put(
    puts: pd.DataFrame,
    spot: float,
    width: float,
    long_offset_pct: float = 0.0,
    expiry: str = "",
    dte: int = 0,
) -> SpreadIdea | None:
    """Debit: buy higher put, sell lower put."""
    liq = _liquid(puts)
    if liq.empty:
        return None
    target = spot * (1 + long_offset_pct)
    long_row = _nearest_strike(liq, target, "any")
    if long_row is None:
        return None
    long_k = float(long_row["strike"])
    lower = liq[liq["strike"] < long_k]
    if lower.empty:
        return None
    short_row = lower.iloc[(lower["strike"] - (long_k - width)).abs().argmin()]
    short_k = float(short_row["strike"])
    if short_k >= long_k:
        return None

    long_leg = _leg_from_row(long_row, "put", "buy", spot)
    short_leg = _leg_from_row(short_row, "put", "sell", spot)
    debit = long_leg.fill - short_leg.fill
    if debit <= 0.05:
        return None
    w = long_k - short_k
    if not _width_ok(w, width) or debit / w > HARD_DEBIT_FILL_HI:
        return None
    max_profit = (w - debit) * 100
    max_loss = debit * 100
    if max_profit <= 0:
        return None
    be = long_k - debit
    rr = max_profit / max_loss
    fill = 1 - debit / w
    score = min(100.0, fill * 90 + rr * 35 + min(long_leg.oi or 0, 2000) / 2000 * 10)
    debit_r = round(debit, 2)
    w_r = round(w, 2)
    half_bb = round(debit_r + 0.5 * (w_r - debit_r), 2)
    max_profit_r = round(max_profit, 2)
    return SpreadIdea(
        name=f"Bear Put Debit · 买{long_k:.0f}/卖{short_k:.0f}",
        code="bear_put",
        structure="Debit Vertical",
        thesis="看空",
        net_credit=None,
        net_debit=debit_r,
        max_profit=max_profit_r,
        max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)],
        width=w_r,
        pop_est=None,
        score=round(score, 1),
        dte=dte,
        expiry=expiry,
        legs=[long_leg, short_leg],
        pricing_mode=_pricing_mode_for_legs([long_leg, short_leg]),
        metric_half_buyback=half_bb,
        metric_half_profit=round(max_profit_r * 0.5, 1),
        notes=[
            f"开仓：买 {long_k:.0f} Put，卖 {short_k:.0f} Put",
            f"净付约 ${debit_r:.2f}/股（买腿ask / 卖腿bid）→ 最大盈 ${max_profit_r:.0f}，最大亏 ${max_loss:.0f}",
            f"打和点 {be:.2f}；跌破 {short_k:.0f} 附近接近满盈",
            "适合明确看空；风险有限",
            f"50%止盈：价差卖出约 ${half_bb:.2f}/股（约赚 ${max_profit_r * 0.5:.0f}/张）",
        ],
        risk_reward=round(rr, 2),
        otm_label=f"长腿行权价 {long_k:.0f}",
    )


# ---------------------------------------------------------------------------
# Cached option chain fetch
# ---------------------------------------------------------------------------

def _fetch_option_chain_uncached(symbol: str, expiry: str) -> dict[str, Any]:
    """Pull raw chain frames; returns dict for Streamlit cache friendliness."""
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiry)
    calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
    puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()
    return {"calls": calls, "puts": puts}


_st_chain_cached = None  # lazily wrap with st.cache_data once


def _get_cached_chain(symbol: str, expiry: str) -> dict[str, Any]:
    """
    期权链缓存：Streamlit 下 st.cache_data(ttl=120s)；
    非 Streamlit 环境直接拉数。
    """
    global _st_chain_cached
    try:
        import streamlit as st

        if _st_chain_cached is None:
            _st_chain_cached = st.cache_data(ttl=120, show_spinner=False)(
                _fetch_option_chain_uncached
            )
        return _st_chain_cached(symbol, expiry)
    except Exception:
        return _fetch_option_chain_uncached(symbol, expiry)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_options_spreads(
    symbol: str,
    target_dte: int = 30,
    width: float | None = None,
    hist_df: pd.DataFrame | None = None,
    bias_label: str | None = None,
) -> OptionsReport:
    """Vertical-only analysis with direction + best spread recommendation."""
    del bias_label  # direction computed from hist_df
    sym = options_symbol(symbol)
    label = INDEX_ETF_WHITELIST.get(sym, "")
    if not is_options_eligible(sym):
        return OptionsReport(
            symbol=sym,
            label=label or "非白名单",
            spot=0.0,
            eligible=False,
            message=(
                f"`{sym}` 不在白名单。请用：{', '.join(sorted(INDEX_ETF_WHITELIST))}。"
            ),
            summary="标的不适合。",
        )

    direction = analyze_direction(hist_df) if hist_df is not None else analyze_direction(pd.DataFrame())

    ticker = yf.Ticker(sym)
    spot = None
    try:
        fi = ticker.fast_info
        spot = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
    except Exception:
        spot = None
    if not spot:
        try:
            h = ticker.history(period="5d")
            if h is not None and not h.empty:
                spot = float(h["Close"].iloc[-1])
        except Exception:
            pass
    if not spot:
        return OptionsReport(
            symbol=sym,
            label=label,
            spot=0.0,
            eligible=True,
            message="无法获取现价。",
            direction=direction,
            summary="无现价。",
        )
    spot = float(spot)

    try:
        exps = list(ticker.options or [])
    except Exception:
        exps = []
    if not exps:
        return OptionsReport(
            symbol=sym,
            label=label,
            spot=spot,
            eligible=True,
            message="无期权到期日。",
            direction=direction,
            summary="无期权链。",
        )

    expiry = _pick_expiry(exps, target_dte=target_dte)
    if not expiry:
        return OptionsReport(
            symbol=sym,
            label=label,
            spot=spot,
            eligible=True,
            expiries=exps[:16],
            message="无合适到期日。",
            direction=direction,
            summary="无到期日。",
        )
    dte = _dte(expiry)
    after_hours = not _is_us_rth()

    try:
        raw = _get_cached_chain(sym, expiry)
        calls = _parse_chain(raw.get("calls"), spot, "call")
        puts = _parse_chain(raw.get("puts"), spot, "put")
    except Exception as exc:
        return OptionsReport(
            symbol=sym,
            label=label,
            spot=spot,
            eligible=True,
            expiries=exps[:16],
            selected_expiry=expiry,
            dte=dte,
            direction=direction,
            message=f"期权链失败：{exc}",
            summary="期权链失败。",
            after_hours=after_hours,
        )

    # 报价质量：有多少腿有有效 bid/ask
    def _quote_coverage(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 0.0
        has = (df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0)
        return float(has.mean()) if len(df) else 0.0

    cov = 0.5 * (_quote_coverage(calls) + _quote_coverage(puts))
    quote_warning = ""
    pricing_note = "开仓按自然成交估算：卖腿=买价(bid)，买腿=卖价(ask)；比中间价更保守。"
    if after_hours:
        quote_warning = (
            "当前可能不在美股常规交易时段（美东 09:30–16:00）。"
            "盘后 bid/ask 常为空或失真，Yahoo lastPrice 也易过期——"
            "权利金仅供参考，请在盘中用限价单核验。"
        )
        pricing_note += " 盘后若无买卖价则回退中间价/最新价。"
    elif cov < 0.35:
        quote_warning = (
            "当前期权链买卖价覆盖偏低，部分腿用中间价估算，成交价可能偏差较大。"
        )
        pricing_note += " 部分腿无有效 bid/ask，已回退 mid。"

    iv_atm = _atm_iv(calls, puts, spot)
    if iv_atm is None or iv_atm < 0.05:
        hv = _hist_vol(sym, 30)
        if hv:
            iv_atm = hv

    w = width or suggest_width(spot)
    ideas: list[SpreadIdea] = []

    # Scan several OTM / offsets for credit & debit verticals
    for otm in (0.015, 0.025, 0.035, 0.05):
        for builder, args in (
            (build_bull_put, dict(puts=puts, spot=spot, width=w, otm_pct=otm, expiry=expiry, dte=dte)),
            (build_bear_call, dict(calls=calls, spot=spot, width=w, otm_pct=otm, expiry=expiry, dte=dte)),
        ):
            try:
                idea = builder(**args)
                if idea:
                    ideas.append(idea)
            except Exception:
                pass

    for off in (-0.01, 0.0, 0.01, 0.02):
        for builder, args in (
            (build_bull_call, dict(calls=calls, spot=spot, width=w, long_offset_pct=off, expiry=expiry, dte=dte)),
            (build_bear_put, dict(puts=puts, spot=spot, width=w, long_offset_pct=off, expiry=expiry, dte=dte)),
        ):
            try:
                idea = builder(**args)
                if idea:
                    ideas.append(idea)
            except Exception:
                pass

    # Deduplicate by code+strikes
    uniq: dict[str, SpreadIdea] = {}
    for idea in ideas:
        key = (
            f"{idea.code}:"
            + "-".join(f"{lg.side[0]}{lg.right[0]}{lg.strike:.0f}" for lg in idea.legs)
        )
        prev = uniq.get(key)
        if prev is None or idea.score > prev.score:
            uniq[key] = idea
    ideas = list(uniq.values())

    # Attach win rates + liquidity + 热门规则评分
    ideas = [_attach_win_rates(i, spot, iv_atm) for i in ideas]
    ideas = [score_liquidity(i) for i in ideas]
    try:
        from options_methods import enrich_idea_with_methods

        ideas = [enrich_idea_with_methods(i, spot, dte) for i in ideas]
    except Exception:
        pass

    # 补全 50% 止盈买回价（enrich 可能覆盖 metric_half_profit）
    for idea in ideas:
        bb = half_profit_close_price(idea)
        if bb is not None:
            idea.metric_half_buyback = bb
        if idea.metric_half_profit is None:
            idea.metric_half_profit = round(float(idea.max_profit) * 0.5, 1)

    # 硬过滤：流动性 + 权利金/宽度
    before_n = len(ideas)
    ideas = [i for i in ideas if passes_hard_filters(i)]
    filtered_out = before_n - len(ideas)

    # Align score with direction
    preferred = set(direction.preferred_verticals)
    for idea in ideas:
        boost = 0.0
        if idea.code in preferred:
            boost += 18
            if preferred and idea.code == direction.preferred_verticals[0]:
                boost += 10
        # Penalize opposite side
        bull_codes = {"bull_put", "bull_call"}
        bear_codes = {"bear_call", "bear_put"}
        if direction.direction == "看多" and idea.code in bear_codes:
            boost -= 20
        if direction.direction == "看空" and idea.code in bull_codes:
            boost -= 20
        if direction.direction == "中性":
            if idea.structure == "Credit Vertical":
                boost += 8
            else:
                boost -= 5
        # Mild bull: prefer credit over expensive debit when RSI high
        if direction.direction == "看多" and direction.strength != "强" and idea.code == "bull_put":
            boost += 6
        if direction.direction == "看空" and direction.strength != "强" and idea.code == "bear_call":
            boost += 6

        idea.score = float(np.clip(idea.score + boost, 0, 100))
        idea.rank_reason = (
            f"方向={direction.direction}；结构={idea.structure}；"
            f"{'契合首选' if idea.code in preferred else '次选/对冲'}"
        )

    ideas.sort(key=lambda x: x.score, reverse=True)

    best = ideas[0] if ideas else None
    best_alt = None
    if best and len(ideas) > 1:
        for alt in ideas[1:]:
            if alt.code != best.code or alt.structure != best.structure:
                best_alt = alt
                break
        if best_alt is None:
            best_alt = ideas[1]

    # 最高胜率（有利润）
    def _wr(i: SpreadIdea) -> float:
        v = getattr(i, "win_rate_profit", None)
        if v is None:
            v = getattr(i, "pop_est", None)
        return float(v) if v is not None else -1.0

    with_wr = [i for i in ideas if _wr(i) >= 0]
    best_winrate = max(with_wr, key=_wr) if with_wr else None

    # 与当前方向契合的最高胜率
    bull_codes = {"bull_put", "bull_call"}
    bear_codes = {"bear_call", "bear_put"}
    if direction.direction == "看多":
        aligned_pool = [i for i in with_wr if i.code in bull_codes]
    elif direction.direction == "看空":
        aligned_pool = [i for i in with_wr if i.code in bear_codes]
    else:
        aligned_pool = [i for i in with_wr if i.structure == "Credit Vertical"] or with_wr
    best_winrate_aligned = max(aligned_pool, key=_wr) if aligned_pool else best_winrate

    # 参考市面常见策略规则 → 最佳 / 高赢面
    best_playbook = best_playbook_wr = None
    playbook_table: list = []
    try:
        from options_strategy_book import apply_playbook_ranking

        best_playbook, best_playbook_wr, playbook_table = apply_playbook_ranking(
            ideas, spot, dte, direction.direction
        )
        # 用实战分重排展示顺序（保留原 score）
        ideas = sorted(
            ideas,
            key=lambda x: getattr(x, "playbook_combo", x.score),
            reverse=True,
        )
        if best_playbook is not None:
            best = best_playbook
    except Exception:
        playbook_table = []

    if best_winrate:
        for i, idea in enumerate(sorted(with_wr, key=_wr, reverse=True), start=1):
            idea.rank_reason = (
                (idea.rank_reason + "；" if idea.rank_reason else "")
                + f"赢面排名#{i}（{_wr(idea):.1f}%）"
            )

    if iv_atm is not None:
        if iv_atm >= 0.28:
            regime = "波动偏高 · 信用价差权利金厚但尾部风险大"
        elif iv_atm >= 0.16:
            regime = "波动适中 · 适合标准 vertical"
        else:
            regime = "波动偏低 · 信用价差收租薄，debit 也不宜追贵"
    else:
        regime = "波动数据有限"

    action_plan: list[str] = []
    if best and direction:
        action_plan.append(f"1. 方向结论：{direction.direction}（{direction.strength}）")
        action_plan.append(f"2. 首选 vertical：{best.name}")
        if best.net_credit is not None:
            action_plan.append(
                f"3. 开仓方式：信用价差，目标净收 ≥ ${best.net_credit * 0.9:.2f}（限价单）"
            )
        else:
            action_plan.append(
                f"3. 开仓方式：借方价差，目标净付 ≤ ${best.net_debit * 1.1:.2f}（限价单）"
            )
        if best.win_rate_profit is not None:
            action_plan.append(
                f"4. 预估胜率：有利润 ≈ {best.win_rate_profit:.1f}%"
                + (
                    f"，满盈 ≈ {best.win_rate_max:.1f}%"
                    if best.win_rate_max is not None
                    else ""
                )
                + (
                    f"；粗期望 ≈ ${best.expected_value:.0f}/张"
                    if best.expected_value is not None
                    else ""
                )
            )
            action_plan.append(
                f"5. 风控：最大亏损约 ${best.max_loss:.0f}/张；建议风险不超过账户 1%–2%"
            )
            action_plan.append(
                "6. 管理：盈利达最大利润 50%–70% 可提前平仓；靠近短腿考虑止损/调仓"
            )
            n = 7
        else:
            action_plan.append(
                f"4. 风控：最大亏损约 ${best.max_loss:.0f}/张；建议风险不超过账户 1%–2%"
            )
            action_plan.append(
                "5. 管理：盈利达最大利润 50%–70% 可提前平仓；靠近短腿考虑止损/调仓"
            )
            n = 6
        if best_alt:
            alt_wr = (
                f"，胜率≈{best_alt.win_rate_profit:.1f}%"
                if best_alt.win_rate_profit is not None
                else ""
            )
            action_plan.append(f"{n}. 备选：{best_alt.name}{alt_wr}")
            n += 1
        if best_winrate:
            action_plan.append(
                f"{n}. 最高胜率（全体）：{best_winrate.name}"
                f" ≈ {best_winrate.win_rate_profit:.1f}%"
                + (
                    f"（满盈 {best_winrate.win_rate_max:.1f}%）"
                    if best_winrate.win_rate_max is not None
                    else ""
                )
            )
            n += 1
        if (
            best_winrate_aligned
            and best_winrate
            and best_winrate_aligned is not best_winrate
        ):
            action_plan.append(
                f"{n}. 最高胜率（贴合{direction.direction}）："
                f"{best_winrate_aligned.name} ≈ {best_winrate_aligned.win_rate_profit:.1f}%"
            )

    if best:
        wr_txt = (
            f"，赢面≈{best.win_rate_profit:.0f}%"
            if best.win_rate_profit is not None
            else ""
        )
        style = getattr(best, "playbook_style", "") or ""
        style_txt = f"参考「{style}」。" if style else ""
        hi_txt = ""
        use_wr = best_playbook_wr or best_winrate_aligned or best_winrate
        if use_wr and getattr(use_wr, "win_rate_profit", None) is not None:
            hi_txt = (
                f" 最高赢面：**{use_wr.name}**（{use_wr.win_rate_profit:.0f}%"
                f"{' · ' + getattr(use_wr, 'playbook_style', '') if getattr(use_wr, 'playbook_style', '') else ''}）"
            )
        summary = (
            f"{sym} @ {spot:.2f} → 方向 **{direction.direction}**。"
            f"实战推荐：**{best.name}**"
            f"（{wr_txt.lstrip('，')}"
            + (
                f"，收 ${best.net_credit:.2f}"
                if best.net_credit is not None
                else f"，付 ${best.net_debit:.2f}"
            )
            + f"，到期 {expiry}）。"
            + style_txt
            + hi_txt
        )
        # 白话执行
        half_bb = getattr(best, "metric_half_buyback", None)
        half_p = getattr(best, "metric_half_profit", None)
        if best.net_credit is not None:
            open_line = f"3. 今天：卖出价差，大约收 ${best.net_credit:.2f}/股（卖=bid/买=ask）"
            if half_bb is not None:
                close_line = (
                    f"4. 50%止盈：价差买回约 ${half_bb:.2f}/股"
                    + (f"（约赚 ${half_p:.0f}/张）" if half_p is not None else "")
                )
            else:
                close_line = "4. 几天后：买回平仓"
        else:
            open_line = f"3. 今天：买进价差，大约付 ${best.net_debit:.2f}/股（买=ask/卖=bid）"
            if half_bb is not None:
                close_line = (
                    f"4. 50%止盈：价差卖出约 ${half_bb:.2f}/股"
                    + (f"（约赚 ${half_p:.0f}/张）" if half_p is not None else "")
                )
            else:
                close_line = "4. 几天后：卖出平仓"
        if best.win_rate_profit is not None:
            close_line += f"；估算赢面约 {best.win_rate_profit:.0f}%"
        action_plan = [
            f"1. 方向：{direction.direction}（{direction.strength}）",
            f"2. 实战推荐：{best.name}"
            + (f" —— 像「{style}」" if style else ""),
            open_line,
            close_line,
            f"5. 最多赚约 ${best.max_profit:.0f}/张，最多亏约 ${best.max_loss:.0f}/张",
        ]
        if use_wr and use_wr is not best:
            action_plan.append(
                f"6. 若只要最高赢面：改用 {use_wr.name}"
                f"（约 {use_wr.win_rate_profit:.0f}%）"
            )
        plain = getattr(best, "playbook_plain", "")
        src = getattr(best, "playbook_source", "")
        if plain:
            action_plan.append(f"7. 白话：{plain}")
        if src:
            action_plan.append(f"8. 规则参考：{src}（教学归纳，非荐股）")
    else:
        summary = f"{sym} @ {spot:.2f}，方向 {direction.direction}，未能生成可用 vertical。"

    msg = "仅分析 Vertical Spread；" + pricing_note
    if filtered_out > 0:
        msg += f" 已硬过滤剔除 {filtered_out} 个（流动性差或权利金/宽度不合规）。"

    return OptionsReport(
        symbol=sym,
        label=label,
        spot=spot,
        eligible=True,
        message=msg,
        direction=direction,
        expiries=exps[:16],
        selected_expiry=expiry,
        dte=dte,
        iv_atm=iv_atm,
        ideas=ideas,
        best=best,
        best_alt=best_alt,
        best_winrate=best_winrate,
        best_winrate_aligned=best_winrate_aligned,
        best_playbook=best_playbook,
        best_playbook_wr=best_playbook_wr,
        playbook_table=playbook_table,
        regime=regime,
        summary=summary,
        action_plan=action_plan,
        after_hours=after_hours,
        pricing_note=pricing_note,
        quote_warning=quote_warning,
        filtered_out=filtered_out,
    )


def payoff_per_share(idea: SpreadIdea, underlying: float) -> float:
    """
    到期结算盈亏（每股，未乘 100）。
    不需要行权：信用/借方 vertical 都用结算价值，等同多数人平仓或到期自动结算。
    """
    pnl = 0.0
    for leg in idea.legs:
        if leg.side == "buy":
            pnl -= leg.mid
        else:
            pnl += leg.mid
        if leg.right == "call":
            intrinsic = max(underlying - leg.strike, 0.0)
        else:
            intrinsic = max(leg.strike - underlying, 0.0)
        if leg.side == "buy":
            pnl += intrinsic
        else:
            pnl -= intrinsic
    return float(pnl)


def payoff_per_contract(idea: SpreadIdea, underlying: float) -> float:
    """到期盈亏 $/张（标准 100 乘数）。"""
    return payoff_per_share(idea, underlying) * 100.0


def build_payoff_ladder(
    idea: SpreadIdea,
    spot: float,
    step: float | None = None,
    pad: float | None = None,
) -> pd.DataFrame:
    """
    列出标的到各价位时的赚/蚀（按到期结算，不行权也一样）。
    """
    strikes = [lg.strike for lg in idea.legs]
    be_list = list(idea.breakevens or [])
    key_levels = sorted(set(strikes + be_list + [spot]))

    # 自动步长
    if step is None:
        w = idea.width or 5.0
        if w >= 10:
            step = 5.0
        elif w >= 5:
            step = 2.0
        elif w >= 2:
            step = 1.0
        else:
            step = 0.5

    lo = min(key_levels) - (pad if pad is not None else max(idea.width * 1.5, step * 3))
    hi = max(key_levels) + (pad if pad is not None else max(idea.width * 1.5, step * 3))

    # 均匀网格 + 关键价
    prices = set()
    x = lo
    while x <= hi + 1e-9:
        prices.add(round(x / step) * step)
        x += step
    for k in key_levels:
        prices.add(round(float(k), 2))
    prices.add(round(float(spot), 2))

    rows = []
    for px in sorted(prices):
        pnl = payoff_per_contract(idea, px)
        # 标签
        tags = []
        tol = max(step * 0.51, 0.02)
        if any(abs(px - s) <= tol for s in strikes):
            tags.append("行权价")
        if any(abs(px - b) <= tol for b in be_list):
            tags.append("打和点")
        if abs(px - spot) <= tol:
            tags.append("现价")

        if pnl > 0.5:
            result = "賺"
        elif pnl < -0.5:
            result = "蝕"
        else:
            result = "打和"

        # 区间说明
        zone = ""
        if abs(pnl - idea.max_profit) < 1.0:
            zone = "最大利润区"
        elif abs(pnl + idea.max_loss) < 1.0 or abs(pnl - (-idea.max_loss)) < 1.0:
            zone = "最大亏损区"

        rows.append(
            {
                "标的价": round(px, 2),
                "相对现价%": round((px / spot - 1) * 100, 2) if spot else None,
                "到期盈亏$/张": round(pnl, 2),
                "结果": result,
                "区间": zone,
                "标记": " · ".join(tags) if tags else "",
            }
        )
    return pd.DataFrame(rows)


def payoff_zones_summary(idea: SpreadIdea, spot: float) -> dict[str, Any]:
    """用关键价概括：在什么价位以上/以下是赚还是蚀。"""
    strikes = sorted({lg.strike for lg in idea.legs})
    be = idea.breakevens[0] if idea.breakevens else None
    code = idea.code

    # 采样判断方向
    low_px = min(strikes) - idea.width
    high_px = max(strikes) + idea.width
    pnl_low = payoff_per_contract(idea, low_px)
    pnl_high = payoff_per_contract(idea, high_px)

    lines: list[str] = []
    lines.append("说明：一般不需要手动行权；到期结算或提前平仓，盈亏看标的价与打和点。")

    if code == "bull_put":
        # 信用 put：价越高越好
        short_k = max(lg.strike for lg in idea.legs if lg.side == "sell")
        long_k = min(lg.strike for lg in idea.legs if lg.side == "buy")
        lines.append(f"结构：Bull Put Credit（卖 {short_k:.0f} Put / 买 {long_k:.0f} Put）")
        if be is not None:
            lines.append(f"**賺**：到期标的 **> {be:.2f}**（打和点上方）")
            lines.append(f"**蝕**：到期标的 **< {be:.2f}**")
        lines.append(f"**最大利潤**：标的 ≥ {short_k:.0f}（两腿都虚值）≈ ${idea.max_profit:.0f}/张")
        lines.append(f"**最大亏损**：标的 ≤ {long_k:.0f} ≈ ${idea.max_loss:.0f}/张")
    elif code == "bear_call":
        short_k = min(lg.strike for lg in idea.legs if lg.side == "sell")
        long_k = max(lg.strike for lg in idea.legs if lg.side == "buy")
        lines.append(f"结构：Bear Call Credit（卖 {short_k:.0f} Call / 买 {long_k:.0f} Call）")
        if be is not None:
            lines.append(f"**賺**：到期标的 **< {be:.2f}**（打和点下方）")
            lines.append(f"**蝕**：到期标的 **> {be:.2f}**")
        lines.append(f"**最大利潤**：标的 ≤ {short_k:.0f} ≈ ${idea.max_profit:.0f}/张")
        lines.append(f"**最大亏损**：标的 ≥ {long_k:.0f} ≈ ${idea.max_loss:.0f}/张")
    elif code == "bull_call":
        long_k = min(lg.strike for lg in idea.legs if lg.side == "buy")
        short_k = max(lg.strike for lg in idea.legs if lg.side == "sell")
        lines.append(f"结构：Bull Call Debit（买 {long_k:.0f} Call / 卖 {short_k:.0f} Call）")
        if be is not None:
            lines.append(f"**賺**：到期标的 **> {be:.2f}**")
            lines.append(f"**蝕**：到期标的 **< {be:.2f}**")
        lines.append(f"**最大利潤**：标的 ≥ {short_k:.0f} ≈ ${idea.max_profit:.0f}/张")
        lines.append(f"**最大亏损**：标的 ≤ {long_k:.0f}（权利金全损）≈ ${idea.max_loss:.0f}/张")
    elif code == "bear_put":
        long_k = max(lg.strike for lg in idea.legs if lg.side == "buy")
        short_k = min(lg.strike for lg in idea.legs if lg.side == "sell")
        lines.append(f"结构：Bear Put Debit（买 {long_k:.0f} Put / 卖 {short_k:.0f} Put）")
        if be is not None:
            lines.append(f"**賺**：到期标的 **< {be:.2f}**")
            lines.append(f"**蝕**：到期标的 **> {be:.2f}**")
        lines.append(f"**最大利潤**：标的 ≤ {short_k:.0f} ≈ ${idea.max_profit:.0f}/张")
        lines.append(f"**最大亏损**：标的 ≥ {long_k:.0f} ≈ ${idea.max_loss:.0f}/张")
    else:
        if be is not None:
            if pnl_high > pnl_low:
                lines.append(f"**賺**倾向标的走高；打和点 ≈ {be:.2f}")
            else:
                lines.append(f"**賺**倾向标的走低；打和点 ≈ {be:.2f}")

    lines.append(f"现价 {spot:.2f} 若到期不变 → 盈亏约 ${payoff_per_contract(idea, spot):.0f}/张")
    return {
        "lines": lines,
        "breakeven": be,
        "spot_pnl": round(payoff_per_contract(idea, spot), 2),
        "max_profit": idea.max_profit,
        "max_loss": idea.max_loss,
    }


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_option_price(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    right: str,
    r: float = 0.04,
) -> float:
    """Black-Scholes European option mid estimate."""
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return 0.0
    if t_years <= 1e-8:
        if right == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    vol = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / vol
    d2 = d1 - vol
    if right == "call":
        return float(spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2))
    return float(strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1))


def spread_mark_value(
    idea: SpreadIdea,
    spot: float,
    dte_left: int,
    sigma: float,
    r: float = 0.04,
) -> float:
    """
    价差「市值标价」$/股（未×100）：
    - 借方(debit)：多头价值 = 买腿价 - 卖腿价  → 正数
    - 信用(credit)：开仓收的权利金市值 = 卖腿价 - 买腿价  → 正数表示仍可收这么多
    """
    t = max(dte_left, 0) / 365.0
    sig = max(0.08, min(float(sigma), 0.9))
    total = 0.0
    for leg in idea.legs:
        px = bs_option_price(spot, leg.strike, t, sig, leg.right, r=r)
        if leg.side == "buy":
            total += px
        else:
            total -= px
    # debit spread: total > 0 (long premium)
    # credit spread when opened: we received -total at open if total is negative for short-heavy...
    # For credit: sell high mid, buy low mid → entry net_credit = short - long > 0
    # Mark to open same structure now: short_bs - long_bs
    if idea.net_credit is not None:
        # reconstruct credit mark = -total if total = long - short
        return float(-total)
    return float(total)


def build_daily_mark_calendar(
    idea: SpreadIdea,
    spot: float,
    sigma: float | None,
    dte_total: int | None = None,
    hold_days: int | None = None,
    spot_path: str = "flat",
    r: float = 0.04,
) -> pd.DataFrame:
    """
    从今天(第0日)起，逐日估算价差标价与相对入场的盈亏。

    spot_path:
      - flat: 标的价一直等于现价
      - +1% / -1% / +2% / -2%: 标的线性走到该幅度（到 hold 末日）
    """
    dte0 = int(dte_total if dte_total is not None else idea.dte or 30)
    dte0 = max(dte0, 1)
    # 用户可只看前 N 日（例如买14日）
    n = int(hold_days if hold_days is not None else dte0)
    n = max(1, min(n, dte0))

    sig = sigma if sigma and sigma > 0 else 0.20
    sig = max(0.08, min(float(sig), 0.9))

    # 入场标价：优先用链上 mid 合成
    if idea.net_credit is not None:
        entry_mark = float(idea.net_credit)
    elif idea.net_debit is not None:
        entry_mark = float(idea.net_debit)
    else:
        entry_mark = abs(spread_mark_value(idea, spot, dte0, sig, r=r))

    # 目标末日标的
    if spot_path == "flat":
        end_spot = spot
    elif spot_path.startswith("+"):
        end_spot = spot * (1 + float(spot_path[1:].replace("%", "")) / 100.0)
    elif spot_path.startswith("-"):
        end_spot = spot * (1 - float(spot_path[1:].replace("%", "")) / 100.0)
    else:
        end_spot = spot

    today = date.today()
    rows = []
    for day in range(0, n + 1):
        dte_left = max(dte0 - day, 0)
        # 线性插值标的路径
        if n > 0:
            s_t = spot + (end_spot - spot) * (day / n)
        else:
            s_t = spot
        mark = spread_mark_value(idea, s_t, dte_left, sig, r=r)
        mark = max(mark, 0.0)

        # 盈亏 $/张
        if idea.net_credit is not None:
            # 信用：入场收 entry，平仓付 mark → 赚 (entry - mark)*100
            pnl = (entry_mark - mark) * 100.0
            entry_label = entry_mark
        else:
            # 借方：入场付 entry，平仓收 mark → 赚 (mark - entry)*100
            pnl = (mark - entry_mark) * 100.0
            entry_label = entry_mark

        if pnl > 0.5:
            result = "賺"
        elif pnl < -0.5:
            result = "蝕"
        else:
            result = "打和"

        rows.append(
            {
                "第N日": day,
                "日期": (today + timedelta(days=day)).isoformat(),
                "剩余DTE": dte_left,
                "假设标的价": round(s_t, 2),
                "相对现价%": round((s_t / spot - 1) * 100, 2),
                "价差标价$/股": round(mark, 2),
                "入场标价$/股": round(entry_label, 2),
                "标价差$/股": round(mark - entry_label, 2),
                "浮动盈亏$/张": round(pnl, 2),
                "结果": result,
            }
        )
    return pd.DataFrame(rows)


def legs_to_frame(idea: SpreadIdea) -> pd.DataFrame:
    rows = []
    for leg in idea.legs:
        bid = float(leg.bid or 0)
        ask = float(leg.ask or 0)
        mid = float(leg.mid or 0)
        fill = float(getattr(leg, "fill", 0) or mid)
        fill_src = getattr(leg, "fill_source", "") or "mid"
        if bid > 0 and ask > 0 and mid > 0:
            ba = f"{(ask - bid) / mid * 100:.0f}%"
        else:
            ba = "—"
        src_zh = {"bid": "买价bid", "ask": "卖价ask", "mid": "中间价"}.get(
            fill_src, fill_src
        )
        rows.append(
            {
                "方向": "卖出" if leg.side == "sell" else "买入",
                "类型": "看涨" if leg.right == "call" else "看跌",
                "行权价": leg.strike,
                "成交估": round(fill, 2),
                "成交用": src_zh,
                "中间价": round(mid, 2),
                "买价": round(bid, 2),
                "卖价": round(ask, 2),
                "买卖差": ba,
                "未平仓": int(leg.oi or 0),
                "成交量": int(leg.volume or 0),
            }
        )
    return pd.DataFrame(rows)


def ideas_to_frame(ideas: list[SpreadIdea], sort_by: str = "score") -> pd.DataFrame:
    rows = []
    for i in ideas:
        wr_p = getattr(i, "win_rate_profit", None)
        if wr_p is None:
            wr_p = getattr(i, "pop_est", None)
        rows.append(
            {
                "分数": i.score,
                "名称": i.name,
                "赢面%": wr_p,
                "流动性": getattr(i, "liquidity_label", "") or "—",
                "流动性分": getattr(i, "liquidity_score", None),
                "卖出收$": i.net_credit,
                "买进付$": i.net_debit,
                "最多赚$": i.max_profit,
                "最多亏$": i.max_loss,
                "像哪种做法": getattr(i, "playbook_style", ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if sort_by == "winrate":
        df = df.sort_values("赢面%", ascending=False, na_position="last")
    elif sort_by == "liquidity":
        df = df.sort_values("流动性分", ascending=False, na_position="last")
    else:
        df = df.sort_values("分数", ascending=False, na_position="last")
    return df.reset_index(drop=True)
