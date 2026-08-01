"""Option chain fetch, parse, fill pricing, expiry pick, hard filters."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from options_models import Leg, SpreadIdea

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
    # Use positional argmin — never mix label index with iloc
    return _row_nearest_strike(df, target)


def _row_nearest_strike(df: pd.DataFrame, target: float) -> pd.Series:
    """Closest strike row; index-safe (works after boolean filters)."""
    pos = int((df["strike"].astype(float) - float(target)).abs().to_numpy().argmin())
    return df.iloc[pos]


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
