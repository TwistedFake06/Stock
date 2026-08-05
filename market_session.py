"""
US equity session clock + pre/post/overnight quote helpers (Yahoo free data).

Sessions (America/New_York, Mon–Fri):
  - pre_market   04:00–09:30  盘前
  - rth          09:30–16:00  常规
  - after_hours  16:00–20:00  盘后
  - overnight    20:00–04:00  夜盘/隔夜（部分券商可交易；Yahoo 报价常稀疏）
  - closed       周末/节假日粗判

Data: yfinance info fields (preMarket*, postMarket*, regularMarket*) +
optional 1m/5m history with prepost=True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from stock_service import fetch_history_extended, get_ticker, normalize_symbol


@dataclass
class SessionClock:
    """Current US equity session snapshot."""

    session: str  # pre_market | rth | after_hours | overnight | closed | unknown
    label_zh: str
    et_now: str
    weekday: int
    is_weekend: bool
    note: str = ""


@dataclass
class ExtendedQuote:
    """Regular + pre + post (and best live) from Yahoo info."""

    symbol: str
    session: SessionClock
    # Regular session
    regular_price: float | None = None
    regular_change: float | None = None
    regular_change_pct: float | None = None
    regular_prev_close: float | None = None
    regular_open: float | None = None
    regular_high: float | None = None
    regular_low: float | None = None
    regular_volume: float | None = None
    # Pre-market
    pre_price: float | None = None
    pre_change: float | None = None
    pre_change_pct: float | None = None
    pre_time: str | None = None
    # Post / after-hours
    post_price: float | None = None
    post_change: float | None = None
    post_change_pct: float | None = None
    post_time: str | None = None
    # Best display price for "now"
    live_price: float | None = None
    live_label: str = "—"
    live_change: float | None = None
    live_change_pct: float | None = None
    currency: str = ""
    name: str = ""
    bid: float | None = None
    ask: float | None = None
    notes: list[str] = field(default_factory=list)
    available: bool = False


def _et_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # rough fallback UTC-4
        return datetime.utcnow() - timedelta(hours=4)


def us_session_clock(now: datetime | None = None) -> SessionClock:
    """Classify current ET wall-clock into US equity session buckets."""
    n = now or _et_now()
    # strip tz for simple arithmetic display
    try:
        et_str = n.strftime("%Y-%m-%d %H:%M ET")
    except Exception:
        et_str = str(n)
    wd = int(n.weekday())
    weekend = wd >= 5
    mins = int(n.hour) * 60 + int(n.minute)

    if weekend:
        return SessionClock(
            session="closed",
            label_zh="休市（周末）",
            et_now=et_str,
            weekday=wd,
            is_weekend=True,
            note="美股周末休市；盘前/盘后数据可能仍显示上一个交易日残留报价。",
        )

    # Mon–Fri schedule
    if 4 * 60 <= mins < 9 * 60 + 30:
        return SessionClock(
            session="pre_market",
            label_zh="盘前 Pre-Market",
            et_now=et_str,
            weekday=wd,
            is_weekend=False,
            note="04:00–09:30 ET · 流动性薄，价差大，谨慎下单。",
        )
    if 9 * 60 + 30 <= mins < 16 * 60:
        return SessionClock(
            session="rth",
            label_zh="常规 RTH",
            et_now=et_str,
            weekday=wd,
            is_weekend=False,
            note="09:30–16:00 ET · 主交易时段。",
        )
    if 16 * 60 <= mins < 20 * 60:
        return SessionClock(
            session="after_hours",
            label_zh="盘后 After-Hours",
            et_now=et_str,
            weekday=wd,
            is_weekend=False,
            note="16:00–20:00 ET · 盘后波动大，期权 bid/ask 易失真。",
        )
    # 20:00–04:00 overnight
    return SessionClock(
        session="overnight",
        label_zh="夜盘 / 隔夜",
        et_now=et_str,
        weekday=wd,
        is_weekend=False,
        note="20:00–04:00 ET · 部分券商支持隔夜交易；Yahoo 免费数据常不完整或延迟。",
    )


def _f(val: Any) -> float | None:
    if val is None:
        return None
    try:
        x = float(val)
        if x != x:
            return None
        return x
    except (TypeError, ValueError):
        return None


def _fmt_ts(val: Any) -> str | None:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)) and val > 1e9:
            # unix seconds
            dt = datetime.utcfromtimestamp(int(val))
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        return str(val)[:19]
    except Exception:
        return None


def parse_extended_quote(symbol: str, info: dict[str, Any] | None) -> ExtendedQuote:
    """Pull pre/regular/post fields from a yfinance-like info dict."""
    info = info or {}
    sym = normalize_symbol(symbol)
    clock = us_session_clock()
    notes: list[str] = []

    reg = _f(
        info.get("regularMarketPrice")
        or info.get("currentPrice")
        or info.get("last_price")
    )
    prev = _f(
        info.get("regularMarketPreviousClose")
        or info.get("previousClose")
        or info.get("previous_close")
    )
    reg_chg = _f(info.get("regularMarketChange"))
    reg_pct = _f(info.get("regularMarketChangePercent"))
    if reg_pct is not None and abs(reg_pct) < 1 and abs(reg_pct) > 0:
        # sometimes fraction 0.01 = 1%
        if abs(reg_pct) <= 1.0 and reg is not None and prev and prev > 0:
            # ambiguous; if |pct|<=1 and |chg/prev*100| is larger, scale
            pass
    # Yahoo often returns changePercent already as percent (e.g. 1.23) OR as fraction
    if reg_pct is not None and abs(reg_pct) <= 1.0 and reg is not None and prev and prev > 0:
        implied = (reg - prev) / prev * 100
        if abs(implied) > 1.5 * abs(reg_pct * 100) or abs(reg_pct) < 0.5:
            # prefer fraction→percent when looks like fraction
            if abs(reg_pct) <= 0.2:
                reg_pct = reg_pct * 100
    if reg_chg is None and reg is not None and prev is not None:
        reg_chg = reg - prev
    if reg_pct is None and reg is not None and prev and prev != 0:
        reg_pct = (reg - prev) / prev * 100

    pre = _f(info.get("preMarketPrice"))
    pre_chg = _f(info.get("preMarketChange"))
    pre_pct = _f(info.get("preMarketChangePercent"))
    if pre_pct is not None and abs(pre_pct) <= 1.0:
        # fraction style
        if abs(pre_pct) < 0.5:
            pre_pct = pre_pct * 100
    if pre is not None and prev and pre_chg is None:
        pre_chg = pre - prev
    if pre is not None and prev and prev != 0 and pre_pct is None:
        pre_pct = (pre - prev) / prev * 100

    post = _f(info.get("postMarketPrice"))
    post_chg = _f(info.get("postMarketChange"))
    post_pct = _f(info.get("postMarketChangePercent"))
    if post_pct is not None and abs(post_pct) <= 1.0 and abs(post_pct) < 0.5:
        post_pct = post_pct * 100
    if post is not None and reg is not None and post_chg is None:
        post_chg = post - reg
    if post is not None and reg and reg != 0 and post_pct is None:
        post_pct = (post - reg) / reg * 100

    # Live pick by session
    live = reg
    live_label = "常规"
    live_chg = reg_chg
    live_pct = reg_pct
    if clock.session == "pre_market" and pre is not None:
        live, live_label, live_chg, live_pct = pre, "盘前", pre_chg, pre_pct
    elif clock.session in ("after_hours", "overnight") and post is not None:
        live, live_label, live_chg, live_pct = post, "盘后", post_chg, post_pct
    elif clock.session == "overnight" and post is None and pre is not None:
        live, live_label, live_chg, live_pct = pre, "盘前(残留)", pre_chg, pre_pct
    elif clock.session == "rth" and reg is not None:
        live, live_label, live_chg, live_pct = reg, "常规", reg_chg, reg_pct
    elif post is not None:
        live, live_label, live_chg, live_pct = post, "盘后", post_chg, post_pct
    elif pre is not None:
        live, live_label, live_chg, live_pct = pre, "盘前", pre_chg, pre_pct

    if pre is None and post is None:
        notes.append("Yahoo 未返回盘前/盘后价（常见于部分标的或延迟源）。")
    if clock.session == "overnight":
        notes.append("夜盘时段 Yahoo 免费源常无独立报价，多显示盘后最后价。")
    if clock.session in ("pre_market", "after_hours", "overnight"):
        notes.append("扩展时段流动性低：宜限价、减小仓位；与常规收盘价可能偏差大。")

    available = any(x is not None for x in (reg, pre, post, live))
    return ExtendedQuote(
        symbol=sym,
        session=clock,
        regular_price=reg,
        regular_change=reg_chg,
        regular_change_pct=reg_pct,
        regular_prev_close=prev,
        regular_open=_f(info.get("regularMarketOpen") or info.get("open")),
        regular_high=_f(
            info.get("regularMarketDayHigh") or info.get("dayHigh") or info.get("day_high")
        ),
        regular_low=_f(
            info.get("regularMarketDayLow") or info.get("dayLow") or info.get("day_low")
        ),
        regular_volume=_f(info.get("regularMarketVolume") or info.get("volume")),
        pre_price=pre,
        pre_change=pre_chg,
        pre_change_pct=pre_pct,
        pre_time=_fmt_ts(info.get("preMarketTime")),
        post_price=post,
        post_change=post_chg,
        post_change_pct=post_pct,
        post_time=_fmt_ts(info.get("postMarketTime")),
        live_price=live,
        live_label=live_label,
        live_change=live_chg,
        live_change_pct=live_pct,
        currency=str(info.get("currency") or ""),
        name=str(info.get("shortName") or info.get("longName") or sym),
        bid=_f(info.get("bid")),
        ask=_f(info.get("ask")),
        notes=notes,
        available=available,
    )


def fetch_extended_quote(symbol: str, info: dict[str, Any] | None = None) -> ExtendedQuote:
    """
    Build ExtendedQuote. If info missing fields, try a fresh ticker.info pull
    (caller usually passes cached_info).
    """
    sym = normalize_symbol(symbol)
    data = dict(info or {})
    # If pre/post missing, try one light refresh of info
    need = not any(
        k in data and data.get(k) is not None
        for k in ("preMarketPrice", "postMarketPrice", "regularMarketPrice")
    )
    if need or (
        data.get("preMarketPrice") is None and data.get("postMarketPrice") is None
    ):
        try:
            t = get_ticker(sym)
            raw = t.info or {}
            if isinstance(raw, dict):
                # Prefer existing non-null; fill gaps
                for k, v in raw.items():
                    if v is not None and (k not in data or data.get(k) is None):
                        data[k] = v
        except Exception:
            pass
    return parse_extended_quote(sym, data)


def extended_intraday(
    symbol: str,
    *,
    period: str = "5d",
    interval: str = "5m",
) -> pd.DataFrame:
    """OHLCV including pre/post if Yahoo allows (prepost=True)."""
    return fetch_history_extended(symbol, period=period, interval=interval)
