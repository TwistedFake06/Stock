"""Shared UI helpers for Streamlit views (Cloud + local)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import BiasReport, bias_color, bias_emoji
from charts import bias_gauge
import stock_service as _stock_service

# Tolerate stale Streamlit workers that cached an older stock_service
if not hasattr(_stock_service, "filter_us_only"):
    import importlib

    _stock_service = importlib.reload(_stock_service)

DEFAULT_WATCHLIST = _stock_service.DEFAULT_WATCHLIST
filter_us_only = _stock_service.filter_us_only

from ui_mobile import metric_grid, plotly_chart as mobile_plotly

WATCHLIST_FILE = Path(__file__).resolve().parent.parent / "watchlist.json"

# Public re-export (also available from stock_service)
__all__ = [
    "filter_us_only",
    "fmt_number",
    "fmt_pct",
    "get_price_fields",
    "html_plain",
    "load_watchlist",
    "render_bias_banner",
    "save_watchlist",
]


def html_plain(text: object) -> str:
    """Strip markdown bold markers and HTML-escape for unsafe_allow_html blocks."""
    return html.escape(str(text or "").replace("**", ""), quote=True)


def render_bias_banner(report: BiasReport, compact: bool = False) -> None:
    """Render multi/空 conclusion card."""
    color = bias_color(report.bias)
    emoji = bias_emoji(report.bias)
    summary = html_plain(report.summary)
    bias_safe = html_plain(report.bias)
    conf_safe = html_plain(report.confidence)
    if compact:
        st.markdown(
            f"""
            <div class="bias-card">
              <p class="bias-title" style="color:{color};">{emoji} {bias_safe}
                <span style="font-size:1rem;color:#90a4ae;font-weight:500;">
                  · 得分 {report.score:+.0f} · 置信度 {conf_safe}
                </span>
              </p>
              <p class="bias-sub">{summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Stack vertically for phone; still fine on desktop
    st.markdown(
        f"""
        <div class="bias-card">
          <p class="bias-title" style="color:{color};">{emoji} 综合判断：{bias_safe}</p>
          <p class="bias-sub">{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mobile_plotly(bias_gauge(report.score, report.bias))
    st.caption(
        f"多头 **{report.bull_count}** · 空头 **{report.bear_count}** · "
        f"中性 **{report.neutral_count}** · 置信度 **{report.confidence}**"
    )
    snap = report.snapshot or {}
    metric_grid(
        [
            ("收盘", f"{snap['close']:.2f}" if snap.get("close") is not None else "—"),
            ("RSI", f"{snap['rsi']:.1f}" if snap.get("rsi") is not None else "—"),
            (
                "MACD柱",
                f"{snap['macd_hist']:.4f}" if snap.get("macd_hist") is not None else "—",
            ),
            (
                "布林位置",
                f"{snap.get('bb_position'):.0%}" if snap.get("bb_position") is not None else "—",
            ),
        ],
        per_row=2,
    )
    with st.expander("分项信号", expanded=False):
        for sig in report.signals:
            if sig.bias == "看多":
                cls, mark = "sig-bull", "▲ 看多"
            elif sig.bias == "看空":
                cls, mark = "sig-bear", "▼ 看空"
            else:
                cls, mark = "sig-flat", "● 中性"
            name = html_plain(sig.name)
            detail = html_plain(sig.detail)
            st.markdown(
                f"<span class='{cls}'>{mark}</span> · <strong>{name}</strong> — {detail}",
                unsafe_allow_html=True,
            )

    st.info(
        "实盘辅助参考 · **不构成投资建议** · 下单以券商与自身风控为准。"
    )


def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        try:
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                us = filter_us_only([str(x) for x in data])
                if us:
                    return us
        except Exception:
            pass
    return list(DEFAULT_WATCHLIST)


def save_watchlist(items: list[str]) -> None:
    """Persist watchlist when filesystem allows (local); session always holds truth."""
    items = filter_us_only([str(x) for x in items])
    try:
        WATCHLIST_FILE.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Streamlit Cloud / read-only mounts: keep session_state only
        pass


def fmt_number(val, prefix: str = "", suffix: str = "", digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        n = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(n) >= 1e12:
        return f"{prefix}{n / 1e12:.{digits}f}T{suffix}"
    if abs(n) >= 1e8:
        return f"{prefix}{n / 1e8:.{digits}f}亿{suffix}"
    if abs(n) >= 1e6:
        return f"{prefix}{n / 1e6:.{digits}f}M{suffix}"
    return f"{prefix}{n:,.{digits}f}{suffix}"


def fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{float(val):+.2f}%"


def get_price_fields(info: dict) -> dict:
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("last_price")
        or info.get("previousClose")
        or info.get("previous_close")
    )
    prev = info.get("previousClose") or info.get("previous_close") or info.get("regularMarketPreviousClose")
    change = None
    change_pct = None
    if price is not None and prev is not None and prev != 0:
        change = float(price) - float(prev)
        change_pct = change / float(prev) * 100
    return {
        "price": price,
        "prev": prev,
        "change": change,
        "change_pct": change_pct,
        "open": info.get("open") or info.get("regularMarketOpen") or info.get("open"),
        "high": info.get("day_high") or info.get("dayHigh") or info.get("regularMarketDayHigh"),
        "low": info.get("day_low") or info.get("dayLow") or info.get("regularMarketDayLow"),
        "year_high": info.get("year_high") or info.get("fiftyTwoWeekHigh"),
        "year_low": info.get("year_low") or info.get("fiftyTwoWeekLow"),
        "mcap": info.get("market_cap") or info.get("marketCap"),
        "pe": info.get("trailingPE") or info.get("forwardPE"),
        "currency": info.get("currency") or "",
        "name": info.get("shortName") or info.get("longName") or info.get("_symbol", ""),
        "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
    }
