"""Streamlit page: 技术分析."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import macd_chart, price_volume_chart, rsi_chart
from indicators import enrich
from stock_service import fetch_history
from ui_mobile import plotly_chart as mobile_plotly


def render_technical(
    symbol: str,
    period: str,
    interval: str,
    period_label: str,
    interval_label: str,
) -> None:
    st.header(f"技术分析 · `{symbol}`")

    with st.spinner("计算指标..."):
        hist = fetch_history(symbol, period=period, interval=interval)

    if hist.empty:
        st.warning(f"未能获取 `{symbol}` 数据。")
        return

    df = enrich(hist)
    show_sma = st.checkbox("显示均线 SMA", value=True)
    show_bb = st.checkbox("显示布林带", value=True)
    show_vol = st.checkbox("显示成交量", value=True)

    fig = price_volume_chart(
        df,
        title=f"{symbol} · {period_label} · {interval_label}",
        show_sma=show_sma,
        show_bb=show_bb,
        show_volume=show_vol,
    )
    mobile_plotly(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        mobile_plotly(rsi_chart(df), width="stretch")
    with c2:
        mobile_plotly(macd_chart(df), width="stretch")

    last = df.iloc[-1]
    st.subheader("最新指标快照")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("收盘", f"{float(last['Close']):.2f}")

    rsi_v = last["RSI"] if "RSI" in df.columns else None
    k2.metric("RSI", f"{float(rsi_v):.1f}" if rsi_v is not None and pd.notna(rsi_v) else "—")

    macd_v = last["MACD_HIST"] if "MACD_HIST" in df.columns else None
    k3.metric(
        "MACD柱",
        f"{float(macd_v):.4f}" if macd_v is not None and pd.notna(macd_v) else "—",
    )

    sma20 = last["SMA20"] if "SMA20" in df.columns else None
    k4.metric("SMA20", f"{float(sma20):.2f}" if sma20 is not None and pd.notna(sma20) else "—")
