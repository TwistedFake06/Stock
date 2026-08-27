"""個股技術圖表畫面。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from screens.common import disclaimer, sidebar_controls


def render_chart() -> None:
    st.title("個股圖表")
    disclaimer()
    universe, _, features, failed = sidebar_controls()
    if failed:
        st.warning("下載失敗：" + ", ".join(failed))
    ticker = st.selectbox(
        "代碼",
        list(universe),
        format_func=lambda symbol: f"{symbol} · {universe[symbol]}",
        key="workbench_chart_ticker",
    )
    frame = features[features["Ticker"] == ticker].tail(180).copy()
    if frame.empty:
        st.warning("沒有可用歷史資料。")
        return
    for days in (5, 10, 20):
        frame[f"MA{days}"] = frame["Close"].rolling(days).mean()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.52, .16, .16, .16], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=frame.Date, open=frame.Open, high=frame.High, low=frame.Low, close=frame.Close, name="K線"), row=1, col=1)
    for column, color in (("MA5", "#f0b429"), ("MA10", "#3a86ff"), ("MA20", "#e63946")):
        fig.add_trace(go.Scatter(x=frame.Date, y=frame[column], name=column, line={"color": color}), row=1, col=1)
    fig.add_trace(go.Bar(x=frame.Date, y=frame.Volume, name="成交量"), row=2, col=1)
    fig.add_trace(go.Bar(x=frame.Date, y=frame.macd_hist, name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=frame.Date, y=frame.macd, name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=frame.Date, y=frame.macd_signal, name="Signal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=frame.Date, y=frame.rsi_14, name="RSI(14)"), row=4, col=1)
    fig.add_hline(y=70, row=4, col=1, line_dash="dot"); fig.add_hline(y=30, row=4, col=1, line_dash="dot")
    fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    latest = frame.iloc[-1]
    high, low = frame.High.tail(20).max(), frame.Low.tail(20).min()
    position = (latest.Close - low) / (high - low) if high > low else 0
    cols = st.columns(4)
    cols[0].metric("現價", f"{latest.Close:.2f}"); cols[1].metric("20 日高低位置", f"{position:.1%}")
    cols[2].metric("量比", f"{latest.vol_ratio_20:.2f}" if pd.notna(latest.vol_ratio_20) else "-")
    cols[3].metric("20 日線", "站上" if latest.ma20_dist >= 0 else "未站上")
    st.caption("規則訊號：" + ("量能配合且在 20 日線上" if latest.vol_ratio_20 >= 1 and latest.ma20_dist >= 0 else "等待較清晰的量價條件") + "；只作顯示，不會自動下單。")
    st.dataframe(frame[["Date", "Open", "High", "Low", "Close", "Volume"]].tail(10).iloc[::-1], width="stretch", hide_index=True)