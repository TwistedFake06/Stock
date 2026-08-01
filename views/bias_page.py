"""Streamlit page: 多空分析."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import analyze_bias
from charts import macd_chart, price_volume_chart, rsi_chart
from indicators import enrich
from stock_service import compute_returns, fetch_history
from ui_mobile import plotly_chart as mobile_plotly
from views.common import fmt_pct, render_bias_banner


def render_bias_page(
    symbol: str,
    period: str,
    interval: str,
    period_label: str,
    interval_label: str,
) -> None:
    st.header(f"多空分析 · `{symbol}`")
    st.caption(
        f"基于当前所选周期：{period_label} · {interval_label} · "
        "综合均线 / MACD / RSI / 布林带 / 动量 / 量能"
        " · 完整多因子模型（与 GitHub Pages Lite 简化版阈值相同、信号更全）"
    )

    with st.spinner("分析多空..."):
        hist = fetch_history(symbol, period=period, interval=interval)

    if hist.empty:
        st.warning(f"未能获取 `{symbol}` 数据，无法分析。")
    else:
        df = enrich(hist)
        report = analyze_bias(df)
        render_bias_banner(report, compact=False)

        st.subheader("价格与关键指标")
        fig = price_volume_chart(
            df,
            title=f"{symbol} · 多空参考图",
            show_sma=True,
            show_bb=True,
            show_volume=True,
        )
        mobile_plotly(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            mobile_plotly(rsi_chart(df), use_container_width=True)
        with col_b:
            mobile_plotly(macd_chart(df), use_container_width=True)

        st.subheader("信号明细表")
        rows = [
            {
                "指标": s.name,
                "方向": s.bias,
                "权重分": round(s.score, 2),
                "依据": s.detail,
            }
            for s in report.signals
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        rets = compute_returns(df)
        m1, m2, m3 = st.columns(3)
        m1.metric("区间收益率", fmt_pct(rets.get("total_return_pct")))
        m2.metric("年化波动率(估)", fmt_pct(rets.get("volatility_pct")))
        m3.metric("多空得分", f"{report.score:+.1f}", report.bias)
