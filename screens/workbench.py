"""短炒工作台頁面入口。"""
from __future__ import annotations

import streamlit as st


def render_workbench() -> None:
    st.sidebar.divider()
    page = st.sidebar.radio("工作台頁面", ["今日候選", "個股圖表", "模型打分", "簡易回測"], key="workbench_page")
    if page == "今日候選":
        from screens.screener import render_screener
        render_screener()
    elif page == "個股圖表":
        from screens.chart import render_chart
        render_chart()
    elif page == "模型打分":
        from screens.scoring import render_scoring
        render_scoring()
    else:
        from screens.backtest import render_backtest
        render_backtest()